# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.filterwarnings('ignore', '', FutureWarning)

import torch
from torch import nn
from torch.nn.attention import SDPBackend
from torch.nn.functional import scaled_dot_product_attention

from .gpt2_activations import ACT2FN

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def prune_conv1d_layer(layer, index, dim=1):
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer (:class:`~transformers.modeling_utils.Conv1D`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 1): The dimension on which to keep the indices.

    Returns:
        :class:`~transformers.modeling_utils.Conv1D`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, is_causal=None, is_cross_attention=False, ctx_size=None, use_fast_attn=None, scale=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.is_causal = not is_cross_attention and (config.is_causal if is_causal is None else is_causal)
        if self.is_causal:
            self.register_buffer(
                "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.bool)).view(1, 1, n_ctx, n_ctx)
            )
        else:
            self.register_buffer(
                "bias", torch.ones((n_ctx, n_ctx), dtype=torch.bool).view(1, 1, n_ctx, n_ctx)
            )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        if is_cross_attention:
            self.c_attn = Conv1D(n_state * 2, nx if ctx_size is None else ctx_size)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(nx, n_state)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

        # fast attention optimization
        self._fast_attn = None

        if config.use_fast_attn if use_fast_attn is None else use_fast_attn:
            def mem_eff_attn(*args, **kwargs):
                with torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    res = scaled_dot_product_attention(*args, dropout_p=config.attn_pdrop if self.training else 0.0, is_causal=self.is_causal, **kwargs)
                return res

            self._fast_attn = mem_eff_attn

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        if self.is_causal:
            nd, ns = w.size(-2), w.size(-1)
            if nd > ns:
                raise ValueError(f"Error in self-attention: nd > ns ({nd} > {ns}) in causal attention.")
            mask = self.bias[:, :, ns - nd : ns, :ns]
            w = torch.where(mask, w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self, x, layer_past=None, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(x)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(x).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        # fast attention using the latest PyTorch API
        if (self._fast_attn and head_mask is None and not output_attentions
            and (not self.is_causal or query.size(-2) == key.size(-1))):
            try:
                a = self._fast_attn(query, key.transpose(-2, -1), value, attn_mask=attention_mask)
                attn_outputs = []
            except RuntimeError as e:
                if e.args[0].startswith('No available kernel'):
                    a, *attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
                else:
                    raise

        # report error if fast attention is not available
        elif self._fast_attn:
            if head_mask is not None:
                raise RuntimeError("Cannot apply head mask when using fast attention.")
            elif output_attentions:
                raise RuntimeError("Cannot return attention weights when using fast attention.")
            else:
                raise RuntimeError(f'Fast attention is not available for this attention layer '
                                   f'(is_causal={self.is_causal}, query={tuple(query.shape)}, key={tuple(key.shape)}).')

        # fall back to manual attention
        else:
            a, *attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return (a, present, *attn_outputs)  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, nf, nx, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nf, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, add_cross_attention=False, scale=False):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale=scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

        # all cross attentions are full
        if add_cross_attention:
            self.ln_cross_attn = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
            self.cross_attn = Attention(nx, n_ctx, config, is_causal=False, is_cross_attention=True, scale=scale)

        self.mlp = MLP(4 * nx, nx, nx, config)

    def _attn_block(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        output_attn = self.attn(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)
        x = x + a
        return (x, *output_attn[1:])

    def _cross_attn_block(
        self, x, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False
    ):
        output_cross_attn = self.cross_attn(
            self.ln_cross_attn(x),
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions
        )
        a = output_cross_attn[0]
        x = x + a
        return (x, *output_cross_attn[2:])

    def _ff_block(self, x):
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x

    def forward(
        self, x, layer_past=None, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False
    ):
        x, *outputs_attn = self._attn_block(
            x, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask,
            use_cache=use_cache, output_attentions=output_attentions)

        if encoder_hidden_states is not None:
            if not hasattr(self, 'cross_attn'):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `add_cross_attention=True`"
                )
            x, *outputs_cross_attn = self._cross_attn_block(
                x, attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
            outputs_attn += outputs_cross_attn

        x = self._ff_block(x)

        return (x, *outputs_attn)  # x, present, (attentions, cross_attentions)


class CondBlock(Block):
    """Transformer encoder/decoder layer with image attention (method: feed-forward).
    """
    def __init__(self, n_ctx, config, add_cross_attention=False, scale=False):
        super().__init__(n_ctx, config, add_cross_attention=add_cross_attention, scale=scale)

        # add an additional MLP branch for input conditioning
        nx, n_cond = config.n_embd, config.n_cond
        self.ln_cond = nn.LayerNorm(n_cond, eps=config.layer_norm_epsilon)
        self.mlp_cond = MLP(4 * nx, nx, n_cond, config)

    def forward(
        self, x, cond, layer_past=None, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False
    ):
        x, *outputs_attn = super().forward(
            x, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache, output_attentions=output_attentions)

        # apply the additional MLP branch for input conditioning
        m_cond = self.mlp_cond(self.ln_cond(cond))
        x = x + m_cond

        return (x, *outputs_attn)


class CrossAttnCondBlock(Block):
    """Transformer encoder/decoder layer with image attention (method: cross-attention).
    """
    def __init__(self, n_ctx, config, add_cross_attention=False, scale=False):
        super().__init__(n_ctx, config, add_cross_attention=add_cross_attention, scale=scale)

        # add an additional cross attention block for input conditioning
        nx, n_cond = config.n_embd, config.n_cond
        self.ln_cond = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.cross_attn_cond = Attention(nx, n_ctx, config, is_causal=False, is_cross_attention=True, ctx_size=n_cond, scale=scale)

    def _attn_block(self, x_cond, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        x, cond = x_cond
        x, *outputs_attn = super()._attn_block(x, layer_past, attention_mask, head_mask, use_cache, output_attentions)

        # apply cross attention to x and cond
        outputs_cross_attn = self.cross_attn_cond(
            self.ln_cond(x), attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=cond,
            output_attentions=output_attentions)
        a = outputs_cross_attn[0]
        x = x + a
        outputs_attn += outputs_cross_attn[2:]

        return (x, *outputs_attn)

    def forward(
        self, x, cond, layer_past=None, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False
    ):
        outputs = super().forward(
            (x, cond), layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache, output_attentions=output_attentions)

        return outputs


class MergedCrossAttnCondBlock(Block):
    """Transformer encoder/decoder layer with image attention (method: merged cross-attention).
    """
    def __init__(self, n_ctx, config, add_cross_attention=False, scale=False):
        if not add_cross_attention:
            raise RuntimeError("'add_cross_attention=True' is required for a merged cross-attention conditional block")

        super().__init__(n_ctx, config, add_cross_attention=add_cross_attention, scale=scale)

    def forward(
        self, x, cond, layer_past=None, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False
    ):
        if cond is not None:
            raise ValueError("Please pass an explicit 'cond=None' argument to call a merged cross-attention conditional block")

        outputs = super().forward(
            x, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache, output_attentions=output_attentions)

        return outputs
