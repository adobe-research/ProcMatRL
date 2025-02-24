# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import math
import torch
from torch import nn

from .gpt2 import Conv1D, CondBlock, CrossAttnCondBlock, MergedCrossAttnCondBlock

from torch_geometric.data import Data as GraphData, Batch as GraphBatch
from torch_geometric.nn import GCNConv


class CondTransformer(nn.Module):

    BLOCK_TYPE_DICT = {
        'feed_forward': CondBlock,
        'cross_attention': CrossAttnCondBlock,
        'merged_cross_attention': MergedCrossAttnCondBlock
    }

    def __init__(self, config):
        super().__init__()
        self.config = config

        if len(config.seq_vocab_sizes) < 1:
            raise RuntimeError('Must provide one vocabulary size for each sequence.')
        if len(config.output_seq_dims) < 1:
            raise RuntimeError('Must provide dimensions for at least one output sequence.')

        # input embeddings
        self.sequence_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.n_embd) if vocab_size is not None else None
            for vocab_size in config.seq_vocab_sizes
        ])
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer layers
        if config.cond_type not in self.BLOCK_TYPE_DICT:
            raise ValueError(f'Unrecognized input conditioning method: {config.cond_type}')
        BlockType = self.BLOCK_TYPE_DICT[config.cond_type]

        self.add_cross_attention = config.is_decoder
        self.h = nn.ModuleList([
            BlockType(
                n_ctx=config.n_positions,
                config=config,
                add_cross_attention=config.is_decoder or config.cond_type == 'merged_cross_attention',
                scale=True
            ) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, sum(config.output_seq_dims), bias=sum(config.output_seq_dims) > 1)

        self.init_weights()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

    def get_input_embeddings(self):
        return self.sequence_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.sequence_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        sequences,
        cond=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        output_log_probs=None,
        semantic_masks=None,
        action_masks=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
            If `past` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True``) is passed or when ``config.output_hidden_states=True``:
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        # check input validity
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if cond is not None:
            cond = cond.unsqueeze(dim=1) if cond.ndim == 2 else cond
            if cond.ndim < 2:
                raise ValueError('Input condition must be a 2D tensor at least.')
        if past is None:
            past = [None] * len(self.h)

        if labels is not None and len(labels) != len(self.config.output_seq_dims):
            raise RuntimeError(f'Expected {len(self.config.output_seq_dims)} label sequences but got {len(labels)}.')

        if len(sequences) != len(self.sequence_embeddings):
            raise RuntimeError(f'Expected {len(self.sequence_embeddings)} sequences but got {len(sequences)}.')
        if sequences[0].ndim < 2:
            raise RuntimeError('Sequences must be of shape batch size x sequence length [x embedding dimension (only when passing embedded sequences)].')
        if sequences[0].shape[0] == 0:
            raise RuntimeError('Batch size must be > 0.')

        if encoder_hidden_states is not None and not self.add_cross_attention:
            raise RuntimeError(
                'If `encoder_hidden_states` are passed, the Transformer module has to be instantiated with '
                'cross-attention layers by setting `is_decoder=True` in model config.')

        # aggregate input sequence embeddings
        batch_size, seq_len = sequences[0].shape[:2]
        next_param = next(self.sequence_embeddings.parameters(), sequences[0])
        dtype, device = next_param.dtype, next_param.device

        hidden_states = torch.zeros(batch_size, seq_len, self.config.n_embd, dtype=dtype, device=device)
        for si, sequence in enumerate(sequences):
            if self.sequence_embeddings[si] is None:
                if sequence.shape != (batch_size, seq_len, self.config.n_embd):
                    raise RuntimeError(f'Embedded sequence {si} has incorrect shape {tuple(sequence.shape)}.')
                hidden_states = hidden_states + sequence
            else:
                if sequence.shape != (batch_size, seq_len):
                    raise RuntimeError(f'Sequence {si} has incorrect shape {tuple(sequence.shape)}.')
                if sequence.max().item() >= self.sequence_embeddings[si].num_embeddings:
                    raise RuntimeError(f'Sequence {si} has out-of-bound token {sequence.max().item()} >= {self.sequence_embeddings[si].num_embeddings}.')
                hidden_states = hidden_states + self.sequence_embeddings[si](sequence)

        # dropout
        hidden_states = self.drop(hidden_states)

        # attention mask
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise RuntimeError('The sequence mask must have shape  (batch size x max. sequence length)')
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            if attention_mask.dtype != torch.int32:
                raise RuntimeError('Currently only int32 attention masks are supported.')
            attention_mask = (1.0 - attention_mask) * -10000.0

        # special case for merged cross attention
        if self.config.cond_type == 'merged_cross_attention':
            if encoder_hidden_states is None:
                encoder_hidden_states = cond
                encoder_attention_mask = None
            else:
                # check if the condition and the encoder sequence have the same embedding size
                if cond.size(-1) != encoder_hidden_states.size(-1):
                    raise ValueError(f'The input condition has a dimension of {cond.size(-1)} but the encoder sequence '
                                     f'has a dimension of {encoder_hidden_states.size(-1)}')

                # concatenate input condition and encoder hidden states
                encoder_sequence_length = encoder_hidden_states.size(-2)
                encoder_hidden_states = torch.cat((cond, encoder_hidden_states), dim=-2)

                # augment the encoder attention mask
                if encoder_attention_mask is not None:
                    if encoder_attention_mask.shape != (batch_size, encoder_sequence_length):
                        print(encoder_hidden_states.shape, encoder_attention_mask.shape)
                        raise ValueError('The encoder attention mask must have a shape of (batch size x max. encoder sequence length)')
                    cond_attention_mask = torch.ones((batch_size, cond.size(-2)), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)
                    encoder_attention_mask = torch.cat((cond_attention_mask, encoder_attention_mask), dim=-1)

            # leave input conditioning as empty (required)
            cond = None

        # encoder attention mask
        if self.add_cross_attention and encoder_hidden_states is not None and encoder_attention_mask is not None:
            if encoder_attention_mask.shape != (batch_size, encoder_hidden_states.size(-2)):
                raise RuntimeError('The encoder sequence mask must have shape (batchs size x max. encoder sequence length)')
            if encoder_attention_mask.dtype != torch.int32:
                raise RuntimeError('Currently only int32 attention masks are supported')
            encoder_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = [None] * self.config.n_layer #self.get_head_mask(head_mask, self.config.n_layer)

        output_shape = (batch_size, seq_len, hidden_states.size(-1))

        presents = ()
        all_attentions = []
        all_cross_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states, cond,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])
                if self.add_cross_attention:
                    all_cross_attentions.append(outputs[3])

        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.lm_head(hidden_states)

        hidden_states = hidden_states.view(batch_size, seq_len, sum(self.config.output_seq_dims))

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # split output into separate sequences
        hidden_states = torch.split(hidden_states, self.config.output_seq_dims, dim=-1)

        # collect primary outputs
        outputs = (hidden_states,)

        if labels is not None:
            # compute loss
            loss = torch.zeros(batch_size, len(hidden_states), seq_len - 1, dtype=dtype, device=device)
            for si, (hs, label) in enumerate(zip(hidden_states, labels)):
                # shift since ground truth output is the input sequence shifted one step to the left
                # output_seq_logits = hs[:, :-1, :].contiguous() # omit last token (always a padded token past the end of the sequence)
                # label_seq = label[:, 1:].contiguous() # omit start token (the start token is never an output of the transformer)
                # loss[:, si, :] = nn.functional.cross_entropy(
                #     input=output_seq_logits.view(-1, output_seq_logits.size(-1)),
                #     target=label_seq.view(-1),
                #     reduction='none').view(batch_size, seq_len-1)
                loss_seq = nn.functional.cross_entropy(hs[:, :-1].transpose(-2, -1), label[:, 1:], reduction='none')
                loss[:, si, :] = loss_seq

            outputs = (loss,) + outputs

            # compute log probabilities and entropy
            if output_log_probs:
                raw_log_probs, log_probs, entropy = tuple(torch.zeros_like(loss) for _ in range(3))
                for si, (hs, label) in enumerate(zip(hidden_states, labels)):
                    # apply semantic masks
                    hs = torch.where(semantic_masks[si], hs[:, :-1], -1e9) if semantic_masks is not None else hs[:, :-1]
                    log_probs_seq = hs.log_softmax(dim=-1)
                    raw_log_probs[:, si, :] = log_probs_seq.gather(-1, label[:, 1:, None]).squeeze(-1)

                    # apply action masks
                    if action_masks is not None:
                        hs = torch.where(action_masks[si], hs, -1e9)
                        log_probs_seq = hs.log_softmax(dim=-1)
                        log_probs[:, si, :] = log_probs_seq.gather(-1, label[:, 1:, None]).squeeze(-1)
                    else:
                        log_probs[:, si, :] = raw_log_probs[:, si, :]

                    # compute entropy
                    entropy[:, si, :] = -torch.sum(torch.exp(log_probs_seq) * log_probs_seq, dim=-1)

                outputs += (raw_log_probs.sum(dim=1), log_probs.sum(dim=1), entropy.mean(dim=1))

        # other auxiliary outputs
        if use_cache is True:
            outputs += (presents,)

        if output_hidden_states:
            outputs += (all_hidden_states,)

        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = (batch_size, -1) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs += (all_attentions,)

            if self.add_cross_attention:
                cross_attention_output_shape = (batch_size, -1) + all_cross_attentions[0].shape[-2:]
                all_cross_attentions = tuple(t.view(*cross_attention_output_shape) for t in all_cross_attentions)
                outputs += (all_cross_attentions,)

        return outputs  # (loss[optional], last hidden state, log probs[optional], *other auxiliary outputs)


class ImageCondEncoder(CondTransformer):
    """Conditional Transformer encoder for compatibility purposes.
    """
    def __init__(self, config):
        if config.is_decoder:
            raise ValueError("Conditional Transformer encoder must be instantiated with 'is_decoder=False'")
        super().__init__(config)

    def forward(
        self, sequences, cond, past=None, attention_mask=None, head_mask=None,
        use_cache=None, output_attentions=None, output_hidden_states=None, labels=None,
        output_log_probs=None, semantic_masks=None, action_masks=None
    ):
        return super().forward(
            sequences, cond=cond, past=past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, labels=labels,
            output_log_probs=output_log_probs, semantic_masks=semantic_masks, action_masks=action_masks
        )


class ImageCondPointerNet(nn.Module):

    def __init__(self, node_enc_config, pointer_gen_config, use_inner_product=True, check_semantic_validity=False):
        super().__init__()
        self.node_encoder = ImageCondEncoder(node_enc_config)
        self.pointer_generator = CondTransformer(pointer_gen_config)

        self.is_encoder_decoder = pointer_gen_config.is_decoder
        self.use_inner_product = use_inner_product
        self.check_semantic_validity = check_semantic_validity

    def compute_node_embeddings(self, node_sequences, cond=None, node_attention_mask=None):
        # embed the nodes (no positional encoding)
        node_embed = self.node_encoder(sequences=node_sequences, cond=cond, attention_mask=node_attention_mask)[0][0] # bs, len, embd
        node_embed = node_embed * node_attention_mask.unsqueeze(dim=-1) # zero out padded node embeddings (is this necessary?)

        return node_embed

    def forward(self,
                node_sequences,
                edge_sequences,
                cond=None,
                labels=None,
                node_attention_mask=None,
                edge_attention_mask=None,
                output_log_probs=None,
                semantic_masks=None,
                action_masks=None):

        if isinstance(node_sequences, (list, tuple)):
            # node sequences given to be embedded
            node_embed = self.compute_node_embeddings(node_sequences=node_sequences, cond=cond, node_attention_mask=node_attention_mask)
        else:
            # node embedding given directly
            node_embed = node_sequences

        # get node embeddings for the edge (start,end) sequence
        edge_seq_node_embed = torch.gather(node_embed, 1, edge_sequences[0].unsqueeze(-1).repeat((1, 1, node_embed.shape[-1])).long())
        edge_sequences = (edge_seq_node_embed, *edge_sequences[1:])

        # get pointers that can be compared to the node embeddings
        decoder_kwargs = {
            'encoder_hidden_states': node_embed,
            'encoder_attention_mask': node_attention_mask
        } if self.is_encoder_decoder else {}
        pointers = self.pointer_generator(edge_sequences, cond=cond, attention_mask=edge_attention_mask, **decoder_kwargs)[0][0] # bs, edg_len, embd

        # compare pointers to the node embeddings
        if self.use_inner_product:
            inner_prod = torch.matmul(pointers, node_embed.permute(0, 2, 1))

            if self.check_semantic_validity:
                # do not allow edges with padded node embeddings
                inner_prod[~(node_attention_mask.unsqueeze(dim=1).expand(-1, inner_prod.shape[1], -1).bool())] = -1e9

        # directly use pointers as logits
        else:
            inner_prod = pointers

        outputs = (inner_prod,)

        # compute loss
        batch_size, seq_len = inner_prod.shape[:2]

        if labels is not None:
            # shift since ground truth output is the input sequence shifted one step to the left
            # logits = inner_prod[:, :-1, :].contiguous() # omit last token (always a padded token past the end of the sequence)
            # labels = labels[:, 1:].contiguous() # omit start token (the start token is never an output of the transformer)
            # loss = nn.functional.cross_entropy(input=logits.view(-1, logits.size(-1)), target=labels.view(-1), reduction='none').view(batch_size, seq_len-1)
            loss = nn.functional.cross_entropy(inner_prod[:, :-1].transpose(-2, -1), labels[:, 1:], reduction='none')
            loss = loss.view(batch_size, seq_len-1)
            outputs = (loss,) + outputs

            # compute log probabilities and entropy
            if output_log_probs:
                # apply semantic masks
                inner_prod = torch.where(semantic_masks, inner_prod[:, :-1], -1e9) if semantic_masks is not None else inner_prod[:, :-1]
                log_probs_seq = inner_prod.log_softmax(dim=-1)
                raw_log_probs = log_probs_seq.gather(-1, labels[:, 1:, None]).squeeze(-1)

                # apply action masks
                if action_masks is not None:
                    inner_prod = torch.where(action_masks, inner_prod, -1e9)
                    log_probs_seq = inner_prod.log_softmax(dim=-1)
                    log_probs = log_probs_seq.gather(-1, labels[:, 1:, None]).squeeze(-1)
                else:
                    log_probs = raw_log_probs.clone()

                entropy = -torch.sum(torch.exp(log_probs_seq) * log_probs_seq, dim=-1)

                outputs += (raw_log_probs, log_probs, entropy)

        return outputs


class EdgeConditionedNodeEncoder(nn.Module):
    def __init__(self, node_enc_config, node_seq_stop_token, n_gcn):
        super().__init__()
        self.node_encoder = ImageCondEncoder(node_enc_config)
        self.node_seq_stop_token = node_seq_stop_token

        self.n_gcn = n_gcn
        self.gconv = nn.ModuleList()
        gcn_dim = node_enc_config.output_seq_dims[0]
        for _ in range(n_gcn):
            gcn = GCNConv(gcn_dim, gcn_dim)
            self.gconv.append(gcn)

    def compile(self):
        self.node_encoder = torch.compile(self.node_encoder, dynamic=True)
        # TODO: compile gconv
        return self

    def gen_graph_data_from_list_of_edges(self, node_sequences, node_embed, edge_node_inds):
        graph_data_list = []
        for bi in range(len(edge_node_inds)):
            stop_token_inds = torch.where(node_sequences[0][bi] == self.node_seq_stop_token)[0]

            if len(stop_token_inds) == 0:
                node_count = len(node_sequences[0][bi])
            else:
                node_count = stop_token_inds.min().item()
            if len(edge_node_inds[bi]) > 0:
                graph_edge_node_inds = torch.tensor(edge_node_inds[bi]).t().to(node_embed.device)  # edge_count x 2 -> 2 x edge_count
                graph_edge_node_inds = torch.cat([graph_edge_node_inds, graph_edge_node_inds[[1, 0]]], dim=-1)  # add edges in other direction to get bidirectional edges
            else:
                graph_edge_node_inds = torch.zeros((2, 0), dtype=torch.long, device=node_embed.device)
            graph_data_list.append(GraphData(x=node_embed[bi, :node_count], edge_index=graph_edge_node_inds))

        return graph_data_list

    def gen_graph_data_from_padded_edge_tensors(self, node_sequences, node_embed, edge_node_inds):
        graph_data_list = []
        batch_size = edge_node_inds.shape[0]
        for bi in range(batch_size):
            stop_token_inds = torch.where(node_sequences[0][bi] == self.node_seq_stop_token)[0]

            if len(stop_token_inds) == 0:
                node_count = len(node_sequences[0][bi])
            else:
                node_count = stop_token_inds.min().item()

            # get unpadded edge_node_ind
            padded_edge_node_ind = edge_node_inds[bi]
            # check for correctness
            assert padded_edge_node_ind[-1][0] == padded_edge_node_ind[-1][1]
            edge_len = padded_edge_node_ind[-1][0]
            if edge_len > 0:
                unpadded_edge_node_ind = padded_edge_node_ind[:edge_len] # [edge_count, 2]
                graph_edge_node_inds = unpadded_edge_node_ind.t().to(node_embed.device)  # edge_count x 2 -> 2 x edge_count
                graph_edge_node_inds = torch.cat([graph_edge_node_inds, graph_edge_node_inds[[1, 0]]], dim=-1)  # add edges in other direction to get bidirectional edges
            else:
                graph_edge_node_inds = torch.zeros((2, 0), dtype=torch.long, device=node_embed.device)
            graph_data_list.append(GraphData(x=node_embed[bi, :node_count], edge_index=graph_edge_node_inds))

        return graph_data_list

    def forward(self, node_sequences, edge_node_inds, cond=None, node_attention_mask=None):
        node_embed = self.node_encoder(sequences=node_sequences, cond=cond, attention_mask=node_attention_mask)[0][0]

        # compute local graph neighborhood features
        # (node sequences don't have a start token, so no offset of the edge_node_inds is needed)

        # convert graph batch to torch_geometric format
        # (in this format all graphs in the batch are merged into disconnected components of one large graph)
        if isinstance(edge_node_inds, (list, tuple)):
            graph_data_list = self.gen_graph_data_from_list_of_edges(node_sequences, node_embed, edge_node_inds)
        elif isinstance(edge_node_inds, torch.Tensor):
            graph_data_list = self.gen_graph_data_from_padded_edge_tensors(node_sequences, node_embed, edge_node_inds)
        else:
            raise RuntimeError(f'Unknown type of edge_node_inds: {type(edge_node_inds)}')

        # graph_batch.batch:[n_nodes, ]
        # graph_batch.x: [n_nodes, n_hidden]
        # graph_batch.edge_index:[2, n_edges]
        graph_batch = GraphBatch.from_data_list(graph_data_list)

        # perform graph convolutions to update node embeddings
        for i in range(self.n_gcn):
            graph_batch.x = self.gconv[i](graph_batch.x, graph_batch.edge_index)

        # add updated node embeddings as residuals to the node original node embeddings
        node_embed_skip = node_embed
        node_embed = node_embed.clone()
        for bi in range(len(edge_node_inds)):
            batch_updated_node_embed = graph_batch.x[graph_batch.batch == bi]
            node_embed[bi, :batch_updated_node_embed.shape[0]] = node_embed_skip[bi, :batch_updated_node_embed.shape[0]] + batch_updated_node_embed

        return node_embed


class ImageCondGraphConditionalParamEncoder(nn.Module):

    def __init__(self, node_enc_config, generator_config, node_seq_stop_token, n_gcn):
        super().__init__()
        if not generator_config.conditional or generator_config.cond_vocab_size is not None:
            raise RuntimeError('The generator must be configured to be conditional without embedding of the condition.')

        self.node_encoder = EdgeConditionedNodeEncoder(node_enc_config, node_seq_stop_token, n_gcn)
        self.param_decoder = CondTransformer(generator_config)
        self.is_encoder_decoder = generator_config.is_decoder

    def compile(self):
        self.node_encoder = self.node_encoder.compile()
        self.param_decoder = torch.compile(self.param_decoder, dynamic=True)
        return self

    def compute_node_embeddings(self, node_sequences, edge_node_inds, cond, node_attention_mask):
        return self.node_encoder(node_sequences=node_sequences, edge_node_inds=edge_node_inds,
                                 cond=cond, node_attention_mask=node_attention_mask)

    # cond is expected to be per graph if node_subset_inds is given, otherwise per node
    def forward(self,
                node_sequences,
                edge_node_inds,
                gen_sequences,
                param_node_inds,
                cond=None,
                labels=None,
                node_attention_mask=None,
                gen_attention_mask=None,
                output_log_probs=None,
                semantic_masks=None,
                action_masks=None):

        if isinstance(node_sequences, (list, tuple)):
            # node sequences given to be embedded
            node_embed = self.compute_node_embeddings(node_sequences=node_sequences, edge_node_inds=edge_node_inds,
                                                      cond=cond, node_attention_mask=node_attention_mask)
        else:
            # node embedding given directly
            node_embed = node_sequences

        # assemble node embedding sequences
        node_embedding_sequence = torch.take_along_dim(node_embed, param_node_inds.unsqueeze(2), dim=1)

        decoder_kwargs = {
            'output_log_probs': output_log_probs,
            'semantic_masks': semantic_masks,
            'action_masks': action_masks,
            **({
                'encoder_hidden_states': node_embed,
                'encoder_attention_mask': node_attention_mask,
            } if self.is_encoder_decoder else {})
        }
        outputs = self.param_decoder(
            (node_embedding_sequence, *gen_sequences), cond=cond, labels=labels,
            attention_mask=gen_attention_mask, **decoder_kwargs)

        return outputs
