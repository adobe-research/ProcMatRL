# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os.path as pth
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip

from .gpt2 import Conv1D

allow_downsample_tensor = True


def get_clip_dim(model_name):
    if model_name == 'ViT-B/32':
        return 512
    elif model_name == 'ViT-L/14':
        return 768
    else:
        raise RuntimeError(f'Unsupported CLIP model: {model_name}')


class PostProcessor(nn.Module):
    def __init__(self, n_embedding, clip_dim, embed_type):
        super().__init__()
        self.embed_type = embed_type
        self.n_embedding = n_embedding

        if embed_type == 'project':
            self.ln = Conv1D(n_embedding, clip_dim)
        elif embed_type == 'project_resize':
            self.ln = Conv1D(clip_dim, clip_dim)
        else:
            raise ValueError(f'Unknown CLIP embedding preprocessing type: {embed_type}')

    def forward(self, x):
        x = self.ln(x)
        if self.embed_type == 'project_resize':
            x = x.view(*x.shape[:-1], -1, self.n_embedding)
        return x


class CLIPImageEncoder(nn.Module):
    def __init__(self, n_embedding, model_name, device, embed_type=None, normalize=False, bypass_clip=False):
        super().__init__()
        self.clip_encoder = CLIPEncoder(model_name=model_name, device=device) if not bypass_clip else nn.Identity()
        self.normalize = normalize
        self.device = device

        # preprocess CLIP embeddings if needed
        if embed_type is not None:
            self.post_processor = PostProcessor(n_embedding, get_clip_dim(model_name), embed_type)
        else:
            self.post_processor = None

    def _process_embedding(self, x):
        if self.post_processor is not None:
            x = self.post_processor(x)
        return x

    def forward(self, x):
        x = self.clip_encoder(x).float()
        if self.normalize:
            x = F.normalize(x, dim=-1)

        return self._process_embedding(x)


class CLIPImageMultiModalEncoder(CLIPImageEncoder):

    def forward(self, x):
        with torch.no_grad():
            if isinstance(x[0], str):
                embedding = self.clip_encoder(x).float()
            else:  # x[0] is MultiModalInput
                text_prompts, images, index = [], [], []
                for y in x:
                    if isinstance(y.data, str):
                        text_prompts.append(y.data)
                        index.append((True, len(text_prompts) - 1))
                    else:
                        images.append(y.data)
                        index.append((False, len(images) - 1))

                # compute text embedding
                text_embedding = self.clip_encoder(text_prompts).float()
                if self.normalize:
                    text_embedding = F.normalize(text_embedding)

                # compute image embedding
                if len(images) > 0:
                    images = torch.stack(images, dim=0).to(self.device)
                    image_embedding = self.clip_encoder(images).float()
                    if self.normalize:
                        image_embedding = F.normalize(image_embedding)
                else:
                    image_embedding = None

                embedding = []
                for idx in index:
                    embedding.append(text_embedding[idx[1]] if idx[0] else image_embedding[idx[1]])
                embedding = torch.stack(embedding, dim=0)

        return self._process_embedding(embedding)


class CLIPLoss(nn.Module):
    def __init__(self, encoder, y=None, metric='cos', reduce=True):
        super(CLIPLoss, self).__init__()

        self.encoder = encoder
        if y is None:
            self.y_clip = None
        else:
            self.y_clip = self.encoder(y)

        reduction = 'mean' if reduce else 'none'
        if metric == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif metric == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif metric == 'cos':
            self.criterion = CosineLoss(reduction=reduction)
        else:
            raise NotImplementedError(f"Unknown loss type: {metric}")

    def forward(self, x, y=None):
        x_clip = self.encoder(x)
        y_clip = self.encoder(y) if y is not None else self.y_clip
        if y_clip.shape[0] != 1:
            raise RuntimeError('Target should be a single sample.')
        if x_clip.shape[0] != y_clip.shape[0]:
            y_clip = y_clip.expand_as(x_clip)
        return self.criterion(x_clip, y_clip)


class CLIPEncoder(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.model_name = model_name
        self.device = device

        # load model and preprocessor
        self.model, self.pil_preprocess = clip.load(model_name, device)
        self.model.requires_grad_(False)
        self.tensor_preprocess = Preprocessor(model_name, device)

    def forward(self, x):
        if isinstance(x, Image.Image):
            return self.model.encode_image(self.pil_preprocess(x).unsqueeze(0).to(self.device))
        elif isinstance(x, torch.Tensor):
            if x.ndim == 4:
                return self.model.encode_image(self.tensor_preprocess(x))
            elif x.ndim == 2:
                return x
            else:
                raise RuntimeError(f'Tensor dimension is incorrect: {x.ndim} (accept 2 or 4)')
        elif isinstance(x, (tuple, list, str)):
            return self.model.encode_text(clip.tokenize(x).to(self.device))
        else:
            raise RuntimeError(f'Unknown input type: {type(x)}')


class Preprocessor(nn.Module):
    def __init__(self, model_name, device):
        super(Preprocessor, self).__init__()

        # resize = T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=None)
        # center_crop = T.CenterCrop(size=(224, 224))
        # normalized = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        if model_name in ['ViT-B/32', 'ViT-L/14']:
            self.size = 224
            mean = torch.as_tensor((0.48145466, 0.4578275, 0.40821073), device=device)[None, :, None, None]
            std = torch.as_tensor((0.26862954, 0.26130258, 0.27577711), device=device)[None, :, None, None]
        else:
            raise RuntimeError(f"Unknown model name {model_name}")

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        if x.shape[2] != self.size and x.shape[3] != self.size:
            # warnings.warn('F.interpolate may lead to significant difference in CLIP performance.')
            if allow_downsample_tensor:
                x = F.interpolate(x, size=(self.size, self.size), mode='bicubic', antialias=True)
                x = x.clamp_(0.0, 1.0)
            else:
                raise RuntimeError('F.interpolate is disabled. Please downsample the image using PIL')
        x = (x - self.mean) / self.std
        return x


class CosineLoss(nn.Module):
    def __init__(self, reduction):
        super(CosineLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.reduction = reduction
        assert self.reduction in ['mean', 'none'], f'Unknown reduction type: {self.reduction}'

    def forward(self, x, y):
        z = 1 - self.cos(x, y)
        if self.reduction == 'mean':
            z = torch.mean(z)
        return z


def classify(image_filenames, text_desc, classify_image2text='image'):
    model_name = 'ViT-B/32'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    images = []
    for image_filename in image_filenames:
        image = preprocess(Image.open(image_filename)).unsqueeze(0).to(device)
        images.append(image)
    images = torch.cat(images, dim=0)
    text = clip.tokenize(text_desc).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(images, text)

    np.set_printoptions(precision=3, suppress=True)
    if classify_image2text:
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        for image_filename, prob in zip(image_filenames, probs):
            print(f'{pth.basename(image_filename)}: ', prob)

    else:
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        for txt, prob in zip(text_desc, probs):
            print(f'{txt}: ', prob)
