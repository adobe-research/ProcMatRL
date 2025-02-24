# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights
from torchvision.models.vgg import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self, vgg_path):
        super(FeatureExtractor, self).__init__()

        # load pre-trained VGG
        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load(vgg_path))
        self.vgg.requires_grad_(False)

        # normalizer
        imagenet_mean = torch.as_tensor((0.40760392, 0.45795686, 0.48501961))[None, :, None, None]
        self.register_buffer('imagenet_mean', imagenet_mean)

    def normalize(self, image):
        image = image[:, [2, 1, 0], :, :]  # turn to BGR
        image = image - self.imagenet_mean  # subtract imagenet mean
        image = image * 255.0
        return image

    def forward(self, im, layers, detach=False):
        if im.shape[1] == 1:
            im = torch.cat((im, im, im), dim=1)

        im = self.normalize(im)
        out_features = self.vgg(im, layers, detach)

        return out_features


class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, detach):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])

        if not detach:
            return [out[key] for key in out_keys]
        else:
            return [out[key].detach() for key in out_keys]


class VGGTextureDescriptor(nn.Module):
    """Texture descriptor evaluation based on a pretrained VGG19 network.
    """
    def __init__(self, device='cpu', td_level=2):
        """Initialize the texture descriptor evaluator.

        Args:
            device (DeviceType, optional): Device placement of the texture descriptor network.
                Defaults to 'cpu'.
        """
        super().__init__()
        self.device = device
        self.td_level = td_level
        self.img_size = 224

        # Record intermediate results from the feature extraction network to compute the texture
        # descriptor
        self.features = []

        # Set up the feature extraction network
        self._setup_model()

        # Image statistics for normalizing an input texture
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(-1, 1, 1)

    def _setup_model(self):
        """Initialize the texture feature extraction model.
        """
        # Get a pretrained VGG19 network and set it to evaluation state
        model: nn.Sequential = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device)
        model.eval()

        # Disable network training
        for param in model.parameters():
            param.requires_grad_(False)

        # Change the max pooling to average pooling
        for i, module in enumerate(model):
            if isinstance(module, nn.MaxPool2d):
                model[i] = nn.AvgPool2d(kernel_size=2)

        # The forward hook function for capturing output at a certain network layer
        def forward_hook(module, input, output):
            self.features.append(output)

        # Register the forward hook function
        for i in (4, 9, 18, 27):
            model[i].register_forward_hook(forward_hook)

        self.model = model

    def _texture_descriptor(self, img):
        """Compute the texture descriptor of an input image of shape `(B, C, H, W)`.

        Args:
            img (Tensor): A mini-batch of images.

        Returns:
            Tensor: Texture descriptors of input images, in a shape of `(B, feature_size)`.
        """
        # Normalize the input image
        img = (img - self.mean) / self.std

        # Run the VGG feature extraction network
        self.features.clear()
        self.features.append(self.model(img))

        def gram_matrix(img_feature):
            mat = img_feature.flatten(-2)
            gram = torch.matmul(mat, mat.transpose(-2, -1)) / mat.shape[-1]
            return gram.flatten(1)

        # Compute the Gram matrices using recorded features
        # The feature descriptor has a shape of (B, F), where F is feature length
        return torch.cat([gram_matrix(img_feature) for img_feature in self.features], dim=1)

    def forward(self, img):
        """Compute the texture descriptor of an input image at multiple scales.

        Args:
            img (Tensor): A mini-batch of images whose shape is `(B, C, H, W)`.

        Raises:
            ValueError: Texture descriptor level is not an integer or holds a negative value.

        Returns:
            Tensor: Multi-scale texture descriptors of input images, with a shape of
                `(B, feature_size * (td_level + 1))`.
        """
        if not isinstance(self.td_level, int) or self.td_level < 0:
            raise ValueError('The texture descriptor level must be a non-negative integer')

        # Verify input image resolution
        if img.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f'Input image resolution must be {self.img_size}x{self.img_size}')

        # Compute the texture descriptor at native resolution
        img = img.contiguous()
        tds = [self._texture_descriptor(img)]

        # Repeat for downscaled images
        for level in range(1, self.td_level + 1):
            img_scaled = F.avg_pool2d(img, kernel_size=2**level, stride=2**level)
            tds.append(self._texture_descriptor(img_scaled))

        td_output = torch.cat(tds, dim=1)

        return td_output
