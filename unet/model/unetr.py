from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet import Decoder, ConvBlock2d, Upsampler2d
from model.vit import get_vision_transformer

#
# UNETR IMPLEMENTATION [Vision Transformer (ViT from IJEPA) + UNet Decoder from `torch_em`]
#

class FeatAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # bs, seq_len, dims = x.shape
        x = x.permute((0, 2, 1))
        return self.avg_pool(x).squeeze()

class UNETR(nn.Module):

    def _load_encoder_from_checkpoint(self, backbone, encoder, checkpoint):
        encoder_state = checkpoint
        self.encoder.load_state_dict(encoder_state)

    def __init__(
        self,
        img_size: int = 1024,
        backbone: str = "ijepa",
        encoder: Optional[Union[nn.Module, str]] = "vit_b",
        decoder: Optional[nn.Module] = None,
        out_channels: int = 1,
        encoder_checkpoint: Optional[Union[str, OrderedDict]] = None,
        final_activation: Optional[Union[str, nn.Module]] = None,
        use_skip_connection: bool = True,
        embed_dim: Optional[int] = None,
        use_conv_transpose=True,
        task="segmentation",
        type_cls="HP"
    ) -> None:
        super().__init__()
        self.task = task
        self.type_cls = type_cls
        self.use_skip_connection = use_skip_connection
        self.avg_pool = FeatAvgPool()

        if isinstance(encoder, str):  # "vit_b" / "vit_l" / "vit_h"
            print(f"Using {encoder} from {backbone.upper()}")
            self.encoder = get_vision_transformer(img_size=img_size, backbone=backbone, model=encoder)
            if encoder_checkpoint is not None:
                self._load_encoder_from_checkpoint(backbone, encoder, encoder_checkpoint)

            in_chans = self.encoder.in_chans
            if embed_dim is None:
                embed_dim = self.encoder.embed_dim

        else:  # `nn.Module` ViT backbone
            self.encoder = encoder

            have_neck = False
            for name, _ in self.encoder.named_parameters():
                if name.startswith("neck"):
                    have_neck = True

            if embed_dim is None:
                if have_neck:
                    embed_dim = self.encoder.neck[2].out_channels  # the value is 256
                else:
                    embed_dim = self.encoder.patch_embed.proj.out_channels

            try:
                in_chans = self.encoder.patch_embed.proj.in_channels
            except AttributeError:  # for getting the input channels while using vit_t from MobileSam
                in_chans = self.encoder.patch_embed.seq[0].c.in_channels

        # parameters for the decoder network
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        # choice of upsampler - to use (bilinear interpolation + conv) or conv transpose
        _upsampler = SingleDeconv2DBlock if use_conv_transpose else Upsampler2d

        if decoder is None:
            self.decoder = Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=ConvBlock2d,
                sampler_impl=_upsampler
            )
        else:
            self.decoder = decoder

        if use_skip_connection:
            self.deconv1 = Deconv2DBlock(embed_dim, features_decoder[0])
            self.deconv2 = nn.Sequential(
                Deconv2DBlock(embed_dim, features_decoder[0]),
                Deconv2DBlock(features_decoder[0], features_decoder[1])
            )
            self.deconv3 = nn.Sequential(
                Deconv2DBlock(embed_dim, features_decoder[0]),
                Deconv2DBlock(features_decoder[0], features_decoder[1]),
                Deconv2DBlock(features_decoder[1], features_decoder[2])
            )
            self.deconv4 = ConvBlock2d(in_chans, features_decoder[-1])
        else:
            self.deconv1 = Deconv2DBlock(embed_dim, features_decoder[0])
            self.deconv2 = Deconv2DBlock(features_decoder[0], features_decoder[1])
            self.deconv3 = Deconv2DBlock(features_decoder[1], features_decoder[2])
            self.deconv4 = Deconv2DBlock(features_decoder[2], features_decoder[3])

        self.base = ConvBlock2d(embed_dim, features_decoder[0])

        self.out_conv = nn.Conv2d(features_decoder[-1], self.out_channels, 1)

        self.deconv_out = _upsampler(
            scale_factor=2, in_channels=features_decoder[-1], out_channels=features_decoder[-1]
        )

        self.decoder_head = ConvBlock2d(2 * features_decoder[-1], features_decoder[-1])

        # self.fc3 = nn.Linear(768, 256)
        self.fc3_1 = nn.Linear(768, 1)

        # self.fc1 = nn.Linear(768, 512)
        self.fc1_1 = nn.Linear(768, 10)

        # self.fc2 = nn.Linear(768, 512)
        self.fc2_1 = nn.Linear(768, 7)

        self.final_activation = self._get_activation(final_activation)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def _get_activation(self, activation):
        return_activation = None
        if activation is None:
            return None
        if isinstance(activation, nn.Module):
            return activation
        if isinstance(activation, str):
            return_activation = getattr(nn, activation, None)
        if return_activation is None:
            raise ValueError(f"Invalid activation: {activation}")
        return return_activation()

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.encoder.img_size, self.encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, x):
        # original_shape = x.shape[-2:]

        use_skip_connection = getattr(self, "use_skip_connection", True)

        if self.task == "segmentation":
            encoder_outputs = self.encoder(x, return_features=True)
        else:
            encoder_outputs = self.encoder(x)

        if isinstance(encoder_outputs[-1], list):
            # `encoder_outputs` can be arranged in only two forms:
            #   - either we only return the image embeddings
            #   - or, we return the image embeddings and the "list" of global attention layers
            z12, from_encoder = encoder_outputs
        else:
            z12 = encoder_outputs
        if self.task == "segmentation":
            if use_skip_connection:
                from_encoder = from_encoder[::-1]
                z9 = self.deconv1(from_encoder[0])
                z6 = self.deconv2(from_encoder[1])
                z3 = self.deconv3(from_encoder[2])
                z0 = self.deconv4(x)

            else:
                z9 = self.deconv1(z12)
                z6 = self.deconv2(z9)
                z3 = self.deconv3(z6)
                z0 = self.deconv4(z3)

            updated_from_encoder = [z9, z6, z3]

            x = self.base(z12)
            x = self.decoder(x, encoder_inputs=updated_from_encoder)
            x = self.deconv_out(x)

            x = torch.cat([x, z0], dim=1)
            x = self.decoder_head(x)

            x = self.out_conv(x)
            if self.final_activation is not None:
                x = self.final_activation(x)

            # x = self.postprocess_masks(x, input_shape, original_shape)
            return x
        elif self.task == "classification":
            if self.type_cls == "HP":
                # print(f"z12.shape: {z12.shape}")
                x = self.avg_pool(z12)
                # print(f"z12.shape: {z12.shape}")
                # x = self.fc3(x)
                x = self.fc3_1(x)
                return x
            elif self.type_cls == "vitri":
                x = self.avg_pool(z12)
                # x = self.fc1(x)
                x = self.fc1_1(x)
                return x
        elif self.task == "multitask":
            if use_skip_connection:
                from_encoder = from_encoder[::-1]
                z9 = self.deconv1(from_encoder[0])
                z6 = self.deconv2(from_encoder[1])
                z3 = self.deconv3(from_encoder[2])
                z0 = self.deconv4(x)

            else:
                z9 = self.deconv1(z12)
                z6 = self.deconv2(z9)
                z3 = self.deconv3(z6)
                z0 = self.deconv4(z3)

            updated_from_encoder = [z9, z6, z3]

            out_1 = self.avg_pool(z12)
            # out_1 = self.fc1(out_1)
            out_1 = self.fc1_1(out_1)

            out_2 = self.avg_pool(z12)
            # out_2 = self.fc2(out_2)
            out_2 = self.fc2_1(out_2)

            out_3 = self.avg_pool(z12)
            # out_3 = self.fc3(out_3)
            out_3 = self.fc3_1(out_3)

            x = self.base(z12)
            x = self.decoder(x, encoder_inputs=updated_from_encoder)
            x = self.deconv_out(x)

            x = torch.cat([x, z0], dim=1)
            x = self.decoder_head(x)

            x = self.out_conv(x)
            if self.final_activation is not None:
                x = self.final_activation(x)

            return out_1, out_2, out_3, x

#
#  ADDITIONAL FUNCTIONALITIES
#


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)
        )

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_conv_transpose=True):
        super().__init__()
        _upsampler = SingleDeconv2DBlock if use_conv_transpose else Upsampler2d
        self.block = nn.Sequential(
            _upsampler(scale_factor=2, in_channels=in_channels, out_channels=out_channels),
            SingleConv2DBlock(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

if __name__ == "__main__":
    unetr = UNETR(img_size=256, backbone="ijepa", encoder="vit-b")