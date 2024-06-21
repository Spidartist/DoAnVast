from torch import nn
import torch
import torch.nn as nn
import torch.fft as fft
from torchvision import models, datasets, transforms
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn


#
# Model Internal Post-processing
#
# Note: these are mainly for bioimage.io models, where postprocessing has to be done
# inside of the model unless its defined in the general spec


# TODO think about more (multicut-friendly) boundary postprocessing
# e.g. max preserving smoothing: bd = np.maximum(bd, gaussian(bd, sigma=1))
class AccumulateChannels(nn.Module):
    def __init__(
        self,
        invariant_channels,
        accumulate_channels,
        accumulator
    ):
        super().__init__()
        self.invariant_channels = invariant_channels
        self.accumulate_channels = accumulate_channels
        assert accumulator in ("mean", "min", "max")
        self.accumulator = getattr(torch, accumulator)

    def _accumulate(self, x, c0, c1):
        res = self.accumulator(x[:, c0:c1], dim=1, keepdim=True)
        if not torch.is_tensor(res):
            res = res.values
        assert torch.is_tensor(res)
        return res

    def forward(self, x):
        if self.invariant_channels is None:
            c0, c1 = self.accumulate_channels
            return self._accumulate(x, c0, c1)
        else:
            i0, i1 = self.invariant_channels
            c0, c1 = self.accumulate_channels
            return torch.cat([x[:, i0:i1], self._accumulate(x, c0, c1)], dim=1)


def affinities_to_boundaries(aff_channels, accumulator="max"):
    return AccumulateChannels(None, aff_channels, accumulator)


def affinities_with_foreground_to_boundaries(aff_channels, fg_channel=(0, 1), accumulator="max"):
    return AccumulateChannels(fg_channel, aff_channels, accumulator)


def affinities_to_boundaries2d():
    return affinities_to_boundaries((0, 2))


def affinities_with_foreground_to_boundaries2d():
    return affinities_with_foreground_to_boundaries((1, 3))


def affinities_to_boundaries3d():
    return affinities_to_boundaries((0, 3))


def affinities_with_foreground_to_boundaries3d():
    return affinities_with_foreground_to_boundaries((1, 4))


def affinities_to_boundaries_anisotropic():
    return AccumulateChannels(None, (1, 3), "max")


POSTPROCESSING = {
    "affinities_to_boundaries_anisotropic": affinities_to_boundaries_anisotropic,
    "affinities_to_boundaries2d": affinities_to_boundaries2d,
    "affinities_with_foreground_to_boundaries2d": affinities_with_foreground_to_boundaries2d,
    "affinities_to_boundaries3d": affinities_to_boundaries3d,
    "affinities_with_foreground_to_boundaries3d": affinities_with_foreground_to_boundaries3d,
}


#
# Base Implementations
#

class UNetBase(nn.Module):
    """
    """
    def __init__(
        self,
        encoder,
        base,
        decoder,
        out_conv=None,
        final_activation=None,
        postprocessing=None,
        check_shape=True,
    ):
        super().__init__()
        if len(encoder) != len(decoder):
            raise ValueError(f"Incompatible depth of encoder (depth={len(encoder)}) and decoder (depth={len(decoder)})")

        self.encoder = encoder
        self.base = base
        self.decoder = decoder

        if out_conv is None:
            self.return_decoder_outputs = False
            self._out_channels = self.decoder.out_channels
        elif isinstance(out_conv, nn.ModuleList):
            if len(out_conv) != len(self.decoder):
                raise ValueError(f"Invalid length of out_conv, expected {len(decoder)}, got {len(out_conv)}")
            self.return_decoder_outputs = True
            self._out_channels = [None if conv is None else conv.out_channels for conv in out_conv]
        else:
            self.return_decoder_outputs = False
            self._out_channels = out_conv.out_channels
        self.out_conv = out_conv
        self.check_shape = check_shape
        self.final_activation = self._get_activation(final_activation)
        self.postprocessing = self._get_postprocessing(postprocessing)

    @property
    def in_channels(self):
        return self.encoder.in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def depth(self):
        return len(self.encoder)

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

    def _get_postprocessing(self, postprocessing):
        if postprocessing is None:
            return None
        elif isinstance(postprocessing, nn.Module):
            return postprocessing
        elif postprocessing in POSTPROCESSING:
            return POSTPROCESSING[postprocessing]()
        else:
            raise ValueError(f"Invalid postprocessing: {postprocessing}")

    # load encoder / decoder / base states for pretraining
    def load_encoder_state(self, state):
        self.encoder.load_state_dict(state)

    def load_decoder_state(self, state):
        self.decoder.load_state_dict(state)

    def load_base_state(self, state):
        self.base.load_state_dict(state)

    def _apply_default(self, x):
        self.encoder.return_outputs = True
        self.decoder.return_outputs = False

        x, encoder_out = self.encoder(x)
        x = self.base(x)
        x = self.decoder(x, encoder_inputs=encoder_out[::-1])

        if self.out_conv is not None:
            x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        if self.postprocessing is not None:
            x = self.postprocessing(x)

        return x

    def _apply_with_side_outputs(self, x):
        self.encoder.return_outputs = True
        self.decoder.return_outputs = True

        x, encoder_out = self.encoder(x)
        x = self.base(x)
        x = self.decoder(x, encoder_inputs=encoder_out[::-1])

        x = [x if conv is None else conv(xx) for xx, conv in zip(x, self.out_conv)]
        if self.final_activation is not None:
            x = [self.final_activation(xx) for xx in x]

        if self.postprocessing is not None:
            x = [self.postprocessing(xx) for xx in x]

        # we reverse the list to have the full shape output as first element
        return x[::-1]

    def _check_shape(self, x):
        spatial_shape = tuple(x.shape)[2:]
        depth = len(self.encoder)
        factor = [2**depth] * len(spatial_shape)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            msg = f"Invalid shape for U-Net: {spatial_shape} is not divisible by {factor}"
            raise ValueError(msg)

    def forward(self, x):
        # cast input data to float, hotfix for modelzoo deployment issues, leaving it here for reference
        # x = x.float()
        if getattr(self, "check_shape", True):
            self._check_shape(x)
        if self.return_decoder_outputs:
            return self._apply_with_side_outputs(x)
        else:
            return self._apply_default(x)


def _update_conv_kwargs(kwargs, scale_factor):
    # if the scale factor is a scalar or all entries are the same we don"t need to update the kwargs
    if isinstance(scale_factor, int) or scale_factor.count(scale_factor[0]) == len(scale_factor):
        return kwargs
    else:  # otherwise set anisotropic kernel
        kernel_size = kwargs.get("kernel_size", 3)
        padding = kwargs.get("padding", 1)

        # bail out if kernel size or padding aren"t scalars, because it"s
        # unclear what to do in this case
        if not (isinstance(kernel_size, int) and isinstance(padding, int)):
            return kwargs

        kernel_size = tuple(1 if factor == 1 else kernel_size for factor in scale_factor)
        padding = tuple(0 if factor == 1 else padding for factor in scale_factor)
        kwargs.update({"kernel_size": kernel_size, "padding": padding})
        return kwargs


class Encoder(nn.Module):
    def __init__(
        self,
        features,
        scale_factors,
        conv_block_impl,
        pooler_impl,
        anisotropic_kernel=False,
        **conv_block_kwargs
    ):
        super().__init__()
        if len(features) != len(scale_factors) + 1:
            raise ValueError("Incompatible number of features {len(features)} and scale_factors {len(scale_factors)}")

        conv_kwargs = [conv_block_kwargs] * len(scale_factors)
        if anisotropic_kernel:
            conv_kwargs = [_update_conv_kwargs(kwargs, scale_factor)
                           for kwargs, scale_factor in zip(conv_kwargs, scale_factors)]

        self.blocks = nn.ModuleList(
            [conv_block_impl(inc, outc, **kwargs)
             for inc, outc, kwargs in zip(features[:-1], features[1:], conv_kwargs)]
        )
        self.poolers = nn.ModuleList(
            [pooler_impl(factor) for factor in scale_factors]
        )
        self.return_outputs = True

        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return len(self.blocks)

    def forward(self, x):
        encoder_out = []
        for block, pooler in zip(self.blocks, self.poolers):
            x = block(x)
            encoder_out.append(x)
            x = pooler(x)

        if self.return_outputs:
            return x, encoder_out
        else:
            return x


class Decoder(nn.Module):
    def __init__(
        self,
        features,
        scale_factors,
        conv_block_impl,
        sampler_impl,
        anisotropic_kernel=False,
        **conv_block_kwargs
    ):
        super().__init__()
        if len(features) != len(scale_factors) + 1:
            raise ValueError("Incompatible number of features {len(features)} and scale_factors {len(scale_factors)}")

        conv_kwargs = [conv_block_kwargs] * len(scale_factors)
        if anisotropic_kernel:
            conv_kwargs = [_update_conv_kwargs(kwargs, scale_factor)
                           for kwargs, scale_factor in zip(conv_kwargs, scale_factors)]

        self.blocks = nn.ModuleList(
            [conv_block_impl(inc, outc, **kwargs)
             for inc, outc, kwargs in zip(features[:-1], features[1:], conv_kwargs)]
        )
        self.samplers = nn.ModuleList(
            [sampler_impl(factor, inc, outc) for factor, inc, outc
             in zip(scale_factors, features[:-1], features[1:])]
        )
        self.return_outputs = False

        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return len(self.blocks)

    # FIXME this prevents traces from being valid for other input sizes, need to find
    # a solution to traceable cropping
    def _crop(self, x, shape):
        shape_diff = [(xsh - sh) // 2 for xsh, sh in zip(x.shape, shape)]
        crop = tuple([slice(sd, xsh - sd) for sd, xsh in zip(shape_diff, x.shape)])
        return x[crop]
        # # Implementation with torch.narrow, does not fix the tracing warnings!
        # for dim, (sh, sd) in enumerate(zip(shape, shape_diff)):
        #     x = torch.narrow(x, dim, sd, sh)
        # return x

    def _concat(self, x1, x2):
        return torch.cat([x1, self._crop(x2, x1.shape)], dim=1)

    def forward(self, x, encoder_inputs):
        if len(encoder_inputs) != len(self.blocks):
            raise ValueError(f"Invalid number of encoder_inputs: expect {len(self.blocks)}, got {len(encoder_inputs)}")

        decoder_out = []
        for block, sampler, from_encoder in zip(self.blocks, self.samplers, encoder_inputs):
            x = sampler(x)
            x = block(self._concat(x, from_encoder))
            decoder_out.append(x)

        if self.return_outputs:
            return decoder_out + [x]
        else:
            return x


def get_norm_layer(norm, dim, channels, n_groups=32):
    if norm is None:
        return None
    if norm == "InstanceNorm":
        kwargs = {"affine": True, "track_running_stats": True, "momentum": 0.01}
        return nn.InstanceNorm2d(channels, **kwargs) if dim == 2 else nn.InstanceNorm3d(channels, **kwargs)
    elif norm == "OldDefault":
        return nn.InstanceNorm2d(channels) if dim == 2 else nn.InstanceNorm3d(channels)
    elif norm == "GroupNorm":
        return nn.GroupNorm(min(n_groups, channels), channels)
    elif norm == "BatchNorm":
        return nn.BatchNorm2d(channels) if dim == 2 else nn.BatchNorm3d(channels)
    else:
        raise ValueError(f"Invalid norm: expect one of 'InstanceNorm', 'BatchNorm' or 'GroupNorm', got {norm}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim,
                 kernel_size=3, padding=1, norm="InstanceNorm"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = nn.Conv2d if dim == 2 else nn.Conv3d

        if norm is None:
            self.block = nn.Sequential(
                conv(in_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                conv(out_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                get_norm_layer(norm, dim, in_channels),
                conv(in_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                get_norm_layer(norm, dim, out_channels),
                conv(out_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class Upsampler(nn.Module):
    def __init__(self, scale_factor,
                 in_channels, out_channels,
                 dim, mode):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor

        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.conv = conv(in_channels, out_channels, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode=self.mode, align_corners=False)
        x = self.conv(x)
        return x


#
# 2d unet implementations
#

class ConvBlock2d(ConvBlock):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, dim=2, **kwargs)


class Upsampler2d(Upsampler):
    def __init__(self, scale_factor,
                 in_channels, out_channels,
                 mode="bilinear"):
        super().__init__(scale_factor, in_channels, out_channels,
                         dim=2, mode=mode)


class UNet2d(UNetBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        initial_features=32,
        gain=2,
        final_activation=None,
        return_side_outputs=False,
        conv_block_impl=ConvBlock2d,
        pooler_impl=nn.MaxPool2d,
        sampler_impl=Upsampler2d,
        postprocessing=None,
        check_shape=True,
        **conv_block_kwargs,
    ):
        features_encoder = [in_channels] + [initial_features * gain ** i for i in range(depth)]
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]

        if return_side_outputs:
            if isinstance(out_channels, int) or out_channels is None:
                out_channels = [out_channels] * depth
            if len(out_channels) != depth:
                raise ValueError()
            out_conv = nn.ModuleList(
                [nn.Conv2d(feat, outc, 1) for feat, outc in zip(features_decoder[1:], out_channels)]
            )
        else:
            out_conv = None if out_channels is None else nn.Conv2d(features_decoder[-1], out_channels, 1)

        super().__init__(
            encoder=Encoder(
                features=features_encoder,
                scale_factors=scale_factors,
                conv_block_impl=conv_block_impl,
                pooler_impl=pooler_impl,
                **conv_block_kwargs
            ),
            decoder=Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=conv_block_impl,
                sampler_impl=sampler_impl,
                **conv_block_kwargs
            ),
            base=conv_block_impl(
                features_encoder[-1], features_encoder[-1] * gain,
                **conv_block_kwargs
            ),
            out_conv=out_conv,
            final_activation=final_activation,
            postprocessing=postprocessing,
            check_shape=check_shape,
        )
        self.init_kwargs = {"in_channels": in_channels, "out_channels": out_channels, "depth": depth,
                            "initial_features": initial_features, "gain": gain,
                            "final_activation": final_activation, "return_side_outputs": return_side_outputs,
                            "conv_block_impl": conv_block_impl, "pooler_impl": pooler_impl,
                            "sampler_impl": sampler_impl, "postprocessing": postprocessing, **conv_block_kwargs}


#
# 3d unet implementations
#

class ConvBlock3d(ConvBlock):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, dim=3, **kwargs)


class Upsampler3d(Upsampler):
    def __init__(self, scale_factor,
                 in_channels, out_channels,
                 mode="trilinear"):
        super().__init__(scale_factor, in_channels, out_channels,
                         dim=3, mode=mode)


class AnisotropicUNet(UNetBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        initial_features=32,
        gain=2,
        final_activation=None,
        return_side_outputs=False,
        conv_block_impl=ConvBlock3d,
        anisotropic_kernel=False,  # TODO benchmark which option is better and set as default
        postprocessing=None,
        check_shape=True,
        **conv_block_kwargs,
    ):
        depth = len(scale_factors)
        features_encoder = [in_channels] + [initial_features * gain ** i for i in range(depth)]
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]

        if return_side_outputs:
            if isinstance(out_channels, int) or out_channels is None:
                out_channels = [out_channels] * depth
            if len(out_channels) != depth:
                raise ValueError()
            out_conv = nn.ModuleList(
                [nn.Conv3d(feat, outc, 1) for feat, outc in zip(features_decoder[1:], out_channels)]
            )
        else:
            out_conv = None if out_channels is None else nn.Conv3d(features_decoder[-1], out_channels, 1)

        super().__init__(
            encoder=Encoder(
                features=features_encoder,
                scale_factors=scale_factors,
                conv_block_impl=conv_block_impl,
                pooler_impl=nn.MaxPool3d,
                anisotropic_kernel=anisotropic_kernel,
                **conv_block_kwargs
            ),
            decoder=Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=conv_block_impl,
                sampler_impl=Upsampler3d,
                anisotropic_kernel=anisotropic_kernel,
                **conv_block_kwargs
            ),
            base=conv_block_impl(
                features_encoder[-1], features_encoder[-1] * gain,
                **conv_block_kwargs
            ),
            out_conv=out_conv,
            final_activation=final_activation,
            postprocessing=postprocessing,
            check_shape=check_shape,
        )
        self.init_kwargs = {"in_channels": in_channels, "out_channels": out_channels, "scale_factors": scale_factors,
                            "initial_features": initial_features, "gain": gain,
                            "final_activation": final_activation, "return_side_outputs": return_side_outputs,
                            "conv_block_impl": conv_block_impl, "anisotropic_kernel": anisotropic_kernel,
                            "postprocessing": postprocessing, **conv_block_kwargs}

    def _check_shape(self, x):
        spatial_shape = tuple(x.shape)[2:]
        scale_factors = self.init_kwargs.get("scale_factors", [[2, 2, 2]]*len(self.encoder))
        factor = [int(np.prod([sf[i] for sf in scale_factors])) for i in range(3)]
        if len(spatial_shape) != len(factor):
            msg = f"Invalid shape for U-Net: dimensions don't agree {len(spatial_shape)} != {len(factor)}"
            raise ValueError(msg)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            msg = f"Invalid shape for U-Net: {spatial_shape} is not divisible by {factor}"
            raise ValueError(msg)


class UNet3d(AnisotropicUNet):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        initial_features=32,
        gain=2,
        final_activation=None,
        return_side_outputs=False,
        conv_block_impl=ConvBlock3d,
        postprocessing=None,
        check_shape=True,
        **conv_block_kwargs,
    ):
        scale_factors = depth * [2]
        super().__init__(in_channels, out_channels, scale_factors,
                         initial_features=initial_features, gain=gain,
                         final_activation=final_activation,
                         return_side_outputs=return_side_outputs,
                         anisotropic_kernel=False,
                         postprocessing=postprocessing,
                         conv_block_impl=conv_block_impl,
                         check_shape=check_shape,
                         **conv_block_kwargs)
        self.init_kwargs = {"in_channels": in_channels, "out_channels": out_channels, "depth": depth,
                            "initial_features": initial_features, "gain": gain,
                            "final_activation": final_activation, "return_side_outputs": return_side_outputs,
                            "conv_block_impl": conv_block_impl, "postprocessing": postprocessing, **conv_block_kwargs}

class UnetDownModule(nn.Module):

    """ U-Net downsampling block. """

    def __init__(self, in_channels, out_channels, downsample=True):
        super(UnetDownModule, self).__init__()

        # layers: optional downsampling, 2 x (conv + bn + relu)
        self.maxpool = nn.MaxPool2d((2,2)) if downsample else None
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UnetEncoder(nn.Module):

    """ U-Net encoder. https://arxiv.org/pdf/1505.04597.pdf """

    def __init__(self, num_channels):
        super(UnetEncoder, self,).__init__()
        self.module1 = UnetDownModule(num_channels, 64, downsample=False)
        self.module2 = UnetDownModule(64, 128)
        self.module3 = UnetDownModule(128, 256)
        self.module4 = UnetDownModule(256, 512)
        self.module5 = UnetDownModule(512, 1024)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.module5(x)
        return x


def get_backbone(name, pretrained=True):

    """ Loading backbone, defining names for skip-connections and encoder output. """

    # TODO: More backbones

    # loading backbone model
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        backbone = models.resnet152(pretrained=pretrained)
    elif name == 'vgg16':
        backbone = models.vgg16_bn(pretrained=pretrained).features
    elif name == 'vgg19':
        backbone = models.vgg19_bn(pretrained=pretrained).features
    # elif name == 'inception_v3':
    #     backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
    elif name == 'densenet121':
        backbone = models.densenet121(pretrained=True).features
    elif name == 'densenet161':
        backbone = models.densenet161(pretrained=True).features
    elif name == 'densenet169':
        backbone = models.densenet169(pretrained=True).features
    elif name == 'densenet201':
        backbone = models.densenet201(pretrained=True).features
    elif name == 'unet_encoder':
        backbone = UnetEncoder(3)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        # TODO: consider using a 'bridge' for VGG models, there is just a MaxPool between last skip and backbone output
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    # elif name == 'inception_v3':
    #     feature_names = [None, 'Mixed_5d', 'Mixed_6e']
    #     backbone_output = 'Mixed_7c'
    elif name.startswith('densenet'):
        feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
    elif name == 'unet_encoder':
        feature_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output = 'module5'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output


class UpsampleBlock(nn.Module):

    # TODO: separate parametric and non-parametric classes?
    # TODO: skip connection concatenated OR added

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):
        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)

                                                    
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


class Unet(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self,
                 backbone_name='resnet50',
                 pretrained=True,
                 encoder_freeze=False,
                 classes=1,
                 position_classes=10,
                 damage_classes=7,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True):
        super(Unet, self).__init__()

        # self.conv_test = nn.Conv2d(2, 16, 3)
        # self.dropout_test = nn.Dropout(p=0.1, inplace=True)

        self.backbone_name = backbone_name

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if backbone_name == 'resnet50':
            in_neuron = 2048
        elif backbone_name == 'densenet121':
            in_neuron = 1024
        elif backbone_name == 'vgg19': 
            in_neuron = 512
        
        self.fc1 = nn.Linear(in_neuron, 512)

        self.fc1_1 = nn.Linear(512, position_classes)

        self.fc2 = nn.Linear(in_neuron, 512)

        self.fc2_1 = nn.Linear(512, damage_classes)

        self.fc3 = nn.Linear(in_neuron, 256)
        self.fc3_1 = nn.Linear(256, 1)

        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name, pretrained=pretrained)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm))

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating  inputs with different number of channels later

    def freeze_encoder(self):

        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):

        """ Forward propagation in U-Net. """

        # tmp_input = fft.rfftn(*input, dim=1).real
        # tmp_input = self.conv_test(tmp_input)
        # tmp_input = self.dropout_test(tmp_input)
        # # tmp_input = torch.view_as_complex(tmp_input.clone().detach())
        # # print(tmp_input.shape)
        # tmp_input = fft.irfftn(tmp_input)
        # x, features = self.forward_backbone(tmp_input)

        x, features = self.forward_backbone(*input)

        cls_res = x
        cls_res = self.avgpool(x)
        cls_res = torch.flatten(cls_res, 1)

        out1 = self.fc1(cls_res)
        out1 = self.fc1_1(out1)

        out2 = self.fc2(cls_res)
        out2 = self.fc2_1(out2)

        out3 = self.fc3(cls_res)
        out3 = self.fc3_1(out3)

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)
            # print(skip_name, x.shape)

        x = self.final_conv(x)

        return out1, out2, out3, x

    def forward_backbone(self, x):
        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()

        for name, child in self.backbone.named_children():
            x = child(x)
            # print(name, x.shape)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param


if __name__ == "__main__":
    device = "cuda:0"
    # simple test run
    net = Unet(backbone_name='resnet50')
    net = net.to(device)
    print(net.backbone)
    x = torch.rand(1, 3, 480, 480).to(device)
    y = net(x)