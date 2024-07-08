import copy
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn

from rgenn.groups import MatrixManifold
from rgenn.inr import INRPartConfig
from rgenn.layers import (
    GConv2d,
    GlobalMaxProjection,
    GlobalMeanProjection,
    GMaxPool2D,
    LiftGConv2d,
    gconv_layer,
)
from rgenn.norm_act import create_act_layer, create_norm_layer


class GSimpleBlock(nn.Module):
    def __init__(
        self,
        conv_fn: type[GConv2d],
        in_planes: int,
        planes: int,
        kernel_size: int,
        stride: int = 1,
        act_layer: str | Callable = "gelu",
        norm_layer: str | Callable = "layernorm3d",
        sampled_path: bool = True,
        conv_bias: bool = True,
        inr_cfg: Optional[INRPartConfig] = None,
    ):
        super().__init__()
        self.conv1 = conv_fn(
            in_channels=in_planes,
            out_channels=in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=conv_bias,
            inr_cfg=inr_cfg,
            sampled_path=False,
            grid_type="affine",
        )
        self.norm1 = create_norm_layer(norm_layer, in_planes)
        self.act1 = create_act_layer(act_layer, inplace=True)

        self.conv2 = conv_fn(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=conv_bias,
            inr_cfg=inr_cfg,
            sampled_path=False,
            grid_type="affine",
        )
        self.norm2 = create_norm_layer(norm_layer, planes)
        self.act2 = create_act_layer(act_layer, inplace=True)
        self.act3 = create_act_layer(act_layer, inplace=True)

        self.downsample = None
        if in_planes != planes:
            # TODO: Not fully implemented, see 'attach_samples' below.
            cfg = copy.deepcopy(inr_cfg)
            cfg.hidden_features //= 2
            cfg.kernel_config.hidden_features = cfg.hidden_features
            self.downsample = conv_fn(
                in_channels=in_planes,
                out_channels=planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                sampled_path=False,
                inr_cfg=cfg,
            )

        self.sampled_path = sampled_path

    def register_gsamples(self, num_gsamples: int):
        self.conv1.register_gsamples(num_gsamples)
        self.conv2.register_gsamples(num_gsamples)

        if self.downsample is not None:
            self.downsample.register_gsamples(num_gsamples)

    # TODO: For now, we reuse samples for all convs.  When downsample = None, we
    # have to do this, so that output and input are defined over the same group
    # elements. If downsample != None (need shortcut) we can let each conv
    # layers sample, and reuse the last conv samples for the downsample.
    def attach_samples(
        self, g_tilde: torch.Tensor, g_inv: torch.Tensor, g_tilde_inv
    ) -> None:
        self.conv1.attach_samples(g_tilde=g_tilde, g_inv=g_tilde_inv)
        self.conv2.attach_samples(g_tilde=g_tilde, g_inv=g_tilde_inv)

        if self.downsample is not None:
            self.downsample.attach_samples(g_tilde=g_tilde, g_inv=g_tilde_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = x + shortcut
        x = self.act3(x)
        return x


def collect_sampling_layers(module):
    def collect_layers(module, layers):
        if hasattr(module, "sampled_path") and module.sampled_path:
            layers.append(module)

    layers = []
    module.apply(partial(collect_layers, layers=layers))
    return layers


class SamplingWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_gsamples: int,
        group: Optional[MatrixManifold] = None,
        sample_inference: bool = True,
        deterministic: bool = False,
    ):
        super().__init__()

        self.model = model
        self.num_gsamples = num_gsamples

        if group is None:
            group = self.model.group

        self.group = group
        self.sample_inference = sample_inference
        self.deterministic = deterministic
        self.sample_path = collect_sampling_layers(self.model)

        if self.deterministic:
            # Sample group elements once and reuse.
            with torch.no_grad():
                self.sample()

    def set_numgsamples(self, num_gsamples: int) -> None:
        assert num_gsamples > 0
        self.num_gsamples = num_gsamples

        if hasattr(self.model, "num_gsamples"):
            self.model.num_gsamples = num_gsamples

        for layer in self.sample_path:
            layer.register_gsamples(num_gsamples)

    def sample(self) -> None:
        g_tilde, g_tilde_inv = None, None
        for layer in self.sample_path:
            g, g_inv = self.group.sample_pair(self.num_gsamples)
            layer.attach_samples(g_tilde=g_tilde, g_inv=g_inv, g_tilde_inv=g_tilde_inv)
            g_tilde = g
            g_tilde_inv = g_inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.deterministic and (self.training or self.sample_inference):
            self.sample()
        return self.model(x)


def build_classifier(
    in_features: int,
    hidden_features: int,
    num_classes: int,
    act_layer: str,
    norm_layer: str,
    dropout_rate: float,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden_features, bias=False),
        create_norm_layer(norm_layer, hidden_features),
        create_act_layer(act_layer, inplace=True),
        nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        nn.Linear(hidden_features, num_classes),
    )


class GResNet(nn.Module):
    def __init__(
        self,
        group: MatrixManifold,
        num_gsamples: int,
        inr_cfg: INRPartConfig,
        dims: list[int],
        block: type[GSimpleBlock] = GSimpleBlock,
        in_chans: int = 3,
        stem_kernel: int = 5,
        stem_padding: int = 0,
        blocks_kernel: int = 5,
        num_classes: int = 10,
        global_pool: str = "max",
        act_layer: str = "gelu",
        norm_layer: str = "layernorm",
        norm_layer_conv: str = "layernorm3d",
    ):
        super().__init__()

        self.group = group
        self.num_gsamples = num_gsamples
        self.inr_cfg = inr_cfg

        self.dims = dims
        self.in_chans = in_chans
        self.stem_kernel = stem_kernel
        self.stem_padding = stem_padding
        self.blocks_kernel = blocks_kernel
        self.num_classes = num_classes
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.conv_norm_layer = norm_layer_conv

        lift2d = gconv_layer(LiftGConv2d, group, num_gsamples, inr_cfg)
        conv2d = gconv_layer(GConv2d, group, num_gsamples)

        # Stem/Lifting
        # NB: For some datasets (e.g. affNIST) having padding in the lifting layer
        # strongly decreases performance for some reason.
        self.stem = nn.Sequential(
            *[
                lift2d(
                    in_channels=in_chans,
                    out_channels=dims[0],
                    kernel_size=stem_kernel,
                    padding=stem_padding,
                    bias=False,
                    grid_type="affine",
                ),
                create_norm_layer(self.conv_norm_layer, dims[0]),
                create_act_layer(act_layer, inplace=True),
            ]
        )

        # Residual Blocks
        net = []
        for index in range(1, len(dims)):
            net += [
                block(
                    conv2d,
                    dims[index - 1],
                    dims[index],
                    kernel_size=blocks_kernel,
                    act_layer=act_layer,
                    norm_layer=self.conv_norm_layer,
                    conv_bias=False,
                    inr_cfg=inr_cfg,
                ),
                create_norm_layer(self.conv_norm_layer, dims[index]),
                GMaxPool2D(kernel_size=2, stride=2),
                create_act_layer(act_layer, inplace=True),
            ]

        net += [
            conv2d(
                in_channels=dims[-1],
                out_channels=dims[-1],
                kernel_size=blocks_kernel,
                bias=False,
                inr_cfg=inr_cfg,
                grid_type="affine",
            ),
            create_norm_layer(self.conv_norm_layer, dims[-1]),
            create_act_layer(act_layer, inplace=True),
        ]

        self.blocks = nn.Sequential(*net)

        if global_pool == "max":
            self.pool = GlobalMaxProjection()
        elif global_pool == "mean":
            self.pool = GlobalMeanProjection()
        else:
            raise ValueError(f"Projection must be (max, mean), got {global_pool}")

        self.fc = build_classifier(
            in_features=dims[-1],
            hidden_features=256,
            num_classes=self.num_classes,
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
            dropout_rate=0.3,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
