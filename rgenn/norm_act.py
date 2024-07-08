import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm3d(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x):
        return self.bn(x)


class LayerNorm3d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return F.layer_norm(x, x.shape[-4:])


class LayerNorm3dG(nn.LayerNorm):
    # LayerNorm over channels only, like in ViT, ConvNext etc
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class GroupNorm(nn.Module):
    def __init__(self, channels, eps=1e-5, affine=True):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=1, num_channels=channels, eps=eps, affine=affine
        )

    def forward(self, x):
        return self.norm(x)


_norm_layers = dict(
    layernorm3dg=LayerNorm3dG,
    layernorm3d=LayerNorm3d,
    batchnorm3d=BatchNorm3d,
    groupnorm3d=GroupNorm,
    groupnorm=GroupNorm,
    layernorm=nn.LayerNorm,
    batchnorm=nn.BatchNorm1d,
    none=nn.Identity,
)


_standard_act_layers = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    relu=nn.ReLU,
    selu=nn.SELU,
    gelu=nn.GELU,
    elu=nn.ELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    identity=nn.Identity,
    none=nn.Identity,
)


def get_act_layer(name: type[nn.Module] | str):
    if not isinstance(name, str):
        return name

    return _standard_act_layers[name]


def get_norm_layer(name: type[nn.Module] | str):
    if not isinstance(name, str):
        return name

    return _norm_layers[name]


def create_norm_layer(name: type[nn.Module] | str, features=None, **kwargs):
    norm_layer = get_norm_layer(name)
    if features is None:
        return norm_layer(**kwargs)
    try:
        return norm_layer(features, **kwargs)
    except TypeError:
        return norm_layer(**kwargs)


def create_act_layer(name: type[nn.Module] | str, inplace=None, **kwargs):
    act_layer = get_act_layer(name)
    if inplace is None:
        return act_layer(**kwargs)
    try:
        return act_layer(inplace=inplace, **kwargs)
    except TypeError:
        return act_layer(**kwargs)
