from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from rgenn.groups import MatrixManifold
from rgenn.inr import INR, SIREN, INRPartConfig, SIRENConfig


def circular_mask(
    kernels: torch.Tensor,
    kernel_grids: torch.Tensor,
    extra_dim: int = 2,
    radius: float = 1.0,
) -> torch.Tensor:
    mask = torch.norm(kernel_grids[:, :2], dim=1) > radius
    mask = mask.unsqueeze(0).unsqueeze(extra_dim).expand_as(kernels)
    kernels[mask] = 0
    return kernels


def circular_mask_smooth(
    kernels: torch.Tensor,
    kernel_grids: torch.Tensor,
    max_rel_dist: float = 1.0,
    slope: float = 2.0,
    extra_dim: int = 2,
) -> torch.Tensor:
    smooth_mask = torch.sigmoid(
        slope * (max_rel_dist - torch.norm(kernel_grids, dim=1))
    )
    smooth_mask = smooth_mask.unsqueeze(0).unsqueeze(extra_dim)
    smooth_mask = smooth_mask.expand_as(kernels)
    return torch.mul(kernels, smooth_mask)


def get_grid2d(kernel_size: int, grid_type: str = "affine") -> torch.Tensor:
    if kernel_size == 1:
        # NB: Initializing with 0 will mean that xy component is 0. for the INR in the first layer.
        # so only group information is fed in.
        # For > 1e-6 it seems to induce some equivariance error.
        return torch.zeros(2, 1, 1)

    if grid_type in ("affine", "affine_xy"):
        # https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/4
        coords = F.affine_grid(
            torch.eye(2, 3)[None, ...],
            (1, 1, kernel_size, kernel_size),
            align_corners=False,
        ).squeeze(0)
        if grid_type == "affine":
            return rearrange(coords, "h w r2 -> r2 w h")
        else:
            return rearrange(coords, "h w r2 -> r2 h w")

    elif grid_type == "ij":
        return torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, kernel_size, dtype=torch.get_default_dtype()),
                torch.linspace(-1, 1, kernel_size, dtype=torch.get_default_dtype()),
                indexing="ij",
            )
        )
    else:
        return torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, kernel_size, dtype=torch.get_default_dtype()),
                torch.linspace(-1, 1, kernel_size, dtype=torch.get_default_dtype()),
                indexing="xy",
            )
        )


def pack_regular(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "batch ch gr h w -> batch (ch gr) h w")


def unpack_regular(x: torch.Tensor, num_gsamples: int) -> torch.Tensor:
    return rearrange(x, "batch (ch gr) h w -> batch ch gr h w", gr=num_gsamples)


class GConv2d(nn.Module):
    def __init__(
        self,
        group: MatrixManifold,
        num_gsamples: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        inr_cfg: Optional[INRPartConfig] = None,
        masked: bool = True,
        mask_map: Optional[Callable[..., torch.Tensor]] = None,
        mask_extra_dim: int = 2,
        grid_type: str = "affine",
        sampled_path: bool = True,
        normalized_coordinate: bool = False,
    ):
        super().__init__()
        assert all(arg > 0 for arg in (kernel_size, stride, dilation, groups))
        assert padding >= 0
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        assert grid_type in ("affine", "ij", "xy", "affine_xy"), grid_type

        # (Lie) Group parameters
        self.group = group
        self.num_gsamples = num_gsamples

        # Conv2d parameters
        self.in_channels = in_channels // groups
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding) if padding > 0 else 0
        self.dilation = dilation
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Kernel parameters
        if inr_cfg is None:
            # Default to a 2-layer SIREN with 32 hidden dim.
            inr_cfg = INRPartConfig(
                kernel_class=SIREN,
                hidden_features=64,
                kernel_config=SIRENConfig(hidden_features=64, num_hidden=2),
            )

        self.inr_cfg = inr_cfg
        self.masked = masked
        self.mask_map = mask_map
        self.volume_preserving = self.group.volume_preserving
        self.grid_type = grid_type
        self.sampled_path = sampled_path
        self.normalize_coordinate = normalized_coordinate

        self._mask = None

        if masked:
            if mask_map is None:
                if self.volume_preserving:
                    self._mask = partial(circular_mask, extra_dim=mask_extra_dim)
                else:
                    self._mask = partial(
                        circular_mask_smooth,
                        max_rel_dist=1.0,
                        slope=2.0,
                        extra_dim=mask_extra_dim,
                    )
            else:
                self._mask = partial(mask_map, extra_dim=mask_extra_dim)

        self.inr = self.build_inr()

        # R^2 grid
        self.register_buffer("grid", get_grid2d(kernel_size, grid_type=grid_type))

        # Group samples used to produce the kernel, to be updated via 'attach_samples'
        self.register_gsamples(self.num_gsamples)

    def register_gsamples(self, num_gsamples: int):
        self.num_gsamples = num_gsamples

        # NB: Set the device in case this is changed after e.g. .cuda()
        g_identity = torch.eye(2, device=self.grid.device)[None, ...].repeat(
            num_gsamples, 1, 1
        )
        self.register_buffer("g_tilde", g_identity[:])
        self.register_buffer("g_inv", g_identity[:])

    def build_inr(self) -> INR:
        return INR.from_part_config(
            in_features=self.group.dim + 2,
            in_ch=self.in_channels,
            out_ch=self.out_channels,
            kernel_size=self.kernel_size,
            part_config=self.inr_cfg,
        )

    @torch.no_grad()
    def attach_samples(
        self, g_tilde: torch.Tensor, g_inv: torch.Tensor, **kwargs
    ) -> None:
        self.g_tilde.copy_(g_tilde)
        self.g_inv.copy_(g_inv)
        self.num_gsamples = self.g_inv.shape[0]

    def transformed_grid(
        self, g_tilde: torch.Tensor, g_inv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        in_gr, out_gr = g_tilde.shape[0], g_inv.shape[0]

        h_grid = self.group.outer_act_group(gr_elems=g_inv, gr_grid=g_tilde)
        h_grid = rearrange(h_grid, "out_gr in_gr i j -> (out_gr in_gr) i j")
        h_grid = self.group.left_xi_inv(h_grid)

        if self.normalize_coordinate:
            h_grid /= torch.pi

        h_grid = repeat(
            h_grid,
            "(out_gr in_gr) gr_dim -> out_gr in_gr gr_dim w h",
            in_gr=in_gr,
            out_gr=out_gr,
            gr_dim=self.group.dim,
            h=self.kernel_size,
            w=self.kernel_size,
        )

        r2_grid = self.group.outer_act_r2(g_inv, self.grid)
        r2_grid = repeat(r2_grid, "out_gr r2 w h -> out_gr in_gr r2 w h", in_gr=in_gr)
        r2h_grid = torch.cat((r2_grid, h_grid), dim=2)
        grid = rearrange(
            r2h_grid,
            "out_gr in_gr dim w h -> (out_gr in_gr h w) dim",
        )

        mask = None
        if self.masked:
            mask = rearrange(r2h_grid, "out_gr in_gr dim w h -> out_gr dim in_gr h w")

        return grid, mask

    def forward_kernel(
        self,
        g_tilde: torch.Tensor,
        g_inv: torch.Tensor,
    ) -> torch.Tensor:
        kernel, mask = self.transformed_grid(g_tilde, g_inv)
        kernel = self.inr(kernel)
        kernel = rearrange(
            kernel,
            "(out_gr in_gr h w) (out_ch in_ch) -> out_ch out_gr in_ch in_gr h w",
            out_gr=g_inv.shape[0],
            in_gr=g_tilde.shape[0],
            h=self.kernel_size,
            w=self.kernel_size,
            in_ch=self.in_channels,
            out_ch=self.out_channels,
        )

        if self.masked:
            kernel = self._mask(kernel, mask)

        if not self.volume_preserving:
            kernel_vol = torch.abs(torch.det(g_tilde))
            kernel /= kernel_vol[None, ..., None, None, None, None]

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.forward_kernel(self.g_tilde, self.g_inv)
        kernel = rearrange(
            kernel,
            "out_ch out_gr in_ch in_gr h w -> (out_ch out_gr) (in_ch in_gr) h w",
        )
        x = pack_regular(x)
        x = F.conv2d(
            input=x,
            weight=kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        x = unpack_regular(x, num_gsamples=self.num_gsamples)

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1)
        return x

    def extra_repr(self) -> str:
        fstr = "Gsamples: {} in_ch {} out_ch {} kernel_size {} stride {} padding {}"
        fstr += " grid_type {}, masked {}, bias {}"
        return fstr.format(
            self.num_gsamples,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.grid_type,
            self.masked,
            self.bias is not None,
        )


class LiftGConv2d(GConv2d):
    def __init__(
        self,
        group: MatrixManifold,
        num_gsamples: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        inr_cfg: Optional[INRPartConfig] = None,
        masked: bool = True,
        mask_map: Optional[Callable[..., torch.Tensor]] = None,
        mask_extra_dim: int = 2,
        grid_type: str = "affine",
        sampled_path: bool = True,
    ):
        super().__init__(
            group=group,
            num_gsamples=num_gsamples,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            inr_cfg=inr_cfg,
            masked=masked,
            mask_map=mask_map,
            mask_extra_dim=mask_extra_dim,
            grid_type=grid_type,
            sampled_path=sampled_path,
        )

        if padding > 0:
            print(
                "Warning: Lifting layer padding sometimes negatively affects performance."
            )

        # NB: The following is only true for lifting/projection layers.
        assert (
            kernel_size > 1
        ), "If 1x1 kernel and affine_grid torch zeros, input to inr is 0."

    def build_inr(self) -> INR:
        return INR.from_part_config(
            in_features=2,
            in_ch=self.in_channels,
            out_ch=self.out_channels,
            kernel_size=self.kernel_size,
            part_config=self.inr_cfg,
        )

    @torch.no_grad()
    def attach_samples(self, g_inv: torch.Tensor, **kwargs) -> None:
        self.g_inv.copy_(g_inv)
        self.num_gsamples = self.g_inv.shape[0]

    def transformed_grid(
        self, g_inv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r2_grid = self.group.outer_act_r2(g_inv, self.grid)
        r2_grid = rearrange(r2_grid, "gr r2 w h -> gr r2 h w")
        grid = rearrange(r2_grid, "gr r2 h w -> (gr h w) r2")
        return grid, r2_grid

    def forward_kernel(self, g_inv: torch.Tensor) -> torch.Tensor:
        kernel, mask = self.transformed_grid(g_inv)
        kernel = self.inr(kernel)
        kernel = rearrange(
            kernel,
            "(gr h w) (out_ch in_ch) -> out_ch gr in_ch h w",
            gr=g_inv.shape[0],
            h=self.kernel_size,
            w=self.kernel_size,
            in_ch=self.in_channels,
            out_ch=self.out_channels,
        )

        if self.masked:
            kernel = self._mask(kernel, mask)

        if not self.volume_preserving:
            kernel_vol = torch.abs(torch.det(g_inv))
            kernel *= kernel_vol[None, ..., None, None, None]
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.forward_kernel(self.g_inv)
        kernel = rearrange(
            kernel,
            "out_ch gr in_ch h w -> (out_ch gr) in_ch h w",
        )
        x = F.conv2d(
            input=x,
            weight=kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        x = unpack_regular(x, num_gsamples=self.num_gsamples)

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1)

        return x


class GProj2dS(GConv2d):
    # Equation (136)
    def __init__(
        self,
        group: MatrixManifold,
        num_gsamples: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        inr_cfg: Optional[INRPartConfig] = None,
        masked: bool = True,
        mask_map: Optional[Callable[..., torch.Tensor]] = None,
        grid_type: str = "affine",
        sampled_path: bool = True,
    ):
        super().__init__(
            group=group,
            num_gsamples=num_gsamples,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            inr_cfg=inr_cfg,
            masked=masked,
            mask_map=mask_map,
            mask_extra_dim=1,
            grid_type=grid_type,
            sampled_path=sampled_path,
        )

    def build_inr(self) -> INR:
        return INR.from_part_config(
            in_features=2,
            in_ch=self.in_channels,
            out_ch=self.out_channels,
            kernel_size=self.kernel_size,
            part_config=self.inr_cfg,
        )

    @torch.no_grad()
    def attach_samples(self, g_tilde_inv: torch.Tensor, **kwargs) -> None:
        self.g_inv.copy_(g_tilde_inv)

    def transformed_grid(
        self, g_inv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r2_grid = self.group.outer_act_r2(g_inv, self.grid)
        r2_grid = rearrange(r2_grid, "gr r2 w h -> gr r2 h w")
        grid = rearrange(r2_grid, "gr r2 h w -> (gr h w) r2")
        return grid, r2_grid

    def forward_kernel(self, g_inv: torch.Tensor) -> torch.Tensor:
        kernel, mask = self.transformed_grid(g_inv)
        kernel = self.inr(kernel)
        kernel = rearrange(
            kernel,
            "(in_gr h w) (out_ch in_ch) -> out_ch in_ch in_gr h w",
            in_gr=g_inv.shape[0],
            h=self.kernel_size,
            w=self.kernel_size,
            in_ch=self.in_channels,
            out_ch=self.out_channels,
        )
        if self.masked:
            kernel = self._mask(kernel, mask)

        if not self.volume_preserving:
            kernel_vol = torch.abs(torch.det(g_inv))
            kernel *= kernel_vol[None, ..., None, None, None]

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.forward_kernel(g_inv=self.g_inv)
        kernel = rearrange(
            kernel,
            "out_ch in_ch in_gr h w -> out_ch (in_ch in_gr) h w",
        )
        x = pack_regular(x)
        x = F.conv2d(input=x, weight=kernel, padding=self.padding, stride=self.stride)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x


class GMaxPool2D(nn.Module):
    # Apply maxpooling over spatial dimension.
    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_gsamples = x.shape[2]
        x = pack_regular(x)
        x = F.max_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        x = unpack_regular(x, num_gsamples=num_gsamples)
        return x

    def extra_repr(self) -> str:
        return "kernel_size: {} stride: {}".format(self.kernel_size, self.stride)


class GAvgPool2D(nn.Module):
    # Apply avgpooling over spatial dimension.
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_gsamples = x.shape[2]
        x = pack_regular(x)
        x = F.avg_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        x = unpack_regular(x, num_gsamples=num_gsamples)
        return x

    def extra_repr(self) -> str:
        return "kernel_size: {} stride: {}".format(self.kernel_size, self.stride)


def gconv_layer(
    conv_class: type[GConv2d],
    group: MatrixManifold,
    num_gsamples: int,
    inr_cfg: Optional[INRPartConfig] = None,
):
    # Return type ommitted to avoid typing functools.partial
    return partial(
        conv_class,
        group,
        num_gsamples,
        inr_cfg=inr_cfg,
    )


class GlobalMaxProjection(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=(-3, -2, -1))


class GlobalMeanProjection(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=(-3, -2, -1))


class GroupMaxPooling(nn.Module):
    # Should be equivalent to escnn.nn.GroupPooling
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=2)[0]


class GroupAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=2)
