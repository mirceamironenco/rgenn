from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class KernelConfig:
    hidden_features: int
    num_hidden: int
    in_features: int | None = None

    def update_infeat(self, in_features: int):
        self.in_features = in_features


@dataclass
class INRPartConfig:
    kernel_class: type[nn.Module]
    kernel_config: KernelConfig
    hidden_features: int
    final_bias: bool = True
    final_gain: float = 6.0
    final_mixed_fans: bool = False


# INR - Takes in (transformed) coordinates and produces the convolution kernel.
class INR(nn.Module):
    def __init__(
        self,
        kernel_net: nn.Module,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        in_features: int,
        hidden_features: int,
        final_bias: bool = True,
        final_gain: float = 6.0,
        final_mixed_fans: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert final_gain > 0.0
        self.kernel_net = kernel_net
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.final_bias = final_bias
        self.final_gain = final_gain
        self.final_mixed_fans = final_mixed_fans

        self.final_linear = nn.Linear(hidden_features, in_ch * out_ch, bias=True)

        with torch.no_grad():
            if final_mixed_fans:
                fan_in = (in_ch + out_ch) * (kernel_size**2)
            else:
                fan_in = in_ch * (kernel_size**2)
            uniform_variance = np.sqrt(final_gain / hidden_features) / (np.sqrt(fan_in))
            self.final_linear.weight.data.uniform_(-uniform_variance, uniform_variance)

            if not self.final_bias:
                self.final_linear.bias = None
            else:
                nn.init.constant_(self.final_linear.bias, 0)

    @classmethod
    def from_part_config(
        cls,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        in_features: int,
        part_config: INRPartConfig,
    ):
        # Update kernel config with in_features size
        kernel_config = part_config.kernel_config
        kernel_config.update_infeat(in_features)
        kernel_net = part_config.kernel_class.from_config(kernel_config)

        # Construct INR
        fields = asdict(part_config)
        fields = {k: v for k, v in fields.items() if v is not None}
        return cls(kernel_net, in_ch, out_ch, kernel_size, in_features, **fields)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kernel_net(x)
        x = self.final_linear(x)
        return x


class SineAct(nn.Module):
    def __init__(self, a: float = 1.0, trainable: float = False):
        super().__init__()
        self.register_parameter("a", nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.a * x)


class CosAct(nn.Module):
    def __init__(self, a: float = 1.0, trainable: bool = False):
        super().__init__()
        self.register_parameter("a", nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.a * x)


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float,
        is_first: bool,
        use_bias: bool,
        trainable_act: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w0 = w0
        self.is_first = is_first
        self.use_bias = use_bias
        self.act_layer = torch.jit.script(SineAct(a=w0, trainable=trainable_act))
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if self.is_first:
            w_std = 1.0 / self.in_features
        else:
            w_std = np.sqrt(6.0 / self.in_features) / self.w0

        self.linear.weight.uniform_(-w_std, w_std)

        if not self.is_first and self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_layer(self.linear(x))


class RealGaborLayer(nn.Module):
    """
    Source: https://github.com/vishwa91/wire/blob/main/modules/wire.py
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float,
        s0: float,
        is_first: bool,
        use_bias: bool,
        freqs_act: str = "sin",
        siren_init: bool = False,
        trainable_act: bool = False,
    ):
        super().__init__()
        assert freqs_act in ("sin", "cos")
        self.in_features = in_features
        self.out_features = out_features
        self.w0 = w0
        self.s0 = s0
        self.is_first = is_first
        self.use_bias = use_bias
        self.siren_init = siren_init
        self.freqs = nn.Linear(in_features, out_features, bias=use_bias)
        self.scale = nn.Linear(in_features, out_features, bias=use_bias)

        if freqs_act == "sin":
            self.act_freqs = torch.jit.script(SineAct(a=w0, trainable=trainable_act))
        else:
            self.act_freqs = torch.jit.script(CosAct(a=w0, trainable=trainable_act))

        if self.siren_init:
            self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for layer, multiplier in zip((self.freqs, self.scale), (self.w0, self.s0)):
            if self.is_first:
                w_std = 1.0 / self.in_features
            else:
                w_std = np.sqrt(6.0 / self.in_features) / multiplier
            layer.weight.uniform_(-w_std, w_std)

            if not self.is_first and layer.bias is not None:
                layer.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self.act_freqs(self.freqs(x))
        scale = self.scale(x) * self.s0
        return freq * torch.exp(-(scale**2))


@dataclass
class SIRENConfig(KernelConfig):
    first_w0: float = 10.0
    w0: float = 10.0
    use_bias: bool = True
    trainable_act: bool = False


class SIREN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_hidden: int,
        first_w0: float,
        w0: float,
        use_bias: bool = True,
        trainable_act: bool = False,
    ):
        super().__init__()
        if w0 <= 0.0 or first_w0 <= 0.0:
            raise ValueError("Invalid omega_0 parameter specified for SIREN.")

        net = []
        for index in range(num_hidden + 1):
            layer = SineLayer(
                in_features=hidden_features if index > 0 else in_features,
                out_features=hidden_features,
                w0=first_w0 if index == 0 else w0,
                use_bias=use_bias,
                is_first=index == 0,
                trainable_act=trainable_act,
            )
            net.append(layer)
        self.net = nn.Sequential(*net)

    @classmethod
    def from_config(cls, config: SIRENConfig):
        fields = asdict(config)
        fields = {k: v for k, v in fields.items() if v is not None}
        return cls(**fields)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class WIREConfig(KernelConfig):
    first_w0: float = 10.0
    w0: float = 10.0
    first_s0: float = 10.0
    s0: float = 10.0
    freqs_act: str = "sin"
    use_bias: bool = True
    siren_init: bool = True
    trainable_act: bool = False


class WIRE(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_hidden: int,
        first_w0: float,
        w0: float,
        first_s0: float,
        s0: float,
        freqs_act: str = "sin",
        use_bias: bool = True,
        siren_init: bool = True,
        trainable_act: bool = False,
    ):
        super().__init__()
        net = []
        for index in range(num_hidden + 1):
            net.append(
                RealGaborLayer(
                    in_features=hidden_features if index > 0 else in_features,
                    out_features=hidden_features,
                    w0=first_w0 if index == 0 else w0,
                    s0=first_s0 if index == 0 else s0,
                    is_first=index == 0,
                    freqs_act=freqs_act,
                    use_bias=use_bias,
                    siren_init=siren_init,
                    trainable_act=trainable_act,
                )
            )
        self.net = nn.Sequential(*net)

    @classmethod
    def from_config(cls, config: WIREConfig):
        fields = asdict(config)
        fields = {k: v for k, v in fields.items() if v is not None}
        return cls(**fields)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
