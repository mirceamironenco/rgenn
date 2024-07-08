import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from einops import einsum
from torch.distributions.multivariate_normal import MultivariateNormal

###################
# Utilities for working with SPD matrices.
###################


class rgenn_cache(dict):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def __missing__(self, item: Any) -> Any:
        tensor = self.fn(*item)
        self[item] = tensor
        return tensor

    def __call__(self, *args: Any) -> Any:
        return self[args]


@rgenn_cache
def maximum_dtype(*args):
    # Should work with torch.compile.
    dtype = max(args, key=lambda dt: torch.finfo(dt).bits)
    return dtype


@torch.cuda.amp.autocast(enabled=False)
def eigh(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Allows for float64 for numerical checks
    dtype = maximum_dtype(x.dtype, torch.float32)
    return torch.linalg.eigh(x.to(dtype), UPLO="L")


def sym_expm(x: torch.Tensor) -> torch.Tensor:
    e, v = eigh(x)
    return (v * torch.exp(e).unsqueeze(-2)) @ v.transpose(-1, -2)


def sym_inv_sqrt(x: torch.Tensor) -> torch.Tensor:
    e, v = eigh(x)
    return (v * (torch.pow(e, exponent=-0.5)).unsqueeze(-2)) @ v.transpose(-1, -2)


def sym_logm(x: torch.Tensor) -> torch.Tensor:
    e, v = eigh(x)
    return (v * torch.log(e).unsqueeze(-2)) @ v.transpose(-1, -2)


def sym_sqrt(x: torch.Tensor) -> torch.Tensor:
    e, v = eigh(x)
    return (v * torch.sqrt(e).unsqueeze(-2)) @ v.transpose(-1, -2)


def sym_sqrt_and_inv_sqrt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    e, v = eigh(x)
    sqrt_log_e = 0.5 * torch.log(e)
    sqrt = (v * (torch.exp(sqrt_log_e)).unsqueeze(-2)) @ v.transpose(-1, -2)
    inv_sqrt = (v * (torch.exp(-sqrt_log_e)).unsqueeze(-2)) @ v.transpose(-1, -2)
    return sqrt, inv_sqrt


def antideviator(x: torch.Tensor) -> torch.Tensor:
    tr_mn = batch_trace(x) / x.shape[-1]
    return torch.diag_embed(torch.stack((tr_mn, tr_mn), dim=-1))


def deviator(x: torch.Tensor) -> torch.Tensor:
    return x - antideviator(x)


###################
# Groups
###################
def left_polar_canonical_decomposition(
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # See Section B.7 in the Appendix.
    ggt = g @ g.transpose(-1, -2)
    e, v = eigh(ggt)
    log_sqrt_sym = 0.5 * torch.log(e)
    log_p = (v * log_sqrt_sym.unsqueeze(-2)) @ v.transpose(-1, -2)
    inv_sqrt_p = (v * torch.exp(-log_sqrt_sym).unsqueeze(-2)) @ v.transpose(-1, -2)
    return log_p, inv_sqrt_p @ g


def batch_trace(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    return torch.diagonal(x, dim1=-2, dim2=-1).sum(-1, keepdim=keepdim)


def canonical_inner_product(
    X: torch.Tensor, Y: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    return scale * batch_trace(X.transpose(-1, -2) @ Y)


class MatrixManifold(nn.Module):
    def __init__(self, dim: int, matrix_shape, metric_alpha=1.0, **kwargs):
        super().__init__()
        assert metric_alpha > 0.0
        self._dim = dim
        self._matrix_shape = matrix_shape
        self._metric_alpha = metric_alpha

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def matrix_shape(self):
        return self._matrix_shape

    def belongs(self, g: torch.Tensor):
        raise NotImplementedError

    def expm(self, v: torch.Tensor) -> torch.Tensor:
        return self._expm(v)

    def _expm(self, v: torch.Tensor):
        raise NotImplementedError

    def hat(self, v: torch.Tensor):
        raise NotImplementedError

    def hat_expm(self, v_hat: torch.Tensor) -> torch.Tensor:
        return torch.matrix_exp(v_hat)

    def inner_product(self, X: torch.Tensor, Y: torch.Tensor):
        return canonical_inner_product(X, Y, self._metric_alpha)

    def logm(self, g):
        return self._logm(g)

    def _logm(self, v):
        raise NotImplementedError

    @classmethod
    def outer_act_group(cls, gr_elems, gr_grid):
        return einsum(gr_elems, gr_grid, "in_gr i k, out_gr k j -> in_gr out_gr i j")

    @classmethod
    def outer_act_r2(cls, gr_elems, r2_grid):
        return einsum(
            gr_elems, r2_grid, "in_gr gr_dim r2_dim, r2_dim w h -> in_gr gr_dim w h"
        )

    def vee(self, v_hat):
        raise NotImplementedError


class SOn(MatrixManifold):
    def __init__(self, n: int, metric_alpha: float = 1.0, **kwargs):
        super().__init__(
            dim=n * (n - 1) // 2, matrix_shape=n**2, metric_alpha=metric_alpha, **kwargs
        )
        self.n = n
        self.register_buffer("pi", torch.tensor(3.14159265358979323846))
        self._eps = 1.3e-7
        self._sq_eps = 1e-6
        self.volume_preserving = True

    def belongs(self, g: torch.Tensor) -> bool:
        Id = torch.eye(g.shape[-1], dtype=g.dtype, device=g.device)
        return torch.all(torch.det(g) > 0.0) and torch.allclose(
            g.transpose(-1, -2) @ g, Id
        )


class SO2(SOn):
    def __init__(self, metric_alpha: float = 1.0, **kwargs):
        super().__init__(n=2, metric_alpha=metric_alpha, **kwargs)
        self.basis = self.register_buffer(
            "basis_matrix",
            torch.tensor(
                [
                    [0.0, -1.0 / math.sqrt(metric_alpha * 2.0)],
                    [1.0 / math.sqrt(metric_alpha * 2.0), 0.0],
                ]
            )[None, ...],
        )

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def matrix_to_radians(self, g: torch.Tensor) -> torch.Tensor:
        sin_theta, cos_theta = g[..., 1, 0], g[..., 0, 0]
        # NB: Becuase of signed zero, you can get -0.0 here and when
        # mod \2pi happens, it flips to ~2\pi. What happens more preciesly
        # is that -0.0 is some small number like -1e-7, so we just mask it.
        zeros = torch.zeros_like(sin_theta)
        sin_theta = torch.where(torch.abs(sin_theta) < 1e-6, zeros, sin_theta)
        return self.normalize(torch.atan2(sin_theta, cos_theta))

    @staticmethod
    def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
        pi = torch.tensor(3.14159265358979323846)
        return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0

    @staticmethod
    def rad2deg(tensor: torch.Tensor) -> torch.Tensor:
        pi = torch.tensor(3.14159265358979323846)
        return tensor * 180.0 / pi.to(tensor.device).type(tensor.dtype)

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def _expm(self, v: torch.Tensor) -> torch.Tensor:
        # NB: Don't do mod 2\pi here, as it seems to cause equivaraince error.
        # Because if you do \mod 2\pi and the sign flips, you don't get g, g_inv
        # whenever you do sample_pair. One could do fmod (keeps sign), but it
        # isn't necessary for expm, since periodicity is given by sin/cos.
        theta_sq = v**2
        theta_mask = theta_sq >= self._sq_eps

        # Taylor approximation
        sin_theta = torch.where(
            theta_mask,
            torch.sin(v),
            v * (1.0 - (theta_sq / 6.0) * (1.0 - (theta_sq / 20.0))),
        )
        cos_theta = torch.where(
            theta_mask,
            torch.cos(v),
            1.0 - theta_sq * 0.5 * (1.0 - (theta_sq / 12.0)),
        )
        out = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1)
        out = out.reshape(-1, 2, 2)
        return out

    def _logm(self, g: torch.Tensor) -> torch.Tensor:
        return self.hat(self.matrix_to_radians(g))

    def hat(self, v: torch.Tensor) -> torch.Tensor:
        return v[..., None, None] * self.basis_matrix

    def left_xi_inv(self, g: torch.Tensor) -> torch.Tensor:
        # For compatibility with other groups
        # This map is simply the matrix log + vee here
        return self.vee(self.logm(g)).unsqueeze(-1)

    def sample_pair(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Only for compatibility with other groups.
        # Not used in conv layers, so we hardcode equidistant sampling.
        bound = 2.0 * self.pi
        v = torch.linspace(
            0.0,
            bound * (n_samples - 1.0) / n_samples,
            steps=n_samples,
            device=self.basis_matrix.device,
            dtype=self.basis_matrix.dtype,
        )
        v_out = torch.cat((v, -v), dim=0)
        exp_out = self.expm(v_out)
        return exp_out[:n_samples], exp_out[n_samples:]

    def normalize(self, radian: torch.Tensor) -> torch.Tensor:
        return torch.remainder(radian, 2.0 * self.pi)

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        # Changed from transpose(-2, -1) for numerical tests.
        rad = self.matrix_to_radians(g)
        inv_rad = (-rad) % (2.0 * self.pi)
        return self.expm(inv_rad)

    def vee(self, v_hat: torch.Tensor) -> torch.Tensor:
        return v_hat[..., 1, 0] / self.basis_matrix[0, 1, 0]


class LogUniformSO2:
    def __init__(self, son: SO2, bounds: float = 1.0, **kwargs):
        assert 0.0 < bounds <= 1.0, bounds
        self.son = son
        self.bounds = bounds

    @torch.no_grad()
    def _sample(self, n_samples: int) -> torch.Tensor:
        bound = self.son.pi * self.bounds
        v = bound * (
            2.0
            * torch.rand(
                n_samples,
                device=self.son.basis_matrix.device,
                dtype=self.son.basis_matrix.dtype,
            )
            - 1.0
        )
        return v

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.son.expm(self._sample(n_samples))

    def sample_pair(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        v = self._sample(n_samples)
        v_out = torch.cat((v, -v), dim=0)
        exp_out = self.son.expm(v_out)
        return exp_out[:n_samples], exp_out[n_samples:]


class EquiDistantSO2:
    def __init__(self, son: SO2, bounds: float = 1.0):
        assert 0.0 < bounds <= 1.0, bounds
        self.son = son
        self.bounds = bounds

    @torch.no_grad()
    def _sample(self, n_samples: int) -> torch.Tensor:
        bound = self.son.pi * self.bounds
        v = torch.linspace(
            -bound,
            bound,
            steps=n_samples,
            device=self.son.basis_matrix.device,
            dtype=self.son.basis_matrix.dtype,
        )
        return v

    def sample(self, n_samples: int) -> torch.Tensor:
        v = self._sample(n_samples)
        return self.son.expm(v)

    def sample_pair(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        v = self._sample(n_samples)
        v_out = torch.cat((v, -v), dim=0)
        exp_out = self.son.expm(v_out)
        return exp_out[:n_samples], exp_out[n_samples:]


class HaarUniformSOn:
    """
    Sample SOn matrices from the haar uniform distribution.
    Reference: https://github.com/lezcano/geotorch/blob/master/geotorch/so.py
    """

    def __init__(self, son: SOn, **kwargs):
        self.son = son
        self.dim = son.dim

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        x = torch.randn(
            (n_samples, self.son.n, self.son.n),
            device=self.son.basis_matrix.device,
            dtype=self.son.basis_matrix.dtype,
        )
        q, r = torch.linalg.qr(x)

        # Make uniform (diag r >= 0)
        d = r.diagonal(dim1=-2, dim2=-1).sign()
        q *= d.unsqueeze(-2)

        # Make them have positive determinant by multiplying the
        # first column by -1 (does not change the measure)
        mask = (torch.det(q) >= 0.0).to(torch.get_default_dtype())
        mask[mask == 0.0] = -1.0
        mask = mask.unsqueeze(-1)
        q[..., 0] *= mask
        return q

    def sample_pair(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.sample(n_samples)
        return r, r.transpose(-2, -1)


class SPDBasis(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self._off_diagonal = 1.0 / math.sqrt(2.0 * alpha)
        self._alpha = alpha


class SSPD2Basis(SPDBasis):
    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha=alpha)
        basis = torch.zeros(2, 2, 2)
        basis[0] = torch.tensor([[self._off_diagonal, 0.0], [0.0, -self._off_diagonal]])
        basis[1] = torch.tensor([[0.0, self._off_diagonal], [self._off_diagonal, 0.0]])
        self.register_buffer("basis_matrix", basis)

    def vee(self, v_hat: torch.Tensor) -> torch.Tensor:
        a = v_hat[..., 0, 0] / self._off_diagonal
        b = v_hat[..., 0, 1] / self._off_diagonal
        return torch.stack((a, b), dim=-1)


class SPD2Basis(SPDBasis):
    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha=alpha)
        self._diagonal = math.sqrt(self._alpha)
        basis = torch.zeros(3, 2, 2)
        basis[0] = torch.tensor([[1.0 / self._diagonal, 0.0], [0.0, 0.0]])
        basis[1] = torch.tensor([[0.0, self._off_diagonal], [self._off_diagonal, 0.0]])
        basis[2] = torch.tensor([[0.0, 0.0], [0.0, 1.0 / self._diagonal]])
        self.register_buffer("basis_matrix", basis)

    def vee(self, v_hat: torch.Tensor) -> torch.Tensor:
        a = v_hat[..., 0, 0] * self._diagonal
        b = v_hat[..., 0, 1] / self._off_diagonal
        c = v_hat[..., 1, 1] * self._diagonal
        return torch.stack((a, b, c), dim=1)


class SPD(MatrixManifold):
    def __init__(
        self, n: int, basis: SPDBasis, traceless: bool, metric_alpha: float, **kwargs
    ):
        super().__init__(
            dim=(n * (n + 1) // 2) - int(traceless),
            matrix_shape=n**2,
            metric_alpha=metric_alpha,
            **kwargs,
        )
        self.n = n
        self.basis = basis
        self.traceless = traceless

    def belongs(self, g: torch.Tensor) -> bool:
        is_symmetric = torch.allclose(g, g.transpose(-1, -2)) and (
            g.shape[-1] == g.shape[-2]
        )
        if not is_symmetric:
            return False
        try:
            torch.linalg.cholesky(g)
            if self.traceless:
                return torch.allclose(
                    torch.det(g), torch.tensor(1.0, dtype=g.dtype, device=g.device)
                )

            return True
        except RuntimeError:
            return False

    def _expm(self, v: torch.Tensor) -> torch.Tensor:
        return self.hat_expm(self.hat(v))

    def hat(self, v: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(v, self.basis.basis_matrix, ([-1], [0]))

    def hat_expm(self, v_hat: torch.Tensor) -> torch.Tensor:
        return sym_expm(v_hat)

    def _logm(self, g: torch.Tensor) -> torch.Tensor:
        return sym_logm(g)

    def vee(self, v_hat: torch.Tensor) -> torch.Tensor:
        return self.basis.vee(v_hat)

    def sqrt(self, g: torch.Tensor) -> torch.Tensor:
        return sym_sqrt(g)

    def inv_sqrt(self, g: torch.Tensor) -> torch.Tensor:
        return sym_inv_sqrt(g)


class SPD2(SPD):
    def __init__(self, traceless: bool, metric_alpha: float, **kwargs):
        super().__init__(
            n=2,
            basis=SSPD2Basis(metric_alpha) if traceless else SPD2Basis(metric_alpha),
            traceless=traceless,
            metric_alpha=metric_alpha,
            **kwargs,
        )


class LogNormalSPD2(nn.Module):
    """
    Adapted from:
    github.com/geomstats/geomstats/blob/master/geomstats/distributions/lognormal.py

    [LNGASPD2016] A. Schwartzman,
        "LogNormal distributions and"
        "Geometric Averages of Symmetric Positive Definite Matrices.",

    Currently:
        - The mean is always identity, and a parameter is not available.
        - Only diagonal covariance matrices are allowed.
        - For traceless symmetric matrices I just sample standard symmetric matrices
        and use the decomposition to extract the traceless part.
    """

    def __init__(self, spd: SPD2, bounds: float = 0.2, **kwargs):
        super().__init__()
        assert bounds > 0.0
        self.spd = spd
        self.dim = spd.dim

        if not (isinstance(bounds, tuple) or isinstance(bounds, list)):
            bounds = (bounds,) * self.dim
        else:
            if not len(bounds) == self.dim:
                raise ValueError(
                    f"Expected {self.dim} bounds, got {len(bounds)}: {bounds}"
                )

        self.register_buffer("mean_sym", torch.eye(2))
        self.register_buffer("mean", torch.zeros(self.dim))
        self.register_buffer("cov", torch.diag(torch.tensor(bounds)))
        self._cached_mvn = None

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        if self._cached_mvn is None:
            self._cached_mvn = MultivariateNormal(self.mean, self.cov)
        samples_euclidean = self._cached_mvn.sample((n_samples,))
        symmetric_matrices = self.spd.hat(samples_euclidean)
        if self.spd.traceless:
            symmetric_matrices = deviator(symmetric_matrices)
        return self.spd.hat_expm(symmetric_matrices)

    @torch.no_grad()
    def sample_pair(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_mvn is None:
            self._cached_mvn = MultivariateNormal(self.mean, self.cov)
        samples_euclidean = self._cached_mvn.sample((n_samples,))
        v_out = torch.cat((samples_euclidean, -samples_euclidean), dim=0)
        symmetric_matrices = self.spd.hat(v_out)
        if self.spd.traceless:
            symmetric_matrices = deviator(symmetric_matrices)
        exp_out = self.spd.hat_expm(symmetric_matrices)
        return exp_out[:n_samples], exp_out[n_samples:]

    @torch.no_grad()
    def sample_sym(
        self, mean_vec: torch.Tensor, cov: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        # Generate a symmetric matrix using a normal distribution in the tangent space.
        samples_euclidean = MultivariateNormal(mean_vec, cov).sample((n_samples,))
        return self.spd.hat(samples_euclidean)


class GLBasis(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self._off_diagonal = 1.0 / math.sqrt(2.0 * alpha)
        self._alpha = alpha

    def vee(self, v_hat: torch.Tensor):
        raise NotImplementedError


class GL2Basis(GLBasis):
    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha=alpha)
        self._diagonal = math.sqrt(self._alpha)
        basis = torch.zeros(4, 2, 2)
        basis[0] = torch.tensor([[1.0 / self._diagonal, 0.0], [0.0, 0.0]])
        basis[1] = torch.tensor([[0.0, self._off_diagonal], [self._off_diagonal, 0.0]])
        basis[2] = torch.tensor([[0.0, 0.0], [0.0, 1.0 / self._diagonal]])
        basis[3] = torch.tensor([[0.0, -self._off_diagonal], [self._off_diagonal, 0.0]])
        self.register_buffer("basis_matrix", basis)

    def vee(self, v_hat: torch.Tensor) -> torch.Tensor:
        a = v_hat[..., 0, 0] * self._diagonal
        b = (v_hat[..., 0, 1] + v_hat[..., 1, 0]) * (0.5 / self._off_diagonal)
        c = v_hat[..., 1, 1] * self._diagonal
        d = (v_hat[..., 0, 1] - v_hat[..., 1, 0]) * (-0.5 / self._off_diagonal)
        return torch.stack((a, b, c, d), dim=-1)


class GLn(MatrixManifold):
    # Assumes we are only working with connected component GL+(n, R)
    def __init__(
        self,
        n: int,
        dim: int,
        metric_alpha: float,
        basis: nn.Module,
        spd: Optional[SPD] = None,
        orthogonal: Optional[SOn] = None,
        spd_sampler: Optional[nn.Module] = None,
        orthogonal_sampler: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(
            dim=dim, matrix_shape=n**2, metric_alpha=metric_alpha, **kwargs
        )
        self.n = n
        self.basis = basis

        # SPD factor in cartan decomposition
        self.spd = spd

        # Orthogonal factor in cartan decomposition
        self.orthogonal = orthogonal

        # Samplers for the corresponding components
        self.spd_sampler = spd_sampler
        self.orthogonal_sampler = orthogonal_sampler

        # Record if it is a volume-preserving transformation
        if spd is not None:
            self.volume_preserving = self.spd.traceless
        else:
            self.volume_preserving = True

    def belongs(self, g: torch.Tensor) -> bool:
        return torch.all(torch.det(g) > 0.0) and (g.shape[-1] == g.shape[-2])

    def _expm(self, v: torch.Tensor) -> torch.Tensor:
        return self.hat_expm(self.hat(v))

    def hat(self, v: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(v, self.basis.basis_matrix, ([-1], [0]))

    def vee(self, v_hat: torch.Tensor) -> torch.Tensor:
        return self.basis.vee(v_hat)

    @classmethod
    def left_polar_decomposition_group(
        cls, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # chi^{-1}: GL(n) -> SPD(n) x SO(n)
        g_spd_squared = g @ g.transpose(-1, -2)
        g_spd, g_spd_inv = sym_sqrt_and_inv_sqrt(g_spd_squared)
        return g_spd, g_spd_inv @ g

    @classmethod
    def left_polar_decomposition_cartan(
        cls, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # \varPhi^{-1}: G \to m x H
        return left_polar_canonical_decomposition(g)

    @classmethod
    def left_polar_group(cls, p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # chi: M x SO(n) \to G
        # where M is either SPD(n) or SSPD(n)
        return p @ r

    @classmethod
    def right_polar_group(cls, p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r @ p

    def left_polar_cartan(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # \varPhi: m x H \to G
        return self.spd.hat_expm(x) @ r

    def left_xi_inv_no_vee(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # xi^{-1}: G \to \mathfrak{g}
        log_p, rotation = self.left_polar_decomposition_cartan(g)
        log_r = self.orthogonal.logm(rotation)
        return log_p, log_r

    def left_xi_inv(self, g: torch.Tensor) -> torch.Tensor:
        log_p, log_r = self.left_xi_inv_no_vee(g)
        return self.vee(log_p + log_r)

    def left_sqrt_polar_group(self, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # \varphi: M x SO(n) \to G
        return self.spd.sqrt(s) @ r

    def left_sqrt_inv_polar_group(
        self, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # \varphi^{-1} = \psi: G \to M x SO(n)
        g_spd_squared = g @ g.transpose(-1, -2)
        inv_sqrt = self.spd.inv_sqrt(g_spd_squared)
        return g_spd_squared, inv_sqrt @ g

    def left_exp_riemannian(self, v: torch.Tensor) -> torch.Tensor:
        v_hat = self.hat(v)
        v_hat_t = v_hat.transpose(-2, -1)
        return torch.bmm(torch.matrix_exp(v_hat_t), torch.matrix_exp(v_hat - v_hat_t))

    def sample_product_polar(self, n_samples: int) -> torch.Tensor:
        p, r = (
            self.spd_sampler.sample(n_samples),
            self.orthogonal_sampler.sample(n_samples),
        )
        return self.left_polar_group(p, r)

    def sample_product_polar_sqrt(self, n_samples: int) -> torch.Tensor:
        p, r = (
            self.spd_sampler.sample(n_samples),
            self.orthogonal_sampler.sample(n_samples),
        )
        return self.left_polar_group(self.spd.sqrt(p), r)

    def sample_pair(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        # sample (g, g^-1) pair
        psd, psd_inv = self.spd_sampler.sample_pair(n_samples)
        rot, rot_inv = self.orthogonal_sampler.sample_pair(n_samples)
        # A = S^{1/2}R
        return self.left_polar_group(self.spd.sqrt(psd), rot), self.right_polar_group(
            self.spd.sqrt(psd_inv), rot_inv
        )

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        # NB: You should never need to use this for the conv layers, it is slow.
        return torch.linalg.inv(g)

    def __repr__(self):
        return "{}, metric_alpha {}, spd sampler {}, orth sampler {}".format(
            self.__class__.__name__,
            self._metric_alpha,
            self.spd_sampler.__class__.__name__,
            self.orthogonal_sampler.__class__.__name__,
        )


class GL2(GLn):
    # GL2 with trace metric
    def __init__(
        self,
        metric_alpha: float,
        spd: Optional[SPD] = None,
        orthogonal: Optional[SOn] = None,
        spd_sampler: Optional[nn.Module] = None,
        orthogonal_sampler: Optional[nn.Module] = None,
        **kwargs,
    ):
        if spd is not None and spd.traceless:
            raise ValueError("Cannot have traceless SPD factor for GL2.")

        super().__init__(
            n=2,
            dim=4,
            metric_alpha=metric_alpha,
            basis=GL2Basis(metric_alpha),
            spd=spd,
            orthogonal=orthogonal,
            spd_sampler=spd_sampler,
            orthogonal_sampler=orthogonal_sampler,
            **kwargs,
        )


class SLn(GLn):
    def __init__(
        self,
        n: int,
        basis: nn.Module,
        metric_alpha: float,
        spd: Optional[SPD] = None,
        orthogonal: Optional[SOn] = None,
        spd_sampler: Optional[nn.Module] = None,
        orthogonal_sampler: Optional[nn.Module] = None,
        **kwargs,
    ):
        if spd is not None and not spd.traceless:
            raise ValueError("Cannot have non-traceless SPD factor for SLn")

        super().__init__(
            n=n,
            dim=n**2 - 1,
            metric_alpha=metric_alpha,
            basis=basis,
            spd=spd,
            orthogonal=orthogonal,
            spd_sampler=spd_sampler,
            orthogonal_sampler=orthogonal_sampler,
        )

    def belongs(self, g: torch.Tensor) -> bool:
        return torch.allclose(
            torch.det(g), torch.tensor(1.0, dtype=g.dtype, device=g.device)
        ) and (g.shape[-1] == g.shape[-2])


class SL2Basis(GLBasis):
    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)
        basis = torch.zeros(3, 2, 2)
        basis[0] = torch.tensor([[0.0, self._off_diagonal], [self._off_diagonal, 0.0]])
        basis[1] = torch.tensor([[0.0, -self._off_diagonal], [self._off_diagonal, 0.0]])
        basis[2] = torch.tensor([[self._off_diagonal, 0.0], [0.0, -self._off_diagonal]])
        self.register_buffer("basis_matrix", basis)

    def vee(self, v_hat: torch.Tensor) -> torch.Tensor:
        a = (0.5 / self._off_diagonal) * (v_hat[..., 0, 1] + v_hat[..., 1, 0])
        b = (-0.5 / self._off_diagonal) * (v_hat[..., 0, 1] - v_hat[..., 1, 0])
        c = v_hat[..., 0, 0] / self._off_diagonal
        return torch.stack((a, b, c), dim=-1)


class SL2(SLn):
    def __init__(
        self,
        metric_alpha: float,
        spd: Optional[SPD] = None,
        orthogonal: Optional[SOn] = None,
        spd_sampler: Optional[nn.Module] = None,
        orthogonal_sampler: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(
            n=2,
            basis=SL2Basis(alpha=metric_alpha),
            metric_alpha=metric_alpha,
            spd=spd,
            orthogonal=orthogonal,
            spd_sampler=spd_sampler,
            orthogonal_sampler=orthogonal_sampler,
        )
