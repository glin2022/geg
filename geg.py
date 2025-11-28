
import torch
from torch import Tensor
from torch.optim import Optimizer


def _sinhc(z: Tensor) -> Tensor:
    """
    sinhc(z) = sinh(z)/z, uses a Taylor series for z~0.
    """
    eps = 1e-8
    out = torch.empty_like(z)
    mask = z.abs() < eps
    if mask.any():
        z2 = (z[mask]) ** 2
        out[mask] = 1.0 + z2 / 6.0 + (z2 * z2) / 120.0
    if (~mask).any():
        out[~mask] = torch.sinh(z[~mask]) / z[~mask]
    return out



def _log_geom(
    x: Tensor,
    geometry: str,
    q: float,
    kappa: float,
    eps: float,
    a: float,
    b: float,
) -> Tensor:
    """
    four mirror maps
    - geometry='eg'      : standard ln(x)
    - geometry='tsallis' : Tsallis q-log
    - geometry='kappa'   : Kaniadakis κ-log
    - geometry='geg'   : general Euler (a,b)-log
    All geometries assume x > 0; x >= eps on code.
    """
    x = torch.clamp(x, min=eps)

    if geometry == "eg":
        return torch.log(x)

    elif geometry == "tsallis":
        # Tsallis q-log: (x^{1-q} - 1) / (1 - q)
        # if q~1, equal to eg, ln(x)
        if abs(q - 1.0) < 1e-6:
            return torch.log(x)
        return (x.pow(1.0 - q) - 1.0) / (1.0 - q)

    elif geometry == "kappa":
        # Kaniadakis κ-log: (x^κ - x^{-κ}) / (2 κ)
        # if κ~0, equal to eg, ln(x)
        if abs(kappa) < 1e-6:
            return torch.log(x)
        return (x.pow(kappa) - x.pow(-kappa)) / (2.0 * kappa)

    elif geometry == "geg":
        # GEG-log: log_{a,b}(x) = exp(r * t) * t * sinhc(k * t),
        # where t = ln (x), r = (a+b)/2, k = (a-b)/2
        t = torch.log(x)
        if abs(a - b) < 1e-8:
            # if a=b: log_{a,a}(x) = x^a * ln x
            return x.pow(a) * t
        r = 0.5 * (a + b)
        k = 0.5 * (a - b)
        t = t.to(dtype=x.dtype, device=x.device)
        r_t = r * t
        k_t = k * t
        return torch.exp(r_t) * t * _sinhc(k_t)

    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def _exp_euler_newton(
    y: Tensor,
    a: float,
    b: float,
    eps: float,
    max_iter: int = 15,
    tol: float = 1e-6,
) -> Tensor:
    """
    The approximation of Euler (a,b)-exp by Newton's method.
    To solve log_{a,b}(x) = y, let t = ln x, solve F(t) = 0:

    F(t) = exp(r t) * t * sinhc(k t) - y
    F'(t) = exp(r t) * (r t sinhc(k t) + cosh(k t))

    - max_iter = 15, save computation
    - limit t in [-10, 10] to prevent exp overflow
    """
    device = y.device
    dtype = y.dtype

    if abs(a) < 1e-8 and abs(b) < 1e-8:
        return torch.exp(y)

    r = 0.5 * (a + b)
    k = 0.5 * (a - b)

    # Newton's method
    t = torch.clamp(y.clone().to(device=device, dtype=dtype), min=-5.0, max=5.0)

    for _ in range(max_iter):
        t = t.clamp(min=-10.0, max=10.0)

        k_t = k * t
        s = _sinhc(k_t)
        ert = torch.exp(r * t)
        F = ert * t * s - y
        dF = ert * (r * t * s + torch.cosh(k_t))

        # avoid 0 in dF
        dF = torch.where(dF.abs() < 1e-12, dF.sign() * 1e-12, dF)
        dF = torch.where(dF.abs() < 1e-13, 1e-12, dF)
        step = F / dF

        t_new = t - step
        if step.abs().max() < tol * (1.0 + t.abs().max()):
            t = t_new
            break
        t = t_new

    x = torch.exp(t)
    return torch.clamp(x, min=eps)


def exp_euler_lagrange(x: Tensor, a: float, b: float, eps: float) -> Tensor:
    """
    In Eq. (49):
        exp_{a,b}(x) ≈ 1 + x
            + 1/2 (1 - a - b) x^2
            + [ 1/2 (1 - a - b)^2
                + 1/6 (-2 + 3a - a^2 + 3b - a*b - b^2)] x^3
            - 1/24 (a + 3b - 1)
                * (2a + 2b - 1)
                * (3a + b - 1) x^4
        + O(x^5)
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x

    c2 = 0.5 * (1.0 - a - b)
    c3 = 0.5 * (1.0 - a - b) ** 2 \
         + (1.0 / 6.0) * (-2.0 + 3.0 * a - a ** 2 + 3.0 * b - a * b - b ** 2)
    c4 = -(1.0 / 24.0) * (a + 3.0 * b - 1.0) \
         * (2.0 * a + 2.0 * b - 1.0) \
         * (3.0 * a + b - 1.0)

    x = 1.0 + x + c2 * x2 + c3 * x3 + c4 * x4

    return torch.clamp(x, min=eps)


def _exp_geom(
    y: Tensor,
    geometry: str,
    q: float,
    kappa: float,
    eps: float,
    a: float,
    b: float,
) -> Tensor:

    if geometry == "eg":
        # standard EG：exp(y)
        return torch.exp(y)

    elif geometry == "tsallis":
        # Tsallis q-exp: [1 + (1-q) y]_+^{1/(1-q)}
        if abs(q - 1.0) < 1e-6:
            return torch.exp(y)
        base = 1.0 + (1.0 - q) * y
        base = torch.clamp(base, min=0.0)
        return base.pow(1.0 / (1.0 - q))

    elif geometry == "kappa":
        # Kaniadakis κ-exp: [κ y + sqrt(1 + κ^2 y^2)]^{1/κ}
        if abs(kappa) < 1e-6:
            return torch.exp(y)
        inner = kappa * y + torch.sqrt(1.0 + (kappa * y) ** 2)
        inner = torch.clamp(inner, min=1e-30)
        return inner.pow(1.0 / kappa)

    elif geometry == "geg":
            # approximation of exp_{a,b} by Newton's method
        return _exp_euler_newton(y, a=a, b=b, eps=eps, max_iter=10)
            # approximation of exp_{a,b} by Lagrange's method
        # return exp_euler_lagrange(y, a=a, b=b, eps=eps)

    else:
        raise ValueError(f"Unknown geometry: {geometry}")


class GEG(Optimizer):
    """
    u_{t+1} = exp_geom( log_geom(u_t) - lr * g )
    v_{t+1} = exp_geom( log_geom(v_t) + lr * g )
    p_{t+1} = u_{t+1} - v_{t+1}

    where g = dL/dp (+ weight_decay * p)

    ----
    geometry : 'eg' / 'tsallis' / 'kappa' / 'geg'
    """

    def __init__(
        self,
        params,
        lr=0.1,
        geometry="eg",
        q=1.2, kappa=0.3, a=-0.3, b=0.6,
        weight_decay=5e-4,
        eps=1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if geometry not in ("eg", "tsallis", "kappa", "geg"):
            raise ValueError(f"geometry must be 'eg', 'tsallis', 'kappa' or 'geg', got {geometry}")

        defaults = dict(
            lr=lr,
            geometry=geometry,
            q=q, kappa=kappa, a=a, b=b,
            weight_decay=weight_decay,
            eps=eps,
        )
        super(GEG, self).__init__(params, defaults)

        # initialize u, v
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if "u" not in state or "v" not in state:
                        with torch.no_grad():
                            u0 = torch.clamp(p.data, min=0.0).clone().detach()
                            v0 = torch.clamp(-p.data, min=0.0).clone().detach()
                            u0.add_(eps)
                            v0.add_(eps)
                            state["u"] = u0
                            state["v"] = v0

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        """

        for group in self.param_groups:
            lr = group["lr"]
            geometry = group["geometry"]
            q = group["q"]
            kappa = group["kappa"]
            a = group["a"]
            b = group["b"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # L2 weight decay
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                u = state["u"]
                v = state["v"]

                if u.device != p.device or u.dtype != p.dtype:
                    u = u.to(device=p.device, dtype=p.dtype)
                    state["u"] = u
                if v.device != p.device or v.dtype != p.dtype:
                    v = v.to(device=p.device, dtype=p.dtype)
                    state["v"] = v

                u = torch.clamp(u, min=eps)
                v = torch.clamp(v, min=eps)

                # map to mirror space and do the update
                log_u = _log_geom(u, geometry, q, kappa, eps, a, b)
                log_v = _log_geom(v, geometry, q, kappa, eps, a, b)

                log_u_new = log_u - lr * g
                log_v_new = log_v + lr * g

                # map back to original space
                u_new = _exp_geom(log_u_new, geometry, q, kappa, eps, a, b)
                v_new = _exp_geom(log_v_new, geometry, q, kappa, eps, a, b)

                u_new = torch.clamp(u_new, min=eps)
                v_new = torch.clamp(v_new, min=eps)

                state["u"].copy_(u_new)
                state["v"].copy_(v_new)

                # update p = u - v
                p.copy_(u_new - v_new)

class GEGP(Optimizer):
    """
    GEG Positive:
    GEG optimizer under nonnegative constraints.
    """
    def __init__(
        self,
        params,
        lr=0.1,
        geometry="eg",
        q=1.2, kappa=0.3, a=-0.3, b=0.6,
        weight_decay=5e-4,
        eps=1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if geometry not in ("eg", "tsallis", "kappa", "geg"):
            raise ValueError(f"geometry must be 'eg', 'tsallis', 'kappa' or 'geg', got {geometry}")

        defaults = dict(
            lr=lr,
            geometry=geometry,
            q=q, kappa=kappa, a=a, b=b,
            weight_decay=weight_decay,
            eps=eps,
        )
        super(GEGP, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            lr = group["lr"]
            geometry = group["geometry"]
            q = group["q"]
            kappa = group["kappa"]
            a = group["a"]
            b = group["b"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # L2 weight decay
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                # Enforce positivity
                p.clamp_(min=eps)

                log_p = _log_geom(p, geometry, q, kappa, eps, a, b)
                log_p_new = log_p - lr * g
                p_new = _exp_geom(log_p_new, geometry, q, kappa, eps, a, b)

                p_new = torch.clamp(p_new, min=eps)

                p.copy_(p_new)
