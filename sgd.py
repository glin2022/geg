import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    r"""
    A simple re-implementation of SGD optimizer with optional momentum,
    weight decay (L2 regularization), and Nesterov momentum.
    """

    def __init__(
        self,
        params,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        dampening=0.0,
        nesterov=False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError(
                "Nesterov momentum requires a positive momentum and zero dampening."
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                # L2 weight decay
                if weight_decay != 0.0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Momentum
                if momentum != 0.0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = d_p.clone().detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # Gradient descent update
                p.add_(d_p, alpha=-lr)