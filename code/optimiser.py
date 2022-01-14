import torch
import torch.optim._functional as F


class DirNewton(torch.optim.Optimizer):
    def __init__(self, params, initial_lr=1e-3, ddecay=0.1, weight_decay=0) -> None:
        default = dict(initial_lr=initial_lr, ddecay=ddecay, weight_decay=weight_decay)
        self.prev_loss = None

        # adds params to param_groups and inserts defaults to groups if they do
        # not have custom parameters
        super().__init__(params, default)

        for group in self.param_groups:
            self.state[group]["lr"] = group["initial_lr"]

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

    @torch.no_grad
    def step(self, closure=None):
        with torch.enable_grad():
            loss = float(closure())

        loss_delta = None
        if self.prev_loss is not None:
            loss_delta = loss - self.prev_loss

        grad_norm_squared = 0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffer_list = []
            lr = self.state[group]["lr"]
            ddecay = group["ddecay"]
            weight_decay = group["weight_decay"]


            for param in group["params"]:
                if param.grad is not None:
                    params_with_grad.append(param)
                    grads.append(param.grad)
                    grad_norm_squared += torch.sum(torch.pow(param.grad, 2.))

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

                F.sgd(
                    params_with_grad,
                    grads,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=0,
                    dampening=0,
                    nesterov=False
                )


        for group in self.param_groups:
            old_dderivative_estimate = 1/self.state[group]["lr"]
            new_dderivative_estimate = 2 * old_dderivative_estimate * (1 + loss_delta / grad_norm_squared)
            self.state[group]["lr"] = 1 / (
                (1 - ddecay)*old_dderivative_estimate
                + ddecay*new_dderivative_estimate
            )
        
        return loss
