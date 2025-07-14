import copy
import torch
import numpy as np
from optimizers.utils import MTLOptimizer
from scipy.optimize import minimize


class CAGrad(MTLOptimizer):
    def __init__(
        self,
        optimizer,
        tasks,
        c=0.5,
        specialized_parameters=None,
        scheduler=None,
    ):
        super().__init__(optimizer, tasks, scheduler=scheduler)
        self.n_tasks = len(tasks)
        self.normalize = "l2"
        self.specialized_params = specialized_parameters
        self.alpha = c
        self.rescale = 1

    def _get_update_direction(self, grads, shared, objectives, shapes=None):
        shared_grads = torch.stack([g_i[shared] for g_i in grads]).t()

        GG = shared_grads.t().mm(shared_grads).cpu()  # [num_tasks, num_tasks]

        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (self.alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c * np.sqrt(x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1)) + 1e-8)
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x

        ww = torch.Tensor(w_cpu).to(shared_grads.device)
        gw = (shared_grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = shared_grads.mean(1) + lmbda * gw

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)  # ignores gradient on the shared weights
        for idx in range(self.n_tasks):
            merged_grad[~shared] += grads[idx][~shared]

        if self.rescale == 0:
            merged_grad[shared] = g
        elif self.rescale == 1:
            merged_grad[shared] = g / (1 + self.alpha**2)
        else:
            merged_grad[shared] = g / (1 + self.alpha)

        weights = 1 / self.n_tasks + (c / (gw_norm + 1e-8)) * ww

        return merged_grad, self.weight_to_dict(weights.cpu().numpy(), "weights")
        