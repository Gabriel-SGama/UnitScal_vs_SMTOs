import torch
import numpy as np
import cvxpy as cp
import copy
from optimizers.utils import MTLOptimizer


class NashMTL(MTLOptimizer):
    # code from https://github.com/AvivNavon/nash-mtl
    def __init__(
        self,
        optimizer,
        tasks,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
        specialized_parameters=None,
        scheduler=None,
    ):
        super().__init__(optimizer, tasks, scheduler=scheduler)

        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm
        self.n_tasks = len(tasks)
        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.n_tasks)
        self.nash_iter = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def set_auxiliaries(self, **kwargs):
        self.shared_parameters = kwargs["shared_parameters"]

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.n_tasks,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.n_tasks, self.n_tasks), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(-cp.log(self.alpha_param[i] * self.normalization_factor_param) - cp.log(G_alpha[i]) <= 0)
        obj = cp.Minimize(cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param)
        self.prob = cp.Problem(obj, constraint)

    def custom_backwards(self, objectives):
        # If the loss hasn't been averaged for the mini-batch, do it now.
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean(dim=0) for obj in objectives]

        weighted_loss, weights = self.get_weighted_loss(objectives)

        weighted_loss.backward()

        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.shared_parameters, self.max_norm)

        return self.weight_to_dict(weights.cpu().numpy(), "weights")

    def get_weighted_loss(
        self,
        losses,
        **kwargs,
    ):
        if self.nash_iter == 0:
            self._init_optim_problem()

        if (self.nash_iter % self.update_weights_every) == 0:
            self.nash_iter += 1
            custom_grad, shapes, shared = self._pack_grad(losses, retain_graph=True)
            shared_grads = copy.deepcopy([g_i[shared] for g_i in custom_grad])

            G = torch.stack(tuple(shared_grads))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.nash_iter += 1
            alpha = self.prvs_alpha

        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        return weighted_loss, alpha
