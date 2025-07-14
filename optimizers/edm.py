import copy
import torch
import numpy as np
from optimizers.utils import MTLOptimizer, batch_average, standard_head_backward
from optimizers.min_norm import MinNormSolver, normalize_gradients


class EDM(MTLOptimizer):
    # EDM https://arxiv.org/abs/2007.06937
    # code based on: https://github.com/amkatrutsa/edm & https://github.com/isl-org/MultiObjectiveOptimization

    def __init__(
        self,
        optimizer,
        tasks,
        specialized_parameters=None,
        scheduler=None,
    ):
        super().__init__(optimizer, tasks, scheduler=scheduler)
        self.n_tasks = len(tasks)
        self.normalize = "l2"
        self.specialized_params = specialized_parameters
        self.weight_momentum = None

    def _get_update_direction(self, grads, shared, objectives, shapes=None):
        data = {}
        shared_grads = copy.deepcopy([g_i[shared] for g_i in grads])

        norm_factors = [shared_grads[i].norm() for i in range(len(shared_grads))]

        for norm_fac in norm_factors:
            if norm_fac == 0:
                merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)  # ignores gradient on the shared weights

                for idx in range(len(grads)):
                    merged_grad[~shared] += grads[idx][~shared]

                return merged_grad, data

        normalized_shared_grads = [shared_grads[i] / norm_factors[i] for i in range(len(shared_grads))]

        with torch.no_grad():
            # Compute convex combination coefficients on gradients of shared parameters.
            weight_coef, min_norm = MinNormSolver.find_min_norm_element_FW(normalized_shared_grads)

        common_direction = torch.zeros_like(shared_grads[0])
        for i in range(len(shared_grads)):
            common_direction += weight_coef[i] * normalized_shared_grads[i]

        gamma = 1 / sum(weight_coef[i] / norm_factors[i] for i in range(len(shared_grads)))
        common_direction *= gamma

        data.update(self.weight_to_dict(weight_coef, "edm"))
        data.update(self.weight_to_dict(norm_factors, "norm"))

        weight_coef_final = [gamma * weight_coef[i] / norm_factors[i] for i in range(len(self._tasks))]

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = common_direction

        for idx in range(len(grads)):
            merged_grad[~shared] += grads[idx][~shared]

        data.update(self.weight_to_dict(weight_coef_final, "edm_weights"))

        return merged_grad, data
