import torch
import numpy as np
from optimizers.utils import MTLOptimizer, batch_average, standard_head_backward
from optimizers.min_norm import MinNormSolver, normalize_gradients


class MGDA(MTLOptimizer):
    # MGDA: https://arxiv.org/pdf/1810.04650
    def __init__(self, optimizer, tasks, specialized_parameters=None, scheduler=None, normalize="loss+", ub=False):
        # normalize: normalization type, see function normalize_gradients. loss and loss+ require non-negative losses
        # loss+ is the default as in https://github.com/isl-org/MultiObjectiveOptimization/blob/master/sample.json
        super().__init__(optimizer, tasks, scheduler=scheduler)
        assert normalize in ["l2", "loss", "loss+", "sig-loss+", "none"]
        self.normalize = normalize
        self.ub = ub
        self.shared_repr = None
        self.specialized_params = specialized_parameters
        self._alpha_to_log = None

    def _get_update_direction(self, grads, shared, objectives, shapes=None, return_alpha=False):
        if not return_alpha:
            merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
            for idx in range(len(grads)):
                # Use plain gradients for task-specific parameters
                merged_grad[~shared] += grads[idx][~shared].clone()
            shared_grads = [g_i[shared] for g_i in grads]
        else:
            shared_grads = grads

        # Normalize all gradients, this is optional and not included in the paper.
        normalize_gradients(shared_grads, objectives, self.normalize)

        with torch.no_grad():
            # Compute convex combination coefficients on gradients of shared parameters.
            # TODO: in the paper they claim to use FW, but their code uses projected gradient descent.
            #  the authors provide find_min_norm_element_FW but do not seem to use it
            alpha, min_norm = MinNormSolver.find_min_norm_element(shared_grads)
        self._alpha_to_log = alpha
        if return_alpha:
            return alpha

        for idx in range(len(grads)):
            # Compute the affine combination for shared parameters
            merged_grad[shared] += shared_grads[idx].mul_(alpha[idx])
        return merged_grad, self.weight_to_dict(alpha, "weights")

    def custom_backwards(self, objectives):
        spec_params_flag = self.specialized_params is not None
        # Allow to switch between MGDA and MGDA-UB (see original MGDA paper)
        if not self.ub:
            # MGDA
            return MTLOptimizer.custom_backwards(self, objectives)

        # MGDA-UB
        objectives = batch_average(objectives)
        # Use approximation given by the gradient of the loss w.r.t. shared parameters to find convex combination
        # coefficients, then backward on the scaled sum
        z_grads = []
        for obj in objectives:
            z_grads.append(torch.autograd.grad(obj, self.shared_repr, only_inputs=True, retain_graph=True)[0])
        self.shared_repr = None
        # the implementation of find_min_norm_element implicitly linearizes the gradient of z
        # (which is of size batch_size x repr_size)
        alpha = self._get_update_direction(z_grads, None, objectives, return_alpha=True)
        del z_grads
        mgda_objective = sum([obj * alpha[idx] for idx, obj in enumerate(objectives)])
        mgda_objective.backward(retain_graph=spec_params_flag)

        if spec_params_flag:
            # Overwrite gradients of specialized parameters
            standard_head_backward(objectives, self.specialized_params)

        return self.weight_to_dict(alpha, "weights")

    def set_auxiliaries(self, **kwargs):
        # Pass any auxiliary parameter.
        if self.ub:
            self.shared_repr = kwargs["shared_repr"]
