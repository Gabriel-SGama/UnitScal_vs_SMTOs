import copy
import torch
import numpy as np
from optimizers.utils import MTLOptimizer, batch_average, standard_head_backward
from optimizers.min_norm import MinNormSolver, normalize_gradients


class CDTT(MTLOptimizer):
    # EDM (or Central direction) + task tensioners: https://arxiv.org/abs/2204.06698
    # git: https://github.com/tiemink/MTL_TaskTensioner
    MAX_BATCH_ITER = 10

    def __init__(self, optimizer, tasks, alpha, specialized_parameters=None, scheduler=None):
        super().__init__(optimizer, tasks, scheduler=scheduler)
        self.alpha = alpha
        self.n_tasks = len(tasks)
        self.normalize = "l2"
        self.specialized_params = specialized_parameters
        self.iterations = 0
        self.grad_mean_prev = []
        self.grad_accumulation = torch.zeros((self.n_tasks, self.MAX_BATCH_ITER), dtype=torch.float).cpu()

    def _accumulate_gradient_vector_slide(self, norm_factors: list) -> None:
        index = self.iterations % self.MAX_BATCH_ITER
        for i in range(self.n_tasks):
            self.grad_accumulation[i][index] = norm_factors[i]

    def _compute_tension(self, parametrization):
        a = self.alpha
        tension = a / (1 + np.exp(-parametrization * np.exp(1) + np.exp(1))) + 1 - a
        return max(0, tension)

    def _angle_between_vectors(self, vector1, vector2):
        rad = torch.arccos(torch.dot(vector1, vector2) / (torch.norm(vector1) * torch.norm(vector2))).item()
        return rad

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

        self._accumulate_gradient_vector_slide(norm_factors)
        self.iterations += 1

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

        weight_coef_final = [gamma * weight_coef[i] / norm_factors[i] for i in range(len(self._tasks))] # final weights from EDM

        if self.iterations >= self.MAX_BATCH_ITER:
            # compute mean of gradients
            grad_accumulation_mean = [torch.sum(self.grad_accumulation[i]) for i in range(len(shared_grads))]
            if self.grad_mean_prev:
                tension_vector = []
                all_tension_task = []
                all_diff_vec_norm = []
                for i in range(len(shared_grads)):
                    grad_parametrization = (
                        grad_accumulation_mean[i] / self.grad_mean_prev[i] + torch.log10(objectives[i]).cpu()
                    )
                    # Compute tension factor
                    tension_task = self._compute_tension(grad_parametrization.item())
                    all_tension_task.append(tension_task)
                    diff_vec = shared_grads[i] - common_direction
                    diff_vec_norm = torch.linalg.norm(diff_vec)
                    all_diff_vec_norm.append(diff_vec_norm)
                    unit_vector = diff_vec / diff_vec_norm

                    if tension_vector == []:
                        tension_vector = unit_vector * tension_task
                    else:
                        tension_vector += unit_vector * tension_task

                tension_vector = common_direction + tension_vector
                weight_coef_final = [coef + tension_task/vec_norm - coef*sum([all_tension_task[k]/all_diff_vec_norm[k] for k in range(len(weight_coef_final))]) for j, (coef, tension_task, vec_norm) in enumerate(zip(weight_coef_final, all_tension_task, all_diff_vec_norm))]

                ########################
                ###### check direction (∇sh,m ·v_n ≥ 0, ∀m = 1, . . . , M)
                ##############################
                for i in range(len(shared_grads)):
                    angle = self._angle_between_vectors(shared_grads[i], tension_vector) * 180 / np.pi
                    if angle > 90:
                        shared_grad_norm = torch.linalg.norm(shared_grads[i])
                        unit_vector_param = shared_grads[i] / shared_grad_norm
                        dot_tension_w_unit = torch.dot(tension_vector, unit_vector_param)
                        w = tension_vector - dot_tension_w_unit * unit_vector_param
                        w_norm = torch.linalg.norm(w)
                        w_unit = w / w_norm
                        alpha_angle = np.pi - self._angle_between_vectors(
                            shared_grads[i], (tension_vector - shared_grads[i])
                        )
                        tension_vector = np.tan(alpha_angle) * shared_grad_norm * w_unit
                        
                        weight_coef_final = [weight_coef_final[j] * shared_grad_norm * np.tan(alpha_angle) / w_norm for j in range(len(weight_coef_final))]
                        weight_coef_final[i] -= dot_tension_w_unit * np.tan(alpha_angle) / w_norm
                ##########################################

                common_direction = tension_vector

            self.grad_mean_prev = grad_accumulation_mean

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = common_direction

        for idx in range(len(grads)):
            merged_grad[~shared] += grads[idx][~shared]

        data.update(self.weight_to_dict(weight_coef_final, "cdtt_weights"))
        return merged_grad, data
