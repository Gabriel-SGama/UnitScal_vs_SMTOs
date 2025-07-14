import torch
from optimizers.utils import MTLOptimizer
from supervised_experiments.loaders.QM9 import task_to_qm9_idx


class UW(MTLOptimizer):
    # Uncertainty loss from https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    def __init__(self, optimizer, tasks, scheduler=None, weights=None):
        super().__init__(optimizer, tasks, scheduler=scheduler)

        for t in tasks:
            assert (
                t in ["CL", "CR", "RL", "RR", "S", "D", "I"] or t in task_to_qm9_idx.keys()
            ), f"Need to define the UW for task {t}"

        self.eta = torch.nn.Parameter(torch.Tensor([0.0] * len(tasks)), requires_grad=True)
        self.fractions_terms = torch.nn.Parameter(
            torch.Tensor([2 if (t in ["RL", "RR"] or t in task_to_qm9_idx.keys()) else 1.0 for t in tasks]),
            requires_grad=False,
        )
        self.power_terms = torch.nn.Parameter(
            torch.Tensor(
                [2 if (t in ["CL", "CR", "RL", "RR", "S"] or t in task_to_qm9_idx.keys()) else 1.0 for t in tasks]
            ),
            requires_grad=False,
        )

        self.add_extra_params(self.eta)

    def get_parameters(self):
        return self.eta

    def custom_backwards(self, objectives):
        # If the loss hasn't been averaged for the mini-batch, do it now.
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean(dim=0) for obj in objectives]

        objective = torch.zeros_like(objectives[0])
        for idx, obj in enumerate(objectives):
            objective += (
                self.eta[idx] + (obj * torch.exp(-self.power_terms[idx] * self.eta[idx])) / self.fractions_terms[idx]
            )

        objective.backward()

        return self.weight_to_dict(self.eta, "eta")
