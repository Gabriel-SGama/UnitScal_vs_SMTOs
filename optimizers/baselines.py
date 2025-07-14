import torch
from optimizers.utils import MTLOptimizer


class Baseline(MTLOptimizer):
    # Use the optimizer on a linear scalarization (by default, unit scalarization).
    def __init__(self, optimizer, tasks, scheduler=None, weights=None):
        # weights is None (meaning all ones) or a list of scalars
        super().__init__(optimizer, tasks, scheduler=scheduler)
        if weights is None:
            weights = [1.0 / len(tasks) for _ in range(len(tasks))]

        self.weights = weights

    def custom_backwards(self, objectives):
        # If the loss hasn't been averaged for the mini-batch, do it now.
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean(dim=0) for obj in objectives]

        objective = torch.zeros_like(objectives[0])
        for idx, obj in enumerate(objectives):
            objective += self.weights[idx] * obj
        objective.backward()

        return self.weight_to_dict(self.weights, "weights")
