# Code from https://github.com/Cranial-XIX/FAMO
import torch
import torch.nn.functional as F

from optimizers.utils import MTLOptimizer


class FAMO(MTLOptimizer):
    """
    Fast Adaptive Multitask Optimization.
    """

    def __init__(
        self,
        optimizer,
        device: torch.device,
        tasks,
        scheduler=None,
        gamma: float = 0.01,  # the regularization coefficient
        w_lr: float = 0.025,  # the learning rate of the task logits
        max_norm: float = 1.0,  # the maximum gradient norm
    ):
        super().__init__(optimizer, tasks, scheduler=scheduler)
        self.n_tasks = len(tasks)
        self.min_losses = torch.zeros(self.n_tasks).to(device)
        self.w = torch.tensor([0.0] * self.n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.device = device

    def set_min_losses(self, losses):
        self.min_losses = losses

    def set_auxiliaries(self, **kwargs):
        self.shared_parameters = kwargs["shared_parameters"]

    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        scaling_weights = (z / (c * (losses - self.min_losses) + 1e-8)).detach()
        return loss, scaling_weights

    def update(self, curr_loss):
        if curr_loss[0].dim() > 0 and curr_loss[0].shape[0] > 1:
            curr_loss = torch.cat([(obj.mean(dim=0)).view(1) for obj in curr_loss])

        # specific case from the QM9 dataset - needs to be tensor to allow math operations
        if isinstance(curr_loss, list):
            curr_loss = torch.cat(curr_loss)

        delta = (self.prev_loss - self.min_losses + 1e-8).log() - (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1), self.w, grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def custom_backwards(
        self,
        objectives,
    ):
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = torch.cat([(obj.mean(dim=0)).view(1) for obj in objectives])

        # specific case from the QM9 dataset - needs to be tensor to allow math operations
        if isinstance(objectives, list):
            objectives = torch.cat(objectives)

        loss, scaling_weights = self.get_weighted_loss(losses=objectives)
        if self.max_norm > 0 and self.shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(self.shared_parameters, self.max_norm)
        loss.backward()

        return self.weight_to_dict(scaling_weights.cpu().numpy(), "weights")
