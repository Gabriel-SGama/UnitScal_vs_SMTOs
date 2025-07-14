import copy
import torch
import numpy as np


class MTLOptimizer:
    """
    Implements basic operations for customized MTL Optimizers.
    Note that while the optimizer's parameters can be a list of tensors, internally everything is flattened into a
    single vector (see _pack_grad and _flatten_grad).
    """

    def __init__(
        self,
        optimizer,
        tasks,
        scheduler=None,
        weight_interval=1,
    ):
        self._optim = optimizer
        self._tasks = tasks
        self._sched = scheduler
        self._weight_interval = weight_interval
        self._opt_iter = 0
        return

    @property
    def optimizer(self):
        return self._optim

    def compute_gradient_metrics(self, objectives, tasks):
        if len(tasks) == 1:
            return {}

        # NOTE: involves an additional backward pass per task, but it's just for logging (done less frequently)
        # return l2 norm of sum of original grads on the shared parameters
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean(dim=0) for obj in objectives]
        grad, _, shared = self._pack_grad(objectives, retain_graph=True)
        shared_grads = copy.deepcopy([g_i[shared] for g_i in grad])

        avg_direction = torch.zeros_like(shared_grads[0])
        for idx, t in enumerate(tasks):
            avg_direction += shared_grads[idx] / len(tasks)

        task_string = "_".join(tasks)
        gradient_data = {}
        gradient_data[f"dot_avg_dir_{task_string}"] = 0

        for idx, t in enumerate(tasks):
            gradient_data[f"dot_avg_dir_{task_string}"] += torch.dot(shared_grads[idx], avg_direction).item() / (
                len(tasks) * torch.linalg.norm(shared_grads[idx]) * torch.linalg.norm(avg_direction)
            )

        return gradient_data

    def compute_norm_sum_grads(self, objectives):
        # NOTE: involves an additional backward pass per task, but it's just for logging (done less frequently)
        # return l2 norm of sum of original grads on the shared parameters
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean(dim=0) for obj in objectives]
        grad, _, shared = self._pack_grad(objectives, retain_graph=True)
        norm_sum_grads = torch.sqrt(sum([g_i[shared] for g_i in grad]).pow(2).sum())
        self.zero_grad()
        return norm_sum_grads, shared

    def add_extra_params(self, params):
        self._optim.add_param_group({"params": params})

    def zero_grad(self):
        """
        clear the gradient of the parameters
        """
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        """
        update the parameters with the gradient
        """
        self._optim.step()
        if self._sched is not None:
            self._sched.step()

    def iterate(self, objectives, **kwargs):
        """
        compute the new customized update direction, apply it, zero the gradients
        """
        self.set_auxiliaries(**kwargs)
        if self._opt_iter % self._weight_interval == 0:
            data = self.custom_backwards(objectives)
        else:
            data = self._custom_weight_momentum_backward(objectives)

        self.step()
        self.zero_grad()
        self._opt_iter += 1
        return data

    def set_auxiliaries(self, **kwargs):
        # Pass any auxiliary parameter.
        pass

    def _custom_weight_momentum_backward(self, objectives):
        raise NotImplementedError("children classes of MTLOptimizer need to implement custom_weight_momentum_backward")

    def custom_backwards(self, objectives):
        """
        Calculate the gradient of the parameters with a MTL algorithm that computes a custom update direction.
        The direction is stored as gradient within the variables over which to optimize.

        input:
        - objectives: a list of objectives
        """

        # If the loss hasn't been averaged for the mini-batch, do it now.
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean() for obj in objectives]

        custom_grad, shapes, shared = self._pack_grad(objectives)
        custom_grad, data = self._get_update_direction(custom_grad, shared, objectives)
        custom_grad = self._unflatten_grad(custom_grad, shapes[0])
        self._set_grad(custom_grad)

        return data

    def _get_update_direction(self, grads, shared, objectives, shapes=None):
        # algorithm-dependent method to compute the update direction for MTL
        raise NotImplementedError("children classes of MTLOptimizer need to implement _get_update_direction")

    def _set_grad(self, grads):
        """
        set the modified gradients to the network
        """
        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives, retain_graph=False):
        """
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """
        grads, shapes = [], []
        shared = None
        m = len(objectives)

        for idx, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=(idx < (m - 1)) or retain_graph)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))

            # Infer which parameters are shared and which aren't
            if shared is None:
                shared = self._flatten_grad(has_grad, shape)
            else:
                shared = shared * self._flatten_grad(has_grad, shape)
            shapes.append(shape)

        self._optim.zero_grad(set_to_none=True)
        return grads, shapes, shared.bool()

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx : idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        """
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def weight_to_dict(self, weights, name):
        with torch.no_grad():
            weights_data = copy.deepcopy(weights)

            if torch.is_tensor(weights_data):
                weights_data = weights_data.cpu().numpy()

            for i in range(len(self._tasks)):
                if torch.is_tensor(weights_data[i]):
                    weights_data[i] = weights_data[i].cpu().numpy()

            return {name + "_" + self._tasks[i]: weight_data for i, weight_data in enumerate(weights_data)}


def standard_head_backward(objectives, specialized_params):
    # Perform standard backward pass (of unit scalarization) over the specialized parameters.
    # Overwrites any existing grad on these parameters.
    spec_grad = torch.autograd.grad(sum(objectives), specialized_params, only_inputs=True)
    for idx, sparam in enumerate(specialized_params):
        sparam.grad = spec_grad[idx]


def batch_average(objectives):
    # If the loss hasn't been averaged for the mini-batch, do it now.
    if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
        objectives = [obj.mean(dim=0) for obj in objectives]
    return objectives
