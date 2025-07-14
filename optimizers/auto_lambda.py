# Code from https://github.com/lorenmt/auto-lambda

import torch
from torch import optim
import copy

from optimizers.utils import MTLOptimizer


class AutoLambda(MTLOptimizer):
    def __init__(
        self,
        args,
        model,
        optimizer,
        device,
        train_tasks,
        pri_tasks,
        loss_fn,
        lr_relation,
        weight_init=0.1,
        scheduler=None,
    ):
        super().__init__(optimizer, train_tasks, scheduler=scheduler)
        self.model = model
        self.model_ = copy.deepcopy(model)
        self.meta_weights = torch.tensor([weight_init] * len(train_tasks), requires_grad=True, device=device)
        self.meta_opt = optim.Adam([self.meta_weights], lr=lr_relation * args.lr)
        self.pri_tasks = pri_tasks
        self.loss_fn = loss_fn

    def foward_on_dict_model(self, model, input_x):
        rep, _ = model["rep"](input_x, None)

        pred = []
        for t in self._tasks:
            out_t, _ = model[t](rep, None)
            pred.append(out_t)

        return pred

    def virtual_step(self, train_x, train_y, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        # TODO: check if many to many is working
        if isinstance(train_x, list):  # multi-domain setting [many-to-many]
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:  # single-domain setting [one-to-many]
            train_pred = self.foward_on_dict_model(self.model, train_x)

        train_loss = self.model_fit(train_pred, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        model_params = [p for v in self.model.values() for p in list(v.parameters())]
        model_params_ = [p for v in self.model_.values() for p in list(v.parameters())]
        gradients = torch.autograd.grad(loss, model_params)

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(model_params, model_params_, gradients):
                if "momentum" in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    m = model_optim.state[weight].get("momentum_buffer", 0.0) * model_optim.param_groups[0]["momentum"]
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]["weight_decay"] * weight))

    def unrolled_backward(self, train_x, train_y, val_x, val_y, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """
        self.meta_opt.zero_grad()

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self._tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]

        # compute validation data loss on primary tasks
        if type(val_x) == list:
            val_pred = [self.model_(x, t) for t, x in enumerate(val_x)]
        else:
            val_pred = self.foward_on_dict_model(self.model_, val_x)

        val_loss = self.model_fit(val_pred, val_y)
        loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_params_ = [p for v in self.model_.values() for p in list(v.parameters())]

        model_weights_ = tuple(model_params_)
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, train_x, train_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = -alpha * h

        self.meta_opt.step()

    def compute_hessian(self, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            model_params = [p for v in self.model.values() for p in list(v.parameters())]
            for p, d in zip(model_params, d_model):
                p += eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.foward_on_dict_model(self.model, train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            model_params = [p for v in self.model.values() for p in list(v.parameters())]
            for p, d in zip(model_params, d_model):
                p -= 2 * eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.foward_on_dict_model(self.model, train_x)

        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            model_params = [p for v in self.model.values() for p in list(v.parameters())]
            for p, d in zip(model_params, d_model):
                p += eps * d

        hessian = [(p - n) / (2.0 * eps) for p, n in zip(d_weight_p, d_weight_n)]

        return hessian

    def model_fit(self, pred, targets):
        """
        define task specific losses
        """
        loss = [self.loss_fn[t](pred[i], targets[t]) for i, t in enumerate(self._tasks)]
        return loss

    def custom_backwards(self, objectives):
        # If the loss hasn't been averaged for the mini-batch, do it now.
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean(dim=0) for obj in objectives]

        objective = torch.zeros_like(objectives[0])
        for idx, (w, obj) in enumerate(zip(self.meta_weights, objectives)):
            objective += w * obj
        objective.backward()

        return self.weight_to_dict(self.meta_weights.detach().cpu().numpy(), "weights")
