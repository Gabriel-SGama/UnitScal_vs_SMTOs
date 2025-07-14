# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/

import torch
import torch.nn.functional as F

from supervised_experiments.loaders.cityscapes_labels import city_ignore_index


def nll(pred, gt, average=True):
    if average:
        return F.nll_loss(pred, gt)
    else:
        return F.nll_loss(pred, gt, reduction="none")


def msel(pred, gt, average=True):
    if average:
        loss = F.mse_loss(pred, gt)
    else:
        loss = F.mse_loss(pred, gt, reduction="none")
        loss = loss.mean(dim=[1, 2])

    return loss


def msel_qm9(pred, gt, average=True):
    return F.mse_loss(pred, gt, reduction="none").mean(0)


def cross_entropy2d(log_p, target, weight=None, average=True):
    if average:
        loss = F.nll_loss(log_p, target.long(), ignore_index=city_ignore_index, weight=weight)
    else:
        loss = F.nll_loss(log_p, target.long(), ignore_index=city_ignore_index, weight=weight, reduction="none")
        loss = loss.mean(dim=[1, 2])  # to get a scalar per pic

    return loss


# also used for the cityscapes disparity loss
def l1_loss_depth(input, target, average=True):
    mask = (torch.sum(target, dim=1) != 0).unsqueeze(1)

    if average:
        lss = F.l1_loss(input[mask], target[mask])
    else:
        # Get the losses per batch entry.
        lss = torch.zeros((mask.shape[0],), device=mask.device)
        for idx in range(mask.shape[0]):
            lss[idx] = F.l1_loss(input[idx][mask[idx]], target[idx][mask[idx]])
    return lss


def dot_loss_normal(input, target, average=True):
    mask = (torch.sum(target, dim=1) != 0).unsqueeze(1)
    mask = mask.expand(-1, 3, -1, -1)
    # normal loss: dot product
    loss = 1 - (input[mask] * target[mask]).mean()

    return loss


def l1_loss_instance(input, target, average=True):
    mask = target != city_ignore_index

    if mask.data.sum() < 1:
        # no instance pixel
        print("no instance pixel")
        return torch.zeros(1, requires_grad=True)

    if average:
        lss = F.l1_loss(input[mask], target[mask])
    else:
        lss = F.l1_loss(input[mask], target[mask], reduction="none")
    return lss


def get_loss(dataset, tasks):
    loss_fn = {}

    if "mnist" in dataset:
        for t in tasks:
            if t[0] == "C":
                loss_fn[t] = nll
            elif t[0] == "R":
                loss_fn[t] = msel

        return loss_fn

    if "city" in dataset:
        if "S" in tasks:
            loss_fn["S"] = cross_entropy2d
        if "I" in tasks:
            loss_fn["I"] = l1_loss_instance
        if "D" in tasks:
            loss_fn["D"] = l1_loss_depth
        return loss_fn

    if "QM9" in dataset:
        for t in tasks:
            loss_fn[t] = msel_qm9

        return loss_fn
