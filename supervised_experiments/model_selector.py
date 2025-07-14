# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/

from supervised_experiments.models.multi_lenet import MultiLeNetE, MultiLeNetC, MultiLeNetR

from supervised_experiments.models.resnet_dilated import ResnetDilated
from supervised_experiments.models.cityscapes import DeepLabHead
from supervised_experiments.models import resnet_cityscapes
from supervised_experiments.models.QM9_net import QM9_enc, QM9_dec
import torch.nn as nn

mnist_models = {"lenet": {}}
city_models = {"resnet50": 2048, "resnet18": 512}


# change group of parameters to args
def get_model(args, tasks, device, model_name, parallel=False):
    dataset = args.dataset
    shape = args.shape
    n_classes = args.n_classes
    p = args.p
    conv_p = args.conv_p
    model = {}

    if "mnist" in dataset:
        if model_name not in mnist_models.keys():
            raise NotImplementedError(f"{model_name} not implemented")

        model["rep"] = MultiLeNetE(p=p)
        if parallel:
            model["rep"] = nn.DataParallel(model["rep"])
        model["rep"].to(device)

        for t in tasks:
            if t[0] == "C":  # classification task
                model[t] = MultiLeNetC(p=p)
            elif t[0] == "R":  # reconstruction task
                model[t] = MultiLeNetR(p=p)

            if parallel:
                model[t] = nn.DataParallel(model[t])
            model[t].to(device)

        return model

    if "city" in dataset:
        if model_name not in city_models.keys():
            raise NotImplementedError(f"{model_name} is not implemented for cityscapes")

        out_channels = city_models[model_name]

        model["rep"] = ResnetDilated(resnet_cityscapes.__dict__[model_name](pretrained=True), conv_p=conv_p)
        model["rep"].to(device)

        # img shape order is reversed
        temp_shape = shape.copy()
        temp_shape.reverse()

        if "S" in tasks:
            model["S"] = DeepLabHead(out_channels, n_classes, log_softmax_out=True, p=p, img_size=temp_shape)
            model["S"].to(device)
        if "D" in tasks:
            model["D"] = DeepLabHead(out_channels, 1, log_softmax_out=False, p=p, img_size=temp_shape)
            model["D"].to(device)
        if "I" in tasks:
            model["I"] = DeepLabHead(out_channels, 2, log_softmax_out=False, p=p, img_size=temp_shape)
            model["I"].to(device)

        return model

    if "QM9" in dataset:
        model["rep"] = QM9_enc(num_features=11, dim=64)
        model["rep"].to(device)

        for t in tasks:
            model[t] = QM9_dec(dim=64)
            model[t].to(device)

        return model
