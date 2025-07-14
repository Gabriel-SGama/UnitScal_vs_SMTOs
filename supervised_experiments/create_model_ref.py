import random
import torch
import numpy as np

from supervised_experiments.model_selector import get_model
from supervised_experiments import utils


def check_models_equality(model1, model2):
    # Make sure both models are in evaluation mode

    for key1, key2 in zip(model1, model2):
        model1[key1].eval()
        model2[key2].eval()

        # Check if the models have the same architecture
        if model1[key1].__class__ != model2[key2].__class__:
            print("Models have different architectures.")
            return False
        else:
            # Compare the weights of the models
            for (name1, param1), (name2, param2) in zip(
                model1[key1].named_parameters(), model2[key2].named_parameters()
            ):
                print(torch.mean(torch.abs(param1 - param2)))
                if not torch.equal(param1, param2):
                    print("Models have different weights.")
                    return False

    print("models are equal")


if __name__ == "__main__":
    utils.set_seeds(0)

    args_dict = {
        "args": {
            "dataset": "mnist",
            "model": "lenet",
            "tasks": "CL_CR",
            "shape": [256, 512],
            "n_classes": 19,
            "conv_p": 0,
            "p": 0,
        }
    }

    args = utils.dict_to_argparser(args_dict)
    tasks = args.tasks.split("_")

    model1 = get_model(args, tasks, "cpu", args.model)

    # filler optimizer
    model_params = [p for v in model1.values() for p in list(v.parameters())]
    spec_params = [p for t in tasks for p in list(model1[t].parameters())]

    optimizer = torch.optim.Adam(model_params)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    utils.save_model(model1, optimizer, scheduler, tasks, -1, args, "test/", "model1")
    utils.load_saved_model(model1, tasks, "test/model1.pkl")

    utils.set_seeds(0)

    model2 = get_model(args, tasks, "cpu", args.model)

    check_models_equality(model1, model2)
