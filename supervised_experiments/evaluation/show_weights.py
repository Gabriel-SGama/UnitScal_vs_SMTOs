import os
import argparse
import copy
import json
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from tueplots import bundles, figsizes, fontsizes

import matplotlib.pyplot as plt

from supervised_experiments.evaluation.utils import labels_opt, caption_dict
from supervised_experiments import utils

plt.rcParams.update(bundles.icml2024(usetex=True, column="full"))

with open("supervised_experiments/configs.json") as config_params:
    configs = json.load(config_params)

weight_files = glob(configs["utils"]["results_storage"] + "analyse/save_weights/*/*/*/*/*/val_0.pkl")

print("number of files: ", len(weight_files))

for weight_file in weight_files:
    file_data = torch.load(weight_file, map_location="cpu")
    opt_name = file_data["args"]["optimizer"]
    weight_key = labels_opt[opt_name]["weight_key"]
    stats_dict = file_data["stats"]

    key = utils.create_data_model_key_from_data_dict(file_data)[0] + "_" + opt_name
    tasks = file_data["args"]["tasks"].split("_")

    print(f"\n\n-----------------KEY: {key} | TASKS: {tasks}-----------------")
    normalizing_factor = []
    weight_key = labels_opt[opt_name]["weight_key"]

    for j in range(len(stats_dict)):
        normalizing_factor.append(sum([stats_dict[j][weight_key + "_" + t] for t in tasks]))

    weights_values = {f"{weight_key}_{t}": [] for t in tasks}
    momentum_weights_values = {f"{weight_key}_{t}": [] for t in tasks}
    for t in zip(tasks):
        t = t[0]
        weights_values[weight_key + "_" + t] = [
            stats_dict[j][weight_key + "_" + t] / normalizing_factor[j] for j in range(len(stats_dict))
        ]
        momentum_weights_values[weight_key + "_" + t].append(weights_values[weight_key + "_" + t][0])

        for j in range(1, len(stats_dict)):
            momentum_weights_values[weight_key + "_" + t].append(
                0.9 * momentum_weights_values[weight_key + "_" + t][j - 1]
                + 0.1 * weights_values[weight_key + "_" + t][j]
            )

    total_sum = sum(
        [momentum_weights_values[weight_key + "_" + t][-1] for t in tasks]
    )  # corrects minimum precision error
    [print(weight_key + "_" + t + ":", momentum_weights_values[weight_key + "_" + t][-1] / total_sum) for t in tasks]
