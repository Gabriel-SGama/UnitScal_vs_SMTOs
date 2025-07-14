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

plt.rcParams.update(bundles.icml2024(usetex=True, column="half"))

argparser = argparse.ArgumentParser(description="Plotting Gradients")
argparser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=["mnist", "cityscapes"],
    help="dataset to use",
)
args = argparser.parse_args()

plot_ratio = 0.5/0.3
fontsize = {
    "font.size": 8 * plot_ratio,
    "axes.labelsize": 8 * plot_ratio,
    "legend.fontsize": 6 * plot_ratio,
    "xtick.labelsize": 6 * plot_ratio,
    "ytick.labelsize": 6 * plot_ratio,
    "axes.titlesize": 8 * plot_ratio,
}

plt.rcParams.update(fontsize)

with open("supervised_experiments/configs.json") as config_params:
    configs = json.load(config_params)

if args.dataset == "mnist":
    file_path = "data/saved_results/analyse/save_weights/mnist/lenet/CL_RR/cagrad/lr_0.001_bs_256_dlr_True_p_0.0_cr_0.8/val_0.pkl"
else:
    file_path = "data/saved_results/analyse/save_weights/cityscapes_sh_512_256_aug_nc_19/resnet18/S_D_I/imtl/lr_0.001_bs_32_dlr_True_p_0.0_wd_0.0001/val_0.pkl"

file_data = torch.load(file_path)

opt_name = file_data["args"]["optimizer"]
weight_key = labels_opt[opt_name]["weight_key"]
stats_dict = file_data["stats"]

key = utils.create_data_model_key_from_data_dict(file_data)[0] + "_" + opt_name

print(f"KEY: {key}")
tasks = file_data["args"]["tasks"].split("_")
normalizing_factor = []
weight_key = labels_opt[opt_name]["weight_key"]
for j in range(len(stats_dict)):
    normalizing_factor.append(sum([stats_dict[j][weight_key + "_" + t] for t in tasks]))

weights_values = {f"{weight_key}_{t}": [] for t in tasks}
momentum_weights_values = {f"{weight_key}_{t}": [] for t in tasks}
for t in zip(tasks):
    t = t[0]
    weights_values[weight_key+ "_" + t] = [
        stats_dict[j][weight_key+ "_" + t] / normalizing_factor[j] for j in range(len(stats_dict))
    ]
    momentum_weights_values[weight_key+ "_" + t].append(weights_values[weight_key+ "_" + t][0])

    for j in range(1, len(stats_dict)):
        momentum_weights_values[weight_key+ "_" + t].append(
            0.9 * momentum_weights_values[weight_key+ "_" + t][j - 1] + 0.1 * weights_values[weight_key+ "_" + t][j]
        )

    plot = plt.plot(weights_values[weight_key+ "_" + t], alpha=0.3, linewidth=plot_ratio, label=t)
            
    plt.plot(momentum_weights_values[weight_key+ "_" + t], color=plot[0].get_color(), linestyle="dotted", linewidth=plot_ratio)

# title_name = "MNIST - CAGrad" if "mnist" in file_path else "Cityscapes - EDM"
# plt.title(title_name)
plt.xlabel("Epoch")
plt.ylabel("Weight value")
plt.xlim(0, len(stats_dict) - 1)
plt.legend()
plt.grid()
plt.savefig(f"plots/momentum_weights_{key}.png", dpi=1000)
plt.savefig(f"plots/momentum_weights_{key}.pdf", dpi=1000)