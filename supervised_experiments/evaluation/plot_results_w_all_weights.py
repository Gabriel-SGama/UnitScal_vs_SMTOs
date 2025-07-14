import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
from tueplots import bundles, figsizes, fontsizes

from supervised_experiments import utils
from supervised_experiments import metrics
from supervised_experiments.evaluation import utils as eval_utils
from supervised_experiments.evaluation.utils import labels_opt, caption_dict

plt.rcParams.update(bundles.icml2024(usetex=True, column="half"))

figsize = bundles.icml2024(usetex=True, column="half")["figure.figsize"]
figsize_subplot = (figsize[0] * np.sqrt(2), figsize[1] / np.sqrt(2))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--model", type=str, default="lenet")
parser.add_argument("--tasks", type=str, default="CL_CR")
parser.add_argument("--analysis_test", type=str, default="")
args = parser.parse_args()

assert args.dataset == "mnist"
assert args.model == "lenet"
assert args.tasks in ["CL_CR", "RL_RR"]
assert args.analysis_test == ""

os.makedirs("plots", exist_ok=True)

with open("supervised_experiments/configs.json") as config_params:
    configs = json.load(config_params)

with open("supervised_experiments/evaluation/multi_task_best_results_test.json") as multi_task_data_file:
    multi_task_test_data = json.load(multi_task_data_file)

# To complete the 'get_metrics' function:
with open("supervised_experiments/evaluation/single_task_reference_val.json") as single_task_reference_file:
    single_task_reference = json.load(single_task_reference_file)

weight_file_paths = glob(
    configs["utils"]["results_storage"]
    + f"analyse/save_weights/mnist/lenet/{args.tasks}/*/*/val_0.pkl"
)

data_model_key, _ = utils.create_data_model_key_from_arg(args)
caption = caption_dict[args.dataset]

tasks_str = args.tasks
tasks = tasks_str.split("_")

mtm_metric_name_val = "val_multi_task_metric_" + tasks_str
mtm_metric_name_test = "test_multi_task_metric_" + tasks_str
mtm_metric_name_plot = "\\nabla_{MTM}"

metrics_info = metrics.get_metrics(args, configs, tasks, single_task_reference=single_task_reference, plot_flag=True)

optimizers = list(multi_task_test_data[data_model_key][tasks_str].keys())

data_values = dict.fromkeys(optimizers)
data_info = multi_task_test_data[data_model_key][tasks_str]

best_metric_data_split_tasks, best_mtm_data = eval_utils.get_best_metric_data(
    tasks, optimizers, data_info, metrics_info, scale=100
)

opt_labels_order = [opt for opt in labels_opt.keys() if opt in optimizers]
opt_labels_plot = [labels_opt[opt_name]["plot_name"] for opt_name in opt_labels_order]
opt_color_plot = [labels_opt[opt_name]["color"] for opt_name in opt_labels_order]

data_mtm_plot = [best_mtm_data[name] for name in opt_labels_order]

box = plt.boxplot(
    data_mtm_plot,
    patch_artist=True,
    labels=opt_labels_plot,
)

fig, axes = plt.subplots(1, 2, figsize=figsize_subplot, gridspec_kw={"width_ratios": [0.65, 0.35]})

baseline_median = np.median(data_mtm_plot[0])
axes[0].axhline(y=baseline_median, color="r", linestyle="dashed")
yticklabels = axes[0].get_yticklabels()
trans = transforms.blended_transform_factory(yticklabels[0].get_transform(), axes[0].transData)
box = axes[0].boxplot(data_mtm_plot, patch_artist=True, labels=opt_labels_plot)
axes[0].tick_params(axis="x", rotation=-20)
axes[0].tick_params(axis="y")
axes[0].grid(True)
for patch, color in zip(box["boxes"], opt_color_plot):
    patch.set_facecolor(color)
axes[0].set_ylabel("Multi-task Metric (\%)")

# --------------------------- WEIGHT PLOT ---------------------------
for weight_file in weight_file_paths:
    file_data = torch.load(weight_file)
    opt_name = file_data["args"]["optimizer"]
    weight_key = labels_opt[opt_name]["weight_key"]
    
    weight_array_task_1 = np.array([stat[f"{weight_key}_{tasks[0]}"] for stat in file_data["stats"]])
    weight_array_task_2 = np.array([stat[f"{weight_key}_{tasks[1]}"] for stat in file_data["stats"]])
    
    metric_array = np.array([stat[f"{mtm_metric_name_val}"] for stat in file_data["stats"]])

    norm_factor = weight_array_task_1 + weight_array_task_2
    weight_array_task_1 /= norm_factor

    axes[1].plot(np.abs(weight_array_task_1 - 0.5), label=labels_opt[opt_name]["plot_name"], color=labels_opt[opt_name]["color"])

axes[1].grid(True)
axes[1].set_xlim(0, len(file_data["stats"]) - 1)
axes[1].set_ylabel("Error")
axes[1].set_xlabel("Epoch")

plt.savefig(f"plots/{args.dataset}/combined_figure_{args.tasks}.png", dpi=1000)
plt.savefig(f"plots/{args.dataset}/combined_figure_{args.tasks}.pdf", dpi=1000)
