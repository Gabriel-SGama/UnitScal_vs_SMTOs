import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tueplots import bundles, figsizes, fontsizes

from supervised_experiments import utils

parser = argparse.ArgumentParser(description="Plotting Gradients")
parser.add_argument(
    "--all",
    action="store_true",
)
args = parser.parse_args()


plt.rcParams.update(bundles.icml2024(usetex=True, column="full" if args.all else "half"))

with open("supervised_experiments/evaluation/multi_task_best_results_test_grad_metrics.json") as multi_task_data_file:
    multi_task_data = json.load(multi_task_data_file)

steps_dict = [
    {
        "Name": "R50",
        "key": "cityscapes_sh_256_128_aug_nc_7_resnet50",
        "tasks": "S_D",
        "MTM": "test_multi_task_metric_S_D",
    },
    {
        "Name": "R18",
        "key": "cityscapes_sh_256_128_aug_nc_7_resnet18",
        "tasks": "S_D",
        "MTM": "test_multi_task_metric_S_D",
    },
    {
        "Name": "Inst. Seg.",
        "key": "cityscapes_sh_256_128_aug_nc_7_resnet18",
        "tasks": "S_D_I",
        "MTM": "test_multi_task_metric_S_D_I",
    },
    {
        "Name": "HR",
        "key": "cityscapes_sh_512_256_aug_nc_7_resnet18",
        "tasks": "S_D_I",
        "MTM": "test_multi_task_metric_S_D_I",
    },
    {
        "Name": "19C",
        "key": "cityscapes_sh_512_256_aug_nc_19_resnet18",
        "tasks": "S_D_I",
        "MTM": "test_multi_task_metric_S_D_I",
    },
]

grad_metrics_data = dict.fromkeys([step_dict["key"] + "_" + step_dict["tasks"] for step_dict in steps_dict], {})
step_names = [steps_dict[i]["Name"] for i in range(len(steps_dict))]

optimizers = ["baseline", "imtl"]

baseline_mean = []
baseline_std = []
smto_mean = []
smto_std = []
diff = []

for step_dict in steps_dict:
    key = step_dict["key"]
    tasks = step_dict["tasks"]
    metric_name = step_dict["MTM"]

    for opt in optimizers:
        file_paths = glob(multi_task_data[key][tasks][opt]["path"] + "/test_*.pkl")
        best_metric_values = []
        for file_path in file_paths:
            file_data = torch.load(file_path)
            best_metric = utils.find_best_metric(file_data, metric_name, True)
            best_metric_values.append(best_metric)

        best_metric_values = np.array(best_metric_values) * 100.0
        mean_value = np.mean(best_metric_values)
        std_value = np.std(best_metric_values)

        grad_metrics_data[key + "_" + tasks][opt] = {"mean": mean_value, "std": std_value}

    baseline_mean.append(grad_metrics_data[key + "_" + tasks]["baseline"]["mean"])
    baseline_std.append(grad_metrics_data[key + "_" + tasks]["baseline"]["std"])
    smto_mean.append(grad_metrics_data[key + "_" + tasks]["imtl"]["mean"])
    smto_std.append(grad_metrics_data[key + "_" + tasks]["imtl"]["std"])

    diff.append(smto_mean[-1] - baseline_mean[-1])


baseline_mean = np.array(baseline_mean)
baseline_std = np.array(baseline_std)
smto_mean = np.array(smto_mean)
smto_std = np.array(smto_std)
diff = np.array(diff)

fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.tick_params(axis="both")
ax1.set_xlim(0, len(steps_dict) - 1)
ax2 = ax1.twinx()
ax1.plot(baseline_mean, color="blue", label="Unit. Scal.")
ax1.plot(smto_mean, color="red", label="IMTL")
ax1.set_ylabel("Multi task metric (\%)")

ax1.fill_between(
    list(range(baseline_mean.shape[0])),
    baseline_mean + 2*baseline_std,
    baseline_mean - 2*baseline_std,
    color="blue",
    alpha=0.15,
)
ax1.fill_between(list(range(smto_mean.shape[0])), smto_mean + smto_std, smto_mean - smto_std, color="red", alpha=0.15)

ax2.plot(diff, color="black", label="Difference", linestyle="--")
ax2.set_ylabel("Difference")

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles = handles1 + handles2
labels = labels1 + labels2

ax1.legend(handles, labels, loc="lower right")

ax1.set_xticks(np.arange(len(steps_dict)), step_names)

os.makedirs("plots/cityscapes", exist_ok=True)
plt.savefig("plots/cityscapes/grad_metrics.png", dpi=1200)
plt.savefig("plots/cityscapes/grad_metrics.pdf", dpi=1200)

# --------------Grad Dot Product--------------
plt.figure()
grad_file_paths = glob("data/saved_results/analyse/grad_metrics_correct/*/*/*/baseline/*/val_0.pkl")
grad_file_paths = [path for path in grad_file_paths if "cityscapes" not in path]
for step_dict in steps_dict: # cityscapes grad metrics have multiple runs for hyperparameter tuning
    grad_file_paths.append((multi_task_data[step_dict["key"]][step_dict["tasks"]]["baseline"]["path"] + "/val_0.pkl").replace("grad_metrics", "grad_metrics_correct"))

cityscapes_idx = 0
for path in grad_file_paths:
    file_data = torch.load(path, map_location="cpu")
    dataset = file_data["args"]["dataset"]
    tasks = file_data["args"]["tasks"]
    model = file_data["args"]["model"]
    opt = file_data["args"]["optimizer"]
    tag = f"{dataset}_{model}_{tasks}" if "QM9" not in dataset else f"{dataset}"
    
    label_name = dataset
    # if "QM9" in dataset and not args.all:
    #     continue

    if "mnist" in dataset:
        if tasks != "CL_RR" and not args.all:
            continue
        label_name = f"MNIST {tasks}"

    if "cityscapes" in dataset:
        shape = file_data["args"]["shape"]
        n_classes = file_data["args"]["n_classes"]
        if not (((tasks == "S_D_I" and shape[0] == 512 and shape[1] == 256 and n_classes == 19) or model == "resnet50") or args.all):
            cityscapes_idx += 1
            continue
        
        tag += f"_{shape[0]}_{shape[1]}_{file_data['args']['n_classes']}"
        label_name = f"City {step_names[cityscapes_idx]}"
        cityscapes_idx += 1
    cosine_similarity = np.array([stats[f"dot_avg_dir_{tasks}"] for stats in file_data["stats"]])
    x_percent = np.linspace(0, 100, len(cosine_similarity))
    plt.plot(x_percent, cosine_similarity, label=label_name)

plt.xlabel("Training (\%)")
plt.ylabel("Cosine similarity")
plt.tick_params(axis="both")
if args.all:
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=5 if args.all else 3, frameon=True)
else:
    plt.legend(loc="upper right", frameon=True)

plt.xlim(0, 100)
plt.grid()

if args.all:
    plt.savefig("plots/cityscapes/cosine_similarity_all.png", dpi=1200)
    plt.savefig("plots/cityscapes/cosine_similarity_all.pdf", dpi=1200)
else:
    plt.savefig("plots/cityscapes/cosine_similarity.png", dpi=1200)
    plt.savefig("plots/cityscapes/cosine_similarity.pdf", dpi=1200)
