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

from supervised_experiments import utils

plt.rcParams.update(bundles.icml2024(usetex=True, column="full"))

with open("supervised_experiments/configs.json") as config_params:
    configs = json.load(config_params)

nash_folder = (
    configs["utils"]["results_storage"]
    + "analyse/smto_nash_weight/QM9/mpnn/mu_alpha_homo_lumo_r2_zpve_U0Atom_UAtom_HAtom_GAtom_Cv/nash/"
)

tasks = nash_folder.split("/")[-3].split("_")

tasks_labels = [
    r"$\mu$",
    r"$\alpha$",
    r"$\epsilon_{homo}$",
    r"$\epsilon_{lumo}$",
    r"$R^2$",
    r"$ZPVE$",
    r"$U_0$",
    r"$U$",
    r"$H$",
    r"$G$",
    r"$C_v$",
]


nash_files_path = glob(nash_folder + "*/val_0.pkl")
nash_files_path = sorted(nash_files_path, key=lambda s: float(s.split("/")[-2].split("_")[1]))[::-1]

plt.rcParams.update(figsizes.icml2024_full(nrows=1, ncols=2))
fig, axs = plt.subplots(1, len(nash_files_path), sharex=True, sharey=True)

mtm_key = "val_multi_task_metric_mu_alpha_homo_lumo_r2_zpve_U0Atom_UAtom_HAtom_GAtom_Cv"

for i, path in enumerate(nash_files_path):
    file_data = torch.load(path)
    learning_rate = file_data["args"]["lr"]

    weights_values = {f"weights_{t}": [] for t in tasks}
    momentum_weights_values = {f"weights_{t}": [] for t in tasks}
    stats_dict = file_data["stats"]

    normalizing_factor = []
    for j in range(len(stats_dict)):
        normalizing_factor.append(sum([stats_dict[j]["weights_" + t] for t in tasks]))

    for t, t_label in zip(tasks, tasks_labels):
        weights_values["weights_" + t] = [
            stats_dict[j]["weights_" + t] / normalizing_factor[j] for j in range(len(stats_dict))
        ]
        momentum_weights_values["weights_" + t].append(weights_values["weights_" + t][0])

        for j in range(1, len(stats_dict)):
            momentum_weights_values["weights_" + t].append(
                0.9 * momentum_weights_values["weights_" + t][j - 1] + 0.1 * weights_values["weights_" + t][j]
            )

        plot = axs[i].plot(weights_values["weights_" + t], label=t_label, alpha=0.3)
        axs[i].plot(
            momentum_weights_values["weights_" + t],
            linestyle="dotted",
            color=plot[0].get_color(),
        )

    axs[i].set_title(f"LR: {learning_rate}")

    mtm_data = [stats_dict[j][mtm_key] for j in range(len(stats_dict))]
    axs[i].tick_params(axis="x")
    axs[i].tick_params(axis="y")
    axs[i].tick_params()
    axs[i].grid(True, axis="both")
    axs[i].set_xlim(0, len(stats_dict) - 1)

    if i != 0:
        axs[i].tick_params(axis="y", which="both", left=False, labelleft=False)

    print("\n----------------learning_rate: ", learning_rate, "----------------")
    print("Momentum weights: ")
    total_sum = sum([momentum_weights_values["weights_" + t][-1] for t in tasks])
    [print("weights_" + t + ":", momentum_weights_values["weights_" + t][-1] / total_sum) for t in tasks]


axs[0].set_ylabel("Weight value", color="black")
fig.supxlabel("Epoch", color="black", y=0.08)

handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=(len(tasks) + 1) // 2)
plt.tight_layout()
fig.subplots_adjust(top=0.74)

os.makedirs("plots/QM9/fixed_weights", exist_ok=True)

plt.savefig("plots/QM9/fixed_weights/nash_weights.png", dpi=500)
plt.savefig("plots/QM9/fixed_weights/nash_weights.pdf", dpi=500)
