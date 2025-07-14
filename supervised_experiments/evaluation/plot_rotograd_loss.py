import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tueplots import bundles, figsizes, fontsizes

from supervised_experiments import utils

plt.rcParams.update(bundles.icml2024(usetex=True, column="full"))
plt.rcParams.update(figsizes.icml2024_full(nrows=1, ncols=2))


root = "data/saved_results/analyse/debug_rotf2"
paths = glob(root + "/*/*/*/*/*/*val*.pkl")


samples_dict = {
    "mnist": {
        "name": "mnist",
        "plot_name": "MNIST CL_RR",
        "plot_data": [],
    },
    "cityscapes": {
        "name": "cityscapes_sh_512_256_aug_nc_19",
        "plot_name": "City 19C",
        "plot_data": [],
    },
    "QM9":{
        "name": "QM9",
        "plot_name": "QM9",
        "plot_data": [],
    },
}


for path in paths:
    data = torch.load(path)
    
    rotation_loss = np.array([data["stats"][i]["rotograd_rotation_loss"].cpu() for i in range(len(data["stats"]))])
    samples_dict[data["args"]["dataset"]]["plot_data"].append(rotation_loss)


# Plotting
plt.figure()
for key in samples_dict.keys():
    plot_data = samples_dict[key]["plot_data"]
    plot_data_mean = np.mean(np.array(plot_data), axis=0)
    plot_data_mean_log = np.log(plot_data_mean)
    std_log = np.std(np.log(np.array(plot_data)), axis=0)
    plot_data_std = np.std(np.array(plot_data), axis=0)
    x_axis = np.linspace(0, 100, len(plot_data_mean))
    plt.plot(x_axis, plot_data_mean, label=samples_dict[key]["plot_name"])
    plt.fill_between(
        x_axis,
        plot_data_mean + 2*plot_data_std,
        plot_data_mean - 2*plot_data_std,
        alpha=0.15,
    )
# log scale
plt.yscale("log")
plt.xlabel("Training (\%)")
plt.ylabel("Rotation Loss")
plt.legend()
plt.savefig("plots/rotograd_loss.png", dpi=1000)
plt.savefig("plots/rotograd_loss.pdf", dpi=1000)