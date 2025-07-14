import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from glob import glob
from tqdm import tqdm
from tueplots import bundles, figsizes, fontsizes

from supervised_experiments import utils

labels_opt = {
    "baseline": {"plot_name": "Unit. Scal.", "weight_key": "","color": "#1f77b4"},
    "auto_lambda": {"plot_name": "Auto-Lambda", "weight_key": "weights","color": "#ff7f0e"},
    "cagrad": {"plot_name": "CAGrad", "weight_key": "weights","color": "#2ca02c"},
    "cdtt": {"plot_name": "CDTT", "weight_key": "cdtt_weights","color": "#d62728"},
    "edm": {"plot_name": "EDM", "weight_key": "edm_weights","color": "#9467bd"},
    "famo": {"plot_name": "FAMO", "weight_key": "weights","color": "#8c564b"},
    "graddrop": {"plot_name": "GradDrop", "weight_key": "","color": "#e377c2"},
    "imtl": {"plot_name": "IMTL", "weight_key": "IMTL-G_weights","color": "#7f7f7f"},
    "mgda-ub": {"plot_name": "MGDA-UB", "weight_key": "weights","color": "#bcbd22"},
    "nash": {"plot_name": "Nash-MTL", "weight_key": "weights","color": "#17becf"},
    "fixed_weights_nash_0.005": {"plot_name": "0.005", "weight_key": "","color": "#ff7f0e"},
    "fixed_weights_nash_0.001": {"plot_name": "0.001", "weight_key": "","color": "#2ca02c"},
    "fixed_weights_nash_0.0005": {"plot_name": "0.0005", "weight_key": "","color": "#d62728"},
    "fixed_weights_nash_0.0001": {"plot_name": "0.0001", "weight_key": "","color": "#9467bd"},
    "pcgrad": {"plot_name": "PCGrad", "weight_key": "","color": "#aec7e8"},
    "rlw-dirichlet": {"plot_name": "RLW-Dirichlet", "weight_key": "dirichlet_weights","color": "#ffbb78"},
    "rlw-normal": {"plot_name": "RLW-Normal", "weight_key": "normal_weights","color": "#98df8a"},
    "rotograd": {"plot_name": "RotoGrad", "weight_key": "","color": "#ff9896"},
    "si": {"plot_name": "SI", "weight_key": "","color": "#c5b0d5"},
    "uw": {"plot_name": "UW", "weight_key": "eta","color": "#c49c94"},
    "fixed_weights_cagrad": {"plot_name": "Fixed Weights", "weight_key": "","color": "#c49c94"},
    "fixed_weights_edm": {"plot_name": "Fixed Weights", "weight_key": "","color": "#c49c94"},
}

caption_dict = {
    "mnist": "MNIST",
    "cityscapes": "Cityscapes",
    "QM9": "QM9",
}


def get_best_metric_data(tasks, optimizers, data_info, metrics_info, scale=1):
    best_metric_data_split_tasks = {
        opt: {t: {metric_key: [] for metric_key in metrics_info[t]} for t in tasks} for opt in optimizers
    }

    best_mtm_data = {opt: [] for opt in optimizers}

    for opt in tqdm(optimizers):
        path = data_info[opt]["path"]

        file_paths = glob(path + "/test*.pkl")

        for file in file_paths:
            file_data = torch.load(file)
            tasks_str = file_data["args"]["tasks"]
            tasks = tasks_str.split("_")

            mtm_name = f"test_multi_task_metric_{tasks_str}"
            best_mtm_data[opt].append(utils.find_best_metric(file_data, mtm_name, True) * scale)

            for t in tasks:
                metrics_names = metrics_info[t]

                for metric_key in metrics_names:
                    best_metric = utils.find_best_metric(
                        file_data, f"test_metric_{metric_key}_{t}", metrics_names[metric_key]
                    )

                    best_metric_data_split_tasks[opt][t][metric_key].append(best_metric)

    return best_metric_data_split_tasks, best_mtm_data


def compute_mean_rank(tasks, data_df, best_metric_data_split_tasks, optimizers, df_ascending_columns, opt_labels_plot):
    rank_data = data_df.rank(ascending=False)
    rank_data[df_ascending_columns] = data_df[df_ascending_columns].rank(ascending=True)
    mean_rank_by_task = pd.DataFrame(index=opt_labels_plot, columns=tasks)

    for opt_index, opt_label_plot in enumerate(opt_labels_plot):
        total_n_columns = 0
        for t in tasks:
            mean_rank_value = rank_data.iloc[
                opt_index,
                total_n_columns : total_n_columns + len(best_metric_data_split_tasks[optimizers[0]][t].keys()),
            ].mean()
            mean_rank_by_task.loc[opt_label_plot, t] = mean_rank_value
            total_n_columns += len(best_metric_data_split_tasks[optimizers[0]][t].keys())

    return mean_rank_by_task.mean(axis=1)

def precision_round(value, total_size):
    n_decimal = int(max(np.log10(np.abs(value)), 0)) + 1
    precision = max(total_size - n_decimal, 1)
    return f"{np.round(value, precision):.{precision}f}"

def create_latex_table(
    tasks,
    optimizers,
    best_metric_data_split_tasks,
    data_mtm_plot,
    metrics_info,
    mtm_metric_name_plot,
    opt_labels_plot,
    opt_labels_order,
):
    task_metrics_test_latex = []
    df_ascending_columns = []
    for t in tasks:
        for metric_name in best_metric_data_split_tasks[optimizers[0]][t].keys():
            task_metrics_test_latex.append(
                r"\centering "
                + t
                + (("-" + metric_name) if len(best_metric_data_split_tasks[optimizers[0]][t].keys()) > 1 else "")
            )
            if not metrics_info[t][metric_name]:
                df_ascending_columns.append(task_metrics_test_latex[-1])

    task_metrics_test_latex.append(r"\centering $" + mtm_metric_name_plot + r"$(\%) $\uparrow$")

    data_df = pd.DataFrame(index=opt_labels_plot, columns=task_metrics_test_latex)

    total_size = 4

    for opt_index, (opt_label_plot, opt_label) in enumerate(zip(opt_labels_plot, opt_labels_order)):
        total_n_columns = 0
        for task_idx, t in enumerate(tasks):
            for metric_idx, metric_name in enumerate(best_metric_data_split_tasks[optimizers[0]][t].keys()):

                mean_value = np.mean(np.array(best_metric_data_split_tasks[opt_label][t][metric_name]))
                n_decimal = int(max(np.log10(np.abs(mean_value)), 0)) + 1
                precision = max(total_size - n_decimal, 1)

                # data_df.loc[
                #     opt_label_plot,
                #     task_metrics_test_latex[total_n_columns],
                # ] = f"{np.round(mean_value, precision):.{precision}f}"
                
                data_df.loc[
                    opt_label_plot,
                    task_metrics_test_latex[total_n_columns],
                ] = mean_value
                
                total_n_columns += 1

        mean_value = np.mean(np.array(data_mtm_plot[opt_index]))
        n_decimal = int(max(np.log10(np.abs(mean_value)), 0)) + 1
        precision = max(total_size - n_decimal, 1)

        # data_df.loc[opt_label_plot, task_metrics_test_latex[-1]] = f"{np.round(mean_value, precision):.{precision}f}"
        data_df.loc[opt_label_plot, task_metrics_test_latex[-1]] = mean_value

    return data_df, df_ascending_columns, total_n_columns


def create_table_w_extra_data(
    tasks,
    optimizers,
    best_metric_data_split_tasks,
    single_task_reference,
    data_model_key,
    data_mtm_plot,
    metrics_info,
    mtm_metric_name_plot,
    opt_labels_plot,
    opt_labels_order,
    scale=100,
):
    task_metrics_test_latex = []
    df_ascending_columns = []
    for t in tasks:
        for metric_name in best_metric_data_split_tasks[optimizers[0]][t].keys():
            task_metrics_test_latex.append(
                r"\centering "
                + t
                + (("-" + metric_name) if len(best_metric_data_split_tasks[optimizers[0]][t].keys()) > 1 else "")
            )
            if not metrics_info[t][metric_name]:
                df_ascending_columns.append(task_metrics_test_latex[-1])

    task_metrics_test_latex.append(r"\centering$ avg $\uparrow$")

    data_df = pd.DataFrame(index=opt_labels_plot, columns=task_metrics_test_latex)

    total_size = 4

    for opt_index, (opt_label_plot, opt_label) in enumerate(zip(opt_labels_plot, opt_labels_order)):
        total_n_columns = 0
        for task_idx, t in enumerate(tasks):
            for metric_idx, metric_name in enumerate(best_metric_data_split_tasks[optimizers[0]][t].keys()):
                curr_data_model_key = data_model_key

                if "cityscapes" in data_model_key and "S" in tasks and t != "S":
                    data_str_split = data_model_key.split("_")
                    n_class_idx = data_str_split.index("nc")
                    # removes 'nc' and number of classes of the argument before searching key in saved results
                    del data_str_split[n_class_idx]
                    del data_str_split[n_class_idx]

                    curr_data_model_key = "_".join(data_str_split)

                best_single_task_metric = single_task_reference[curr_data_model_key][t][metric_name]["value"]
                factor = 1 if metrics_info[t][metric_name] else -1
                multi_task_metric_task = scale * factor * (np.array(best_metric_data_split_tasks[opt_label][t][metric_name]) - best_single_task_metric) / best_single_task_metric
 
                data_df.loc[
                    opt_label_plot,
                    task_metrics_test_latex[total_n_columns],
                ] = multi_task_metric_task
                
                total_n_columns += 1
        
        data_df.loc[
            opt_label_plot,
            task_metrics_test_latex[total_n_columns],
        ] = np.mean(np.array(data_mtm_plot[opt_index]))
                

    mean_data_df = data_df.applymap(np.mean)
    new_columns = {
        r"\centering min \Delta \uparrow": np.min,
        r"\centering max \Delta \uparrow": np.max, 
        r"\centering med \Delta \uparrow": np.median,
        r"\centering std \Delta \downarrow": np.std,
        r"\centering avg \Delta \uparrow": np.mean, 
    }

    repetitive_tasks_df = pd.DataFrame(index=opt_labels_plot, columns=new_columns.keys())

    for column_name, func in new_columns.items():
        repetitive_tasks_df[column_name] = mean_data_df.apply(func, axis=1)

    mean_data_df = mean_data_df.applymap(lambda x: precision_round(x, total_size))
    repetitive_tasks_df = repetitive_tasks_df.applymap(lambda x: precision_round(x, total_size))
    
    return mean_data_df, repetitive_tasks_df



def save_boxplot_mtm(args, data_mtm_plot, opt_labels_plot, opt_color_plot, root="", column="half", save=True):
    assert column in ["half", "full"], f"Column must be either 'half' or 'full', but got {column}."

    plt.rcParams.update(bundles.icml2024(usetex=True, column=column))
    if column == "full":
        plt.rcParams.update(figsizes.icml2024_full(nrows=1, ncols=2))
    data_model_key, _ = utils.create_data_model_key_from_arg(args)

    baseline_median = np.median(data_mtm_plot[0])
    plt.axhline(y=baseline_median, color="r", linestyle="dashed")

    box = plt.boxplot(
        data_mtm_plot,
        patch_artist=True,
        labels=opt_labels_plot,
    )

    plt.xticks(rotation=-20)
    plt.grid(True)

    for patch, color in zip(box["boxes"], opt_color_plot):
        plt.setp(patch, facecolor=color)

    plt.ylabel("Multi-task Metric (\%)")

    if save:
        os.makedirs(root + "plots/" + args.dataset + "/" + args.analysis_test, exist_ok=True)
        plt.savefig(
            root
            + "plots/"
            + args.dataset
            + "/"
            + args.analysis_test
            + "/box_plot_"
            + data_model_key
            + "_"
            + args.tasks
            + ".png",
            dpi=1000,
        )

        plt.savefig(
            root
            + "plots/"
            + args.dataset
            + "/"
            + args.analysis_test
            + "/box_plot_"
            + data_model_key
            + "_"
            + args.tasks
            + ".pdf",
            dpi=1000,
        )
