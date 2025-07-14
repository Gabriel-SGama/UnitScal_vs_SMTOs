import os
import json
import torch
import numbers
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd

from tqdm import tqdm
from glob import glob

from supervised_experiments import utils
from supervised_experiments import metrics
from supervised_experiments.evaluation import utils as eval_utils
from supervised_experiments.evaluation.utils import labels_opt, caption_dict


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--model", type=str, default="lenet")
parser.add_argument("--tasks", type=str, default="CL_RR")
parser.add_argument(
    "--shape",
    type=int,
    nargs="*",
    help="shape for Cityscapes",
)
parser.add_argument("--n_classes", type=int, default=19)
parser.add_argument("--aug", action="store_true")
parser.add_argument("--analysis_test", type=str, default="")
parser.add_argument("--column", type=str, default="full")
parser.add_argument("--eval_optimizers", type=str, nargs="*", default=["all"], help="optimizers to evaluate", choices=["all"].extend(labels_opt.keys()))
parser.add_argument("--rm_optimizers", type=str, nargs="*", default=None, help="optimizers to remove from evaluation")
parser.add_argument("--save", action="store_true", help="save the plots")
args = parser.parse_args()


separator = "_" if args.analysis_test != "" else ""
caption = caption_dict[args.dataset]

os.makedirs("plots", exist_ok=True)
plt.rcParams["figure.figsize"] = [18, 5.0]
plt.rcParams["figure.autolayout"] = True

with open("supervised_experiments/configs.json") as config_params:
    configs = json.load(config_params)

with open(
    f"supervised_experiments/evaluation/multi_task_best_results_test{separator}{args.analysis_test}.json"
) as multi_task_data_file:
    multi_task_test_data = json.load(multi_task_data_file)

with open(
    f"supervised_experiments/evaluation/multi_task_best_results_val{separator}{args.analysis_test}.json"
) as multi_task_data_file:
    multi_task_val_data = json.load(multi_task_data_file)

# To complete the get metrics function:
# with open(f"supervised_experiments/evaluation/single_task_reference_val.json") as single_task_reference_file:
#     single_task_reference = json.load(single_task_reference_file)

with open(f"supervised_experiments/evaluation/single_task_reference_test.json") as single_task_reference_file:
    single_task_reference = json.load(single_task_reference_file)

data_model_key, _ = utils.create_data_model_key_from_arg(args)

tasks_str = args.tasks
tasks = tasks_str.split("_")

mtm_metric_name_val = "val_multi_task_metric_" + tasks_str
mtm_metric_name_test = "test_multi_task_metric_" + tasks_str
mtm_metric_name_plot = "\\nabla_{MTM}"

metrics_info = metrics.get_metrics(args, configs, tasks, single_task_reference=single_task_reference, plot_flag=True)

print("metrics_info: ", metrics_info)

if args.analysis_test == "FwLe0.9":
    with open(f"supervised_experiments/evaluation/multi_task_best_results_val.json") as multi_task_data_file:
        multi_task_val_data_default = json.load(multi_task_data_file)

    multi_task_val_data[data_model_key][tasks_str]["nash"] = multi_task_val_data_default[data_model_key][tasks_str][
        "nash"
    ]
    with open(f"supervised_experiments/evaluation/multi_task_best_results_test.json") as multi_task_data_file:
        multi_task_test_data_default = json.load(multi_task_data_file)

    multi_task_test_data[data_model_key][tasks_str]["nash"] = multi_task_test_data_default[data_model_key][tasks_str][
        "nash"
    ]

optimizers = list(multi_task_test_data[data_model_key][tasks_str].keys())

data_values = dict.fromkeys(optimizers)
data_info = multi_task_test_data[data_model_key][tasks_str]

best_metric_data_split_tasks, best_mtm_data = eval_utils.get_best_metric_data(
    tasks, optimizers, data_info, metrics_info, scale=100.0
)

optimizers_to_eval = list(labels_opt.keys()) if args.eval_optimizers[0] == "all" else args.eval_optimizers
if args.rm_optimizers is not None:
    optimizers_to_eval = [opt for opt in optimizers_to_eval if opt not in args.rm_optimizers]
optimizers = [opt for opt in optimizers if opt in optimizers_to_eval]

opt_labels_order = [opt for opt in labels_opt.keys() if opt in optimizers]
opt_labels_plot = [labels_opt[opt_name]["plot_name"] for opt_name in opt_labels_order]
opt_color_plot = [labels_opt[opt_name]["color"] for opt_name in opt_labels_order]

data_mtm_plot = [best_mtm_data[name] for name in opt_labels_order]

eval_utils.save_boxplot_mtm(
    args,
    data_mtm_plot,
    opt_labels_plot,
    opt_color_plot,
    column=args.column,
    save=args.save,
)

data_df, df_ascending_columns, total_n_columns = eval_utils.create_latex_table(
    tasks,
    optimizers,
    best_metric_data_split_tasks,
    data_mtm_plot,
    metrics_info,
    mtm_metric_name_plot,
    opt_labels_plot,
    opt_labels_order,
)

mean_data_df, repetitive_tasks_df = eval_utils.create_table_w_extra_data(
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
)

print(f"\n-----------LATEX MEAN MULTI-TASK METRIC TABLE SKETCH {caption}-----------")
colum_format = "*{" + str(len(mean_data_df.columns) + 2) + "}{c}"
print(mean_data_df.to_latex(index=True, caption="Mean Muiti-Task metric on " + caption + " dataset", column_format=colum_format))

print(f"\n-----------LATEX VARIOUS METRIC TABLE SKETCH {caption}-----------")
colum_format = "*{" + str(len(repetitive_tasks_df.columns) + 2) + "}{c}"
print(repetitive_tasks_df.to_latex(index=True, caption="Various metrics on " + caption + " dataset", column_format=colum_format))


final_mean_rank = eval_utils.compute_mean_rank(
    tasks, data_df, best_metric_data_split_tasks, optimizers, df_ascending_columns, opt_labels_plot
)
total_mean_rank_size = 3

for opt_index, (opt_label_plot, opt_label) in enumerate(zip(opt_labels_plot, opt_labels_order)):
    n_decimal = int(max(np.log10(final_mean_rank[opt_index]), 0)) + 1
    precision = max(total_mean_rank_size - n_decimal, 1)
    data_df.loc[opt_label_plot, r"MR $\downarrow$"] = f"{np.round(final_mean_rank[opt_index], precision):.{precision}f}"

colum_format = f"c*{total_n_columns + 2}"

data_df = data_df.applymap(lambda x: eval_utils.precision_round(x, 4) if isinstance(x, numbers.Number) else x)

print(f"\n-----------LATEX TABLE SKETCH {caption}-----------")
print(data_df.to_latex(index=True, caption="Muiti task metric on " + caption + " dataset", column_format=colum_format))