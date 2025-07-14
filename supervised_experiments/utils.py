import os
import random
import torch
import logging
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm

import rotograd

import supervised_experiments.metrics as metrics
import supervised_experiments.losses as losses_f


def save_model(models, optimizer, scheduler, tasks, epoch, args, folder, name):
    os.makedirs(folder, exist_ok=True)

    state = {
        "epoch": epoch + 1,
        "model_rep": models["rep"].state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
    }

    for t in tasks:
        key_name = "model_{}".format(t)
        state[key_name] = models[t].state_dict()

    torch.save(state, f"{folder}{name}.pkl")


def load_saved_model(models, tasks, path):
    state = torch.load(path)
    models["rep"].load_state_dict(state["model_rep"])
    for t in tasks:
        models[t].load_state_dict(state[f"model_{t}"])

    return models


# ----------------------------------------------------------------
# --------------------PATH/NAME CREATION--------------------------
# ----------------------------------------------------------------
def dict_to_argparser(dict_data):
    parser = argparse.ArgumentParser()

    for arg_name, default_value in dict_data["args"].items():
        parser.add_argument(f"--{arg_name}", default=default_value)

    parsed_args = parser.parse_args()

    return parsed_args


def create_data_model_key_from_data_dict(file_data):
    tasks_str = file_data["args"]["tasks"]
    tasks = tasks_str.split("_")

    data_model_key = file_data["args"]["dataset"]

    if "city" in file_data["args"]["dataset"]:
        data_model_key += "_sh_" + "_".join([str(val) for val in file_data["args"]["shape"]])

    if "city" in file_data["args"]["dataset"]:
        if file_data["args"]["aug"]:
            data_model_key += "_aug"

        if "S" in tasks:
            data_model_key += "_nc_" + str(file_data["args"]["n_classes"])

    data_model_key += "_" + file_data["args"]["model"]

    return data_model_key, tasks_str


def create_data_model_key_from_arg(args):
    args_data = {}
    args_data["args"] = vars(args)
    return create_data_model_key_from_data_dict(args_data)


def make_name(args, name_type="group", results_folder=""):
    task_string = args.tasks
    tasks = args.tasks.split("_")

    name = ""
    analysis_test_str = ""
    if args.analysis_test != "":
        if name_type == "group":
            analysis_test_str = "_" + args.analysis_test
        elif name_type == "folder":
            analysis_test_str = "/analyse/" + args.analysis_test + "/"
        else:
            raise NotImplementedError("Creation of name type only to group or foder")

    separator = "_" if name_type == "group" else "/"

    if name_type == "folder":
        name = results_folder + analysis_test_str
        analysis_test_str = ""

    dataset_str = args.dataset

    if "city" in args.dataset:
        dataset_str += "_sh_" + "_".join([str(val) for val in args.shape])

    if "city" in args.dataset:
        if args.aug:
            dataset_str += "_aug"

        if "S" in tasks:
            dataset_str += f"_nc_{args.n_classes}"

    name += f"{dataset_str}{separator}{args.model}{separator}{task_string}{separator}{args.optimizer}{analysis_test_str}{separator}lr_{args.lr}_bs_{args.batch_size}_dlr_{args.decay_lr}_p_{args.p}"

    if "city" in args.dataset:
        name += f"_wd_{args.weight_decay}"

    if "cdtt" in args.optimizer:
        name += f"_alpha_{args.alpha}"

    if "cagrad" in args.optimizer:
        name += f"_cr_{args.conv_rate}"

    if args.optimizer in ["auto_lambda"]:
        try:
            name += f"_alr_{args.aux_lr}"
        except:
            name += f"_lrr_{args.lr_relation}"

    if "rotograd" in args.optimizer:
        name += f"_lrr_{args.lr_relation}"

    if "famo" in args.optimizer:
        name += f"_ga_{args.gamma}"

    if "rotograd" in args.optimizer:
        feature_size = args.feature_size[0] if isinstance(args.feature_size, list) else args.feature_size
        name += f"_fs_{feature_size}"

    if args.optimizer == "cdtt_norm":
        name += f"_ts_{args.tensor_scale}"

    if args.weight_interval > 1:
        name += f"_wi_{args.weight_interval}_wm_{args.weight_momentum}"

    if name_type == "folder":
        name += "/"

    return name


def get_param_from_path(args, root_path):
    root_path_split = root_path.split("/")
    dataset = root_path_split[-5]
    args.model = root_path_split[-4]

    dataset_split = dataset.split("_")

    args.dataset = dataset_split[0]

    index = root_path.split("_").index("bs")
    args.batch_size = int(root_path.split("_")[index + 1])
    args.aug = "aug" in dataset

    if "nc" in dataset_split:
        index = dataset_split.index("nc")
        args.n_classes = int(dataset_split[index + 1])

    if "sh" in dataset_split:
        index = dataset_split.index("sh")
        args.shape = [int(dataset_split[index + 1]), int(dataset_split[index + 2])]

    return args


def create_dataset_models_tasks_keys(files):
    datasets_models = {}

    for file_name in tqdm(files):
        dataset_model, task = create_data_model_key_from_data_dict(torch.load(file_name))
        if dataset_model not in datasets_models:
            datasets_models[dataset_model] = [task]
            continue

        if task not in datasets_models[dataset_model]:
            datasets_models[dataset_model].append(task)

    return datasets_models


# ----------------------------------------------------------------
# --------------------METRIC EVALUATION---------------------------
# ----------------------------------------------------------------
def find_best_metric(file_data, metric_name, positive):
    # find best metric of one run
    best_metric = -np.inf if positive else np.inf

    for data in file_data["stats"]:
        metric = data[metric_name]
        best_metric = max(metric, best_metric) if positive else min(metric, best_metric)

    if torch.is_tensor(best_metric):
        best_metric = best_metric.item()

    return best_metric


def run_evaluation(
    args, configs, tasks, model, loader, epoch, DEVICE, mode, logger, single_task_reference=None, rotograd_flag=False
):
    loss_fn = losses_f.get_loss(args.dataset, tasks)
    metric = metrics.get_metrics(args, configs, tasks, single_task_reference, mode)
    eval_stats = {}

    if rotograd_flag:
        model.eval()
    else:
        for m in model:
            model[m].eval()

    with torch.no_grad():
        losses_per_epoch = {t: 0.0 for t in tasks}
        num_eval_batches = 0
        for batch_eval in loader:
            eval_images, eval_labels = split_batch_data(args, batch_eval, DEVICE, tasks)

            if rotograd_flag:
                with rotograd.cached():  # Speeds-up computations by caching Rotograd's parameters
                    preds = model(eval_images)

                    for i, t in enumerate(tasks):
                        out_t_eval = preds[0][i]
                        loss_t = loss_fn[t](out_t_eval, eval_labels[t])
                        losses_per_epoch[t] += loss_t.item()  # for logging purposes
                        metric[t].update(out_t_eval, eval_labels[t])

            else:
                val_rep, _ = model["rep"](eval_images, None)
                for t in tasks:
                    out_t_eval, _ = model[t](val_rep, None)
                    loss_t = loss_fn[t](out_t_eval, eval_labels[t])
                    losses_per_epoch[t] += loss_t.item()  # for logging purposes
                    metric[t].update(out_t_eval, eval_labels[t])

            num_eval_batches += 1

    # Print the stored (averaged across batches) validation losses and metrics, per task.
    clog = f"epochs {epoch}/{args.num_epochs}:"
    metric_results = {}
    for t in tasks:
        metric_results[t] = metric[t].get_result()
        metric[t].reset()
        clog += f" {mode}_loss {t} = {losses_per_epoch[t] / num_eval_batches:5.4f}"
        for metric_key in metric_results[t]:
            clog += f" {mode} metric-{metric_key} {t} = {metric_results[t][metric_key]:5.4f}"
        clog += " |||"

    for i, t in enumerate(tasks):
        eval_stats[f"{mode}_loss_{t}"] = losses_per_epoch[t] / num_eval_batches
        for metric_key in metric_results[t]:
            eval_stats[f"{mode}_metric_{metric_key}_{t}"] = metric_results[t][metric_key]

    if "mtm" in metric:
        mtm_val, mtm_dict = metric["mtm"].multi_task_metric(eval_stats)
        eval_stats.update(mtm_dict)
        clog += f" multi task metric = {mtm_val:.4f}"

    logger.info(clog)

    return eval_stats


# ----------------------------------------------------------------
# -------------------------I/O & OTHERS---------------------------
# ----------------------------------------------------------------
def read_data_files(data_folder, mode="val", analysis_test=""):
    """Reads data files and split them in single task and multi task"""

    files = None

    if analysis_test == "":
        files = sorted(glob(data_folder + "*/*/*/*/*/" + mode + "_*.pkl"))
    else:
        files = sorted(glob(data_folder + "analyse/" + analysis_test + "/*/*/*/*/*/*" + mode + "_*.pkl"))

    # filters multi task files by name
    single_task = [file_name for file_name in files if file_name.split("/")[-4].count("_") == 0]
    [files.remove(file_name) for file_name in single_task]

    files = sorted(
        files, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1])
    )  # slows down reading, but assures the files are in order

    print("number of single task files", len(single_task))
    print("number of multi task files", len(files))

    return single_task, files


def create_logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def split_batch_data(args, data, device, tasks):
    if "QM9" not in args.dataset:
        val_images = data[0].to(device)
        labels_val = {t: data[i + 1].to(device) for i, t in enumerate(tasks)}

        return val_images, labels_val

    # QM9 dataset:
    data = data.to(device)
    labels_val = {t: data.y[:, i].unsqueeze(-1) for i, t in enumerate(tasks)}

    return data, labels_val


def set_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    g = torch.Generator()
    g.manual_seed(random_seed)

    return g
