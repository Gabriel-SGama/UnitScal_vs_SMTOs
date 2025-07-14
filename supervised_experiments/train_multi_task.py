# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/train_multi_task.py

import os
import argparse
import copy
import json
import random
import numpy as np
import wandb
import torch

from itertools import cycle
from timeit import default_timer as timer

from supervised_experiments import utils

import supervised_experiments.losses as losses_f
import supervised_experiments.datasets as datasets
import supervised_experiments.metrics as metrics
import supervised_experiments.model_selector as model_selector

from optimizers.pcgrad import PCGrad
from optimizers.imtl import IMTL
from optimizers.mgda import MGDA
from optimizers.rlw import RLW
from optimizers.graddrop import GradDrop
from optimizers.edm import EDM
from optimizers.task_tensioners import CDTT
from optimizers.nash_mtl import NashMTL
from optimizers.auto_lambda import AutoLambda
from optimizers.cagrad import CAGrad
from optimizers.baselines import Baseline
from optimizers.famo import FAMO
from optimizers.scale_inv import SI
from optimizers.uw import UW


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_multi_task(args, random_seed, run_iter):
    # Set random seeds.
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    g = torch.Generator()
    g.manual_seed(random_seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config_file) as config_params:
        configs = json.load(config_params)

    single_task_reference = None
    if os.path.exists("supervised_experiments/evaluation/single_task_reference_val.json"):
        with open("supervised_experiments/evaluation/single_task_reference_val.json") as single_task_file:
            single_task_reference = json.load(single_task_file)

    tasks = args.tasks.split("_")

    results_folder_path = utils.make_name(args, name_type="folder", results_folder=configs["utils"]["results_storage"])
    model_folder_path = utils.make_name(args, name_type="folder", results_folder=configs["utils"]["model_storage"])

    os.makedirs(results_folder_path, exist_ok=True)
    if args.store_models:
        os.makedirs(model_folder_path, exist_ok=True)

    group_name = utils.make_name(args, name_type="group")

    logger = utils.create_logger("Main")
    if not args.debug:
        # avoid race conditions when using multiple GPU's
        wandb_dir = "wandb/" + results_folder_path
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(project="multi_task_analysis", group=group_name, config=args, reinit=True, dir=wandb_dir)

    train_loader, val_loader = datasets.get_dataset(
        args,
        configs,
        tasks=tasks,
        generator=g,
        worker_init_fn=seed_worker,
        train=True,
    )

    loss_fn = losses_f.get_loss(args.dataset, tasks)
    metric = metrics.get_metrics(args, configs, tasks, single_task_reference, "val", device=DEVICE)

    model = model_selector.get_model(args, tasks, device=DEVICE, model_name=args.model)
    model_params = [p for v in model.values() for p in list(v.parameters())]
    spec_params = [p for t in tasks for p in list(model[t].parameters())]

    optimizer = torch.optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)

    if args.decay_lr:
        if "QM9" not in args.dataset:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.7, patience=5, min_lr=0.00001
            )
    else:
        scheduler = None

    if args.optimizer == "pcgrad":
        mtl_opt = PCGrad(optimizer, tasks)
    elif args.optimizer == "imtl":
        # Note that this operates on the gradient of the shared parameters by default (for RL, set ub=False)
        mtl_opt = IMTL(optimizer, tasks, spec_params)
    elif args.optimizer in ["mgda", "mgda-ub"]:
        mtl_opt = MGDA(optimizer, tasks, spec_params, normalize="loss+", ub=(args.optimizer == "mgda-ub"))
    elif args.optimizer in ["graddrop", "ran-graddrop"]:
        mtl_opt = GradDrop(optimizer, tasks, spec_params, use_sign=(args.optimizer == "graddrop"))
    elif "rlw" in args.optimizer:
        mtl_opt = RLW(optimizer, tasks, distribution=args.optimizer.split("-")[1])
    elif "edm" in args.optimizer:
        mtl_opt = EDM(optimizer, tasks)
    elif args.optimizer == "cdtt":
        mtl_opt = CDTT(optimizer, tasks, args.alpha)
    elif args.optimizer == "nash":
        mtl_opt = NashMTL(optimizer, tasks)
    elif args.optimizer == "auto_lambda":
        # Only trained on the multi task setting (no auxiliary option)
        mtl_opt = AutoLambda(args, model, optimizer, DEVICE, tasks, tasks, loss_fn, args.lr_relation)
    elif args.optimizer == "cagrad":
        mtl_opt = CAGrad(optimizer, tasks, args.conv_rate)
    elif args.optimizer == "famo":
        mtl_opt = FAMO(optimizer, DEVICE, tasks, gamma=args.gamma)
    elif args.optimizer == "si":
        mtl_opt = SI(optimizer, tasks, weights=args.losses_weights)
    elif args.optimizer == "uw":
        mtl_opt = UW(optimizer, tasks)
    elif (
        "fixed_weights" in args.optimizer
    ):  # it's the same optimizer as Unit. Scal., but is saved with a different name
        mtl_opt = Baseline(optimizer, tasks, weights=args.losses_weights)
    else:
        mtl_opt = Baseline(optimizer, tasks)

    stats_frequency = 100
    train_val_stats = []  # saving validation stats per epoch

    running_metric_keys = list(metric[tasks[0]].metrics_names.keys()) if len(tasks) == 1 else ["mtm"]
    best_models_copy = dict.fromkeys(running_metric_keys)
    best_results = dict.fromkeys(running_metric_keys)
    epoch_of_best_models = dict.fromkeys(running_metric_keys)
    is_time_exp = args.analysis_test == "time_measurement_exp"

    if len(tasks) == 1:
        running_metric = metric[tasks[0]]

        for key in running_metric_keys:
            best_results[key] = -np.inf if running_metric.metrics_names[key] else np.inf

    elif "mtm" not in metric:
        raise NotImplementedError(
            "if not using the multi task metric for more than one task, add another form of evaluation"
        )
    else:
        best_results = {"mtm": -np.inf}

    for epoch in range(args.num_epochs):
        start = timer()
        opt_mean_time = 0
        epoch_stats = {}

        print(f"Epoch {epoch} Started")

        for m in model:
            model[m].train()

        losses_per_epoch = {t: 0.0 for t in tasks}
        mean_opt_data = {}
        n_iter = 0
        grad_stat_iter = 0
        grad_metric_mean = None

        if args.optimizer == "auto_lambda":
            val_iter = cycle(iter(val_loader))

        for cidx, batch in enumerate(train_loader):
            n_iter += 1
            # Read targets and images for the batch.
            images, labels = utils.split_batch_data(args, batch, DEVICE, tasks)

            # code from: https://github.com/lorenmt/auto-lambda
            if args.optimizer == "auto_lambda":
                batch_val = next(val_iter)
                val_images, labels_val = utils.split_batch_data(args, batch_val, DEVICE, tasks)

                # meta_optimizer.zero_grad() # Done inside the function
                mtl_opt.unrolled_backward(
                    images, labels, val_images, labels_val, optimizer.param_groups[0]["lr"], optimizer
                )
                # meta_optimizer.step() # Done inside the function

            # Compute per-task losses.
            def losses_from_model(cmodel, average=False):
                losses = []
                rep, _ = cmodel["rep"](images, None)
                for t in tasks:
                    out_t, _ = cmodel[t](rep, None)
                    # the losses are averaged within the MTL optimizers, possibly after manipulations per datapoint
                    loss_t = loss_fn[t](out_t, labels[t], average=average)
                    losses.append(loss_t)  # to backprop on

                return losses, rep

            losses, rep = losses_from_model(model)
            for idx, t in enumerate(tasks):
                losses_per_epoch[t] += losses[idx].mean().detach()  # for logging purposes

            if ((cidx - 1) % stats_frequency) == 0 and "grad_metrics" in args.analysis_test:
                grad_metric = mtl_opt.compute_gradient_metrics(losses, tasks)
                if grad_metric_mean is None:
                    grad_metric_mean = grad_metric
                else:
                    for grad_key in grad_metric.keys():
                        grad_metric_mean[grad_key] += grad_metric[grad_key]

                grad_stat_iter += 1

            opt_start = timer()
            data = mtl_opt.iterate(losses, shared_repr=rep, shared_parameters=model["rep"].parameters())

            for key in data.keys():
                if key in mean_opt_data:
                    mean_opt_data[key] += data[key]
                else:
                    mean_opt_data[key] = data[key]

            if args.optimizer == "famo":
                new_losses, rep = losses_from_model(model)
                mtl_opt.update(new_losses)

            opt_mean_time += timer() - opt_start

        for key in data.keys():
            mean_opt_data[key] /= n_iter

        epoch_stats.update(mean_opt_data)

        if args.decay_lr:
            if "QM9" not in args.dataset:
                scheduler.step()
            else:
                scheduler.step(sum([loss for loss in losses]))

        if "grad_metric" in args.analysis_test:
            for grad_key in grad_metric_mean.keys():
                grad_metric_mean[grad_key] /= grad_stat_iter
            epoch_stats.update(grad_metric_mean)

        epoch_stats["opt_mean_time"] = opt_mean_time / n_iter
        if is_time_exp:
            # Measure time only, no need to log training/validation stats.
            torch.cuda.synchronize()
            epoch_runtime = timer() - start
            logger.info(f"epochs {epoch}/{args.num_epochs}: runtime: {epoch_runtime}")
            epoch_stats["runtime"] = epoch_runtime

            if not args.debug:
                wandb.log(epoch_stats, step=epoch)

            train_val_stats.append(epoch_stats)
            continue

        # Print the stored (averaged across batches) training losses, per task.
        clog = f"epochs {epoch}/{args.num_epochs}:"
        for t in tasks:
            clog += f" train_loss {t} = {losses_per_epoch[t] / n_iter:5.4f}"
        logger.info(clog)

        for i, t in enumerate(tasks):
            epoch_stats[f"train_loss_{t}"] = losses_per_epoch[t] / n_iter

        # Evaluate the model on the validation set.
        epoch_stats.update(
            utils.run_evaluation(
                args,
                configs,
                tasks,
                model,
                val_loader,
                epoch,
                DEVICE,
                "val",
                logger,
                single_task_reference=single_task_reference,
            )
        )

        if "val_multi_task_metric_" + "_".join(tasks) in epoch_stats:
            improved = {"mtm": False}
            mtm_val = epoch_stats["val_multi_task_metric_" + "_".join(tasks)]
            if mtm_val > best_results["mtm"]:
                best_results["mtm"] = mtm_val
                improved["mtm"] = True
                best_models_copy["mtm"] = copy.deepcopy(model)

                epoch_of_best_models["mtm"] = epoch

        elif len(tasks) == 1:
            running_metric = metric[tasks[0]]
            improved = dict.fromkeys(running_metric.metrics_names, False)
            for key in running_metric.metrics_names:
                metric_val = epoch_stats["val_metric_" + key + "_" + tasks[0]]

                if (
                    running_metric.metrics_names[key]
                    and metric_val > best_results[key]
                    or not running_metric.metrics_names[key]
                    and metric_val < best_results[key]
                ):
                    best_results[key] = metric_val
                    improved[key] = True
                    best_models_copy[key] = copy.deepcopy(model)

                    epoch_of_best_models[key] = epoch

        else:
            # raise NotImplementedError(f"more than one task, need to use a multi task metric")
            improved = {"mtm": False}

        if not args.debug:
            wandb.log(epoch_stats, step=epoch)

        train_val_stats.append(epoch_stats)

        end = timer()
        print(f"Epoch ended in {end - start}s")

        # Save best evaluation model
        for key in improved:
            if args.store_models and improved[key]:
                utils.save_model(
                    best_models_copy[key],
                    optimizer,
                    scheduler,
                    tasks,
                    epoch,
                    args,
                    model_folder_path,
                    "best_model_" + key + "_" + str(run_iter),
                )

    # Save training/validation results
    torch.save(
        {"stats": train_val_stats, "args": vars(args)},
        f"{results_folder_path}val_{run_iter}.pkl",
    )

    if is_time_exp:
        return

    # run and save results on test dataset
    test_loader, _ = datasets.get_dataset(
        args,
        configs,
        tasks=tasks,
        generator=g,
        worker_init_fn=seed_worker,
        train=False,
    )

    single_task_reference_test = None
    if os.path.exists("supervised_experiments/evaluation/single_task_reference_test.json"):
        with open("supervised_experiments/evaluation/single_task_reference_test.json") as single_task_file:
            single_task_reference_test = json.load(single_task_file)

    test_stats = []
    epochs_evaluated = []
    running_metric_keys = metric[tasks[0]].metrics_names if len(tasks) == 1 else ["mtm"]
    for key in running_metric_keys:
        if epoch_of_best_models[key] in epochs_evaluated:  # skip repeated epochs
            continue

        stats = utils.run_evaluation(
            args,
            configs,
            tasks,
            best_models_copy[key],
            test_loader,
            epoch_of_best_models[key],
            DEVICE,
            "test",
            logger,
            single_task_reference=single_task_reference_test,
        )
        stats["epoch"] = epoch_of_best_models[key]
        test_stats.append(stats)
        epochs_evaluated.append(epoch_of_best_models[key])

    torch.save({"stats": test_stats, "args": vars(args)}, f"{results_folder_path}test_{run_iter}.pkl")

    # Save last model
    if args.store_models:
        utils.save_model(
            model, optimizer, scheduler, tasks, epoch, args, model_folder_path, "last_model_" + str(run_iter)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./dataset", help="Path to dataset folder")
    parser.add_argument("--label", type=str, default="", help="wandb group")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="which dataset to use",
        choices=["mnist", "cityscapes", "QM9"],
    )
    parser.add_argument("--tasks", type=str, default="", help="tasks")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--p", type=float, default=0.0, help="Task dropout probability")
    parser.add_argument(
        "--conv_p", type=float, default=0.0, help="Task dropout in the encoder of the cityscapes architecture"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Epochs to train for.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="pcgrad",
        help="Optimiser to use",
        choices=[
            "pcgrad",
            "baseline",
            "fixed_weights",
            "imtl",
            "mgda",
            "mgda-ub",
            "graddrop",
            "ran-graddrop",
            "rlw-uniform",
            "rlw-normal",
            "rlw-dirichlet",
            "rlw-random_normal",
            "rlw-bernoulli",
            "rlw-constrained_bernoulli",
            "edm",
            "cdtt",
            "nash",
            "auto_lambda",
            "cagrad",
            "famo",
            "si",
            "uw",
            "fixed_weights_nash_0.005",
            "fixed_weights_nash_0.001",
            "fixed_weights_nash_0.0005",
            "fixed_weights_nash_0.0001",
            "fixed_weights_cagrad",
            "fixed_weights_imtl",
            "fixed_weights_edm",
        ],
    )

    parser.add_argument("--debug", action="store_true", help="Debug mode: disables wandb.")
    parser.add_argument("--model", type=str, required=True, help="model for task")
    parser.add_argument("--store_models", action="store_true", help="Whether to store  models at fixed frequency.")
    parser.add_argument("--decay_lr", action="store_true", help="Whether to decay the lr with the epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization.")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of experiment repetitions.")

    # SMTO specifics hypeparamters
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha parameter for cdtt")
    parser.add_argument("--conv_rate", type=float, default=0.5, help="Convergence rate parameter for CAGrad")
    parser.add_argument("--tensor_scale", type=float, default=0.5, help="Percentage of max value allowed by tensor")
    parser.add_argument(
        "--lr_relation",
        type=float,
        default=0.1,
        help="Percentage of the args.lr used to optimize the meta weights of AutoLambda",
    )
    parser.add_argument("--gamma", type=float, default=0.01, help="Regularization term of FAMO")

    parser.add_argument("--start_index", type=int, default=0, help="Start offset for the run.")
    parser.add_argument("--random_seed", type=int, default=1, help="Start random seed to employ for the run.")
    parser.add_argument("--config_file", type=str, default="supervised_experiments/configs.json")
    parser.add_argument("--aug", action="store_true", help="enables augmentation for cityscapes")
    parser.add_argument(
        "--n_classes", type=int, default=19, help="number of classes for semantic segmentation", choices=[7, 13, 19]
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="*",
        help="shape for Cityscapes",
    )
    parser.add_argument(
        "--losses_weights",
        type=float,
        nargs="*",
        help="Weights to use for losses. Be sure that the ordering is correct! (ordering defined as in tasks arg).",
    )

    parser.add_argument("--weight_interval", type=int, default=1, help="Iteration interval between SMTO iteration")
    parser.add_argument("--weight_momentum", type=float, default=0.9, help="Weight momentum to smooth SMTO result")
    parser.add_argument(
        "--analysis_test",
        type=str,
        default="",
        help="testing something that should be saved separately",
    )

    args = parser.parse_args()

    for i in range(args.n_runs):
        train_multi_task(args, args.random_seed + i + args.start_index, i + args.start_index)
