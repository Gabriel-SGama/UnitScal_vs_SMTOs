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
from supervised_experiments.metrics import tasks_dict

import supervised_experiments.losses as losses_f
import supervised_experiments.datasets as datasets
import supervised_experiments.metrics as metrics
import supervised_experiments.model_selector as model_selector

import rotograd
from rotograd import RotoGrad


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

    my_model = model_selector.get_model(args, tasks, device=DEVICE, model_name=args.model)

    feature_size = args.feature_size
    post_shape = args.post_shape if args.post_shape != None else ()
    rot_model = RotoGrad(
        my_model["rep"],
        [my_model[key] for key in tasks],
        feature_size,
        post_shape=post_shape,
    )

    rot_model.to(DEVICE)

    model_list = [my_model[key] for key in my_model.keys()]
    optimizer = torch.optim.Adam(
        [{"params": m.parameters()} for m in model_list]
        + [{"params": rot_model.parameters(), "lr": args.lr_relation * args.lr}],
        lr=args.lr,
    )

    if args.decay_lr:
        if "QM9" not in args.dataset:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.7, patience=5, min_lr=0.00001
            )
    else:
        scheduler = None

    train_val_stats = []  # saving validation stats per epoch
    assert len(tasks) > 1, "Use supervised_experiments/train_multi_task.py file to train single task models"

    best_results = {"mtm": -np.inf}

    is_time_exp = args.analysis_test == "time_measurement_exp"
    for epoch in range(args.num_epochs):
        start = timer()
        opt_mean_time = 0
        epoch_stats = {}

        print(f"Epoch {epoch} Started")

        losses_per_epoch = {t: 0.0 for t in tasks}
        n_iter = 0
        rot_model.train()

        for cidx, batch in enumerate(train_loader):
            n_iter += 1
            images, labels = utils.split_batch_data(args, batch, DEVICE, tasks)

            # Compute per-task losses.
            def losses_from_model(cmodel, average=True):
                optimizer.zero_grad()

                with rotograd.cached():  # Speeds-up computations by caching Rotograd's parameters
                    preds = cmodel(images)

                    losses = [loss_fn[t](preds[0][i], labels[t], average=average) for i, t in enumerate(tasks)]
                    cmodel.backward(losses)

                return losses

            losses = losses_from_model(rot_model)

            opt_start = timer()
            optimizer.step()
            opt_mean_time += timer() - opt_start

            for idx, t in enumerate(tasks):
                losses_per_epoch[t] += losses[idx].mean().detach()  # for logging purposes

        if args.decay_lr:
            if "QM9" not in args.dataset:
                scheduler.step()
            else:
                scheduler.step(sum([loss for loss in losses]))

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
                rot_model,
                val_loader,
                epoch,
                DEVICE,
                "val",
                logger,
                single_task_reference=single_task_reference,
                rotograd_flag=True,
            )
        )

        if "val_multi_task_metric_" + "_".join(tasks) in epoch_stats:
            mtm_val = epoch_stats["val_multi_task_metric_" + "_".join(tasks)]
            if mtm_val > best_results["mtm"]:
                best_results["mtm"] = mtm_val

                single_task_reference_test = None
                if os.path.exists("supervised_experiments/evaluation/single_task_reference_test.json"):
                    with open("supervised_experiments/evaluation/single_task_reference_test.json") as single_task_file:
                        single_task_reference_test = json.load(single_task_file)

                # run and save results on test dataset
                test_loader, _ = datasets.get_dataset(
                    args,
                    configs,
                    tasks=tasks,
                    generator=g,
                    worker_init_fn=seed_worker,
                    train=False,
                )

                test_stats = []
                stats = utils.run_evaluation(
                    args,
                    configs,
                    tasks,
                    rot_model,
                    test_loader,
                    epoch,
                    DEVICE,
                    "test",
                    logger,
                    single_task_reference=single_task_reference_test,
                    rotograd_flag=True,
                )
                stats["epoch"] = epoch
                test_stats.append(stats)

        else:
            raise NotImplementedError(f"rotograd only supports multiple task")

        if not args.debug:
            wandb.log(epoch_stats, step=epoch)

        train_val_stats.append(epoch_stats)

        end = timer()
        print(f"Epoch ended in {end - start}s")

        # TODO: implement model saving for rotograd
        # Save best evaluation model
        # for key in improved:
        #     if args.store_models and improved[key]:
        #         utils.save_model(
        #             best_models_copy[key],
        #             optimizer,
        #             scheduler,
        #             tasks,
        #             epoch,
        #             args,
        #             model_folder_path,
        #             "best_model_" + key + "_" + str(run_iter),
        #         )

    # Save training/validation results
    torch.save(
        {"stats": train_val_stats, "args": vars(args)},
        f"{results_folder_path}val_{run_iter}.pkl",
    )

    if is_time_exp:
        return

    torch.save({"stats": test_stats, "args": vars(args)}, f"{results_folder_path}test_{run_iter}.pkl")

    # TODO: implement model saving for rotograd
    # Save last model
    # if args.store_models:
    #     utils.save_model(
    #         model, optimizer, scheduler, tasks, epoch, args, model_folder_path, "last_model_" + str(run_iter)
    #     )


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
        default="rotograd",
        help="Optimiser to use",
        choices=[
            "rotograd",
        ],
    )

    parser.add_argument("--debug", action="store_true", help="Debug mode: disables wandb.")
    parser.add_argument("--model", type=str, required=True, help="model for task")
    parser.add_argument("--store_models", action="store_true", help="Whether to store  models at fixed frequency.")
    parser.add_argument("--decay_lr", action="store_true", help="Whether to decay the lr with the epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization.")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of experiment repetitions.")

    # SMTO specifics hypeparamters
    parser.add_argument(
        "--lr_relation",
        type=float,
        default=0.1,
        help="Percentage of the args.lr used to optimize the rotation matrix of RotoGrad",
    )
    # parser.add_argument("--feature_size", type=int, help="Features to be rotate by RotoGrad")

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
        help="shape for cityscapes dataset",
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        help="Features to be rotate by RotoGrad",
    )

    parser.add_argument("--post_shape", default=None, type=int, nargs=2, help="Post shape for RotoGrad")
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
