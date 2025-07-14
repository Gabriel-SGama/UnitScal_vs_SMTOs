# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import copy
import numpy as np
import torch
import torch.nn.functional as F

import supervised_experiments.utils as utils
from supervised_experiments.loaders.cityscapes_labels import city_ignore_index
from supervised_experiments.loaders.QM9 import task_to_qm9_idx, QM9_dataset

# task dict: Stores the most relevant metric
# positive: higher is better
# name: metric used for the multi task metric
tasks_dict = {}
tasks_dict["CL"] = {"positive": True, "name": "metric_acc_CL", "plot_name": "Acc CL"}
tasks_dict["CR"] = {"positive": True, "name": "metric_acc_CR", "plot_name": "Acc CR"}
tasks_dict["S"] = {"positive": True, "name": "metric_mIOU_S", "plot_name": "mIoU"}
tasks_dict["RL"] = {"positive": False, "name": "metric_MSE_RL", "plot_name": "MSE RL"}
tasks_dict["RR"] = {"positive": False, "name": "metric_MSE_RR", "plot_name": "MSE RR"}
tasks_dict["D"] = {"positive": False, "name": "metric_l1_abs_D", "plot_name": "L1-abs D"}
tasks_dict["I"] = {"positive": False, "name": "metric_l1_abs_I", "plot_name": "L1-abs I"}


class multiTaskMetric(object):
    def __init__(self, args, tasks, single_task_reference, split, met):
        self.single_task_reference = single_task_reference
        self.tasks = tasks
        self.split = split
        self.evaluate_mtm = True
        self.met = met  # metric dict

        self.data_model_key, _ = utils.create_data_model_key_from_arg(args)

        # change key to dict due to the number of class of the cityscapes dataset
        self.data_model_key_dict = dict.fromkeys(tasks)
        for t in tasks:
            self.data_model_key_dict[t] = self.data_model_key
            if "city" in args.dataset and "S" in tasks and t != "S":
                data_str_split = self.data_model_key.split("_")
                n_class_idx = data_str_split.index("nc")
                # removes 'nc' and number of classes of the argument before searching key in saved results
                del data_str_split[n_class_idx]
                del data_str_split[n_class_idx]

                self.data_model_key_dict[t] = "_".join(data_str_split)

        # assert single_task_reference is not None, "need to run single task and the evaluation script first"
        for t in tasks:
            if self.data_model_key_dict[t] not in single_task_reference.keys():
                print(
                    f"Setup {self.data_model_key_dict[t]} not in single task reference file, multi task metric will always return 0 and models wont be saved properly"
                )
                self.evaluate_mtm = False
                continue

            if t not in single_task_reference[self.data_model_key_dict[t]].keys():
                print(
                    f"Task {t} not in {self.data_model_key_dict[t]} in single task reference file, multi task metric will always return 0 and models wont be saved properly"
                )
                self.evaluate_mtm = False
                continue

            # print("metric: ", single_task_reference[self.data_model_key_dict[t]][t])

    def multi_task_metric(self, epoch_stats):
        # multi task metric
        if not self.evaluate_mtm:
            return 0, {}

        multi_task_metric = 0

        for idx, t in enumerate(self.tasks):
            if t not in self.single_task_reference[self.data_model_key_dict[t]]:
                return 0, {}

            metrics_info = self.met[t].metrics_names
            multi_task_metric_task = 0

            for metric_key in metrics_info:
                factor = 1 if metrics_info[metric_key] else -1

                metric_val = epoch_stats[self.split + "_metric_" + metric_key + "_" + t]

                # every metric should be the same type to avoid issues with convertions
                if torch.is_tensor(metric_val):
                    metric_val = metric_val.item()

                baseline_metric_val = self.single_task_reference[self.data_model_key_dict[t]][t][metric_key]["value"]

                multi_task_metric_task += factor * (metric_val - baseline_metric_val) / baseline_metric_val
            multi_task_metric += multi_task_metric_task / len(metrics_info.keys())

        multi_task_metric /= len(self.tasks)
        return multi_task_metric, {self.split + "_multi_task_metric_" + "_".join(self.tasks): multi_task_metric}


class AccuracyMetric:
    def __init__(self):
        self.accuracy = 0.0
        self.num_updates = 0.0
        self.metrics_names = {"acc": True}

    def reset(self):
        self.accuracy = 0.0
        self.num_updates = 0.0

    def update(self, pred, gt):
        predictions = pred.data.max(1, keepdim=True)[1]
        self.accuracy += predictions.eq(gt.data.view_as(predictions)).sum()
        self.num_updates += predictions.shape[0]

    def get_result(self):
        return {"acc": "init" if self.num_updates == 0 else self.accuracy / self.num_updates}


class L1Metric:
    def __init__(self, instance_flag=False):
        self.l1_abs = 0.0
        self.l1_rel = 0.0
        self.num_updates = 0.0
        self.instance_flag = instance_flag
        self.metrics_names = {"l1_abs": False}
        if not self.instance_flag:
            self.metrics_names.update({"l1_rel": False})

    def reset(self):
        self.l1_abs = 0.0
        self.l1_rel = 0.0
        self.num_updates = 0.0

    def update(self, pred, gt):
        # Adapted from https://github.com/lorenmt/mtan/blob/master/im2im_pred/utils.py
        if self.instance_flag:
            mask = gt != city_ignore_index
        else:
            mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1).to(pred.device)

        x_pred_true = pred.masked_select(mask)
        x_output_true = gt.masked_select(mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        self.l1_abs += abs_err.sum()

        if not self.instance_flag:
            rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
            self.l1_rel += rel_err.sum()

        self.num_updates += torch.sum(mask)

    def get_result(self):
        if self.instance_flag:
            return {
                "l1_abs": "init" if self.num_updates == 0 else self.l1_abs / self.num_updates,
            }

        return {
            "l1_abs": "init" if self.num_updates == 0 else self.l1_abs / self.num_updates,
            "l1_rel": "init" if self.num_updates == 0 else self.l1_rel / self.num_updates,
        }


class MeanSquaredErrorMetric:
    def __init__(self, mask_flag=False):
        self.l_mse = 0.0
        self.mask_flag = mask_flag
        self.num_updates = 0.0
        self.metrics_names = {"MSE": False}

    def reset(self):
        self.l_mse = 0.0
        self.num_updates = 0.0

    def update(self, pred, gt):
        if self.mask_flag:
            raise NotImplementedError("Masking not implemented for MSE metric")

        self.l_mse += F.mse_loss(pred, gt)
        self.num_updates += 1

    def get_result(self):
        return {"MSE": "init" if self.num_updates == 0 else self.l_mse / self.num_updates}


class IOUMetric:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = torch.zeros((n_classes, n_classes))
        self.num_updates = 0.0
        self.metrics_names = {"acc": True, "mIOU": True}

    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes))
        self.num_updates = 0.0

    def update(self, pred, gt):
        target = gt.flatten().long()
        pred = pred.argmax(1).flatten()
        n = self.n_classes
        k = (target >= 0) & (target < n)
        inds = n * target[k] + pred[k]
        self.confusion_matrix += torch.bincount(inds, minlength=n**2).reshape(n, n).cpu()

    def get_result(self):
        acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        iou = torch.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(1) + self.confusion_matrix.sum(0) - torch.diag(self.confusion_matrix)
        )
        return {"acc": acc, "mIOU": iou.mean()}


class MAEMetric:
    def __init__(self, task, std, device):
        self.num_updates = 0
        self.l_mae = 0
        self.std = torch.Tensor(std).to(device)
        self.metrics_names = {"MAE": False}

        if task_to_qm9_idx[task] in [2, 3, 6, 12, 13, 14, 15]:
            self.scale = 1000
        else:
            self.scale = 1

    def reset(self):
        self.num_updates = 0
        self.l_mae = 0

    def update(self, pred, gt):
        self.l_mae += F.l1_loss(pred * self.std, gt * self.std, reduction="none").sum(0)  # MAE
        self.num_updates += len(gt)

    def get_result(self):
        return {"MAE": self.scale * self.l_mae.item() / self.num_updates}


def get_metrics(args, configs, tasks, single_task_reference=None, split=None, plot_flag=False, device="cpu"):
    # Returns a dictionary of metrics, and a function whose output aggregates all metrics
    # model_saver is a dict of functions of the metric dict: we save a model each time a larger value of this function
    # is attained
    met = {}

    if "mnist" in args.dataset:
        for t in tasks:
            if t[0] == "C":
                met[t] = AccuracyMetric()
            elif t[0] == "R":
                met[t] = MeanSquaredErrorMetric()

    if "city" in args.dataset:
        if "S" in tasks:
            met["S"] = IOUMetric(n_classes=args.n_classes)
        if "I" in tasks:
            met["I"] = L1Metric(instance_flag=True)
        if "D" in tasks:
            met["D"] = L1Metric()

    if "QM9" in args.dataset:
        temp_dataset = QM9_dataset(configs["QM9"]["path"], tasks, split="train")
        for i, t in enumerate(tasks):
            std = temp_dataset.get_std(i)
            met[t] = MAEMetric(t, std, device)

    # multit task metric
    if len(tasks) > 1:
        met["mtm"] = multiTaskMetric(args, tasks, single_task_reference, split, copy.deepcopy(met))

    if plot_flag:
        metrics_info = {}
        for t in tasks:
            metrics_info[t] = met[t].metrics_names

        return metrics_info

    return met
