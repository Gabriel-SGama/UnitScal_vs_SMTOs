import argparse
import copy
import json
import numpy as np
import torch

from tqdm import tqdm
from glob import glob

import supervised_experiments.metrics as metrics
from supervised_experiments import utils


def create_single_task_reference_test(best_metric_of_all_val):
    # already copys path and name fields
    best_metric_of_all_test = copy.deepcopy(best_metric_of_all_val)

    for data_model_key in best_metric_of_all_val:
        data_tasks = best_metric_of_all_val[data_model_key].keys()

        for task in data_tasks:
            for metric_key in best_metric_of_all_val[data_model_key][task]:
                metric_name = f"test_metric_{metric_key}_{task}"
                positive = best_metric_of_all_val[data_model_key][task][metric_key]["positive"]

                # get path to the test data
                path = best_metric_of_all_val[data_model_key][task][metric_key]["path"]
                file_paths = glob(path + "/test_*.pkl")

                # reset values
                best_metric_of_all_test[data_model_key][task][metric_key]["value"] = 0
                best_metric_of_all_test[data_model_key][task][metric_key]["count"] = len(file_paths)
                del best_metric_of_all_test[data_model_key][task][metric_key][
                    "compare_value"
                ]  # not necessary for the test dataset

                del best_metric_of_all_test[data_model_key][task][metric_key][
                    "compare_count"
                ]  # not necessary for the test dataset

                for path in file_paths:
                    file_data = torch.load(path)

                    # ensures that is a numpy value
                    best_metric = utils.find_best_metric(file_data, metric_name, positive)
                    best_metric_of_all_test[data_model_key][task][metric_key]["value"] += best_metric

                best_metric_of_all_test[data_model_key][task][metric_key]["value"] /= len(file_paths)

    # saves test data in a .json file
    with open(f"supervised_experiments/evaluation/single_task_reference_test.json", "w") as fp:
        json.dump(best_metric_of_all_test, fp)


def create_single_task_reference_val(single_task_files, configs):
    print("creating single task reference")

    data_dict = {}  # stores the mean metric data of each task
    datasets_models = {}  # stores the possible keys

    # compute mean values for each
    for file in tqdm(single_task_files):
        file_data = torch.load(file)

        name = utils.make_name(utils.dict_to_argparser(file_data))

        task = file_data["args"]["tasks"]
        metric = metrics.get_metrics(utils.dict_to_argparser(file_data), configs, [task], None, "val")
        dataset_name = file_data["args"]["dataset"]

        if name not in data_dict:
            path = "/".join(file.split("/")[:-1])
            dataset_model, _ = utils.create_data_model_key_from_data_dict(file_data)
            data_dict[name] = {
                "data_model": dataset_model,
                "task": task,
                "metrics": list(metric[task].metrics_names.keys()),
            }

            for key in metric[task].metrics_names:
                data_dict[name].update(
                    {
                        key: {"path": path, "count": 0, "compare_count": 0, "value": 0, "compare_value": 0},
                    }
                )

        for key in metric[task].metrics_names:
            metric_name = f"val_metric_{key}_{task}"
            positive = metric[task].metrics_names[key]

            data_dict[name][key]["count"] += 1
            data_dict[name][key]["value"] += utils.find_best_metric(file_data, metric_name, positive)
            data_dict[name][key]["positive"] = positive

            if data_dict[name][key]["compare_count"] < configs[dataset_name]["n_val_runs"]:
                data_dict[name][key]["compare_value"] += utils.find_best_metric(file_data, metric_name, positive)
                data_dict[name][key]["compare_count"] += 1
                index = file.split("/")[-1].split("_")[-1].split(".")[0]
                if index not in ["0", "1", "2", "3", "4"]:
                    print(file)

        dataset_model, task = utils.create_data_model_key_from_data_dict(file_data)
        if dataset_model not in datasets_models:
            datasets_models[dataset_model] = {task: metric[task].metrics_names}

        elif task not in datasets_models[dataset_model]:
            datasets_models[dataset_model].update({task: metric[task].metrics_names})

    print(datasets_models)

    best_metric_of_all = {key: {task: {} for task in datasets_models[key]} for key in datasets_models}

    for key in datasets_models:
        for task in datasets_models[key]:
            best_metric_of_all[key][task] = {}
            for metric_key in datasets_models[key][task]:
                start_value = -np.inf if datasets_models[key][task][metric_key] else np.inf
                best_metric_of_all[key][task].update(
                    {
                        metric_key: {
                            "path": "",
                            "count": 0,
                            "compare_count": 0,
                            "value": start_value,
                            "compare_value": start_value,
                        }
                    }
                )

    for data_key in tqdm(data_dict):
        data_model_key = data_dict[data_key]["data_model"]
        task = data_dict[data_key]["task"]

        for metric_key in data_dict[data_key]["metrics"]:
            positive = data_dict[data_key][metric_key]["positive"]

            compare_metric_val = (
                data_dict[data_key][metric_key]["compare_value"] / data_dict[data_key][metric_key]["compare_count"]
            )

            if (
                positive and compare_metric_val > best_metric_of_all[data_model_key][task][metric_key]["compare_value"]
            ) or (
                not positive
                and compare_metric_val < best_metric_of_all[data_model_key][task][metric_key]["compare_value"]
            ):
                best_metric_of_all[data_model_key][task][metric_key] = data_dict[data_key][metric_key]

                best_metric_of_all[data_model_key][task][metric_key]["compare_value"] /= best_metric_of_all[
                    data_model_key
                ][task][metric_key]["compare_count"]

                best_metric_of_all[data_model_key][task][metric_key]["value"] /= best_metric_of_all[data_model_key][
                    task
                ][metric_key]["count"]

    with open(f"supervised_experiments/evaluation/single_task_reference_val.json", "w") as fp:
        json.dump(best_metric_of_all, fp)

    return best_metric_of_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="supervised_experiments/configs.json")
    args = parser.parse_args()

    with open(args.config_file) as config_params:
        configs = json.load(config_params)

    single_task_files, _ = utils.read_data_files(configs["utils"]["results_storage"], mode="val")

    # creates single task reference for the validation dataset and selects the best config
    best_metric_of_all = create_single_task_reference_val(single_task_files, configs)

    # calculates the mean metric of the best config in the validation dataset
    create_single_task_reference_test(best_metric_of_all)
