import argparse
import copy
import json
import numpy as np
import torch

from tqdm import tqdm
from glob import glob

from supervised_experiments import utils


def create_multi_task_reference_test(best_metric_of_all_val, analysis_test):
    # already copys path and name fields
    best_metric_of_all_test = copy.deepcopy(best_metric_of_all_val)

    for data_model_key in best_metric_of_all_val:
        data_tasks = best_metric_of_all_val[data_model_key].keys()

        for tasks in data_tasks:
            for opt in best_metric_of_all_val[data_model_key][tasks]:
                metric_name = f"test_multi_task_metric_{tasks}"

                positive = True

                # get path to the test data
                path = best_metric_of_all_val[data_model_key][tasks][opt]["path"]
                file_paths = glob(path + "/test_*.pkl")

                # reset values
                best_metric_of_all_test[data_model_key][tasks][opt]["value"] = 0
                best_metric_of_all_test[data_model_key][tasks][opt]["count"] = len(file_paths)
                del best_metric_of_all_test[data_model_key][tasks][opt][
                    "compare_value"
                ]  # not necessary for the test dataset

                del best_metric_of_all_test[data_model_key][tasks][opt][
                    "compare_count"
                ]  # not necessary for the test dataset

                for path in file_paths:
                    file_data = torch.load(path)
                    # ensures that is a numpy value
                    best_metric = utils.find_best_metric(file_data, metric_name, positive)
                    best_metric_of_all_test[data_model_key][tasks][opt]["value"] += best_metric

                best_metric_of_all_test[data_model_key][tasks][opt]["value"] /= len(file_paths)

    # saves test data in a .json file
    separator = "_" if analysis_test != "" else ""

    with open(
        f"supervised_experiments/evaluation/multi_task_best_results_test{separator}{analysis_test}.json", "w"
    ) as fp:
        json.dump(best_metric_of_all_test, fp)


def create_multi_task_reference_val(multi_task_files, configs, analysis_test):
    data_dict = {}  # stores the mean metric data of each task
    datasets_models = {}  # stores the possible keys

    # compute mean values for each
    for file in tqdm(multi_task_files):
        file_data = torch.load(file, map_location="cpu")

        name = utils.make_name(utils.dict_to_argparser(file_data))

        dataset_name = file_data["args"]["dataset"]

        tasks_str = file_data["args"]["tasks"]
        tasks = tasks_str.split("_")
        metric_name = f"val_multi_task_metric_{tasks_str}"

        if name not in data_dict:
            path = "/".join(file.split("/")[:-1])
            dataset_model, _ = utils.create_data_model_key_from_data_dict(file_data)
            data_dict[name] = {
                "data_model": dataset_model,
                "opt": file_data["args"]["optimizer"],
                "tasks": file_data["args"]["tasks"],
                "path": path,
                "count": 0,
                "compare_count": 0,
                "value": 0,
                "compare_value": 0,
            }

        data_dict[name]["count"] += 1
        data_dict[name]["value"] += utils.find_best_metric(file_data, metric_name, True)

        if data_dict[name]["compare_count"] < configs[dataset_name]["n_val_runs"]:
            data_dict[name]["compare_value"] += utils.find_best_metric(file_data, metric_name, True)
            data_dict[name]["compare_count"] += 1

        dataset_model, tasks = utils.create_data_model_key_from_data_dict(file_data)
        if dataset_model not in datasets_models:
            datasets_models[dataset_model] = {tasks: [file_data["args"]["optimizer"]]}

        elif tasks not in datasets_models[dataset_model]:
            datasets_models[dataset_model].update({tasks: [file_data["args"]["optimizer"]]})

        elif file_data["args"]["optimizer"] not in datasets_models[dataset_model][tasks]:
            datasets_models[dataset_model][tasks].append(file_data["args"]["optimizer"])

    print("================================================")
    print(datasets_models)
    print("================================================")

    best_metric_of_all = {
        key: {tasks: {opt: {} for opt in datasets_models[key][tasks]} for tasks in datasets_models[key]}
        for key in datasets_models
    }

    for key in datasets_models:
        for tasks in datasets_models[key]:
            for opt in datasets_models[key][tasks]:
                start_value = -np.inf
                best_metric_of_all[key][tasks][opt] = {
                    "path": "",
                    "count": 0,
                    "compare_count": 0,
                    "value": start_value,
                    "compare_value": start_value,
                }

    for data_key in tqdm(data_dict):
        data_model_key = data_dict[data_key]["data_model"]
        tasks = data_dict[data_key]["tasks"]
        opt = data_dict[data_key]["opt"]

        positive = True

        compare_metric_val = data_dict[data_key]["compare_value"] / data_dict[data_key]["compare_count"]

        if (positive and compare_metric_val > best_metric_of_all[data_model_key][tasks][opt]["compare_value"]) or (
            not positive and compare_metric_val < best_metric_of_all[data_model_key][tasks][opt]["compare_value"]
        ):
            best_metric_of_all[data_model_key][tasks][opt] = copy.copy(data_dict[data_key])
            best_metric_of_all[data_model_key][tasks][opt]["compare_value"] = compare_metric_val
            best_metric_of_all[data_model_key][tasks][opt]["value"] /= best_metric_of_all[data_model_key][tasks][opt][
                "count"
            ]

            del best_metric_of_all[data_model_key][tasks][opt]["data_model"]
            del best_metric_of_all[data_model_key][tasks][opt]["opt"]
            del best_metric_of_all[data_model_key][tasks][opt]["tasks"]

    separator = "_" if analysis_test != "" else ""

    with open(
        f"supervised_experiments/evaluation/multi_task_best_results_val{separator}{analysis_test}.json", "w"
    ) as fp:
        json.dump(best_metric_of_all, fp)

    return best_metric_of_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="supervised_experiments/configs.json")
    parser.add_argument("--analysis_test", type=str, default="")
    args = parser.parse_args()

    with open(args.config_file) as config_params:
        configs = json.load(config_params)

    _, multi_task_files = utils.read_data_files(
        configs["utils"]["results_storage"], mode="val", analysis_test=args.analysis_test
    )

    # creates single task reference for the validation dataset and selects the best config
    best_metric_of_all = create_multi_task_reference_val(multi_task_files, configs, args.analysis_test)

    # print(best_metric_of_all)

    # calculates the mean metric of the best config in the validation dataset
    create_multi_task_reference_test(best_metric_of_all, args.analysis_test)
