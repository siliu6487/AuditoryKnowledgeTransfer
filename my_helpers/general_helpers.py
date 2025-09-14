import argparse
import json
import os

import configs


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with config file")
    parser.add_argument(
        "--config",
        type=str,
        default=f"{os.getcwd()}/exp_configs/2.json",
        help="Path to JSON configuration file"
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def override_default_configs(json_config):
    """
    Update attributes in configs module with values from json_config.
    """
    for key, value in json_config.items():
        if hasattr(configs, key):
            if getattr(configs, key) != value:
                print(f"Overriding {key}: {getattr(configs, key)} → {value}")
                setattr(configs, key, value)
        else:
            print(f"Warning: {key} not in configs.py, skipping.")


def save_new_json(result_save_dir, curr_result):
    num_prev_results = len(sorted(os.listdir(result_save_dir)))
    if num_prev_results > 0:
        prev_result = sorted(os.listdir(result_save_dir))[-1]
        with open(os.path.join(result_save_dir, prev_result), "r") as f:
            prev_result = json.load(f)
        same_context = prev_result['context_dict'] == curr_result['context_dict']
        same_data = prev_result['data_info'] == curr_result['data_info']
        same_hyparam = prev_result['hyparams'] == curr_result['hyparams']
        same_test_folds = len(prev_result['test_results']['accuracies']) == \
                          len(curr_result['test_results']['accuracies'])

        save_new_json = not(same_context and same_data and same_hyparam and same_test_folds)
    else:
        save_new_json = True

    return save_new_json
