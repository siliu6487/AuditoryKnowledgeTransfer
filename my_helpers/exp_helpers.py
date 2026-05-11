import argparse
import itertools
import os
import shutil
from typing import Dict, Any

import numpy as np

import configs
from my_helpers.data_helpers import generate_bibd
from my_helpers.general_helpers import load_json


def get_exp_config_dir(config_dir, data_name, experiment_name, source_type,
                       limit_beh=None, limit_tool=None, clean_dir=False):
    if limit_tool is not None:
        num = len(limit_tool)
        assert num in [1, 5]
        if num == 1:
            lim_tool_name = f"_tool{limit_tool[0]}"
        else:
            lim_tool_name = f"_{num}tool"
    if limit_beh is not None:
        num = len(limit_beh)
        assert num in [1, 5]
        if num == 1:
            lim_beh_name = f"_beh{limit_beh[0].split('-')[0]}"
        else:
            lim_beh_name = f"_{num}beh"

    data_folder_name = data_name.split(".")[0]
    config_dir = os.path.join(config_dir, f"{data_folder_name}/{experiment_name}-{source_type}_to1")
    if limit_beh is not None and experiment_name == "cross_tool":
        config_dir += lim_beh_name
    if limit_tool is not None and experiment_name == "cross_behavior":
        config_dir += lim_tool_name

    if clean_dir:
        if os.path.exists(config_dir):
            shutil.rmtree(config_dir)

    os.makedirs(config_dir, exist_ok=True)

    return config_dir


def set_default_context_hyparams(one_batch: bool, shuffle: bool, epoch_encoder: int, epoch_classifier: int):
    context_dict = {
        "source_tool_list": configs.source_tool_list,
        "target_tool_list": configs.target_tool_list,
        "source_beh_list": configs.source_beh_list,
        "target_beh_list": configs.target_beh_list,
        "test_object_list": configs.new_object_list,
        "shared_object_list": configs.old_object_list,
    }
    hyparams = {
        "one_batch": one_batch,  # one batch data during training
        "shuffle": shuffle,  # shuffle train data

        "batch_size": 128,  # doesn't matter when one_batch is True
        "epoch_encoder": epoch_encoder,
        "epoch_classifier": epoch_classifier,

        "lr_encoder": configs.lr_encoder,
        "lr_classifier": configs.lr_classifier,

        "temperature": 0.5,
        "encoder_input_dim": 512,
        "encoder_hidden_dim": 256,
        "encoder_output_dim": 128,

    }

    return context_dict, hyparams


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file"
    )
    args = parser.parse_args()

    return args


def load_exp_config_and_update(context_dict, hyparams):
    args = parse_args()
    exp_config = load_json(args.config)
    for k, v in exp_config.items():
        if k in context_dict.keys():
            context_dict[k] = v
        if k in hyparams.keys():
            hyparams[k] = v
    return exp_config


def get_result_dir(base_dir: str, context_dict: dict, transfer_type: str, modality: str, data_name: str):
    cross_exp_name = "other_cross_type"
    target_name = "target"
    source_name = "source"
    same_context = "mixed_context"
    # cross tool
    if context_dict['source_tool_list'] != context_dict['target_tool_list']:
        if context_dict['source_beh_list'] == context_dict['target_beh_list']:
            cross_exp_name = "cross_tool"
            target_name = f"target_{context_dict['target_tool_list'][0]}"
            num_source = len(context_dict['source_tool_list'])
            source_name = f"source_other_{num_source}tools" if num_source > 1 \
                else f"source_{context_dict['source_tool_list'][0]}"
            same_context = context_dict['source_beh_list'][0]
    # cross beh
    elif context_dict['source_beh_list'] != context_dict['target_beh_list']:
        if context_dict['source_tool_list'] == context_dict['target_tool_list']:
            cross_exp_name = "cross_behavior"
            target_name = f"target_{context_dict['target_beh_list'][0]}"
            num_source = len(context_dict['source_beh_list'])
            source_name = f"source_other_{num_source}behs" if num_source > 1 \
                else f"source_{context_dict['source_beh_list'][0]}"
            same_context = context_dict['source_tool_list'][0]

    data_folder_name = data_name.split('.')[0]
    result_save_dir = os.path.join(base_dir, transfer_type, cross_exp_name, same_context, target_name, source_name,
                                   modality, data_folder_name)

    return result_save_dir


def make_result_dir(context_dict: dict, transfer_type: str, modality: str, data_name: str):
    result_save_dir = get_result_dir(base_dir="test_result", context_dict=context_dict,
                                     transfer_type=transfer_type, modality=modality, data_name=data_name)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    return result_save_dir


def make_init_result(transfer_type: str, audio_data_name: str, num_test_fold: int,
                     context_dict: dict, hyparams: dict) -> dict:
    results: Dict[str, Any] = {
        "transfer_type": transfer_type,
        "data_info": {
            "modalities": ["audio"],
            "audio_data_name": audio_data_name,
        },
        "test_results": {
            "avg_accuracy": None,
            "accuracies": [],
        }
    }

    for i in range(num_test_fold):
        results['test_results'][f"set{i + 1}"] = {
            "test_obj_list": [],
            "test_truth0_and_preds1": [],
        }

    results.update({
        "context_dict": context_dict,
        "hyparams": hyparams,
    })

    return results


def general_exp_run_setup(
        transfer_type, modality, audio_data_name,
        one_batch, shuffle, epoch_encoder, epoch_classifier, encoder_output_dim,
        num_test_fold, test_size, forbidden_test_obj_combo, prev_test_fold

):
    # ========= default context and params ========================
    context_dict, hyparams = set_default_context_hyparams(
        one_batch=one_batch, shuffle=shuffle,
        epoch_encoder=epoch_encoder, epoch_classifier=epoch_classifier,
    )
    if encoder_output_dim is not None:
        hyparams['encoder_output_dim'] = encoder_output_dim

    # ========= update context based on loaded config ========================
    try:
        exp_config = load_exp_config_and_update(context_dict=context_dict, hyparams=hyparams)
        audio_data_name = exp_config['data_name']
    except SystemExit as e:
        print(
            "⏭️Missing required arguments. Will not update context and parameter values using experiment config file.")

    assert set(context_dict['shared_object_list']).isdisjoint(context_dict['test_object_list'])
    print(f"context_dict: {context_dict}")
    print(f"hyparams: {hyparams}")

    # ========= make result saving dir ========================
    result_save_dir = make_result_dir(transfer_type=transfer_type, context_dict=context_dict,
                                      modality=modality, data_name=audio_data_name)

    # ========= init results ========================
    results = make_init_result(
        transfer_type=transfer_type, audio_data_name=audio_data_name, num_test_fold=num_test_fold,
        context_dict=context_dict, hyparams=hyparams)

    # ========= make test object sets ========================
    full_obj_list = sorted(context_dict['shared_object_list'] + context_dict['test_object_list'])
    test_obj_lists, _ = generate_bibd(
        all_objs=full_obj_list, k=test_size, forbidden_set=forbidden_test_obj_combo,
        n_blocks=num_test_fold, seed=configs.rand_seed + prev_test_fold)

    return {
        "context_dict": context_dict,
        "hyparams": hyparams,
        "test_obj_lists": test_obj_lists,
        "full_obj_list": full_obj_list,
        "results": results,
        "result_save_dir": result_save_dir
    }


def get_result_cross_tool_5to1(shared_beh, data_name):

    context_dict = {
        "source_tool_list": [],
        "target_tool_list": [],
        "source_beh_list": [shared_beh],
        "target_beh_list": [shared_beh],
        "test_object_list": [],
        "shared_object_list": [],
    }

    cross_tool_pair = list(itertools.permutations(configs.ALL_TOOL_LIST, 2))
    pair_results_bin = {}
    for pair in cross_tool_pair:
        context_dict['source_tool_list'] = [pair[0]]
        context_dict['target_tool_list'] = [pair[1]]
        test_result_dir = get_result_dir(base_dir=base_dir, context_dict=context_dict, transfer_type=transfer_type,
                                         modality="audio", data_name=data_name)
        result = load_json(os.path.join(test_result_dir, "results1.json"))
        pair_results_bin[f"{pair[0]}2{pair[1]}"] = result

