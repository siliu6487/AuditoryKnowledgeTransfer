import copy
import itertools
import json
import os
import shutil
import configs
from my_helpers.exp_helpers import get_exp_config_dir


def create_exp_config(experiment_name: str, source_type: str, data_name: str,
                      config_dir: str = "exp_configs",
                      limit_tool: None or list = None, limit_beh: None or list = None):

    config_dir = get_exp_config_dir(config_dir=config_dir, data_name=data_name, experiment_name=experiment_name,
                                    source_type=source_type, limit_tool=limit_tool, limit_beh=limit_beh,
                                    clean_dir=True)

    exp_config = {
        # cross validation off
        "cross_validate": False,
        "trial_val_portion": 0,

        # training
        "data_name": data_name,
        'multi_class': True,
        "loss_func": 'sincere',
        "use_encoder": True,
        'retrain_encoder': True,
        'retrain_clf': True,
    }
    context_dicts = make_context_dict(experiment_name=experiment_name, source_type=source_type,
                                      limit_tool=limit_tool, limit_beh=limit_beh)
    total_task = context_dicts['total_task']

    for i in range(total_task):
        task_id = i + 1
        exp_config["experiment_name"] = f"{experiment_name}_{task_id}of{total_task}"

        # same behavior for all cross-tool exp; same tool for all cross-beh exp
        if source_type != "single":
            tool_idx, beh_idx = (i % context_dicts['num_tool'], i // context_dicts['num_tool']) \
                if context_dicts['iter_by_tool'] else (i // context_dicts['num_beh'], i % context_dicts['num_beh'])
            print(f"tool_idx, beh_idx: {tool_idx, beh_idx}")
            cd = {"source_tool_list": context_dicts['source_tools'][tool_idx],
                  "target_tool_list": context_dicts['target_tools'][tool_idx],
                  "source_beh_list": context_dicts['source_behs'][beh_idx],
                  "target_beh_list": context_dicts['target_behs'][beh_idx]}
        else:
            cd = {"source_tool_list": context_dicts['source_tools'][i],
                  "target_tool_list": context_dicts['target_tools'][i],
                  "source_beh_list": context_dicts['source_behs'][i],
                  "target_beh_list": context_dicts['target_behs'][i]}

        print(cd)
        exp_config.update(cd)
        # save this config file
        config_path = os.path.join(config_dir, f"{task_id}.json")
        with open(config_path, "w") as f:
            json.dump(exp_config, f, indent=2)
        print(f"Saved config {task_id}/{total_task} -> {config_path}")

    return None


def make_context_dict(experiment_name: str, source_type: str,
                      limit_tool: None or list, limit_beh: None or list):
    assert experiment_name in ['cross_tool', 'cross_behavior'], \
        f"experiment_name not correct: {experiment_name}. eligible: ['cross_tool', 'cross_behavior', 'cross_tool_beh']"
    assert source_type in ["rest", "latin_square_single", "single"], \
        f'source_type not correct: {source_type}. eligible: ["rest", "latin_square_single", "single"]'

    tools = copy.deepcopy(configs.ALL_TOOL_LIST)
    behs = copy.deepcopy(configs.ALL_BEH_LIST)
    if limit_tool:
        tools = limit_tool
    if limit_beh:
        behs = limit_beh

    num_tool = len(tools)
    num_beh = len(behs)

    list_tools = [[t] for t in tools]
    list_behs = [[b] for b in behs]

    iter_by_tool = True
    source_tools = []
    target_tools = []
    source_behs = []
    target_behs = []
    if source_type == "rest":  # all other to 1
        total_task = num_tool * num_beh
        if experiment_name == "cross_tool":
            source_behs, target_behs = list_behs, list_behs
            for i, tool in enumerate(tools):
                target_tools.append([tool])
                source_tools.append(tools[:i] + tools[i + 1:])
        else:
            source_tools, target_tools = list_tools, list_tools
            iter_by_tool = False
            for i, beh in enumerate(behs):
                target_behs.append([beh])
                source_behs.append(behs[:i] + behs[i + 1:])

    elif source_type == "latin_square_single":  # one to one but don't do all combo
        if experiment_name == "cross_tool":
            source_behs, target_behs = list_behs, list_behs
            total_task = num_tool * num_beh
            pairs = latin_square_pairs(tools)
            print(f"pairs: {pairs}")
            for p in pairs:
                source_tools.append([p[0]])
                target_tools.append([p[1]])
        else:
            source_tools, target_tools = list_tools, list_tools
            iter_by_tool = False
            total_task = num_tool * num_beh
            pairs = latin_square_pairs(behs)
            print(f"num pairs: {len(pairs)}")
            for p in pairs:
                source_behs.append([p[0]])
                target_behs.append([p[1]])

    else:  # one-to-one
        if experiment_name == "cross_tool":
            source_behs, target_behs = copy.deepcopy(list_behs), copy.deepcopy(list_behs)
            total_task = num_tool * (num_tool - 1) * num_beh
            pairs = list(itertools.permutations(tools, 2))
            print(f"num pairs: {len(pairs)}")
            for p in pairs:
                for beh in list_behs:
                    source_tools.append([p[0]])
                    target_tools.append([p[1]])
                    source_behs.append(beh)
                    target_behs.append(beh)
        else:
            raise Exception(f"exp not needed")

    context_dict = {
        "total_task": total_task,
        "num_tool": num_tool,
        "num_beh": num_beh,
        "iter_by_tool": iter_by_tool,
        "source_tools": source_tools,
        "target_tools": target_tools,
        "source_behs": source_behs,
        "target_behs": target_behs
    }

    print(context_dict)

    return context_dict


def latin_square_pairs(items):
    """
    Generate len(items) source-target pairs (one per item),
    using a Latin-square style shift so each item is source once and target once.
    No (x, x) self-pairs.
    """
    n = len(items)
    pairs = []
    for i in range(n):
        src = items[i]
        tgt = items[(i + 1) % n]  # shift by 1 → ensures no self-pair
        pairs.append((src, tgt))
    return pairs
