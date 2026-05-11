import itertools
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from my_helpers.exp_helpers import get_result_dir
from my_helpers.general_helpers import load_json


def make_target_tool_matrix(pair_results_clap):
    """
    Build a DataFrame: target tool (rows) × behavior (columns),
    filled with avg_accuracy from pair_results_clap.
    Adds row/col averages.
    """
    data = []
    for key, result in pair_results_clap.items():
        if "test_results" not in result:
            continue

        avg_acc = result["test_results"]["avg_accuracy"]

        # key format: "target_{tgt_t}&{beh}"
        try:
            prefix, rest = key.split("target_")
            tgt_t, beh = rest.split("&")
        except ValueError:
            continue

        data.append((tgt_t, beh, avg_acc))

    df = pd.DataFrame(data, columns=["target_tool", "behavior", "accuracy"])
    df_pivot = df.pivot(index="target_tool", columns="behavior", values="accuracy")

    # add row mean (average across behaviors)
    df_pivot["Avg(tools)"] = df_pivot.mean(axis=1)

    # add column mean (average across tools)
    df_pivot.loc["Avg(behavior)"] = df_pivot.mean(axis=0)

    return df_pivot

def plot_target_tool_matrix(df_matrix, title="Target Tool × Behavior Accuracy"):
    # compute global mean and std (excluding NaN)
    values = df_matrix.values.flatten()
    values = values[~np.isnan(values)]
    mean_val = np.mean(values)
    std_val = np.std(values)

    # update title
    full_title = f"{title}\nMean: {mean_val:.3f}, Std: {std_val:.3f}"

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df_matrix,
        annot=True, fmt=".2f", cmap="Blues", cbar=True,
        linewidths=0.5, linecolor="gray",
        mask=df_matrix.isna()
    )
    plt.title(full_title)
    plt.xlabel("Shared Behavior")
    plt.ylabel("Target Tool")
    plt.tight_layout()
    plt.show()


def make_target_beh_matrix(pair_results_clap):
    data = []
    for key, result in pair_results_clap.items():
        if "test_results" not in result:
            continue

        avg_acc = result["test_results"]["avg_accuracy"]

        # key format: "target_{tgt_t}&{beh}"
        try:
            prefix, rest = key.split("target_")
            tgt_t, beh = rest.split("&")
        except ValueError:
            continue

        data.append((tgt_t, beh, avg_acc))

    df = pd.DataFrame(data, columns=["target_beh", "tool", "accuracy"])
    df_pivot = df.pivot(index="target_beh", columns="tool", values="accuracy")

    # add row mean (average across behaviors)
    df_pivot["Avg(behaviors)"] = df_pivot.mean(axis=1)

    # add column mean (average across tools)
    df_pivot.loc["Avg(tool)"] = df_pivot.mean(axis=0)

    return df_pivot


def plot_target_beh_matrix(df_matrix, title="Target behavior × Tool Accuracy"):
    # compute global mean and std (excluding NaN)
    values = df_matrix.values.flatten()
    values = values[~np.isnan(values)]
    mean_val = np.mean(values)
    std_val = np.std(values)

    # update title
    full_title = f"{title}\nMean: {mean_val:.3f}, Std: {std_val:.3f}"

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df_matrix,
        annot=True, fmt=".2f", cmap="Blues", cbar=True,
        linewidths=0.5, linecolor="gray",
        mask=df_matrix.isna()
    )
    plt.title(full_title)
    plt.xlabel("Shared Tool")
    plt.ylabel("Target Behavior")
    plt.tight_layout()
    plt.show()



def mean_per_target_tool(conf_matrix):
    """
    Given a confusion matrix (source tools as rows, target tools as columns),
    compute mean accuracy per target tool across all source tools.
    """
    # column-wise mean, ignoring NaNs
    return conf_matrix.mean(axis=0)


def make_acronym(tool_name: str) -> str:
    parts = tool_name.split("-")
    return "-".join([p[0] for p in parts])  # e.g. "metal-scissor" -> "m-s"


def plot_confusion_matrix(conf_matrix, title="Tool Transfer Accuracy"):
    # Map index/columns to acronyms
    acronym_map = {tool: make_acronym(tool) for tool in conf_matrix.index}
    conf_matrix_acr = conf_matrix.rename(index=acronym_map, columns=acronym_map)

    # Compute row and column means
    row_means = conf_matrix_acr.mean(axis=1)
    col_means = conf_matrix_acr.mean(axis=0)
    overall_mean = conf_matrix_acr.values.mean()

    # Append averages as new row/col
    conf_with_avg = conf_matrix_acr.copy()
    conf_with_avg["Avg"] = row_means
    conf_with_avg.loc["Avg"] = list(col_means) + [overall_mean]

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_with_avg,
        annot=True, fmt=".2f", cmap="Blues", cbar=True,
        linewidths=0.5, linecolor="gray",
        mask=conf_with_avg.isna()  # mask NaN (so missing pairs look blank)
    )

    m, std = get_stats(conf_matrix)
    plt.title(title + f"\n Mean: {m:.2f}%, SD: {std:.2f}%")
    plt.xlabel("Target Tool")
    plt.ylabel("Source Tool")
    plt.tight_layout()
    plt.show()


def get_stats(conf_matrix):
    # Flatten the confusion matrix to 1D, ignore NaNs
    values = conf_matrix.values.flatten()
    values = values[~np.isnan(values)]

    mean_val = np.mean(values)
    std_val = np.std(values)
    return mean_val, std_val


def make_confusion_matrix(pair_results):
    """
    Convert pair_results into a confusion matrix (DataFrame).
    pair_results: dict with keys like 'metal-scissor2metal-whisk'
                  and values are result dicts with ['test_results']['avg_accuracy'].
    """

    data = {}
    tools = set()

    for pair, result in pair_results.items():
        if "test_results" not in result:
            continue
        avg_acc = result["test_results"]["avg_accuracy"]

        # split "source2target"
        if ">" not in pair:
            continue

        source, target = pair.split(">")
        tools.add(source)
        tools.add(target)

        data[(source, target)] = avg_acc

    tools = sorted(tools)
    df = pd.DataFrame(index=tools, columns=tools, dtype=float)

    for (s, t), acc in data.items():
        df.loc[s, t] = acc

    return df


def compare_to1_vs_5to1(conf_matrix_to1, conf_matrix_5to1, behavior="3-stirring-fast"):
    """
    Compare mean accuracies of 1-to-1 transfer vs 5-to-1 transfer
    for a given behavior.
    """
    # --- 1-to-1 case: mean across all target tools
    vals_to1 = conf_matrix_to1.mean(axis=0)  # mean per target tool
    mean_to1 = vals_to1.mean()
    std_to1 = vals_to1.std()

    # --- 5-to-1 case: extract that behavior
    if behavior not in conf_matrix_5to1.columns:
        raise ValueError(f"Behavior {behavior} not found in 5-to-1 matrix")
    vals_5to1 = conf_matrix_5to1[behavior].dropna()
    mean_5to1 = vals_5to1.mean()
    std_5to1 = vals_5to1.std()

    return {
        "1-to-1": {"mean": mean_to1, "std": std_to1, "per_target": vals_to1},
        "5-to-1": {"mean": mean_5to1, "std": std_5to1, "per_target": vals_5to1}
    }


def diff_5to1_minus_1to1(conf_matrix_to1, conf_matrix_5to1, behavior="3-stirring-fast"):
    """
    Compute (5-to-1 accuracy − 1-to-1 accuracy) per target tool for a given behavior.
    """
    # mean across source tools (for each target) in 1-to-1
    vals_to1 = conf_matrix_to1.mean(axis=0)

    # extract 5-to-1 values for this behavior
    if behavior not in conf_matrix_5to1.columns:
        raise ValueError(f"Behavior {behavior} not found in 5-to-1 matrix")
    vals_5to1 = conf_matrix_5to1[behavior]

    # align indexes (only keep common tools)
    common = vals_to1.index.intersection(vals_5to1.index)
    vals_to1 = vals_to1.loc[common]
    vals_5to1 = vals_5to1.loc[common]

    # difference
    diff = vals_5to1 - vals_to1
    return pd.DataFrame({
        "1-to-1": vals_to1,
        "5-to-1": vals_5to1,
        "diff (5to1-1to1)": diff
    })


def get_result_cross_tool_1to1(base_dir, transfer_type, data_name, tool_list, shared_beh, result_id, modality="audio"):
    context_dict = {
        "source_tool_list": [],
        "target_tool_list": [],
        "source_beh_list": [shared_beh],
        "target_beh_list": [shared_beh],
        "test_object_list": [],
        "shared_object_list": [],
    }

    cross_tool_pair = list(itertools.permutations(tool_list, 2))
    pair_results = {}
    for pair in cross_tool_pair:
        context_dict['source_tool_list'] = [pair[0]]
        context_dict['target_tool_list'] = [pair[1]]
        test_result_dir = get_result_dir(base_dir=base_dir, context_dict=context_dict, transfer_type=transfer_type,
                                         modality=modality, data_name=data_name)
        result = load_json(os.path.join(test_result_dir, f"results{result_id}.json"))
        pair_results[f"{pair[0]}>{pair[1]}"] = result

    return pair_results

def get_result_cross_beh_1to1(base_dir, transfer_type, data_name, beh_list, shared_tool, result_id, modality="audio"):
    context_dict = {
        "source_tool_list": [shared_tool],
        "target_tool_list": [shared_tool],
        "source_beh_list": [],
        "target_beh_list": [],
        "test_object_list": [],
        "shared_object_list": [],
    }

    cross_pair = list(itertools.permutations(beh_list, 2))
    pair_results = {}
    for pair in cross_pair:
        context_dict['source_beh_list'] = [pair[0]]
        context_dict['target_beh_list'] = [pair[1]]
        test_result_dir = get_result_dir(base_dir=base_dir, context_dict=context_dict, transfer_type=transfer_type,
                                         modality=modality, data_name=data_name)
        result = load_json(os.path.join(test_result_dir, f"results{result_id}.json"))
        pair_results[f"{pair[0]}>{pair[1]}"] = result

    return pair_results


def get_results_cross_tool_5to1(data_name, tool_list, shared_beh_list, transfer_type, base_dir, result_id, modality="audio"):
    context_dict = {
        "source_tool_list": [],
        "target_tool_list": [],
        "source_beh_list": [],
        "target_beh_list": [],
    }
    pair_results = {}
    for tgt_t in tool_list:
        src_t_list = sorted(t for t in tool_list if t != tgt_t)
        context_dict['source_tool_list'] = src_t_list
        context_dict['target_tool_list'] = [tgt_t]
        for shared_b in shared_beh_list:
            context_dict['source_beh_list'] = [shared_b]
            context_dict['target_beh_list'] = [shared_b]

            test_result_dir = get_result_dir(base_dir=base_dir, context_dict=context_dict, transfer_type=transfer_type,
                                             modality=modality, data_name=data_name)

            try:
                result = load_json(os.path.join(test_result_dir, f"results{result_id}.json"))
                pair_results[f"target_{tgt_t}&{shared_b}"] = result
            except FileNotFoundError:
                print(f"no result: {test_result_dir}")
                continue

    return pair_results


def get_results_cross_beh_4to1(data_name, beh_list, shared_tool_list, transfer_type, base_dir, result_id, modality="audio"):

    context_dict = {
        "source_tool_list": [],
        "target_tool_list": [],
        "source_beh_list": [],
        "target_beh_list": [],
        "test_object_list": [],
        "shared_object_list": [],
    }

    pair_results = {}
    for target_beh in beh_list:
        context_dict['source_beh_list'] = [b for b in beh_list if b != target_beh]
        context_dict['target_beh_list'] = [target_beh]
        for shared_t in shared_tool_list:
            context_dict["source_tool_list"] = [shared_t]
            context_dict["target_tool_list"] = [shared_t]

            test_result_dir = get_result_dir(base_dir=base_dir, context_dict=context_dict, transfer_type=transfer_type,
                                             modality=modality, data_name=data_name)
            try:
                result = load_json(os.path.join(test_result_dir, f"results{result_id}.json"))
                pair_results[f"target_{target_beh}&{shared_t}"] = result
            except FileNotFoundError:
                print(f"no result: {test_result_dir}")
                continue

    return pair_results


def make_tool_behavior_matrix(pair_results_clap_cB):
    """
    Build DataFrame: tool (rows) × target behavior (columns).
    Adds row/col averages.
    """
    data = []
    for key, result in pair_results_clap_cB.items():
        if "test_results" not in result:
            continue
        avg_acc = result["test_results"]["avg_accuracy"]

        try:
            _, rest = key.split("target_")
            tgt_b, tool = rest.split("&")
        except ValueError:
            continue

        data.append((tool, tgt_b, avg_acc))

    df = pd.DataFrame(data, columns=["tool", "behavior", "accuracy"])
    df_pivot = df.pivot(index="tool", columns="behavior", values="accuracy")

    # add averages
    df_pivot["Avg(behaviors)"] = df_pivot.mean(axis=1)
    df_pivot.loc["Avg(tools)"] = df_pivot.mean(axis=0)

    return df_pivot


def plot_tool_behavior_matrix(df_matrix, title="Tool × Target Behavior Accuracy"):
    # compute global mean & std
    values = df_matrix.values.flatten()
    values = values[~np.isnan(values)]
    mean_val, std_val = np.mean(values), np.std(values)

    full_title = f"{title}\nMean: {mean_val:.3f}, Std: {std_val:.3f}"

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df_matrix,
        annot=True, fmt=".2f", cmap="Blues", cbar=True,
        linewidths=0.5, linecolor="gray",
        mask=df_matrix.isna()
    )
    plt.title(full_title)
    plt.xlabel("Target Behavior")
    plt.ylabel("Tool")
    plt.tight_layout()
    plt.show()
