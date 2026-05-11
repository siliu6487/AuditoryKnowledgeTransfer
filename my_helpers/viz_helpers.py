import copy
import logging
import os.path

import torch
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
import numpy as np

from my_helpers.epx_specific.torch_pipeline import TransferPipeline
import configs
import model
from my_helpers.data_helpers import get_all_embeddings_or_data, restart_label_index_from_zero, create_tool_idx_list, \
    filter_keys_by_func, combine_datasets_embeddings, make_dataset_by_context

# manually order objects by similarity
SIM_OBJECTS_LIST = ['empty', 'water', 'detergent', 'chia-seed', 'cane-sugar', 'salt',
                    'styrofoam-bead', 'split-green-pea', 'wheat', 'chickpea', 'kidney-bean',
                    'wooden-button', 'plastic-bead', 'glass-bead', 'metal-nut-bolt']
SORTED_DATA_OBJ_LIST = sorted(SIM_OBJECTS_LIST)  # for input data from data files, the labels were created in this order


def reduce_dimension(all_embeds):
    # Reduce to 2D with PCA
    if all_embeds.shape[1] > 2:
        print(f"reduce dimension...")
        tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
        xy = tsne.fit_transform(all_embeds)  # (N, 2)
    else:
        # Already 2D or 1D
        if all_embeds.shape[1] == 1:
            xy = np.hstack([all_embeds, np.zeros_like(all_embeds)])
        else:
            xy = all_embeds
    return xy


def extract_plot_points(xy, meta, obj2color, full_obj_list):
    xs, ys, colors, markers, edges, sizes = [], [], [], [], [], []
    for (dataset_type, obj_id, tool_id, beh_id), (x, y) in zip(meta, xy):
        # Color = object
        color = obj2color[full_obj_list[obj_id]]

        # Shape = dataset type
        if dataset_type == "source_train":
            marker = "o"  # circle
            size = 100
        else:
            marker = "s"  # square
            size = 100

        # Edge = test dataset
        edgecolor = "black" if dataset_type == "target_test" else "none"
        if dataset_type == "target_test": size = 150

        xs.append(x)
        ys.append(y)
        colors.append(color)
        markers.append(marker)
        edges.append(edgecolor)
        sizes.append(size)
    return xs, ys, colors, markers, edges, sizes


def plot_datasets(context_dict, full_obj_list, datasets=None, encoder=None, save_fig_path=None, save_name="figure.jpg"):
    if datasets is None:
        source_train_dataset, target_train_dataset, target_test_dataset = make_dataset_by_context(
            context_dict, full_obj_list)
        datasets = {
            "source_train": source_train_dataset,
            "target_train": target_train_dataset,
            "target_test": target_test_dataset,
        }
    all_embeds, meta = combine_datasets_embeddings(datasets, encoder)
    xy = reduce_dimension(all_embeds)

    # Setup colormap for objects
    cmap = plt.get_cmap("tab20", len(full_obj_list))
    obj2color = {obj: cmap(i) for i, obj in enumerate(full_obj_list)}

    # Extract plotting info
    xs, ys, colors, markers, edges, sizes = extract_plot_points(
        xy, meta, obj2color, full_obj_list
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for x, y, c, m, e, s in zip(xs, ys, colors, markers, edges, sizes):
        ax.scatter(x, y, c=[c], marker=m, edgecolors=e, s=s, alpha=0.8)

    # Shape legend (source vs target vs test)
    cross_type = "Tool" if context_dict['source_tool_list'] != context_dict['target_tool_list'] \
        else "Behavior"
    shape_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"Source {cross_type} (All Obj)",
                   markerfacecolor="gray", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label=f"Target {cross_type} (Shared Obj)",
                   markerfacecolor="gray", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label=f"Target {cross_type} (Transfer Obj)",
                   markerfacecolor="gray", markeredgecolor="black", markersize=10),
    ]
    if len(datasets['target_train']) == 0:
        shape_handles = [shape_handles[0]] + [shape_handles[2]]

    ax.legend(handles=shape_handles, fontsize=15)

    # Discrete colormap for objects
    cmap = plt.get_cmap("tab20", len(full_obj_list))
    bounds = np.arange(len(full_obj_list) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Create a ScalarMappable so we can attach a colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #
    # # Add colorbar with object labels
    cbar = plt.colorbar(sm, ticks=np.arange(len(full_obj_list)) + 0.5, ax=ax)
    cbar.ax.set_yticklabels(full_obj_list, fontsize=15)  # object names
    cbar.set_label("Objects", fontsize=15)

    # Black box around plot
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1)

    # Tick marks
    ax.tick_params(axis="both", which="both", direction="out", length=6, width=1.2, colors="black")
    # ax.minorticks_on()

    ax.set_title("Dataset Visualization (t-SNE reduced)", fontsize=20)
    ax.set_xlabel("t-SNE 1", fontsize=15)
    ax.set_ylabel("t-SNE 2", fontsize=15)
    ax.grid(False)
    ax.set_facecolor("white")
    plt.tight_layout()
    if save_fig_path is not None:
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
        plt.savefig(f"{save_fig_path}/{save_name}", dpi=300, format="jpeg", bbox_inches="tight")
        print(f"figure saved as: {save_fig_path}/{save_name}")


    plt.show()


def _map_objects_to_colors(obj_list, labels):
    labels = np.squeeze(labels)
    # take the subset of all colors (from SIM_OBJECTS_LIST) for obj_list
    subset_obj_idx = [SIM_OBJECTS_LIST.index(obj) for obj in obj_list]
    logging.debug(f"subset_obj_idx: {subset_obj_idx}, unique labels: {np.unique(labels)}")
    sim_colors = plt.colormaps['tab20b'](np.linspace(0, 1, len(SIM_OBJECTS_LIST)))  # fix colors for all 15 objects
    subset_colors = sim_colors[sorted(subset_obj_idx)]  # select the subset colors, value has to be
    cmap = ListedColormap(subset_colors)  # Discrete colormap
    bounds = list(range(len(obj_list) + 1))  # Boundaries for discrete colors
    norm = BoundaryNorm(bounds, len(obj_list))  # Normalize to discrete boundaries

    # map current label (ordered by obj_list) to align with color (ordered by SIM_OBJECTS_LIST)
    # Step 1: Remove duplicates from subset_obj_idx while preserving the order
    unique_subset_idx = list(dict.fromkeys(subset_obj_idx))  # [9, 2, 14, 13, 8]

    # Step 2: Create a mapping from original labels to indices in the sorted subset
    sim_obj_idx = [SIM_OBJECTS_LIST.index(obj_list[int(l)]) for l in labels]
    label_to_subset_idx = {label: sim_obj_idx[i] for i, label in enumerate(labels)}  # Map labels to subset indices
    sorted_unique_idx = sorted(unique_subset_idx)  # Sort the unique subset indices
    index_mapping = {idx: i for i, idx in enumerate(sorted_unique_idx)}  # Map sorted indices to new positions

    # Step 3: Map original labels to new positions
    mapped_labels = np.array([index_mapping[label_to_subset_idx[label]] for label in labels])
    return mapped_labels, cmap, norm, subset_obj_idx, label_to_subset_idx


def _get_curr_labels_on_cbar(subset_obj_idx, label_to_subset_idx):
    cbar_labels = np.array(SIM_OBJECTS_LIST)[sorted(subset_obj_idx)]
    reversed_mp = {sim_idx: label for label, sim_idx in label_to_subset_idx.items()}
    original_labels = [reversed_mp[sim_idx] for sim_idx in sorted(subset_obj_idx)]
    tick_labels = [f"{cbar_label}\n(label {orig_label})" for cbar_label, orig_label in
                   zip(cbar_labels, original_labels)]
    return tick_labels


def viz_data(trans_cls, encoder: model.encoder or None, data_dim=None,
             loss_func=configs.loss_func, source_tool_list=configs.source_tool_list,
             target_tool_list=configs.target_tool_list,
             new_object_list=configs.new_object_list, old_object_list=configs.old_object_list,
             source_beh_list=configs.source_beh_list, target_beh_list=configs.target_beh_list,
             modality_list=configs.modality_list, show_fig=True,
             trail_list=configs.trail_list, viz_l2_norm=configs.viz_l2_norm, save_fig=configs.save_fig):
    assert (encoder or data_dim) is not None
    if encoder is not None:
        encoder = copy.deepcopy(encoder)
        encoder.l2_norm = True if viz_l2_norm else False  # check if project and viz embeddings that are l2 normed
    all_object_list = old_object_list + new_object_list
    data_groups, label_groups, _ = get_all_embeddings_or_data(
        trans_cls=trans_cls, encoder=encoder, data_dim=data_dim,
        source_tool_list=source_tool_list, target_tool_list=target_tool_list,
        source_beh_list=source_beh_list, target_beh_list=target_beh_list,
        old_object_list=old_object_list, new_object_list=new_object_list,
        modality_list=modality_list, trail_list=trail_list)
    # all_emb order: source_old, source_new, target_old, target_new
    source_data_train, source_label_train = data_groups[0], label_groups[0]
    source_data_test, source_label_test = data_groups[1], label_groups[1]
    target_data_train, target_label_train = data_groups[2], label_groups[2]
    target_data_test, target_label_test = data_groups[3], label_groups[3]

    if source_tool_list == target_tool_list and source_beh_list == target_beh_list:  # if there's no knowledge transfer
        assert new_object_list == []  # no new object
        assert np.equal(source_data_train, target_data_train).all()
        assert np.equal(source_data_test, target_data_test).all()

        # viz ALL data in 2D space
    len_list = [len(l_group) for l_group in label_groups]
    tool_labels = create_tool_idx_list(source_label_len=len_list[0] + len_list[1],
                                       target_label_train_len=len_list[2], target_label_test_len=len_list[3])
    data_descpt = "Encoder Input Data (All Objects)" if encoder is None else \
        f"Encoder Output Data (All Objects) - {loss_func} Loss"
    if source_beh_list == target_beh_list and source_tool_list != target_tool_list:  # tool transfer
        subtitle = (f"behavior: {source_beh_list}, target tool: {target_tool_list} \n"
                    f"source tool: {source_tool_list}")
    elif source_beh_list != target_beh_list and source_tool_list == target_tool_list:  # behavior transfer
        subtitle = (f"tool: {source_tool_list}, \n target behavior: {target_beh_list} \n"
                    f"source behavior: {source_beh_list}")
    else:
        subtitle = (f"target tool: {target_tool_list}, target behavior: {target_beh_list} \n"
                    f"source tool: {source_tool_list} \n source behavior: {source_beh_list}")

    if len(tool_labels) != 0:
        logging.debug(f"Viz data for all objects...")
        _viz_embeddings(embeds=np.vstack(data_groups), labels=np.vstack(label_groups), loss_func=loss_func,
                        viz_l2_norm=viz_l2_norm, tool_labels=tool_labels, obj_list=all_object_list,
                        save_fig=save_fig, title=f"{data_descpt}", show_fig=show_fig, subtitle=subtitle)
    # Viz train data in 2D space
    data = np.vstack([source_data_train, source_data_test, target_data_train])
    if len(data) != 0:
        logging.debug(f"Viz data for train objects...")
        labels = np.vstack([source_label_train, source_label_test, target_label_train])
        tool_labels = create_tool_idx_list(source_label_len=len(source_label_train) + len(source_label_test),
                                           target_label_train_len=len(target_label_train))
        data_descpt = "Encoder Input Data (Train Objects)" if encoder is None else \
            f"Encoder Output Data (Train Objects) - {loss_func} Loss"
        _viz_embeddings(embeds=data, labels=labels, tool_labels=tool_labels, loss_func=loss_func, show_fig=show_fig,
                        viz_l2_norm=viz_l2_norm, obj_list=all_object_list, save_fig=save_fig, title=f"{data_descpt}",
                        subtitle=subtitle)

    # viz test data in 2D space
    data = np.vstack([source_data_test, target_data_test])
    if len(data) != 0:
        logging.debug(f"Viz data for test objects...")
        labels = np.squeeze(np.vstack([source_label_test, target_label_test]))
        labels = restart_label_index_from_zero(labels)  # labels should start from 0 to match the new object list
        tool_labels = create_tool_idx_list(source_label_len=len(source_label_test),
                                           target_label_test_len=len(target_label_test))
        data_descpt = "Encoder Input Data (Test Objects)" if encoder is None else \
            f"Encoder Output Data (Test Objects) - {loss_func} Loss"
        _viz_embeddings(viz_l2_norm=viz_l2_norm, embeds=data, labels=labels, loss_func=loss_func, show_fig=show_fig,
                        tool_labels=tool_labels, obj_list=new_object_list, save_fig=save_fig, title=f"{data_descpt}",
                        subtitle=subtitle)


def _viz_embeddings(embeds: np.ndarray, labels: np.ndarray, tool_labels: list, obj_list: list, loss_func: str,
                    viz_l2_norm, save_fig: bool, title='', subtitle='', show_curr_label=False,
                    show_fig=True, save_path=r'./figs/'):
    """
    visualize embeddings in 2D space and use object as color and tool as marker.
    :param embeds: embedding by trial/sample:  [n_sample, emb_len]. The section of rows has to follow this order:
        ['Source Tool(old and/or new obj)', 'Target Tool(old)', 'Target Tool(new)']
    :param tool_labels: tool index for embeddings
    :param labels: [n_sample, 1 or 0], will be squeezed
    :param obj_list: list of unique object names, order aligns with labeling: e.g., ["wheat", ...] -> label 0 is "wheat"
    :param loss_func: name for the loss function, e.g. "TL"
    :param viz_l2_norm: whether visualize l2 normed embeddings, if True, embeddings will be on a unit circle
    :param save_fig: whether save this figure
    :param title: content for the first line of title
    :param subtitle: second line of title: "... Visualization of {save_name}\n {subtitle}"
    :param show_curr_label: show the actual label (e.g., label for classifier) for the object on color bar.
        the color bar is fixed based on SIM_OBJECTS_LIST, so we need to re-align labels with the colorbar tick labels

    :return:
    """

    assert (np.max(labels) == len(obj_list) and np.min(labels) == 0) or len(np.unique(obj_list)) == len(obj_list)
    labels = np.squeeze(labels)
    tool_labels = np.array(tool_labels)

    logging.debug(f"➡️ _viz_embeddings.. viz_l2_norm: {viz_l2_norm}")
    logging.debug(f"embeds shape: {embeds.shape}, all_emb max: {np.max(embeds):.3f}, min: {np.min(embeds):.3f}")
    logging.debug(f"labels: \n      {labels}")
    logging.debug(f"tool_labels: \n      {tool_labels}")
    half = int(len(embeds) / 2)

    # make sure Dimensionality is 2D
    if embeds.shape[1] > 2:

        tsne = TSNE(n_components=2, random_state=configs.rand_seed, init='pca', learning_rate="auto")
        embeds_2d = tsne.fit_transform(embeds)
        if viz_l2_norm:
            norms = np.linalg.norm(embeds_2d, axis=1, keepdims=True)
            embeds_2d /= norms
        logging.debug(f"dimension reduced embeds_2d max: {np.max(embeds_2d):.3f}, min: {np.min(embeds_2d):.3f}")
    else:
        embeds_2d = embeds
    half = int(len(embeds_2d) / 2)

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 12
    context_order_list = ['Source', 'Target (Train)', 'Target (Test)']
    markers = ['o', 's', 's']  # Markers for tools. Circle for source, square for target test

    # take the subset of all colors (from SIM_OBJECTS_LIST) for obj_list
    mapped_labels, cmap, norm, subset_obj_idx, label_to_subset_idx = _map_objects_to_colors(obj_list, labels)

    # for the color bar reference only, in case the last batch of objects for plt are a subset of obj_list
    scatter = plt.scatter(
        embeds_2d[:, 0], embeds_2d[:, 1],
        c=mapped_labels,
        cmap=cmap,
        norm=norm,
        s=0  # Use invisible points
    )

    for tool_label, marker in enumerate(markers):
        mask = tool_labels == tool_label
        masked_labels = labels[mask]
        if len(masked_labels) == 0:
            continue
        shrink = 1  # if on unit sphere, change tools' radius, so they don't block each other
        edge_color = None  # add edge to test object marker
        size = 50
        if tool_label in [1, 2]:  # target embedding
            size = 80
            if viz_l2_norm:
                shrink = 0.95  # inside source embedding circle
            if tool_label == 2:  # test object
                edge_color = 'black'
        plt.scatter(
            embeds_2d[mask, 0] * shrink, embeds_2d[mask, 1] * shrink,
            c=mapped_labels[mask],
            cmap=cmap,
            norm=norm,
            marker=marker,
            edgecolor=edge_color,
            s=size,
            alpha=1.0,
            label=f"{context_order_list[tool_label]}"
        )

    cbar = plt.colorbar(scatter, ticks=np.arange(len(obj_list)) + 0.5)  # ticks at the center of each color
    cbar.ax.tick_params(which='minor', length=0)  # Hide minor ticks on each color's boundary
    cbar.set_label("Objects", rotation=270, labelpad=20)
    # show labels for color bar
    if show_curr_label:  # map color bar label to input labels
        cbar.ax.set_yticklabels(_get_curr_labels_on_cbar(subset_obj_idx, label_to_subset_idx))
    else:  # colors are sorted by sim object, so should the object names
        cbar.ax.set_yticklabels(np.array(SIM_OBJECTS_LIST)[sorted(subset_obj_idx)])

    # plt.legend(title="Tool Type", loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.legend()
    save_name = title if title else f"Shared Space-{loss_func} loss"
    viz_discpt = "T-SNE Reduced " if embeds.shape[1] > 2 else ""
    plt.title(f"Visualization of {save_name} - Vector Size={embeds.shape[-1]}\n {subtitle}", fontsize=10)
    plt.xlabel(f"{viz_discpt}Dimension 1")
    plt.ylabel(f"{viz_discpt}Dimension 2")
    plt.grid(True)
    # plt.tight_layout()

    if save_fig:
        plt.savefig(save_path + save_name + '.jpeg', bbox_inches='tight')
    if show_fig:
        plt.show(block=False)
    plt.close()


def viz_test_objects_embedding(
        transfer_class, Encoder, Classifier, test_accuracy, viz_l2_norm=configs.viz_l2_norm, save_fig=configs.save_fig,
        source_beh_list=configs.source_beh_list, target_beh_list=configs.target_beh_list,
        new_object_list=configs.new_object_list,
        source_tool_list=configs.source_tool_list, target_tool_list=configs.target_tool_list,
        encoder_output_dim=configs.encoder_output_dim, task_descpt='', save_path=r'./figs/', show_fig=True,
        clf_exp_name=configs.clf_exp_name, trail_list=configs.trail_list,
        modality_list=configs.modality_list):
    logging.debug(f"viz_test_objects_embedding: viz_l2_norm: {viz_l2_norm} ")
    object_list = new_object_list
    # Step 1: Generate embedded data
    clf_input_l2_norm = transfer_class.enc_l2_norm
    Encoder.l2_norm = True if viz_l2_norm else False

    all_emb, all_labels, _ = get_all_embeddings_or_data(
        trans_cls=transfer_class, encoder=Encoder, target_tool_list=target_tool_list, source_tool_list=source_tool_list,
        new_object_list=new_object_list, old_object_list=[], data_dim=None,
        target_beh_list=target_beh_list, source_beh_list=source_beh_list,
        modality_list=modality_list, trail_list=trail_list)
    source_emb, source_y = all_emb[1], all_labels[1]
    target_emb, target_y = all_emb[3], all_labels[3]
    all_emb = np.vstack([source_emb, target_emb])
    logging.debug(f"embeds shape: {all_emb.shape}, all_emb max: {np.max(all_emb):.3f}, min: {np.min(all_emb):.3f}")
    logging.debug(f"source_y shape: {source_y.shape}, target_y shape: {target_y.shape}")

    # Step 2: make sure the dimension is 2
    if source_emb.shape[1] > 2:  # Apply T-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=configs.rand_seed, init='pca', learning_rate="auto")
        all_embeddings_2d = tsne.fit_transform(all_emb)
        if viz_l2_norm:
            norms = np.linalg.norm(all_embeddings_2d, axis=1, keepdims=True)
            all_embeddings_2d /= norms
        source_embeddings_2d = all_embeddings_2d[:len(source_emb)]
        new_embeddings_2d = all_embeddings_2d[-len(target_emb):]
    elif source_emb.shape[1] == 2:
        source_embeddings_2d = source_emb
        new_embeddings_2d = target_emb
        all_embeddings_2d = all_emb
    else:
        raise Exception(f"embedding shape is not correct: {all_emb.shape}")

    # Step 3: Create a grid for decision boundary on 2d space
    if viz_l2_norm:
        buffer = 0.1
        x_min, x_max = -1 - buffer, 1 + buffer
        y_min, y_max = -1 - buffer, 1 + buffer
    else:
        x_buffer = 0.05 * (all_embeddings_2d[:, 0].max() - all_embeddings_2d[:, 0].min())
        y_buffer = 0.05 * (all_embeddings_2d[:, 1].max() - all_embeddings_2d[:, 1].min())
        x_min, x_max = all_embeddings_2d[:, 0].min() - x_buffer, all_embeddings_2d[:, 0].max() + x_buffer
        y_min, y_max = all_embeddings_2d[:, 1].min() - y_buffer, all_embeddings_2d[:, 1].max() + y_buffer
    n_points = 500
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # (n_points*n_points, 2)

    # Step 4: Train a classifier in the 2D T-SNE space
    if all_emb.shape[1] == 2:  # use the trained classifier directly because input shape matches
        if clf_input_l2_norm:  # clf was trained on l2 normed data
            norms = np.linalg.norm(grid_points, axis=1, keepdims=True)
            grid_points /= norms
        grid_proba = Classifier(torch.tensor(grid_points, dtype=torch.float32, device=configs.device))
        grid_proba = grid_proba.cpu().detach().numpy()  # (n_points*n_points, C)
    else:  # use LogisticRegression to classify data on the t-sne transformed space (approximate the decision boundaries of the Classifier)
        secondary_classifier = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
        secondary_classifier.fit(source_embeddings_2d, np.squeeze(source_y))
        grid_proba = secondary_classifier.predict_proba(grid_points)  # Shape: [n_grid_points, C]

    grid_classes = np.argmax(grid_proba, axis=1)  # (n_points*n_points, )

    # Step 5: Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 12

    levels = np.arange(-1, len(object_list))  # the lowest level as -1 to show all colors for label starting from 0!!!
    grid_classes, cmap, norm, subset_obj_idx, label_to_subset_idx = _map_objects_to_colors(object_list, grid_classes)
    plt.contourf(xx, yy, grid_classes.reshape(xx.shape), cmap=cmap, levels=levels,
                 alpha=0.3)  # contourf flips the array upside down

    # Step 6: Overlay the original and test embeddings
    color_label_source, *_ = _map_objects_to_colors(object_list, source_y)
    color_label_target, *_ = _map_objects_to_colors(object_list, target_y)

    # source
    scatter_original = plt.scatter(
        source_embeddings_2d[:, 0], source_embeddings_2d[:, 1],
        c=color_label_source, cmap=cmap, norm=norm, s=50, alpha=1.0, label="Source"
    )

    # Target. if on unit sphere, shrink the target tool embeddings' sphere, so they don't block the source embeddings
    shrink = 0.95 if viz_l2_norm else 1
    plt.scatter(
        new_embeddings_2d[:, 0] * shrink, new_embeddings_2d[:, 1] * shrink,
        c=color_label_target, cmap=cmap, norm=norm, edgecolor='k', s=80, alpha=1.0, marker='s', label="Target"
    )

    # Add a color bar and legend
    cbar = plt.colorbar(scatter_original, ticks=np.arange(len(object_list)) + 0.5)  # ticks at the center of each color
    cbar.ax.tick_params(which='minor', length=0)  # Hide minor ticks on each color's boundary
    cbar.ax.set_yticklabels(np.array(SIM_OBJECTS_LIST)[sorted(subset_obj_idx)])
    cbar.set_label("Classes", rotation=270, labelpad=20)

    if encoder_output_dim > 2:
        subtitle = ("Approximated decision boundary based on the clusters in this 2D space. \n"
                    f"Actual predictions might not match the background. Accuracy: {test_accuracy * 100:.1f}%")
    else:
        subtitle = ("Exact decision boundary from the trained Classifier.\n"
                    f"Actual predictions match the background color. Accuracy: {test_accuracy * 100:.1f}%")
    subtitle += f" Random Guess: {100 / len(object_list):.1f}%"
    plt.title(f"Visualization with Classifier Decision Boundary on Test Set - {transfer_class.encoder_loss_fuc} "
              f"- Vector Size={all_emb.shape[-1]}\n{subtitle}\n{task_descpt}", fontsize=10)
    viz_discpt = "T-SNE Reduced " if all_emb.shape[1] > 2 else ""
    plt.xlabel(f"{viz_discpt}Dimension 1")
    plt.ylabel(f"{viz_discpt}Dimension 2")

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    if viz_l2_norm:
        plt.xlim(-1 - 0.1, 1 + 0.1)
        plt.ylim(-1 - 0.1, 1 + 0.1)
    if save_fig:
        save_name = f"classifier_decision_boundary_{transfer_class.encoder_loss_fuc}"
        plt.savefig(save_path + save_name + '.jpeg', bbox_inches='tight')
    if show_fig:
        plt.show(block=False)
    plt.close()


def smooth_line(line, window_size=configs.smooth_wind_size):
    """ for unstable learning progression, smooth the line"""
    # Define the smoothing kernel
    kernel = np.ones(window_size) / window_size

    # # Apply padding to the line (reflect padding here)
    # padded_line = np.pad(line, pad_width=(window_size // 2,), mode='reflect')

    # Perform the convolution
    smoothed_line = np.convolve(line, kernel, mode='valid')
    smoothed_line = np.hstack([line[:window_size - 1], smoothed_line])  # fill the beginning part with orig value
    return smoothed_line


def window_line(line, window_size=configs.smooth_wind_size):
    """ for unstable learning progression,  draw the line as non-overlapping sliding average windows"""
    wind_line = []
    wind_idx = 0
    for i in range(len(line)):
        if i + 1 >= window_size and (i + 1) % window_size == 0:
            wind_idx += 1
            if wind_idx * window_size <= len(line):
                window_val = np.mean(line[window_size * (wind_idx - 1):window_size * wind_idx])
                wind_line.append([window_val] * window_size)
    wind_line = np.hstack(wind_line)
    gap = len(line) - len(wind_line)
    if gap > 0:
        final_mean = [np.mean(line[window_size * wind_idx:])] * gap
        wind_line = np.hstack([wind_line, final_mean])
    return wind_line


def plot_learning_progression(record, type, TL_margin, loss_func, sincere_temp, lr_classifier, lr_encoder,
                              encoder_output_dim, encoder_hidden_dim, save_name='test', show_fig=True,
                              save_path=r'./figs/', plot_every=1, save_fig=True):  # type-> 'encoder', 'classifier'
    logging.debug(f"➡️ plot_learning_progression for {type}: {save_name}...")
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12
    color_group = ['C0', 'C1', 'C3']
    if type == 'encoder':
        encoder_param = TL_margin if loss_func == "TL" else sincere_temp
        encoder_param_name = "margin" if loss_func == "TL" else "temperature"
        xaxis = np.arange(1, record.shape[1] + 1)
        plt.plot(xaxis[::plot_every], record[0, ::plot_every], color_group[0], label='train loss')
        if record[1, -1] != 0:
            plt.plot(xaxis[::plot_every], record[1, ::plot_every], color_group[1], label='val loss')
            plt.plot(xaxis[::plot_every], window_line(record[1])[::plot_every], color_group[2],
                     label='val loss (avg window)')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(f'Encoder Training Loss Progression - Loss: {loss_func} \n '
                  f'lr: {lr_encoder}, {encoder_param_name}: {encoder_param}'
                  f', emb_size: {encoder_hidden_dim} - {encoder_output_dim}', fontsize=10)
        plt.grid()
        plt.legend()
        if save_fig:
            plt.savefig(save_path + save_name + '.jpeg', bbox_inches='tight')
        if show_fig:
            plt.show(block=False)
        plt.close()
    elif type == 'classifier':
        xaxis = np.arange(1, record.shape[1] + 1)
        plt.plot(xaxis[::plot_every], record[0, ::plot_every], color_group[0], label='train loss')
        if record[1, -1] != 0:
            plt.plot(xaxis[::plot_every], record[1, ::plot_every], color_group[1], label='val loss')
        plt.xlabel('epochs')
        plt.title(f'Classifier Training Loss Progression - Encoder Loss: {loss_func} \n '
                  f'lr: {lr_classifier}', fontsize=10)
        plt.ylabel('loss')
        plt.grid()
        plt.legend()
        if save_fig:
            plt.savefig(save_path + save_name + '.jpeg', bbox_inches='tight')
        if show_fig:
            plt.show(block=False)
        plt.close()
    else:
        logging.warning(f'invalid model type: {type}, plot not available.')


def plot_data_histogram(data_flat, bins=20):
    # Plot histogram with 20 bins
    plt.figure(figsize=(8, 5))
    plt.hist(data_flat, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Data Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data Values')
    plt.grid(True)

    # Show the plot
    plt.show(block=False)


def viz_encoder_input_data(transfer_obj, data_dim, all_params, context_dict):
    viz_params = filter_keys_by_func(all_params, viz_data)
    viz_params.update({
        "trans_cls": transfer_obj, "encoder": None, "data_dim": data_dim,
        "new_object_list": context_dict['enc_new_objs'], "old_object_list": context_dict['enc_old_objs'],
        "source_tool_list": context_dict['enc_source_tools'],
        "target_tool_list": context_dict['enc_target_tools'],
        "source_beh_list": context_dict['enc_source_behs'],
        "target_beh_list": context_dict['enc_target_behs']
    })
    viz_data(**viz_params)  # TODO: update behaviors, combine tool and behavior for target and source


def viz_encoder_output_data(transfer_obj, encoder, loss_func, all_params, context_dict):
    viz_params = filter_keys_by_func(all_params, viz_data)
    viz_params.update({
        "trans_cls": transfer_obj, "encoder": encoder, "loss_func": loss_func,
        "new_object_list": context_dict['enc_new_objs'], "old_object_list": context_dict['enc_old_objs'],
        "source_tool_list": context_dict['enc_source_tools'],
        "target_tool_list": context_dict['enc_target_tools'],
        "source_beh_list": context_dict['enc_source_behs'],
        "target_beh_list": context_dict['enc_target_behs']
    })
    viz_data(**viz_params)


def viz_decision_boundary(transfer_obj, encoder, clf, test_acc, all_params, context_dict):
    viz_params = filter_keys_by_func(all_params, viz_test_objects_embedding)
    viz_params.update({
        "transfer_class": transfer_obj, "Encoder": encoder, "Classifier": clf, "test_accuracy": test_acc,
        "encoder_output_dim": all_params['encoder_output_dim'], "new_object_list": context_dict['clf_objs'],
        "source_beh_list": context_dict['clf_source_behs'], "target_beh_list": context_dict['clf_target_behs'],
        "source_tool_list": context_dict['clf_source_tools'], "target_tool_list": context_dict['clf_target_tools'],
        "task_descpt": f"source:{context_dict['clf_source_tools'][0], context_dict['clf_source_behs'][0]};"
                       f"target:{context_dict['clf_target_tools'][0], context_dict['clf_target_behs'][0]} \n "
                       f"encoder exp: {all_params['encoder_exp_name']}, clf exp: {all_params['clf_exp_name']}"
    })
    viz_test_objects_embedding(**viz_params)
