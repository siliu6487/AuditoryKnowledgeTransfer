import logging
import time

import torch

import configs
import model
from my_helpers.data_helpers import setup_context_for_experiment, fill_missing_hyper_params, filter_keys_by_func, \
    get_config_params, validate_transfer
from my_helpers.viz_helpers import viz_encoder_input_data, viz_encoder_output_data, viz_decision_boundary
from transfer_class import ToolKnowledgeTransfer


def run_pipeline(loss_func: str, data_name: str, multi_class: bool,
                 curr_context: dict = None, pipe_settings: dict = None, hyparams: dict = None) -> dict:
    """
    Run the transfer learning pipeline: train encoder on target tool (with shared/old obj) and source tool(with all obj)
        --> freeze encoder -->  train classifier on source tool (new obj) --> test classifier on target tool (new obj)

    :param loss_func: triplet loss as "TL" or sincere loss as "sincere"
    :param data_name: name of the dataset
    :param multi_class: classifier predicts object (multi-class) or object attributes (multi-label)
    :param curr_context: context for the transfer task, such as behaviors, source tool and target tool, etc.
                         if not fully specified, will be automatically filled up by values from configs.py file.
    :param pipe_settings: setting for the transfer task, such as encoder_exp_name, retrain_clf etc.
                         if not fully specified, will be automatically filled up by values from configs.py file.
    :param hyparams: hyperparameters for the transfer task, such as encoder_output_dim, sincere_temp etc.
                         if not fully specified, will be automatically filled up by values from configs.py file.
    :return: dictionary, at least with "test_acc": accuracy from the classifier tested on target tool

    A structured way to manage and pass parameters to functions:
    1. Filters relevant keys from a dictionary based on a function's expected parameters, preventing unexpected arguments.
    2. Updates the filtered dictionary with explicitly specified values that are required for the function.
    3. Passes the dictionary to the function, allowing:
        - Default values in the target function to remain unchanged unless explicitly updated.
        - easier to identify the task specific changes without overwriting or duplicating unchanged parameters.
        - Simplifying tracking of changes to defaults arguments
            e.g., when orig_context['trail_list']==[0,1,2,3,4], full pipeline's trail_list will be [0,1,2,3,4]
                    without explicitly updating to any function with trail_list parameter
        - Avoidance of explicitly specifying all parameters, which can make the code verbose and harder to maintain.
    """
    start_time = time.time()
    pipeline_results = {}

    # %% 0. set up parameters and fill with defaults from configs.py
    all_params = get_config_params()
    if curr_context is not None:
        all_params.update(curr_context)
    validate_transfer(all_params)
    if pipe_settings is not None:
        all_params.update(pipe_settings)
    if hyparams is not None:
        all_params.update(hyparams)  # change default values first
        # functions take hyparams parameter as a whole dictionary,
        # so there's no need to specify one parameter for each hyperparameter
        all_params.update({"hyparams": fill_missing_hyper_params(hyparams, param_model="both")})  # fill up the param list

    context_dict = setup_context_for_experiment(**filter_keys_by_func(all_params, setup_context_for_experiment))
    logging.debug(f"context_dict: {context_dict}")

    transfer = ToolKnowledgeTransfer(encoder_loss_fuc=loss_func, data_name=data_name, multi_class=multi_class)

    # Assuming modalities are concatenated
    # Assuming data from all behaviors and tools are pre-processed to the same length
    input_dim = 0
    data_dim = 0
    for modality in all_params['modality_list']:
        trial_batch = transfer.orig_data[all_params['source_beh_list'][0]][
            all_params['source_tool_list'][0]][modality]
        x_sample = trial_batch[context_dict['clf_objs'][0]]['X'][0]
        input_dim += len(x_sample)
        data_dim = x_sample.shape[-1]

    if all_params['viz_dataset']:
        logging.info("👀visualize initial data ...")
        viz_encoder_input_data(transfer_obj=transfer, data_dim=data_dim, all_params=all_params, context_dict=context_dict)
    # %%

    # %% 1. encoder
    if all_params['use_encoder']:
        if all_params['retrain_encoder']:
            logging.info(f"👉 ------------ Training representation encoder using {loss_func} loss ------------ ")
            encoder_time = time.time()
            enc_params = filter_keys_by_func(all_params, transfer.train_encoder)
            enc_params.update({
                "new_object_list": context_dict['enc_new_objs'], "old_object_list": context_dict['enc_old_objs'],
                "source_tool_list": context_dict['enc_source_tools'], "target_tool_list": context_dict['enc_target_tools'],
                "source_beh_list": context_dict['enc_source_behs'], "target_beh_list": context_dict['enc_target_behs'],
                "trail_list": context_dict['enc_train_trail_list']
            })
            encoder = transfer.train_encoder(**enc_params)
            if all_params['save_temp_model']:
                torch.save(encoder.state_dict(), all_params['enc_pt_folder'] + all_params['encoder_pt_name'])
            logging.info(f"⏱️Time used for encoder training: {round((time.time() - encoder_time) // 60)} "
                         f"min {(time.time() - encoder_time) % 60:.1f} sec.")
        else:  # look for checkpoint
            encoder = model.encoder(input_size=input_dim, hidden_size=all_params['hyparams']['encoder_hidden_dim'],
                                    output_size=all_params['hyparams']['encoder_output_dim']).to(all_params['device'])
            encoder.load_state_dict(torch.load(all_params['enc_pt_folder'] + all_params['encoder_pt_name'],
                                               map_location=torch.device(all_params['device'])))
        if all_params['viz_share_space']:
            logging.info("👀visualize embeddings in shared latent space...")
            viz_encoder_output_data(transfer_obj=transfer, encoder=encoder, loss_func=loss_func,
                                    all_params=all_params, context_dict=context_dict)
    else:
        encoder = None

    # %% 2. classifier
    if all_params['retrain_clf']:
        logging.info(f"👉 ------------ Training classification head ------------ ")
        clf_time = time.time()
        clf_params = filter_keys_by_func(all_params, transfer.train_classifier)
        clf_params.update({
            'Encoder': encoder,
            "new_object_list": context_dict['clf_objs'],
            "source_tool_list": context_dict['clf_source_tools'],
            "source_beh_list": context_dict['clf_source_behs'],
            "trail_list": context_dict['enc_train_trail_list']
        })
        clf = transfer.train_classifier(**clf_params)
        if all_params['save_temp_model']:
            torch.save(clf.state_dict(), all_params['clf_pt_folder'] + all_params['clf_pt_name'])

        logging.info(f"⏱️Time used for classifier training: {round((time.time() - clf_time) // 60)} "
                     f"min {(time.time() - clf_time) % 60:.1f} sec.")

    else:  # look for checkpoint
        clf = model.classifier(input_size=all_params['hyparams']['encoder_output_dim'],
                               output_size=transfer.num_clf_class).to(all_params['device'])
        clf.load_state_dict(
            torch.load(all_params['clf_pt_folder'] + all_params['clf_pt_name'],
                       map_location=torch.device(all_params['device'])))

    # %% 3. evaluation
    logging.info(f"👉 ------------ Evaluating the classifier ------------ ")
    eval_params = filter_keys_by_func(all_params, transfer.eval_classifier)
    eval_params.update({
        "Encoder": encoder, "Classifier": clf, "tool_list": context_dict['clf_target_tools'],
        "behavior_list": context_dict['clf_target_behs'], "return_pred": True,
        "trail_list": context_dict['clf_val_trial_list'], "new_object_list": context_dict['clf_objs']
    })
    print(f"test eval_params: {eval_params}")

    test_result = transfer.eval_classifier(**eval_params)

    print(f"test_result: {test_result}")
    print(f"test_acc: {test_result['test_accuracy']}")
    logging.info(f"test accuracy: {test_result['test_accuracy'] * 100:.2f}%")
    logging.info(f"⏱️total time used: {round((time.time() - start_time) // 60)} "
                 f"min {(time.time() - start_time) % 60:.1f} sec.")

    if all_params['viz_decision_boundary']:
        logging.info("👀visualize decision boundary in shared latent space...")
        viz_decision_boundary(transfer_obj=transfer, encoder=encoder, clf=clf, test_acc=test_result['test_accuracy'],
                              all_params=all_params, context_dict=context_dict)

    pipeline_results["test_acc"] = test_result['test_accuracy']
    return pipeline_results
