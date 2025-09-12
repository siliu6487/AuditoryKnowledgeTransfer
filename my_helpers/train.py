import logging
import os
import random
import time
from typing import List, Union

import numpy as np
import torch

import configs
from my_helpers.pipeline import run_pipeline

encoder_pt_name = f"tmp_myencoder.pt"
clf_pt_name = f"tmp_myclassifier.pt"


def train_k_fold(train_val_obj_list: List[str], number_of_folds: int, loss_func: str, data_name: str,
                 grid: dict, pipe_settings: dict, context: dict, val_size=4, no_overlap_sample=True):
    '''
    cross validation split on objects: some objects for train, the rest objects for val
    Assume that the last 3 objects are unknown to the tool, we use the 12 known ones to split train and val.
    '''
    if no_overlap_sample:
        assert len(train_val_obj_list) % number_of_folds == 0, (f"can't split {len(train_val_obj_list)} objects into "
                                                                f"{number_of_folds} folds without overlap!")
        num_obj_per_fold = len(train_val_obj_list) // number_of_folds
    else:
        num_obj_per_fold = val_size

    best_acc = -1
    best_acc_std = -1
    best_alpha = -1
    best_lr_en = -1
    best_temp = -1
    best_dim = -1
    rand_guess_acc = 1 / num_obj_per_fold

    pipe_settings.update({'save_temp_model': False, 'viz_dataset': False, 'viz_share_space': False, 'save_fig': False})
    logging.info("###########################Grid Search Start###########################")
    search_start_time = time.time()
    num_grid_combo = 0

    for alpha in grid['alpha_list']:
        for temp in grid['temp_list']:
            for lr_encoder in grid['lr_en_list']:
                for encoder_output_dim in grid['encoder_output_dim_list']:
                    num_grid_combo += 1
                    hyparams = {'TL_margin': alpha, 'lr_encoder': lr_encoder, 'sincere_temp': temp,
                                'encoder_output_dim': encoder_output_dim}  # 👈 add params
                    logging.info(f"current search: {hyparams}")
                    logging.info(f"========================= {number_of_folds}fold cv start=========================")
                    acc_list = []
                    cv_start_time = time.time()
                    for fold_idx in range(number_of_folds):
                        random.seed(configs.rand_seed + fold_idx)  # same splits for all hyperparam combos
                        logging.info(f"-----------fold {fold_idx + 1}/{number_of_folds} start-----------")
                        fold_start_time = time.time()
                        if no_overlap_sample:  # normal k fold
                            val_obj_list = train_val_obj_list[fold_idx * num_obj_per_fold: (fold_idx + 1) * num_obj_per_fold]
                        else:  # random select val objs
                            val_obj_list = random.sample(train_val_obj_list, val_size)
                        logging.debug(f"num_grid_combo: {num_grid_combo}, fold_idx: {fold_idx}, val_obj_list: {val_obj_list}")
                        train_obj_list = [item for item in train_val_obj_list if item not in val_obj_list]
                        context.update({'old_object_list': train_obj_list, 'new_object_list': val_obj_list})
                        result = run_pipeline(loss_func=loss_func, data_name=data_name,
                                              curr_context=context, pipe_settings=pipe_settings, hyparams=hyparams,
                                              multi_class=configs.multi_class)

                        acc_list.append(result["test_acc"])
                        logging.info(f"👉 fold {fold_idx + 1}/{number_of_folds} val_obj: {val_obj_list} \n"
                                     f"TL margin: {alpha}, lr: {lr_encoder}, val accuracy: {result['test_acc'] * 100:.2f}%, "
                                     f"random guess accuracy: {rand_guess_acc * 100:.1f}%")

                        logging.info(f"☑️ total time used for {fold_idx+1}th fold: "
                                     f"{round((time.time() - fold_start_time) // 60)} min "
                                     f"{(time.time() - fold_start_time) % 60:.1f} sec.")

                    avg_acc = np.mean(acc_list)
                    std_acc = np.std(acc_list)
                    if avg_acc > best_acc:  # 👈 add params
                        best_acc = avg_acc
                        best_acc_std = std_acc
                        best_alpha = alpha
                        best_lr_en = lr_encoder
                        best_temp = temp
                        best_dim = encoder_output_dim

                    logging.info(f"☑️ total time for {number_of_folds}fold cv: {round((time.time() - cv_start_time) // 60)} min "
                                 f"{(time.time() - cv_start_time) % 60:.1f} sec.")

    logging.info(f"✅ The best avg val accuracy is: {best_acc * 100:.1f}%, best_alpha: {best_alpha}, "
                 f"best_lr_en: {best_lr_en}, best_temp: {best_temp}, best encoder_output_dim: {best_dim}"
                 f" random guess accuracy: {rand_guess_acc * 100:.1f}%")  # 👈 add params

    logging.info(f"✅ total time for grid search for for {num_grid_combo} combinations ({number_of_folds} cv each): "
                 f"{round((time.time() - search_start_time) // 60)} min "
                 f"{(time.time() - search_start_time) % 60:.1f} sec.")

    return {  # 👈 add params
        "best_val_acc": best_acc,
        "best_val_acc_std": best_acc_std,
        "TL_margin": best_alpha,
        "lr_encoder": best_lr_en,
        "sincere_temp": best_temp,
        "encoder_output_dim": best_dim
    }


def train_fixed_param(train_val_obj_list: List[str], test_obj_list: List[str], loss_func: str, data_name: str,
                      hyparams: dict, pipe_settings: dict, context: dict, test_name="fold0"):
    logging.warning("train_fixed_param function currently only applies the best TL alpha and training learning rate"
                    ", all other hyper-params are by default from configs.")

    test_enc_pt_folder = './saved_model/encoder/test/'
    test_clf_pt_folder = './saved_model/classifier/test/'
    if not os.path.exists(test_enc_pt_folder):
        os.makedirs(test_enc_pt_folder)
    if not os.path.exists(test_clf_pt_folder):
        os.makedirs(test_clf_pt_folder)
    pipe_settings.update(
        {'viz_dataset': False, 'viz_share_space': False, 'save_fig': False,
         'enc_pt_folder': test_enc_pt_folder, 'encoder_pt_name': f"{test_name}_{configs.encoder_pt_name}",
         'clf_pt_folder': test_clf_pt_folder, "clf_pt_name": f"{test_name}_{configs.clf_pt_name}"})

    context.update({'old_object_list': train_val_obj_list, 'new_object_list': test_obj_list})
    result = run_pipeline(loss_func=loss_func, data_name=data_name,
                          curr_context=context, pipe_settings=pipe_settings, hyparams=hyparams,
                          multi_class=configs.multi_class)

    return result["test_acc"]
