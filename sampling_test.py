import argparse
import json
import random
import sys
import time
import logging
import os

import numpy as np

import configs
import my_helpers.data_helpers
from my_helpers import train
from my_helpers.data_helpers import fill_missing_hyper_params, fill_missing_context

# %% 0. script setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.getLogger('matplotlib').setLevel(logging.WARNING)  # suppressing DEBUG messages from matplotlib
logging.getLogger("numexpr").setLevel(logging.WARNING)
main_logger = logging.getLogger("main_logger")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
main_logger.addHandler(console_handler)  # main_logger's message will be printed on the console

# for reproducibility
my_helpers.data_helpers.set_torch_seed()

# %% 1. task setup
data_name = configs.data_name
source_tool_list = configs.source_tool_list
target_tool_list = configs.target_tool_list
context = {"source_tool_list": source_tool_list, "target_tool_list": target_tool_list}

loss_func = configs.loss_func  # 👈
exp_name = configs.encoder_exp_name  # 👈
exp_pred_obj = configs.exp_pred_obj  # 👈   # clf predicts new objects only
pipe_settings = {'encoder_exp_name': exp_name, "exp_pred_obj": exp_pred_obj,
                 'clf_exp_name': "default"}

# %% 2. CV setup
# outside test
test_size = 5
num_test_fold = 10  # 👈
prev_test_fold = 0  # 👈 use this to skip used random seed for object sampling

# inside CV
no_overlap_sample = False
plot_learning = True  # 👈
viz_decision_boundary = False  # 👈
pipe_settings.update({
    'retrain_encoder': True, 'retrain_clf': True,
    'plot_learning': plot_learning, "viz_decision_boundary": viz_decision_boundary})

alpha_list = [0.5, 1]  # parameter for triplet Loss: TL margin
temp_list = [0.1, 0.5]  # parameter for sincere Loss: sincere_temp
lr_en_list = [0.001, 0.01]  # learning rate for encoder
encoder_output_dim_list = [128, 32]
grid = {"alpha_list": alpha_list if loss_func == "TL" else [None],  # only search for TL
        "temp_list": temp_list if loss_func == "sincere" else [None],  # only search for sincere
        "lr_en_list": lr_en_list, "encoder_output_dim_list": encoder_output_dim_list}  # 👈add params to tune
total_hyper_params = 1
for v in grid.values():
    total_hyper_params *= len(v)

# %% 3. tracking
top_folder_name = f"./test_result/source_{source_tool_list[0]}/target_{target_tool_list[0]}/{loss_func}/{exp_pred_obj}"
if not os.path.exists(top_folder_name + "/logs"):
    os.makedirs(top_folder_name + "/logs")
print(f"everything will be saved to :{top_folder_name}")
all_obj_list = configs.SORTED_OBJ_LIST
logging.basicConfig(level=logging.DEBUG, filename=top_folder_name + f"/logs/cv_{exp_name}.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')

test_result_dict = {f"{loss_func}": {}}
exp_file_path = os.path.join(top_folder_name, f"test_result_{exp_name}.json")

if os.path.exists(exp_file_path):
    response = input(f"{exp_file_path} already exists, do you want to proceed to overwrite it? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        print("Proceeding with the operation.")
    else:
        print("Operation aborted.")
        sys.exit()
exp_dict = {exp_name: {}}

# %% 4. start CV
main_logger.debug(f"========================= New Run =========================")  # new log starts here
main_logger.info(f"context: {context}")
main_logger.info(f"pipe_settings: {pipe_settings}")
main_logger.info(f"search grid for {loss_func} loss for {total_hyper_params} hyperparameter combos: {grid}")

start_time = time.time()
all_test_acc = []
all_test_obj = []
for i in range(num_test_fold):
    test_fold_name = f"fold{i}"
    random.seed(configs.rand_seed + i + prev_test_fold)
    test_obj_list = random.sample(all_obj_list, test_size)
    all_test_obj.append(test_obj_list)
    test_result_dict[f"{loss_func}"].update({test_fold_name: {"test_obj_list": test_obj_list}})
    train_val_obj_list = [item for item in all_obj_list if item not in test_obj_list]
    main_logger.info(f"test fold {i + 1}/num_test_fold, test_obj_list: {test_obj_list}")

    cv_result = {
        "TL_margin": configs.TL_margin,
        "lr_encoder": configs.lr_encoder,
        "sincere_temp": configs.sincere_temp,
        "encoder_output_dim": configs.encoder_output_dim
    }

    # %% 4.2 Test
    hyparams = cv_result  # irrelevant keys won't be used
    test_accuracy = train.train_fixed_param(train_val_obj_list=train_val_obj_list, test_obj_list=test_obj_list,
                                            loss_func=loss_func, data_name=data_name, context=context,
                                            hyparams=hyparams, pipe_settings=pipe_settings, test_name=test_fold_name)
    all_test_acc.append(test_accuracy)
    main_logger.info(f"✅ test fold {i + 1}/{num_test_fold}, test_accuracy: {test_accuracy * 100:.1f}%")
    test_result_dict[f"{loss_func}"][test_fold_name].update({"test_accuracy": test_accuracy})

main_logger.info(f"avg test accuracies: {(sum(all_test_acc)/len(all_test_acc)*100 ):.2f}%")
main_logger.info(f"test accuracies: {all_test_acc}")
main_logger.info(f"test objects: {all_test_obj}")
main_logger.info(f"✅✅✅ total time used for {num_test_fold} test folds "
                 f"* {total_hyper_params} hyperparams combinations: "
                 f"{round((time.time() - start_time) // 60)} min {(time.time() - start_time) % 60:.1f} sec.")