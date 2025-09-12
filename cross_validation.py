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

parser = argparse.ArgumentParser(description="Run experiment with given parameters.")
parser.add_argument("--exp_name",  type=str,  default="default",
                    help="Name of the experiment (default: 'default')")
parser.add_argument("--exp_pred_obj",  type=str,  default="new",
                    help="fot the classifier, 'new' means train and predict mew object, 'all' for all objects.")
parser.add_argument("--loss_func",  type=str,  default="sincere",
                    help="Name of the loss function (default: 'sincere')")
parser.add_argument("--source_tool",  type=str,  default=configs.source_tool_list[0],
                    help="Name of the source tool (default: configs.source_tool_list[0])")
parser.add_argument("--target_tool",  type=str,  default=configs.target_tool_list[0],
                    help="Name of the source tool (default: configs.target_tool_list[0])")
args = parser.parse_args()

# for reproducibility
my_helpers.data_helpers.set_torch_seed()

# %% 1. task setup
data_name = configs.data_name  # 'audio_16kHz_token_down32_beh3.bin'
source_tool_list = [args.source_tool]
target_tool_list = [args.target_tool]
context = {"source_tool_list": source_tool_list, "target_tool_list": target_tool_list}

loss_func = args.loss_func  # 👈
exp_name = args.exp_name  # 👈
exp_pred_obj = args.exp_pred_obj  # 👈   # clf predicts new objects only
pipe_settings = {'encoder_exp_name': exp_name, "exp_pred_obj": exp_pred_obj,
                 'clf_exp_name': "default"}

# %% 2. CV setup
# outside test
test_size = 5
num_test_fold = 10  # 👈

# inside CV
cross_validate = True
number_of_cv_folds = 5  # 👈 for hyper-param tuning
no_overlap_sample = False
plot_learning = False  # 👈
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
for i in range(num_test_fold):
    test_fold_name = f"fold{i}"
    random.seed(configs.rand_seed + i)
    test_obj_list = random.sample(all_obj_list, test_size)
    test_result_dict[f"{loss_func}"].update({test_fold_name: {"test_obj_list": test_obj_list}})
    train_val_obj_list = [item for item in all_obj_list if item not in test_obj_list]
    main_logger.info(f"test fold {i + 1}/num_test_fold, test_obj_list: {test_obj_list}")

    # %% 4.1 train
    cv_result = train.train_k_fold(train_val_obj_list=train_val_obj_list, number_of_folds=number_of_cv_folds,
                                   loss_func=loss_func, data_name=data_name, grid=grid, context=context,
                                   no_overlap_sample=no_overlap_sample, pipe_settings=pipe_settings)

    test_result_dict[f"{loss_func}"][test_fold_name].update(cv_result)
    # Dump the dictionary as a JSON file
    with open(f"{top_folder_name}/test_result_{exp_name}_temp.json", "w") as json_file:
        json.dump(test_result_dict, json_file, indent=2)

    # %% 4.2 Test
    main_logger.info(f"search for current test set is Done. start testing...")
    hyparams = cv_result  # irrelevant keys won't be used
    test_accuracy = train.train_fixed_param(train_val_obj_list=train_val_obj_list, test_obj_list=test_obj_list,
                                            loss_func=loss_func, data_name=data_name, context=context,
                                            hyparams=hyparams, pipe_settings=pipe_settings, test_name=test_fold_name)

    main_logger.info(f"✅ test fold {i + 1}/{num_test_fold}, test_accuracy: {test_accuracy * 100:.1f}%")
    test_result_dict[f"{loss_func}"][test_fold_name].update({"test_accuracy": test_accuracy})

main_logger.info(f"✅✅✅ total time used for {num_test_fold} test * {number_of_cv_folds} val folds "
                 f"* {total_hyper_params} hyperparams combinations: "
                 f"{round((time.time() - start_time) // 60)} min {(time.time() - start_time) % 60:.1f} sec.")

# %% 5. summarize all test result
accuracy_list = [fold_result['test_accuracy'] for fold_result in test_result_dict[f"{loss_func}"].values()]
hyparams = fill_missing_hyper_params({}, param_model="both")
context = fill_missing_context({})
exp_dict[exp_name] = {
    loss_func: {
        "result": {
            "avg_accuracy": np.mean(accuracy_list),
            "std_accuracy": np.std(accuracy_list),
            "accuracy_list": accuracy_list,
            "source_tool": source_tool_list,
            "target_tool": target_tool_list,
            "experiment":
                {"rand_seed": configs.rand_seed,
                 "num_test_folds": num_test_fold,
                 "num_cv_folds_in_test": number_of_cv_folds,
                 "encoder_exp_name": exp_name,
                 "exp_pred_obj": exp_pred_obj
                 },
            "test_result_dict": test_result_dict,
        },
        "other_info": {
            "other_params": hyparams,
            "data_name": data_name,
            "original_context": context,
        }

    }
}

try:
    with open(exp_file_path, "w") as json_file:
        json.dump(exp_dict, json_file, indent=2)
    os.remove(f"{top_folder_name}/test_result_{exp_name}_temp.json")
finally:
    main_logger.info(f"Done.")


