import json
import os
import time

import configs
from my_helpers.epx_specific.baselines import Baseline2Audio
from my_helpers.exp_helpers import general_exp_run_setup
from my_helpers.data_helpers import set_torch_seed
from my_helpers.general_helpers import save_new_json
from my_helpers.viz_helpers import plot_datasets

modality = "audio"
transfer_type = "baseline2"
audio_data_name = "audio_20s_clap_emb_all.npz"
plot_dataset = True  # 👈
forbidden_test_obj_combo = ("water", "detergent", "empty")  # or None # 👈
if forbidden_test_obj_combo is not None:
    transfer_type += "_filter_obj"

test_size = 5
num_test_fold = 10  # 👈 default 10
prev_test_fold = 0  # 👈 default 0 use this to skip used random seed for object sampling
epoch_encoder = 300  # 👈 control running time. default 300
epoch_classifier = 300  # 👈 control running time. default 300
encoder_output_dim = 128  # 👈 so we can match it with text
one_batch = True  # if true, no minibatch for training, default False
shuffle = False  # shuffle training data, default false

setup = general_exp_run_setup(
    transfer_type=transfer_type, modality=modality, audio_data_name=audio_data_name,
    one_batch=one_batch, shuffle=shuffle, epoch_encoder=epoch_encoder,
    epoch_classifier=epoch_classifier, encoder_output_dim=encoder_output_dim,
    num_test_fold=num_test_fold, test_size=test_size, prev_test_fold=prev_test_fold,
    forbidden_test_obj_combo=forbidden_test_obj_combo
)
context_dict = setup['context_dict']
hyparams = setup['hyparams']
test_obj_lists = setup['test_obj_lists']
full_obj_list = setup['full_obj_list']
results = setup['results']
result_save_dir = setup['result_save_dir']

# if plot_dataset:
#     plot_datasets(context_dict=context_dict, full_obj_list=full_obj_list)

# ========= only need one encoder because source data does not change ========================
pipeline = Baseline2Audio(
        context_dict=context_dict,
        audio_data_name=audio_data_name
    )
set_torch_seed()

print("train encoder...")
enc_result = pipeline.train_encoder(hyparams=hyparams, one_batch=hyparams['one_batch'], full_obj_list=full_obj_list)
# plt.plot(enc_result['all_losses'])
# plt.show()
# print(f"enc_result: {enc_result}")

# ========= start fold training ========================
start_time = time.time()
for i in range(num_test_fold):
    context_dict['test_object_list'] = test_obj_lists[i]
    context_dict['shared_object_list'] = [o for o in full_obj_list if o not in context_dict['test_object_list']]
    results['test_results'][f'set{i + 1}']['test_obj_list'] = context_dict['test_object_list']

    print("train classifier...")
    clf_result = pipeline.train_classifier(hyparams=hyparams, one_batch=hyparams['one_batch'])
    # plt.plot(clf_result['all_losses'])
    # plt.plot(clf_result['all_accuracies'])
    # plt.show()
    # print(f"clf_result: {clf_result}")

    print(f"test classifier...")
    test_result = pipeline.test_classifier()
    print(f"test accuracy: {test_result['accuracy']}")
    print(f"test_obj_list: {test_result['test_obj_list']}")
    print(f"all_truth: {test_result['all_truth']}")
    print(f"all_pred: {test_result['all_pred']}")

    results['test_results']['accuracies'].append(test_result['accuracy'])
    results['test_results'][f"set{i + 1}"]['test_truth0_and_preds1'] = [
        [int(tr), int(pr)] for tr, pr in zip(test_result['all_truth'], test_result['all_pred'])
    ]

a_l = results['test_results']['accuracies']
results['test_results']['avg_accuracy'] = sum(a_l) / len(a_l)

num_results_saved = len(os.listdir(result_save_dir))
if save_new_json(result_save_dir, results):
    config_path = os.path.join(result_save_dir, f"results{num_results_saved + 1}.json")
else:
    config_path = os.path.join(result_save_dir, f"a")  # overwrite

with open(config_path, "w") as f:
    json.dump(results, f, indent=2)

print(f'Saved results -> {config_path}')
print(f"✅✅✅ total time used for {num_test_fold} test folds: "
      f"{round((time.time() - start_time) // 60)} min {(time.time() - start_time) % 60:.1f} sec.")
