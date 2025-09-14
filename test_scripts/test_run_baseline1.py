import json
import os.path
import time

import configs
import my_helpers.data_helpers
from my_helpers.epx_specific.baselines import Baseline1Audio
from my_helpers.exp_helpers import make_init_result, make_result_dir, load_exp_config_and_update, \
    set_default_context_hyparams
from my_helpers.general_helpers import save_new_json
from test_obj_samples import objects_5sample_sets

modality = "audio"
transfer_type = "baseline1"
audio_data_name = "dataset_discretized.bin"  # or "audio_20s_clap_emb_all.npz"
load_arg_config = False  # 👈

test_size = 5
cv_fold = 5  # 👈
num_test_fold = 10  # 👈 default 10
prev_test_fold = 0  # 👈 use this to skip used random seed for object sampling
epoch_encoder = 300  # 👈 control running time. default 300
epoch_classifier = 300  # 👈 control running time. default 300
one_batch = True  # if true, no minibatch for training, default True
shuffle = False  # shuffle training data, default false

# ========= default context and params ========================
context_dict, hyparams = set_default_context_hyparams(
    one_batch=one_batch, shuffle=shuffle,
    epoch_encoder=epoch_encoder, epoch_classifier=epoch_classifier
)

# ========= update context based on loaded config ========================
if load_arg_config:
    exp_config = load_exp_config_and_update(context_dict=context_dict, hyparams=hyparams)
    audio_data_name = exp_config['data_name']

assert set(context_dict['shared_object_list']).isdisjoint(context_dict['test_object_list'])
full_obj_list = sorted(context_dict['shared_object_list'] + context_dict['test_object_list'])
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
# test_obj_lists = objects_5sample_sets
test_obj_lists, _ = my_helpers.data_helpers.generate_bibd(
    all_objs=full_obj_list, k=test_size,
    n_blocks=num_test_fold, seed=configs.rand_seed + prev_test_fold)

# ========= start fold training ========================
start_time = time.time()
for i in range(num_test_fold):
    context_dict['test_object_list'] = test_obj_lists[i]
    context_dict['shared_object_list'] = [o for o in full_obj_list if o not in context_dict['test_object_list']]
    results['test_results'][f'set{i + 1}']['test_obj_list'] = context_dict['test_object_list']

    pipeline = Baseline1Audio(
        context_dict=context_dict,
        audio_data_name=audio_data_name
    )

    my_helpers.data_helpers.set_torch_seed()

    print(f"test fold {i+1}/{num_test_fold}: {cv_fold} fold cross validate...")
    cv_result = pipeline.cross_validate(k=cv_fold, hyparams=hyparams, one_batch=hyparams['one_batch'])
    accuracies = [cv_result[f'fold{fold + 1}']['accuracy'] for fold in range(cv_fold)]
    print(f"accuracy: {sum(accuracies)/len(accuracies)*100:.2f}%. test objects :{context_dict['test_object_list']}")

    results['test_results']['accuracies'].append(sum(accuracies)/len(accuracies))
    results['test_results'][f"set{i + 1}"]['cv_result'] = cv_result

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
