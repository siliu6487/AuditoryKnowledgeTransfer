import json
import os
import time
from matplotlib import pyplot as plt

import configs
from my_helpers.epx_specific.tune_clap import ClapPiplineAudio
from my_helpers.exp_helpers import set_default_context_hyparams, make_result_dir, make_init_result, \
    load_exp_config_and_update
from my_helpers.data_helpers import generate_bibd, set_torch_seed
from my_helpers.general_helpers import save_new_json

modality = "audio"
transfer_type = "CLAP"
audio_data_name = "audio_20s_clap_emb_all.npz"
text_data_name = "clap_text_emb_all.npz"
linear_probe = True  # 👈 use a linear probe clf instead of text retrival
novel_obj = True  # 👈 true true!! text retrieve on novel objects
true_zero_shot = True  # 👈
fine_tune = True  # 👈 True!
load_arg_config = True
forbidden_test_obj_combo = ("water", "detergent", "empty")  # or None # 👈

if novel_obj:
    transfer_type += "_novel"
    linear_probe = False

if linear_probe:
    transfer_type += "_lp"
if true_zero_shot is False:
    transfer_type += "_not0"
if fine_tune:
    transfer_type += "_tuned"
if forbidden_test_obj_combo is not None:
    transfer_type += "_filter_obj"

test_size = 5
num_test_fold = 10  # 👈 default 10
prev_test_fold = 0  # 👈 default 0 use this to skip used random seed for object sampling
epoch_encoder = 300  # 👈 control running time. default 300
epoch_classifier = 300  # 👈 control running time. default 300
encoder_output_dim = 512  # so we can match it with text
one_batch = True  # if true, no minibatch for training, default False
shuffle = False  # shuffle training data, default True

# region ####################### Auto setup #######################
# ========= default context and params ========================
context_dict, hyparams = set_default_context_hyparams(
    one_batch=one_batch, shuffle=shuffle,
    epoch_encoder=epoch_encoder, epoch_classifier=epoch_classifier,
)
hyparams['encoder_output_dim'] = encoder_output_dim

# ========= update context based on loaded config ========================
try:
    exp_config = load_exp_config_and_update(context_dict=context_dict, hyparams=hyparams)
    audio_data_name = exp_config['data_name']
except SystemExit as e:
    print("⏭️Missing required arguments. Will not update context and parameter values using experiment config file.")

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
test_obj_lists, _ = generate_bibd(
    all_objs=full_obj_list, k=test_size, forbidden_set=forbidden_test_obj_combo,
    n_blocks=num_test_fold, seed=configs.rand_seed + prev_test_fold)

# endregion #######################

start_time = time.time()
# ========= train model, source tool only ========================
set_torch_seed()

pipeline = ClapPiplineAudio(
    context_dict=context_dict,
    audio_data_name=audio_data_name,
    text_data_name=text_data_name,
    true_zero_shot=true_zero_shot,
    novel_obj=novel_obj,
)

if true_zero_shot:  # we can train one model for all test object sets!
    if fine_tune:
        print("train encoder...")
        enc_result = pipeline.learn_encoder(hyparams=hyparams, one_batch=hyparams['one_batch'],
                                            full_obj_list=full_obj_list)

    if linear_probe and not novel_obj:
        clf, _ = pipeline.train_classifier(hyparams=hyparams, one_batch=hyparams['one_batch'])
        pipeline.clf = clf
        test_result = pipeline.test_classifier()
    else:
        test_result = pipeline.test_retrival(test_obj_list=full_obj_list, use_enc=fine_tune)  # 0-shot

    print(f"test obj class retrival ...")
    print(f"Full object list test accuracy: {test_result['accuracy']}")
    print(f"all_truth: {test_result['all_truth']}")
    print(f"all_pred: {test_result['all_pred']}")

    results['test_results']['all_objs'] = {"accuracy": test_result['accuracy']}
    results['test_results']['all_objs']['test_truth0_and_preds1'] = [
            [int(tr), int(pr)] for tr, pr in zip(test_result['all_truth'], test_result['all_pred'])
    ]

# ========= start fold training ========================
print(f"start testing for subsets of objects...")
for i in range(num_test_fold):
    test_objs = test_obj_lists[i]
    context_dict['test_object_list'] = test_objs
    context_dict['shared_object_list'] = [o for o in full_obj_list if o not in context_dict['test_object_list']]
    results['test_results'][f'set{i + 1}']['test_obj_list'] = context_dict['test_object_list']

    if true_zero_shot:   # use the encoder trained above on all source data
        if linear_probe and not novel_obj:  # train a classifier only on test object set
            clf, _ = pipeline.train_classifier(hyparams=hyparams, one_batch=hyparams['one_batch'])
            pipeline.clf = clf
            test_result = pipeline.test_classifier()
        else:
            test_result = pipeline.test_retrival(test_obj_list=test_objs, use_enc=fine_tune)
        print(f"test accuracy: {test_result['accuracy']}")
        print(f"all_truth: {test_result['all_truth']}")
        print(f"all_pred: {test_result['all_pred']}")

    else:  # train a new encoder that add target data with shared object
        print("train encoder...")
        enc_result = pipeline.learn_encoder(hyparams=hyparams, one_batch=hyparams['one_batch'],
                                            full_obj_list=full_obj_list)
        print(f"train and test linear probe classifier ...")
        clf,  _ = pipeline.train_classifier(hyparams=hyparams, one_batch=hyparams['one_batch'])
        pipeline.clf = clf
        test_result = pipeline.test_classifier()
        print(f"test accuracy: {test_result['accuracy']}")
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
    config_path = os.path.join(result_save_dir, f"results{num_results_saved}.json")  # overwrite

with open(config_path, "w") as f:
    json.dump(results, f, indent=2)

print(f'Saved results -> {config_path}')
print(f"✅✅✅ total time used for {num_test_fold} test folds: "
      f"{round((time.time() - start_time) // 60)} min {(time.time() - start_time) % 60:.1f} sec.")
