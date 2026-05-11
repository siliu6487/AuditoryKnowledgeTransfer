import torch
import numpy as np

ALL_TOOL_LIST = ['metal-scissor', 'metal-whisk', "plastic-knife", 'plastic-spoon', "wooden-chopstick", 'wooden-fork']
ALL_BEH_LIST = ['2-stirring-slow', '3-stirring-fast', '4-stirring-twist', '5-whisk', '6-poke']
SORTED_OBJ_LIST = sorted(['empty', 'water', 'detergent', 'chia-seed', 'cane-sugar', 'salt',
                          'styrofoam-bead', 'split-green-pea', 'wheat', 'chickpea', 'kidney-bean',
                          'wooden-button', 'plastic-bead', 'glass-bead', 'metal-nut-bolt'])

modality_list = ['audio']

# --------------- transfer context ---------------
# old_object_list = ['chia-seed', 'empty', 'glass-bead', 'plastic-bead', 'wheat',
#                    'salt', 'kidney-bean', 'styrofoam-bead', 'water',  'wooden-button']
# new_object_list = ['metal-nut-bolt', 'cane-sugar', 'chickpea', 'detergent', 'split-green-pea']
# old_object_list = ['salt', 'chia-seed', 'empty', 'water', 'glass-bead', 'plastic-bead',
#                    'kidney-bean', 'styrofoam-bead']
# new_object_list = ['cane-sugar', 'wheat', 'detergent', 'metal-nut-bolt',  'chickpea', 'wooden-button', 'split-green-pea']
new_object_list = ['cane-sugar', 'chia-seed', 'detergent', 'salt', 'wheat']
old_object_list = list(set(SORTED_OBJ_LIST) - set(new_object_list))

# source_beh_list = ['2-stirring-slow']
behavior_list = ['3-stirring-fast']
source_beh_list = ['3-stirring-fast']
target_beh_list = ['3-stirring-fast']
# source_beh_list = ALL_BEH_LIST
# target_beh_list = ALL_BEH_LIST

source_tool_list = ['plastic-spoon', 'wooden-fork', 'metal-whisk', "wooden-chopstick", "plastic-knife"]
# source_tool_list = ALL_TOOL_LIST
target_tool_list = ['metal-scissor']

# trail_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
trail_list = np.repeat(np.arange(10), repeats=1).tolist()
enc_trial_list = [0, 1, 2, 3, 4, 5, 6, 7]  # for baseline1, use the remaining trials to test classifier
trial_val_portion = 0
randomize_trials = True

loss_func = "sincere"  # "TL" for triplet loss or "sincere", or "mulsupcon"
# data_name = "audio_16kHz_token_down32_stir_20s.bin"
data_name = "audio_20s_clap_emb_all.npz"
# data_name = "dataset_discretized.bin"
# tool_emb_name = "tool_binary_embeddings.bin"
# tool_emb_name = "tool_click_box_emb.npz"
tool_emb_name = "tool_beh_mil_emb1.npz"
# data_name = "audio_16kHz_token_down32_beh3.bin"

enc_pt_folder = './saved_model/encoder/'
encoder_pt_name = f"myencoder_{loss_func}.pt"
clf_pt_folder = './saved_model/classifier/'
clf_pt_name = f"myclassifier_{loss_func}.pt"

# --------------- piepline experiments ---------------
multi_class = True   # False means multi-label classification
use_tool_emb = False  # True, false , or "random_tool_trial" concat tool embeddings to input features
# encoder_exp_name = "baseline1"   # no transfer, train on target tool and test on target tool
# encoder_exp_name = "baseline2"  # no transfer, train on source tool(s) and test on target tool
# encoder_exp_name = "baseline2-all"  # no transfer, train on all other tools and test on target tool
# encoder_exp_name = "all"  # all other tools as source tool
encoder_exp_name = "default"  # source to target transfer

clf_exp_name = "default"  # use source tool to train clf on new objects.

exp_pred_obj = "new"  # new or all, classifier only predicts new object
# exp_pred_obj = "all"   # classifier predicts all object

# --------------- pipeline options ---------------
use_encoder = True
retrain_encoder = True
retrain_clf = True
save_temp_model = True

# viz:
plot_learning = True  # loss plot for learning progression
viz_dataset = True
viz_share_space = True
viz_decision_boundary = True
save_fig = False
viz_l2_norm = False  # viz l2 normed data in 2d space

# --------------- tuned hyper-param ---------------
cross_validate = True
lr_encoder = 1e-4  # encoder lr
TL_margin = 0.5  # TL alpha
sincere_temp = 0.5  # SINCERE temperature
mulsupcon_temp = 0.1

# --------------- default hyper params ---------------
# total epoch patience for encoder is early_stop_patience_enc * smooth_wind_size epochs
early_stop_patience_enc = 2  # None or int
smooth_wind_size = 10  # check progression in every <smooth_wind_size> epochs

early_stop_patience_clf = 100  # None or int
tolerance = 5e-3
clf_tolerance = 1e-4

# encoder parameters
encoder_input_dim = 512
encoder_hidden_dim = 256
encoder_output_dim = 128  # 128. 2 makes it easy to visualize real decision boundary
epoch_encoder = 300

# TL loss parameters
pairs_per_batch_per_object = 300  # smaller value fluctuates the loss

# classifier parameters
epoch_classifier = 499  # it's ok to be large with early stopping
lr_classifier = 1e-2

# --------------- device and randomness ---------------
# device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'  # GPU
elif torch.backends.mps.is_available():
    device = 'mps'  # acceleration tool for apple M chip
else:
    device = 'cpu'

data_dtype = torch.float32
label_dtype = torch.int64 if multi_class else torch.float32
rand_seed = 43
