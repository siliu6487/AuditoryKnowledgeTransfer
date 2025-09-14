import torch

ALL_TOOL_LIST = ['metal-scissor', 'metal-whisk', "plastic-knife", 'plastic-spoon', "wooden-chopstick", 'wooden-fork']
ALL_BEH_LIST = ['2-stirring-slow', '3-stirring-fast', '4-stirring-twist', '5-whisk', '6-poke']
SORTED_OBJ_LIST = sorted(['empty', 'water', 'detergent', 'chia-seed', 'cane-sugar', 'salt',
                          'styrofoam-bead', 'split-green-pea', 'wheat', 'chickpea', 'kidney-bean',
                          'wooden-button', 'plastic-bead', 'glass-bead', 'metal-nut-bolt'])
modality_list = ['audio']

# --------------- transfer context ---------------
old_object_list = ['chia-seed', 'empty', 'glass-bead', 'plastic-bead', 'wheat',
                   'salt', 'kidney-bean', 'styrofoam-bead', 'water', 'wooden-button']
new_object_list = ['metal-nut-bolt', 'cane-sugar', 'chickpea', 'detergent', 'split-green-pea']

source_beh_list = ['3-stirring-fast']
target_beh_list = ['3-stirring-fast']

source_tool_list = ['plastic-spoon', 'wooden-fork', 'metal-whisk', "wooden-chopstick", "plastic-knife"]
target_tool_list = ['metal-scissor']

trail_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
enc_trial_list = [0, 1, 2, 3, 4, 5, 6, 7]  # for baseline1, use the remaining trials to test classifier
trial_val_portion = 0
randomize_trials = True

loss_func = "sincere"
data_name = "dataset_discretized.bin"

# --------------- piepline experiments ---------------
multi_class = True
# encoder_exp_name = "baseline1"   # no transfer, train on target tool and test on target tool
# encoder_exp_name = "baseline2"  # no transfer, train on source tool(s) and test on target tool
encoder_exp_name = "default"  # source to target transfer

clf_exp_name = "default"  # use source tool to train clf on new objects.

exp_pred_obj = "new"  # new or all, classifier only predicts new object
# exp_pred_obj = "all"   # classifier predicts all object

# --------------- pipeline options ---------------
use_encoder = True
retrain_encoder = True
retrain_clf = True

# --------------- tuned hyper-param ---------------
cross_validate = False
lr_encoder = 1e-3  # encoder lr
sincere_temp = 0.5  # SINCERE temperature

# --------------- default hyper params ---------------
# total epoch patience for encoder is early_stop_patience_enc * smooth_wind_size epochs
early_stop_patience_enc = 2  # None or int
smooth_wind_size = 10  # check progression in every <smooth_wind_size> epochs
tolerance = 5e-3

early_stop_patience_clf = 20  # None or int
clf_tolerance = 1e-4

# encoder parameters
encoder_hidden_dim = 256
encoder_output_dim = 128
epoch_encoder = 300

# classifier parameters
epoch_classifier = 500  # it's ok to be large with early stopping
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
label_dtype = torch.float32
rand_seed = 43

# --------------- future work ---------------
# tool_emb_name = "tool_beh_mil_emb.npz"
# use_tool_emb = False  # True, false , or "random_tool_trial" concat tool embeddings to input features
