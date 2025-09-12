import logging
import os

import configs
import my_helpers.data_helpers
from my_helpers.pipeline import run_pipeline

# %%  0. setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.getLogger('matplotlib').setLevel(logging.WARNING)  # suppressing DEBUG messages from matplotlib
logging.getLogger("numexpr").setLevel(logging.WARNING)
main_logger = logging.getLogger("main_logger")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
main_logger.addHandler(console_handler)  # main_logger's message will be printed on the console

fig_file_path = './figs'
if not os.path.exists(fig_file_path):
    os.makedirs(fig_file_path)

model_file_path = './saved_model'
if not os.path.exists(model_file_path):
    os.makedirs(model_file_path + "/encoder")
    os.makedirs(model_file_path + "/classifier")

log_file_path = './logs'
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
logging.basicConfig(level=logging.DEBUG, filename=log_file_path + "/log_file_main.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')

main_logger.info(f"input data name: {configs.data_name}")
main_logger.info(f"loss_func: {configs.loss_func}")

# for reproducibility
my_helpers.data_helpers.set_torch_seed()

# %% 1. task setup
main_logger.debug(f"========================= New Run =========================")  # new log starts here

hyparams = {}
pipe_settings = {}
orig_context = {}
run_pipeline(loss_func=configs.loss_func, data_name=configs.data_name, multi_class=configs.multi_class,
             curr_context=orig_context, pipe_settings=pipe_settings, hyparams=hyparams)

