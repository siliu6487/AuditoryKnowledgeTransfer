from my_helpers.experiment_generator import create_exp_config

ALL_BEHS = ['2-stirring-slow', '3-stirring-fast', '4-stirring-twist', '5-whisk', '6-poke']

# ======== binary audio ========
data_name = "dataset_discretized.bin"
limited_beh = ['3-stirring-fast']

# 1
create_exp_config(experiment_name="cross_tool", source_type="single",
                  data_name=data_name, limit_beh=limited_beh)

# 2
create_exp_config(experiment_name="cross_tool", source_type="rest",
                  data_name=data_name, limit_beh=limited_beh)

# ========= audio (clap embedding) ==========
data_name = "audio_20s_clap_emb_all.npz"
# 3
create_exp_config(experiment_name="cross_tool", source_type="rest",
                  data_name=data_name)

# 4
create_exp_config(experiment_name="cross_behavior", source_type="rest",
                  data_name=data_name)

# only do single transfer on this beh
# 5
limited_beh = ['3-stirring-fast']
create_exp_config(experiment_name="cross_tool", source_type="single",
                  data_name=data_name, limit_beh=limited_beh)




