"""
--test_script "test_run_clap.py"
--data_name "audio_20s_clap_emb_all.npz"
--experiment_name "cross_tool"
--source_type "rest"
--limit_beh "3-stirring-fast"
--limit_tool "metal-whisk"


example run on terminal:
python test_scripts/exp_run.py --data_name "dataset_discretized.bin" --experiment_name "cross_tool" --source_type "rest"

python test_scripts/exp_run.py --data_name "audio_20s_clap_emb_all.npz" --experiment_name "cross_tool" --source_type "rest"
python test_scripts/exp_run.py --data_name "audio_20s_clap_emb_all.npz" --experiment_name "cross_behavior" --source_type "rest"

python test_scripts/exp_run.py --data_name "audio_20s_clap_emb_all.npz" --experiment_name "cross_tool" --source_type "single" --limit_beh "3-stirring-fast"

python test_scripts/exp_run.py --data_name "audio_20s_clap_emb_all.npz" --experiment_name "cross_behavior" --source_type "single" --test_script "test_run_sincere_audio.py" --limit_tool "metal-scissor"

python test_scripts/exp_run.py --data_name "audio_20s_clap_emb_all.npz" --experiment_name "cross_tool" --source_type "single" --test_script "test_run_baseline2.py" --limit_beh "2-stirring-slow"
python test_scripts/exp_run.py --data_name "audio_20s_clap_emb_all.npz" --experiment_name "cross_behavior" --source_type "single" --test_script "test_run_baseline2.py" --limit_tool "metal-whisk"

python test_scripts/exp_run.py --data_name "audio_20s_clap_emb_all.npz" --experiment_name "cross_tool" --source_type "single" --test_script "test_run_clap_novelObj.py" --limit_beh "2-stirring-slow"
python test_scripts/exp_run.py --data_name "audio_20s_clap_emb_all.npz" --experiment_name "cross_behavior" --source_type "single" --test_script "test_run_clap_novelObj.py" --limit_tool "metal-whisk"

"""
import argparse
import os
import subprocess
import sys

from my_helpers.exp_helpers import get_exp_config_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with config file")

    parser.add_argument(
        "--test_script",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="cross_tool",
    )
    parser.add_argument(
        "--source_type",
        type=str,
        default="rest",
    )
    parser.add_argument(
        "--limit_beh",
        type=str,
        default=""
    )
    parser.add_argument(
        "--limit_tool",
        type=str,
        default=""
    )

    args = parser.parse_args()
    if args.limit_tool == "":
        args.limit_tool = None
    else:
        args.limit_tool = [args.limit_tool]
    if args.limit_beh == "":
        args.limit_beh = None
    else:
        args.limit_beh = [args.limit_beh]
    config_dir = get_exp_config_dir(config_dir=f"exp_configs", data_name=args.data_name,
                                    experiment_name=args.experiment_name, source_type=args.source_type,
                                    limit_tool=args.limit_tool, limit_beh=args.limit_beh)
    print(f"config_dir: {config_dir}")
    # List all json configs
    config_files = sorted([f for f in os.listdir(config_dir) if f.endswith(".json")])
    print(config_files)
    # Iterate through them
    for config_id, fname in enumerate(config_files, start=1):
        config_path = os.path.join(config_dir, fname)
        # print(f"Running test_scripts/test_run_sincere_audio.py with {config_path} ...")

        subprocess.run(
            ["python", f"test_scripts/{args.test_script}", "--config", config_path],
            check=True
        )
