"""
example run: python exp_run.py --experiment_name "cross_tool" --source_type "rest"
"""
import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with config file")

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
        default=None
    )

    args = parser.parse_args()
    config_dir = f"{os.getcwd()}/exp_configs"
    config_dir = os.path.join(config_dir, f"{args.experiment_name}-{args.source_type}_to1")
    if args.limit_beh is not None:
        config_dir += f"_beh{args.limit_beh.split('-')[0]}"

    # List all json configs
    config_files = sorted([f for f in os.listdir(config_dir) if f.endswith(".json")])

    # Iterate through them
    for config_id, fname in enumerate(config_files, start=1):
        config_path = os.path.join(config_dir, fname)
        print(f"Running sampling_test.py with {config_path} ...")

        subprocess.run(
            ["python", "sampling_test.py", "--config", config_path],
            check=True
        )
