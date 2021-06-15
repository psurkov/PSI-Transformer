import argparse

from omegaconf import OmegaConf


def run_with_config(fn: callable, default_config_path: str = "config.yaml"):
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=default_config_path, type=str, help="Path to YAML config")
    args = args.parse_args()
    config = OmegaConf.load(args.config)

    fn(config)
