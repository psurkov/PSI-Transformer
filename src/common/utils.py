import argparse
import functools
from typing import Callable, Any

from omegaconf import OmegaConf, DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def run_with_config(fn: Callable[[DictConfig], Any], default_config_path: str = "src/common/configs/config_psi.yaml"):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default=default_config_path, type=str, help="Path to YAML config")
    args = arg_parser.parse_args()
    config = OmegaConf.load(args.config)
    assert isinstance(config, DictConfig)

    fn(config)


def _linear_with_warmup(current_step: int, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
):
    return LambdaLR(
        optimizer,
        functools.partial(
            _linear_with_warmup, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        ),
        last_epoch,
    )
