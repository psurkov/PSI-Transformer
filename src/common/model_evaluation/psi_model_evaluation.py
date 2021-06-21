import torch
from omegaconf import DictConfig

from src.common.model_training.pl_models.psi_gpt2 import PSIGPT2


def evaluate(config: DictConfig, pl_ckpt_path: str, holdout: str):
    if holdout == "mock":
        data_path = config.source_data.mock_jsonl
    elif holdout == "train":
        data_path = config.source_data.train_jsonl
    elif holdout == "val":
        data_path = config.source_data.val_jsonl
    elif holdout == "test":
        data_path = config.source_data.test_jsonl
    else:
        ValueError(f"Unknown holdout {holdout}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = PSIGPT2.load_from_checkpoint(pl_ckpt_path, map_location=device)
    