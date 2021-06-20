import os

from omegaconf import DictConfig
from tqdm import tqdm

from src.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from src.psi_datapoint.tree_structures.LineBreaker import LineBreaker
from src.utils import run_with_config


def generate_full_line_dataset(
    config: DictConfig, example_split_symbol: str = "␢", out_dir: str = "flcc_vanilla_data"
) -> None:
    psi_datapoint = PSIDatapointFacade(config)
    os.makedirs(out_dir, exist_ok=True)

    for in_path in [
        config.source_data.mock_jsonl,
        config.source_data.train_jsonl,
        config.source_data.val_jsonl,
        config.source_data.test_jsonl,
    ]:
        out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(in_path))[0]}.txt")
        with open(in_path, "r") as in_file:
            with open(out_path, "w") as out_file:
                for json_string in tqdm(in_file, desc=f"Converting data from {in_path}"):
                    nodes = psi_datapoint.json_to_tree(json_string, to_filter=True)
                    if nodes is not None:
                        out_file.write(f"{example_split_symbol}\n{LineBreaker.program(nodes)}\n")


if __name__ == "__main__":
    run_with_config(generate_full_line_dataset)
