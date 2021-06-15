import argparse
import difflib

from omegaconf import DictConfig, OmegaConf

from psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from utils import run_with_config


def train(config: DictConfig) -> None:
    tree_loader = PSIDatapointFacade(config)
    tree_loader.train()

    # Testing stuff
    with open(config.source_data.mock_jsonl) as f:
        [json_string] = f.readlines()

    tree, ids = tree_loader.transform(json_string)
    if config.psi_pretraining.show_mock:
        print(tree.tree_representation)
    orig_nodes = tree_loader.inverse_transform(tree)
    if config.psi_pretraining.show_mock:
        print(orig_nodes[0].program)
    print(f"Mock tree size: {tree.size}, compressed_size: {tree.compressed_size}, ids amount: {len(ids)}")

    tree_builder = tree_loader.get_tree_builder()
    for id_ in ids:
        # name = tree_builder._tokenizer.bpe_tokenizer.id_to_token(id_)
        assert id_ in tree_builder.get_next_possible_ids()
        tree_builder.add_id(id_)

    built_nodes = tree_loader.inverse_transform(tree_builder.tree)

    if config.psi_pretraining.show_mock:
        print(f"Diff between mock (after transformations) and (after tokenization and detokenization)")
        if built_nodes[0].program != orig_nodes[0].program:
            for text in difflib.unified_diff(built_nodes[0].program.split("\n"), orig_nodes[0].program.split("\n")):
                if text[:3] not in ("+++", "---", "@@ "):
                    print(text)


if __name__ == "__main__":
    run_with_config(train)
