import difflib

from omegaconf import DictConfig
from tqdm import tqdm

from src.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from src.utils import run_with_config


def train(config: DictConfig) -> None:
    tree_loader = PSIDatapointFacade(config)
    tree_loader.train()

    test_trees(config.source_data.mock_jsonl, tree_loader, verbose=True)
    test_trees(config.source_data.val_jsonl, tree_loader, verbose=False)
    test_trees(config.source_data.test_jsonl, tree_loader, verbose=False)


def test_trees(jsonl_path: str, tree_facade: PSIDatapointFacade, verbose: bool) -> None:
    skipped = 0
    total = 0
    errors = 0
    with open(jsonl_path) as f:
        for i, json_string in tqdm(enumerate(f), desc=f"Testing {jsonl_path}..."):
            total += 1
            res = tree_facade.transform(json_string, to_filter=True)
            if res is None:
                skipped += 1
                continue
            tree, ids = res
            tree_builder = tree_facade.get_tree_builder()
            for id_ in ids:
                next_ids = tree_builder.get_next_possible_ids()
                if id_ not in next_ids:
                    name = tree_facade.tokenizer.decode(id_)
                    base_name = name.split("|")[0]
                    if any(tree_facade.tokenizer.decode(id_).startswith(base_name) for id_ in next_ids):
                        print(f"Not expected token {name}, but {base_name} is expected")
                    else:
                        print(f"Not expected token {name} with id {id_} and base name isn't expected!!!")
                    errors += 1
                try:
                    tree_builder.add_id(id_)
                except KeyError:
                    print(f"Could not add token {tree_facade.tokenizer.decode(id_)} with id {id_} to the tree")
                    break

            if verbose:
                print(tree.tree_representation)
                orig_nodes = tree_facade.inverse_transform(tree)
                built_nodes = tree_facade.inverse_transform(tree_builder.tree)

                if built_nodes[0].program != orig_nodes[0].program:
                    print(f"Diff between mock (after transformations) and (after tokenization and detokenization)")
                    for text in difflib.unified_diff(
                        built_nodes[0].program.split("\n"), orig_nodes[0].program.split("\n")
                    ):
                        if text[:3] not in ("+++", "---", "@@ "):
                            print(text)

                print(f"Tree size: {tree.size}, compressed_size: {tree.compressed_size}, ids amount: {len(ids)}")

    print(f"Tested {jsonl_path}: total {total}; skipped {skipped}, errors {errors}")


if __name__ == "__main__":
    run_with_config(train)
