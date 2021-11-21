import glob
import logging
import os
from typing import Tuple

import torch
from omegaconf import OmegaConf

from flccpsi.connectors.connector_base import Connector
from flccpsi.connectors.settings import GenerationSettings
from src.common.model_evaluation.beam_search.psi_gpt_wrapper import PSIGPT2Wrapper
from src.common.model_evaluation.beam_search.sequence_generator import SequenceGenerator
from src.common.model_training.pl_models.psi_gpt2 import PSIGPT2
from src.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from src.psi.psi_datapoint.tree_structures.line_breaker import LineBreaker

logger = logging.getLogger(__name__)


class PSIConnector(Connector):
    def __init__(self, path: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path, pl_path = PSIConnector._get_paths(path)

        config = OmegaConf.load(config_path)
        config.save_path = path
        config.inference_mode = True

        self._facade = PSIDatapointFacade(config, diff_warning=False)
        model = PSIGPT2.load_from_checkpoint(
            pl_path, map_location=device, config=config, actual_vocab_size=self._facade.tokenizer.vocab_size
        ).model
        self._model_wrapper = PSIGPT2Wrapper(config, model)

    def get_suggestions(self, prime: str, filename: str, language: str, settings: GenerationSettings):
        tree, ids = self._facade.transform(prime)
        print(tree.nodes[0].tree_representation)
        tree_builder = self._facade.get_tree_builder(tree)

        sequence_generator = SequenceGenerator(self._model_wrapper, settings.num_iterations, settings.beam_size)
        terminated_hyps, current_hyps = sequence_generator.search_sequence(
            num_iterations=settings.num_iterations, tree_builder=tree_builder
        )

        all_hyps = terminated_hyps if settings.only_full_lines else terminated_hyps + current_hyps
        selected_hyps = sorted(all_hyps, key=lambda h: h.get_normalized_score(), reverse=True)[:10]
        return [LineBreaker.program(hyp.tree_builder.tree.nodes, indent="") for hyp in selected_hyps]

    def cancel(self):
        torch.cuda.empty_cache()

    @staticmethod
    def _get_paths(path: str) -> Tuple[str, str]:
        assert os.path.exists(path) and not os.path.isfile(path)
        config_path = os.path.join(path, "config.yaml")
        [model_path] = glob.glob(os.path.join(path, "*.ckpt"))

        return config_path, model_path


if __name__ == "__main__":

    def main():
        connector = PSIConnector("models/gpt2-psi-java-med-dummy-888/")
        json_string = """{"label":"","AST":[{"node":"java.FILE","children":[1],"token":"<EMPTY>"},{"node":"PACKAGE_STATEMENT","children":[2],"token":"<EMPTY>"},{"node":"PACKAGE_KEYWORD","token":"package"}]}"""
        suggestions = connector.get_suggestions(
            prime=json_string, filename="", language="", settings=GenerationSettings(num_iterations=30)
        )
        print(
            "\n".join(suggestions)
        )

    main()
