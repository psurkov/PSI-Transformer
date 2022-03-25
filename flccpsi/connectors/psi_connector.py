import glob
import json
import logging
import os
from typing import Tuple

import torch
from omegaconf import OmegaConf

from flccpsi.connectors.connector_base import Connector
from flccpsi.connectors.settings import GenerationSettings
from flccpsisrc.common.model_evaluation.beam_search.psi_gpt_wrapper import PSIGPT2Wrapper
from flccpsisrc.common.model_evaluation.beam_search.sequence_generator import SequenceGenerator
from flccpsisrc.common.model_training.pl_models.psi_gpt2 import PSIGPT2
from flccpsisrc.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade

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
            pl_path, map_location=device, config=config, actual_vocab_size=self._facade.vocab_size
        ).model
        self._model_wrapper = PSIGPT2Wrapper(config, model)

    def get_suggestions(self, prime: str, filename: str, language: str, settings: GenerationSettings):
        split_tree = self._facade.json_dict_to_split_tree(json.loads(prime))
        print(split_tree)

        sequence_generator = SequenceGenerator(self._model_wrapper, settings.num_iterations, settings.beam_size)
        terminated_hyps, current_hyps = sequence_generator.search_sequence(
            num_iterations=settings.num_iterations, split_tree=split_tree, rollback_prefix=settings.rollback_prefix
        )

        all_hyps = terminated_hyps if settings.only_full_lines else terminated_hyps + current_hyps
        scores = [h.get_normalized_score() for h in all_hyps]

        selected_hyps = [
            (s, h) for s, h in sorted(zip(scores, all_hyps), key=lambda sh: sh[0], reverse=True) if s > 1e-2
        ]
        return [
            (h.text, s) for s, h in selected_hyps
        ]

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
        connector = PSIConnector("models/cur/")

        json_string = """{"nodes":[{"nodeTypeId":14,"children":[1,4],"placeholders":[]},{"nodeTypeId":21,"children":[2,3],"placeholders":[]},{"nodeTypeId":314,"children":[],"placeholders":[[7982],[8067],[8604]]},{"nodeTypeId":314,"children":[],"placeholders":[[7982],[8067],[7955]]},{"nodeTypeId":603,"children":[5,6,7,8,9,10],"placeholders":[]},{"nodeTypeId":1,"children":[],"placeholders":[[9392]]},{"nodeTypeId":48,"children":[],"placeholders":[]},{"nodeTypeId":49,"children":[],"placeholders":[]},{"nodeTypeId":50,"children":[],"placeholders":[]},{"nodeTypeId":43,"children":[],"placeholders":[]},{"nodeTypeId":1082,"children":[11],"placeholders":[[23,8145,11122]]},{"nodeTypeId":61,"children":[12],"placeholders":[]},{"nodeTypeId":43,"children":[],"placeholders":[]}]}"""
        # [19, 26, 319, 9215, 0, 9300, 0, 9837, 0, 1, 319, 9215, 0, 9300, 0, 9188, 0, 1, 1, 608, 6, 10625, 0, 1, 53, 1, 54, 1, 55, 1, 48, 1, 1087, 1256, 9378, 12355, 0, 66, 48, 1, 1, 1, 1, 1]

        ddddddddddd = """{"nodes":[{"nodeTypeId":14,"children":[1,4],"placeholders":[]},{"nodeTypeId":21,"children":[2,3],"placeholders":[]},{"nodeTypeId":314,"children":[],"placeholders":[[7982],[8067],[8604]]},{"nodeTypeId":314,"children":[],"placeholders":[[7982],[8067],[7955]]},{"nodeTypeId":603,"children":[5,6,7,8,9,10],"placeholders":[]},{"nodeTypeId":1,"children":[],"placeholders":[[9392]]},{"nodeTypeId":48,"children":[],"placeholders":[]},{"nodeTypeId":49,"children":[],"placeholders":[]},{"nodeTypeId":50,"children":[],"placeholders":[]},{"nodeTypeId":43,"children":[],"placeholders":[]},{"nodeTypeId":1082,"children":[11],"placeholders":[[23,8145,11122]]},{"nodeTypeId":61,"children":[12,13],"placeholders":[]},{"nodeTypeId":43,"children":[],"placeholders":[]},{"nodeTypeId":483,"children":[],"placeholders":[[33],[8465]]}]}"""
        # [19, 26, 319, 9215, 0, 9300, 0, 9837, 0, 1, 319, 9215, 0, 9300, 0, 9188, 0, 1, 1, 608, 6, 10625, 0, 1, 53, 1, 54, 1, 55, 1, 48, 1, 1087, 1256, 9378, 12355, 0, 66, 48, 1, 488, 1266, 0, 9698, 0, 1, 1, 1, 1, 1]
        suggestions = connector.get_suggestions(
            prime=json_string, filename="", language="", settings=GenerationSettings(
                num_iterations=10,
                # rollback_prefix=[]
                rollback_prefix=["List", "<", "Integer", ">"]
            )
        )
        print("\n".join(f"{repr(s[0])} --- {s[1]}" for s in suggestions))


    main()
