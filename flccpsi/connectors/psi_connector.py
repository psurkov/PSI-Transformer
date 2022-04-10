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
            (s, h) for s, h in sorted(zip(scores, all_hyps), key=lambda sh: sh[0], reverse=True) if s > 0
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
        connector = PSIConnector("models/new-3000-small-steps/")

        json_string = """{"nodes":[{"nodeTypeId":14,"children":[1,2],"placeholders":[]},{"nodeTypeId":21,"children":[],"placeholders":[]},{"nodeTypeId":717,"children":[3,4,5,6,7,8,17],"placeholders":[]},{"nodeTypeId":1,"children":[],"placeholders":[[45,968,5,1381,6009,96,1040]]},{"nodeTypeId":35,"children":[],"placeholders":[]},{"nodeTypeId":36,"children":[],"placeholders":[]},{"nodeTypeId":37,"children":[],"placeholders":[]},{"nodeTypeId":38,"children":[],"placeholders":[]},{"nodeTypeId":448,"children":[9,10,11],"placeholders":[[301]]},{"nodeTypeId":58,"children":[],"placeholders":[]},{"nodeTypeId":61,"children":[],"placeholders":[]},{"nodeTypeId":69,"children":[12,13,16],"placeholders":[]},{"nodeTypeId":38,"children":[],"placeholders":[]},{"nodeTypeId":361,"children":[14,15],"placeholders":[[1068,1047,13],[301]]},{"nodeTypeId":58,"children":[],"placeholders":[]},{"nodeTypeId":61,"children":[],"placeholders":[]},{"nodeTypeId":42,"children":[],"placeholders":[]},{"nodeTypeId":619,"children":[18,19,20,21],"placeholders":[[9733,5945,2097]]},{"nodeTypeId":58,"children":[],"placeholders":[]},{"nodeTypeId":577,"children":[],"placeholders":[[1055],[45,968,5,1381,6009,96,1047],[1831]]},{"nodeTypeId":61,"children":[],"placeholders":[]},{"nodeTypeId":69,"children":[22,23,39],"placeholders":[]},{"nodeTypeId":38,"children":[],"placeholders":[]},{"nodeTypeId":1014,"children":[24,25,26],"placeholders":[[1831],[148,5945]]},{"nodeTypeId":58,"children":[],"placeholders":[]},{"nodeTypeId":61,"children":[],"placeholders":[]},{"nodeTypeId":80,"children":[27],"placeholders":[]},{"nodeTypeId":69,"children":[28,29,38],"placeholders":[]},{"nodeTypeId":38,"children":[],"placeholders":[]},{"nodeTypeId":569,"children":[30,31,37],"placeholders":[[45,968,5,1381,6009,96,174]]},{"nodeTypeId":58,"children":[],"placeholders":[]},{"nodeTypeId":378,"children":[32,33,36],"placeholders":[[1311]]},{"nodeTypeId":58,"children":[],"placeholders":[]},{"nodeTypeId":329,"children":[34,35],"placeholders":[[1831],[357]]},{"nodeTypeId":106,"children":[],"placeholders":[]},{"nodeTypeId":221,"children":[],"placeholders":[[27,3524,207,272,3190,5046,9352]]},{"nodeTypeId":61,"children":[],"placeholders":[]},{"nodeTypeId":61,"children":[],"placeholders":[]},{"nodeTypeId":42,"children":[],"placeholders":[]},{"nodeTypeId":476,"children":[40,41,42,43],"placeholders":[[1527,7613]]},{"nodeTypeId":58,"children":[],"placeholders":[]},{"nodeTypeId":18,"children":[],"placeholders":[]},{"nodeTypeId":1,"children":[],"placeholders":[[1831]]},{"nodeTypeId":61,"children":[],"placeholders":[]}]}"""

        suggestions = connector.get_suggestions(
            prime=json_string, filename="", language="", settings=GenerationSettings(
                num_iterations=25,
                # rollback_prefix=["if", "(", "done", ".", "com"]
                rollback_prefix=["virtualPrefixTokens", ".", "ad"]
            )
        )
        print("\n".join(f"{repr(s[0])} --- {s[1]}" for s in suggestions))


    main()
