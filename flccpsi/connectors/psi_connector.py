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
        print("tree=" + split_tree.__str__())
        print("prefix=" + settings.rollback_prefix.__str__())

        sequence_generator = SequenceGenerator(self._model_wrapper, settings.num_iterations, settings.beam_size)
        terminated_hyps, current_hyps = sequence_generator.search_sequence(
            num_iterations=settings.num_iterations, split_tree=split_tree, rollback_prefix=settings.rollback_prefix
        )

        all_hyps = terminated_hyps if settings.only_full_lines else terminated_hyps + current_hyps
        scores = [h.get_normalized_score() for h in all_hyps]

        selected_hyps = [
            (s, h) for s, h in sorted(zip(scores, all_hyps), key=lambda sh: sh[0], reverse=True) if s > 0
        ]
        for s, h in selected_hyps:
            print("Suggestion " + s.__str__() + ":")
            print(h.text)
            print(h.types)
            print()
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
        connector = PSIConnector("models/cur/lll/")

        json_string = """{"nodes":[{"nodeTypeId":16,"children":[1,5,26],"placeholders":[]},{"nodeTypeId":17,"children":[2,3,4],"placeholders":[]},{"nodeTypeId":18,"children":[],"placeholders":[]},{"nodeTypeId":264,"children":[],"placeholders":[[170],[6819],[26,603,5],[4710,68],[2462]]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":23,"children":[6,10,14,18,22],"placeholders":[]},{"nodeTypeId":45,"children":[7,8,9],"placeholders":[]},{"nodeTypeId":46,"children":[],"placeholders":[]},{"nodeTypeId":264,"children":[],"placeholders":[[170],[6819],[26,603,5],[1783],[41,603,5,677]]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":45,"children":[11,12,13],"placeholders":[]},{"nodeTypeId":46,"children":[],"placeholders":[]},{"nodeTypeId":283,"children":[],"placeholders":[[186],[3385],[1503],[699]]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":45,"children":[15,16,17],"placeholders":[]},{"nodeTypeId":46,"children":[],"placeholders":[]},{"nodeTypeId":235,"children":[],"placeholders":[[174],[243],[642]]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":45,"children":[19,20,21],"placeholders":[]},{"nodeTypeId":46,"children":[],"placeholders":[]},{"nodeTypeId":235,"children":[],"placeholders":[[174],[243],[158]]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":45,"children":[23,24,25],"placeholders":[]},{"nodeTypeId":46,"children":[],"placeholders":[]},{"nodeTypeId":235,"children":[],"placeholders":[[174],[243],[1327]]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":771,"children":[27],"placeholders":[[41,603,5,959,9721,725]]},{"nodeTypeId":14,"children":[28,29,42,55,59,63],"placeholders":[]},{"nodeTypeId":40,"children":[],"placeholders":[]},{"nodeTypeId":346,"children":[30,32,36],"placeholders":[[29,3475,1585,7028]]},{"nodeTypeId":364,"children":[31],"placeholders":[]},{"nodeTypeId":503,"children":[],"placeholders":[[699]]},{"nodeTypeId":267,"children":[33],"placeholders":[[158]]},{"nodeTypeId":51,"children":[34,35],"placeholders":[]},{"nodeTypeId":503,"children":[],"placeholders":[[699]]},{"nodeTypeId":217,"children":[],"placeholders":[[41,603,5,959,9721,715]]},{"nodeTypeId":15,"children":[37,41],"placeholders":[]},{"nodeTypeId":413,"children":[38],"placeholders":[[642]]},{"nodeTypeId":42,"children":[39,40],"placeholders":[]},{"nodeTypeId":60,"children":[],"placeholders":[]},{"nodeTypeId":63,"children":[],"placeholders":[]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":346,"children":[43,45,49],"placeholders":[[1282,7028]]},{"nodeTypeId":364,"children":[44],"placeholders":[]},{"nodeTypeId":503,"children":[],"placeholders":[[699]]},{"nodeTypeId":267,"children":[46],"placeholders":[[158]]},{"nodeTypeId":51,"children":[47,48],"placeholders":[]},{"nodeTypeId":503,"children":[],"placeholders":[[699]]},{"nodeTypeId":217,"children":[],"placeholders":[[41,603,5,959,9721,715]]},{"nodeTypeId":15,"children":[50,54],"placeholders":[]},{"nodeTypeId":413,"children":[51],"placeholders":[[642]]},{"nodeTypeId":42,"children":[52,53],"placeholders":[]},{"nodeTypeId":60,"children":[],"placeholders":[]},{"nodeTypeId":63,"children":[],"placeholders":[]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":927,"children":[56],"placeholders":[[769,428,3715,1208]]},{"nodeTypeId":15,"children":[57,58],"placeholders":[]},{"nodeTypeId":307,"children":[],"placeholders":[]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":927,"children":[60],"placeholders":[[769,35,71]]},{"nodeTypeId":15,"children":[61,62],"placeholders":[]},{"nodeTypeId":307,"children":[],"placeholders":[]},{"nodeTypeId":22,"children":[],"placeholders":[]},{"nodeTypeId":324,"children":[64,68],"placeholders":[[217]]},{"nodeTypeId":69,"children":[65,66,67],"placeholders":[]},{"nodeTypeId":60,"children":[],"placeholders":[]},{"nodeTypeId":450,"children":[],"placeholders":[[699],[41,603,5,959,9721,715],[1456]]},{"nodeTypeId":63,"children":[],"placeholders":[]},{"nodeTypeId":71,"children":[69],"placeholders":[]},{"nodeTypeId":40,"children":[],"placeholders":[]}]}"""

        suggestions = connector.get_suggestions(
            prime=json_string, filename="", language="", settings=GenerationSettings(
                num_iterations=25,
                # rollback_prefix=["UNKNOWN:vi"]
                rollback_prefix=["IDENTIFIER:virtualPrefixTokens", "STRUCTURE:.", "UNKNOWN:ad"]
                # rollback_prefix=["IDENTIFIER:virtualPrefixTokens", "STRUCTURE:.", "IDENTIFIER:add", "STRUCTURE:("]
            )
        )
        print("\n".join(f"{repr(s[0])} --- {s[1]}" for s in suggestions))


    main()
