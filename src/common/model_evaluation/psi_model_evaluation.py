import dataclasses
import itertools
import json
import os

import editdistance
import torch
from omegaconf import DictConfig, OmegaConf

from src.common.model_evaluation.beam_search.psi_gpt_wrapper import PSIGPT2Wrapper
from src.common.model_evaluation.beam_search.sequence_generator import SequenceGenerator
from src.common.model_evaluation.lines_extractor import extract_lines
from src.common.model_training.pl_models.psi_gpt2 import PSIGPT2
from src.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from src.psi.psi_datapoint.tree_structures.line_breaker import LineBreaker


@dataclasses.dataclass
class Prediction:
    prediction: str
    length: int
    score: float
    is_terminated: bool


@dataclasses.dataclass
class EvaluationResult:
    context: str
    target: str
    preds: list[Prediction]

    @staticmethod
    def fromdict(d: dict) -> "EvaluationResult":
        preds = [
            Prediction(pred["prediction"], pred["length"], pred["score"], pred["is_terminated"])
            for pred in d["predictions"]
        ]
        return EvaluationResult(d["context"], d["target"], preds)


def edit_similarity(a: str, b: str) -> float:
    dist = editdistance.eval(a, b)
    return (1 - (dist / max(len(a), len(b)))) * 100


def evaluate(
    config: DictConfig,
    *,
    pl_ckpt_path: str,
    holdout: str,
    num_iterations: int,
    beam_size: int,
):
    facade = PSIDatapointFacade(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PSIGPT2.load_from_checkpoint(
        pl_ckpt_path, map_location=device, config=config, actual_vocab_size=facade.tokenizer.vocab_size
    ).model
    model_wrapper = PSIGPT2Wrapper(config, model)
    sequence_generator = SequenceGenerator(model_wrapper, num_iterations, beam_size)

    for prompt_part in (0.1, 0.25, 0.5):
        out_file = os.path.join(config.save_path, f"evaluation_{holdout}_prompt{prompt_part}.jsonl")
        with open(out_file, "w") as out:
            for line_example in extract_lines(config, holdout, prompt_part):
                if not line_example.target_str:
                    continue
                terminated_hyps, current_hyps = sequence_generator.search_sequence(
                    num_iterations=num_iterations, tree_builder=line_example.context_tree_builder
                )

                predictions = []
                for hyp in itertools.chain(terminated_hyps, current_hyps):
                    whole_program = LineBreaker.program(hyp.tree_builder.tree.nodes, indent="")
                    context, pred = (
                        whole_program[: len(line_example.context_str)],
                        whole_program[len(line_example.context_str) :],
                    )
                    predictions.append(Prediction(pred, len(hyp.ids), hyp.score, hyp.is_terminated))
                eval_res = EvaluationResult(context, line_example.target_str, predictions)
                out.write(f"{json.dumps(dataclasses.asdict(eval_res))}\n")


if __name__ == "__main__":

    def main():
        config = OmegaConf.load("src/common/configs/config_psi.yaml")
        evaluate(
            config,
            pl_ckpt_path="out/epoch=4-step=271529-val_overall_MRR@5=0.000.ckpt",
            holdout="mock",
            num_iterations=20,
            beam_size=8,
        )

    main()
