import argparse
import dataclasses
import itertools
import json
import os
from typing import List

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from flccpsisrc.common.model_evaluation.beam_search.psi_gpt_wrapper import PSIGPT2Wrapper
from flccpsisrc.common.model_evaluation.beam_search.sequence_generator import SequenceGenerator
from flccpsisrc.common.model_evaluation.lines_extractor import extract_lines
from flccpsisrc.common.model_training.pl_models.psi_gpt2 import PSIGPT2
from flccpsisrc.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from flccpsisrc.psi.psi_datapoint.tree_structures.line_breaker import LineBreaker


@dataclasses.dataclass
class TextHypothesis:
    prediction: str
    length: int
    score: float
    is_terminated: bool


@dataclasses.dataclass
class Prediction:
    context: str
    target: str
    hypotheses: List[TextHypothesis]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Prediction":
        hypotheses = [
            TextHypothesis(pred["prediction"], pred["length"], pred["score"], pred["is_terminated"])
            for pred in d["hypotheses"]
        ]
        return Prediction(d["context"], d["target"], hypotheses)


def predict(
    config: DictConfig,
    *,
    pl_ckpt_path: str,
    holdout: str,
    num_examples: int,
    num_iterations: int,
    beam_size: int,
    seed: int,
):
    facade = PSIDatapointFacade(config, diff_warning=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PSIGPT2.load_from_checkpoint(
        pl_ckpt_path, map_location=device, config=config, actual_vocab_size=facade.tokenizer.vocab_size
    ).model
    model_wrapper = PSIGPT2Wrapper(config, model)
    sequence_generator = SequenceGenerator(model_wrapper, num_iterations, beam_size)

    out_dir = os.path.join(config.save_path, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    for prompt_part in (0.1, 0.25, 0.5):
        out_file = os.path.join(out_dir, f"predictions_{holdout}_prompt{prompt_part}.jsonl")
        with open(out_file, "w") as out:
            for line_example in tqdm(
                extract_lines(config, holdout, prompt_part, num_examples, seed),
                desc=f"Predicting {holdout} with prompt {prompt_part}",
            ):
                terminated_hyps, current_hyps = sequence_generator.search_sequence(
                    num_iterations=num_iterations, tree_builder=line_example.tree_builder
                )

                predictions = []
                for hyp in itertools.chain(terminated_hyps, current_hyps):
                    whole_program = LineBreaker.program(hyp.tree_builder.tree.nodes, indent="")
                    context, pred = (
                        whole_program[: len(line_example.context_str)],
                        whole_program[len(line_example.context_str) :],
                    )
                    predictions.append(TextHypothesis(pred, len(hyp.ids), hyp.score, hyp.is_terminated))
                eval_res = Prediction(context, line_example.target_str, predictions)
                out.write(f"{json.dumps(dataclasses.asdict(eval_res))}\n")


if __name__ == "__main__":

    def main():
        args = argparse.ArgumentParser()
        args.add_argument("-c", "--config", type=str, default="flccpsisrc/common/configs/config_psi.yaml")
        args.add_argument("-m", "--pl_ckpt", type=str, required=True)
        args.add_argument("-d", "--holdout", type=str, required=True)
        args.add_argument("-n", "--num_examples", type=int, required=True)
        args.add_argument("-i", "--num_iters", type=int, default=20)
        args.add_argument("-b", "--beam_size", type=int, default=6)
        args = args.parse_args()

        config = OmegaConf.load(args.config)
        predict(
            config,
            pl_ckpt_path=args.pl_ckpt,
            holdout=args.holdout,
            num_examples=args.num_examples,
            num_iterations=args.num_iters,
            beam_size=args.beam_size,
            seed=42,
        )

    main()
