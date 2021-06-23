import editdistance
import torch
from omegaconf import DictConfig, OmegaConf

from src.common.model_evaluation.beam_search.psi_gpt_wrapper import PSIGPT2Wrapper
from src.common.model_evaluation.beam_search.sequence_generator import SequenceGenerator
from src.common.model_evaluation.lines_extractor import extract_lines
from src.common.model_training.pl_models.psi_gpt2 import PSIGPT2
from src.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from src.psi.psi_datapoint.tree_structures.line_breaker import LineBreaker


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
    len_norm_base: float,
    len_norm_pow: float,
):
    facade = PSIDatapointFacade(config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = PSIGPT2.load_from_checkpoint(
        pl_ckpt_path, map_location=device, config=config, actual_vocab_size=facade.tokenizer.vocab_size
    ).model
    model_wrapper = PSIGPT2Wrapper(config, model)
    sequence_generator = SequenceGenerator(model_wrapper, num_iterations, beam_size)

    for prompt_part in (0.0, 0.2, 0.5):
        for line_example in extract_lines(config, holdout, prompt_part):
            terminated_hyps, current_hyps = sequence_generator.search_sequence(
                num_iterations=num_iterations, tree_builder=line_example.context_tree_builder
            )
            hyps = sorted(
                current_hyps + terminated_hyps,
                key=lambda x: x.get_normalized_score(len_norm_base, len_norm_pow),
                reverse=True,
            )
            whole_program = LineBreaker.program(hyps[0].tree_builder.tree.nodes, indent="")
            assert whole_program.startswith(line_example.context_str)
            context = whole_program[: len(line_example.context_str)]

            print(f"=========\n{context}\n=========\nActual: {repr(line_example.target_str)}\n")

            for hyp in hyps:
                whole_program = LineBreaker.program(hyp.tree_builder.tree.nodes, indent="")
                assert whole_program.startswith(line_example.context_str)
                pred = whole_program[len(line_example.context_str) :]
                sim = edit_similarity(pred, line_example.target_str)
                print(
                    f"Model pred: {repr(pred)}\n"
                    f"Edit similarity: {sim:.2f}, score: {hyp.get_normalized_score(len_norm_base, len_norm_pow)}"
                )


if __name__ == "__main__":

    def main():
        config = OmegaConf.load("src/common/configs/config_psi.yaml")
        evaluate(
            config,
            pl_ckpt_path="out/epoch=4-step=271529-val_overall_MRR@5=0.000.ckpt",
            holdout="mock",
            num_iterations=30,
            beam_size=6,
            len_norm_base=5.0,
            len_norm_pow=0.7,
        )

    main()
