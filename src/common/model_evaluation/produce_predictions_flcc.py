# from flcc.connectors.gpt_connector import GPTConnector
# from flcc.connectors.settings import GenerationSettings
# from flcc.connectors.exceptions import CompletionError
#
#
# import argparse
# import dataclasses
# import json
# import os
#
# from omegaconf import DictConfig, OmegaConf
# from tqdm import tqdm
#
# from src.common.model_evaluation.lines_extractor import extract_lines
# from src.common.model_evaluation.produce_predictions import TextHypothesis, Prediction
#
#
# def predict(
#     config: DictConfig,
#     *,
#     model_path: str,
#     holdout: str,
#     num_examples: int,
#     num_iterations: int,
#     beam_size: int,
#     seed: int,
# ):
#     connector = GPTConnector("gpt2", model_path, os.path.join(model_path, "gitbpe-java-20480.bpe"), language="Java")
#     settings = GenerationSettings(num_iterations, beam_size, min_prefix_dist=0.0)
#
#     out_dir = os.path.join(config.save_path, "evaluation")
#     os.makedirs(out_dir, exist_ok=True)
#
#     for prompt_part in (0.1, 0.25, 0.5):
#         out_file = os.path.join(out_dir, f"predictions_{holdout}_prompt{prompt_part}_flcc.jsonl")
#         with open(out_file, "w") as out:
#             for line_example in tqdm(
#                 extract_lines(config, holdout, prompt_part, num_examples, seed),
#                 desc=f"Predicting {holdout} with prompt {prompt_part}",
#             ):
#                 try:
#                     hyps = connector.get_suggestions(line_example.context_str, "", settings)
#                 except CompletionError as e:
#                     print(e)
#                     hyps = []
#
#                 predictions = []
#                 for pred_text, score in hyps:
#                     predictions.append(TextHypothesis(pred_text, 1, score, False))
#                 eval_res = Prediction(line_example.context_str, line_example.target_str, predictions)
#
#                 out.write(f"{json.dumps(dataclasses.asdict(eval_res))}\n")
#
#
# if __name__ == "__main__":
#
#     def main():
#         args = argparse.ArgumentParser()
#         args.add_argument("-c", "--config", type=str, default="src/common/configs/config_psi.yaml")
#         args.add_argument("-m", "--model_path", type=str, required=True)
#         args.add_argument("-d", "--holdout", type=str, required=True)
#         args.add_argument("-n", "--num_examples", type=int, required=True)
#         args.add_argument("-i", "--num_iters", type=int, default=20)
#         args.add_argument("-b", "--beam_size", type=int, default=6)
#         args = args.parse_args()
#
#         config = OmegaConf.load(args.config)
#         predict(
#             config,
#             model_path=args.model_path,
#             holdout=args.holdout,
#             num_examples=args.num_examples,
#             num_iterations=args.num_iters,
#             beam_size=args.beam_size,
#             seed=42,
#         )
#
#     main()
