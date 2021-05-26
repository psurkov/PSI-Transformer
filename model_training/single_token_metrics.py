from typing import Tuple

import torch


def accuracy_mrr(
    scores: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 5,
    ignore_index: int = -100,
    shift: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates accuracy@1, accuracy@top_k and MRR@top_k given scores (softmax or logits) and ground truth labels.

    :param scores: logits or softmax scores. Must have > 1 dimensions
    :param labels: ground truth labels. Must have one dimension more than scores
    :param top_k: parameter of calculated accracy@top_k and MRR@top_k
    :param ignore_index: index that should be ignored when computing metrics. If you don't need it, just pass negative
    :param shift: whether your model works with sequences and inputs == target. It's always true for HuggingFace models

    :return:
        sum of hits for accuracy@1
        sum of hits for accuracy@k
        sum of MRR for all examples
        number of examples
    """
    assert scores.ndimension() == labels.ndimension() + 1
    assert scores.size()[:-1] == labels.size()
    assert scores.size()[-1] >= top_k

    if shift:
        scores = scores[..., :-1, :]
        labels = labels[..., 1:]

    labels = torch.flatten(labels)
    scores = torch.flatten(scores, end_dim=-2)

    ignored_mask = labels != ignore_index
    labels = labels[ignored_mask]
    scores = scores[ignored_mask, :]

    # Top predictions
    _, top_k_predictions = torch.topk(scores, top_k)
    true_pos = top_k_predictions == labels.unsqueeze(-1).expand_as(top_k_predictions)

    # true_pos shape: (N, top_k)

    acc_top_1_sum = true_pos[:, :1].sum()
    acc_top_k_sum = true_pos.sum()
    mrr_top_k_sum = (true_pos / torch.arange(
            1, true_pos.size(-1) + 1, dtype=torch.float32, device=true_pos.device
        )).sum()

    total_size = torch.zeros_like(acc_top_k_sum) + labels.size(0)
    return acc_top_1_sum, acc_top_k_sum, mrr_top_k_sum, total_size
