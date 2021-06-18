from typing import Optional, Dict

import torch
from torchmetrics import Metric


class AccuracyMRR(Metric):
    def __init__(self, top_k: int, ignore_index: int = -100, shift: bool = False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct_acc_1", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("correct_acc_k", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("correct_mrr_k", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        self._top_k = top_k
        self._ignore_index = ignore_index
        self._shift = shift

    def update(
        self, scores: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Calculates accuracy@1, accuracy@top_k and MRR@top_k given scores (softmax or logits) and ground truth labels.

        :param scores: logits or softmax scores. Must have > 1 dimensions
        :param labels: ground truth labels. Must have one dimension more than scores
        :param mask: mask whether token should be ignored when computing metrics

        :return: Dict with:
            acc@1: sum of hits for accuracy@1
            acc@k: sum of hits for accuracy@k
            MRR@k: sum of MRR for all examples
            total: number of examples
        """
        assert scores.ndimension() == labels.ndimension() + 1
        assert scores.size()[:-1] == labels.size()
        assert scores.size()[-1] >= self._top_k

        if self._shift:
            scores = scores[..., :-1, :]
            labels = labels[..., 1:]

        if mask is None:
            mask = labels != self._ignore_index
        else:
            mask = torch.logical_and(mask, labels != self._ignore_index)
        labels = labels[mask]
        scores = scores[mask]

        # Top predictions
        _, top_k_predictions = torch.topk(scores, self._top_k)
        true_pos = top_k_predictions == labels.unsqueeze(-1).expand_as(top_k_predictions)

        # true_pos shape: (N, top_k)
        acc_top_1_sum = true_pos[:, :1].sum()
        acc_top_k_sum = true_pos.sum()
        mrr_top_k_sum = (
            true_pos / torch.arange(1, true_pos.size(-1) + 1, dtype=torch.float32, device=true_pos.device)
        ).sum()

        total_size = labels.numel()

        self.correct_acc_1 += acc_top_1_sum
        self.correct_acc_k += acc_top_k_sum
        self.correct_mrr_k += mrr_top_k_sum
        self.total += total_size

        return {
            "acc@1": acc_top_1_sum,
            f"acc@{self._top_k}": acc_top_k_sum,
            f"MRR@{self._top_k}": mrr_top_k_sum,
        }

    def compute(self):
        return {
            "acc@1": self.correct_acc_1 / self.total,
            f"acc@{self._top_k}": self.correct_acc_k / self.total,
            f"MRR@{self._top_k}": self.correct_mrr_k / self.total,
        }
