import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class CTCLoss(nn.Module):
    def __init__(self, blank: int = 0, padding_idx: int = 0) -> None:
        super().__init__()
        self.blank = blank
        self.padding_idx = padding_idx

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.permute(0, 2, 1)
        preds = preds.log_softmax(-1)
        batch, seq_len, classes = preds.shape
        preds = rearrange(preds, "n t c -> t n c")

        pred_lengths = torch.full(size=(batch,), fill_value=seq_len, dtype=torch.long)

        targets = torch.where(targets == self.padding_idx, self.blank, targets)
        #target_lengths = torch.count_nonzero(targets, axis=1)
        target_lengths = torch.sum(targets != self.blank, dim=-1)

        return F.ctc_loss(preds, targets, pred_lengths, target_lengths, blank=self.blank, zero_infinity=True)