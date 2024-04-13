import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNLWithLogitsLoss(nn.Module):
    """
    Implements the Multiple Positive Negative Logarithmic Loss (MPNLWithLogitsLoss).

    The loss function is defined as:

        loss = -sum_{i in Prov} log(softmax(z_r)_i)

    where `z_r` represents the logits from the reranker for all passages,
    and `Prov` contains the indices of correct (positive) passages according
    to the ground truth.

    Args:
        logits (torch.Tensor): The logits output by the reranker. Shape: [batch_size, num_passages]
        labels (torch.Tensor): The ground truth labels indicating the correct passages. Shape: [batch_size, num_passages]

    Returns:
        torch.Tensor: The computed loss.
    """

    def __init__(self):
        super(MPNLWithLogitsLoss, self).__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        log_probs = F.log_softmax(logits, dim=1)

        batch_loss = []
        for batch_idx, label in enumerate(labels):
            true_indices = torch.nonzero(label, as_tuple=True)[0]
            true_log_probs = log_probs[batch_idx][true_indices]
            loss = -torch.sum(true_log_probs)
            batch_loss.append(loss)

        loss = torch.mean(torch.stack(batch_loss))
        return loss
