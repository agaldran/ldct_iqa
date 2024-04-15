import torch
import torch.nn.functional as F

class RPS(torch.nn.Module):
    """
        alpha_ce*CE + beta_rps * rps
    Args:
        alpha_ce (float, optional): The balancing weight for CE loss. Defaults to 1.
        alpha_rps (float, optional): The balancing weight for RPS loss. Defaults to 1.
    Assumes tensors of shape:
    logits = BS x C
    target = BS, will be one-hot encoded internally using n_classes = logits.shape[1]
    """
    def __init__(self, alpha_ce: float = 1., beta_rps: float = 1., ce_weight: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.beta_rps = beta_rps
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        self.reduction = reduction

    def forward(self, inputs, targets):

        loss_ce = self.cross_entropy(inputs, targets)

        num_classes = inputs.shape[1]
        labels = torch.nn.functional.one_hot(targets.long(), num_classes=num_classes)
        probs = F.softmax(inputs, dim=1)
        rps_loss = ((torch.cumsum(labels, dim=-1) - torch.cumsum(probs, dim=-1)) ** 2).sum(dim=-1)
        if self.reduction == 'mean':
            rps_loss = rps_loss.mean()
        return self.alpha_ce * loss_ce + self.beta_rps * rps_loss
