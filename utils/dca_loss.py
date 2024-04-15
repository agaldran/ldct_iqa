# this file was downloaded from https://github.com/GB-TonyLiang/DCA
import torch
import torch.nn.functional as F

def cross_entropy_with_dca_loss(logits, labels, weights=None, alpha=1., beta=5.):
    ce = F.cross_entropy(logits, labels, weight=weights)

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    mean_conf = confidences.float().mean()
    acc = accuracies.float().sum()/len(accuracies)
    dca = torch.abs(mean_conf-acc)
    loss = alpha*ce+beta*dca

    return loss


def cross_entropy_with_rsa_loss(logits, labels, weights=None, alpha=1., beta=1., reduction='mean'):
    # assumes tensors of shape:
    # logits = BS x C
    # target = BS, will be one-hot encoded internally using n_classes = logits.shape[1]
    ce = F.cross_entropy(logits, labels, weight=weights, reduction=reduction)

    num_classes = logits.shape[1]
    labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    probs = F.softmax(logits, dim=1)
    rps_loss = ((torch.cumsum(labels, dim=-1) - torch.cumsum(probs, dim=-1)) ** 2).sum(dim=-1)

    if reduction == 'mean':
        rps_loss = rps_loss.mean()
    return alpha*ce + beta*rps_loss
