
import torch
import numpy as np

def mixup_data(x, y, alpha=0.4):
    """
    Returns augmented image lam*x1 + (1-lam)*x2 by mixing x1 and x2
    with coefficient lam simulated from beta(alpha, alpha)
    :param x: (torch.tensor) images
    :param y: (torch.tensor) labels
    :param alpha: (float) parameter of the beta distribution
    :return: (augmented image, label1:label of x1, label2:label of x2, lambda:simulated lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam, criterion=torch.nn.CrossEntropyLoss()):
    """
    loss function for augmented image x = lam*x_a + (1-lam)*x_b
    :param criterion: (torch.nn.criterion) this works for CrossEntropy-like losses
    :param pred: (torch.tensor) prediction for augmented image x
    :param y_a: (torch.tensor) label of x_a
    :param y_b: (torch.tensor) label of x_b
    :param lam: (float) mixing coefficient of x_a and x_b
    :return: loss:torch.tensor
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


