import torch
import torch.nn.functional as F


class RegressionLoss(torch.nn.Module):

    # define class constructor
    def __init__(self):

        # call super class constructor
        super(RegressionLoss, self).__init__()

    # define loss forward pass
    def forward(self, x, y):

        # normalize the predictions
        x = F.normalize(x, dim=1, p=2)
        y = F.normalize(y, dim=1, p=2)

        # compute the regression loss
        loss = 2 - 2 * (x * y).sum(dim=-1)

        return loss