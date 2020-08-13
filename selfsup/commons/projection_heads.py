import torch.nn as nn
import torch.nn.functional as F


class DoubleLinear(nn.Module):
    r"""Implements double-lienar projection head.
    
    Arguments
        feat_dim: dimensionality of the input feature
        out_dim: dimensionality of the output tensor
    """
    def __init__(self, feat_dim, out_dim):
        super(DoubleLinear, self).__init__()
        
        # Set linear transformations
        self.linear1  = nn.Linear(feat_dim, feat_dim)
        self.linear2 = nn.Linear(feat_dim, out_dim)
    
    def forward(self, x):
        r"""Implements the forward function."""
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


def get_proj_head(proj_head_params):
    r"""Return projection head by name."""
    # Case: double linear
    if proj_head_params["proj_head_name"] == "double_linear":
        proj_head = DoubleLinear(proj_head_params["feat_dim"], proj_head_params["out_dim"])
    
    return proj_head