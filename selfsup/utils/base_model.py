import torch.nn as nn


class BaseModel(nn.Module):
    r"""Base model class with a feature extractor module and a projection head.
    
    Arguments:
        feature_extractor (module): feature extractor sub-network
        proj_head (module): projection head sub-network 
    """
    def __init__(self, feature_extractor, proj_head):
        super(BaseModel, self).__init__()

        # Set feature extractor and projection head modules
        self.feature_extractor = feature_extractor
        self.proj_head = proj_head

    def forward(self):
        pass
