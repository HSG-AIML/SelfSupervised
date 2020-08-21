import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from selfsup.utils.base_model import BaseModel
from selfsup.commons.feature_extractors_vision import get_feature_extractor
from selfsup.commons.projection_heads import get_proj_head


class SimCLRModel(BaseModel):
    r"""Impelementation of SimCLR model.
    
    Arguments:
        params: all model params as as dict
    """
    def __init__(self, **params):
        # Set feature extractor and projection head modules
        feature_extractor = get_feature_extractor(params["feature_extractor"])
        proj_head = get_proj_head(params["proj_head"])

        # Set super class modules
        super(SimCLRModel, self).__init__(feature_extractor, proj_head)
    
    def forward(self, x):
        r"""Implements forward pass of the model."""
        # Extract features
        h = self.feature_extractor(x)
        h = h.squeeze()
        
        # Feed features to the projection head
        x = self.proj_head(h)
        
        return h, x
