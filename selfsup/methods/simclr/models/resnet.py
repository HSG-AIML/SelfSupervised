import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet(nn.Module):
    r"""SimCLR Model with ResNet baseline and linear projections on top."""
    def __init__(self, base_model, out_dim):
        super(ResNet, self).__init__()
        # Set baseline model
        if base_model == "resnet18":
            resnet = models.resnet18(pretrained=False)
        elif base_model == "resnet50":
            resnet = models.resnet50(pretrained=False)
        else:
            raise RuntimeError(f"Base model {base_model} not defined.")
        
        # Compute feature vector dim
        num_ftrs = resnet.fc.in_features

        # Feature extractor (encoder) module
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Linear projections
        self.linear1 = nn.Linear(num_ftrs, num_ftrs)
        self.linear2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        # x: B x 3 x H x W

        h = self.features(x) 
        # B x NUM_FEATS x 1 x 1
        h = h.squeeze() 
        # B x NUM_FEATS
        
        x = self.linear1(h)
        x = F.relu(x)
        # B x NUM_FEATS

        x = self.linear2(x)
        # B x OUT_DIM

        return h, x
