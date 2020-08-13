import torch.nn as nn
import torchvision.models as models


def get_resnet(feature_extractor_name):
    r"""Returns feature extractor with ResNet backbone."""
    # Set baseline model
    if feature_extractor_name == "resnet18":
        resnet = models.resnet18(pretrained=False)
    elif feature_extractor_name == "resnet50":
        resnet = models.resnet50(pretrained=False)
    else:
        raise RuntimeError(f"Base model {feature_extractor_name} not defined.")
    
    # Feature extractor (encoder) module
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    
    return feature_extractor


def get_feature_extractor(feature_extractor_params):
    r"""Returns feature extractor by name from a list of common feature extactors."""
    feature_extractor_name = feature_extractor_params["feature_extractor_name"]
    if feature_extractor_name.startswith("resnet"):
        feature_extractor = get_resnet(feature_extractor_name)
    else:
        raise RuntimeError(f"Model name {feature_extractor_name} not defined.")
    
    return feature_extractor