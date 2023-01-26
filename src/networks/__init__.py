import torch.nn

from src.networks.model_types import *
from torchvision import models
import math


def xavier_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            bound = math.sqrt(2) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)


def kaiming_init(model, factor=0.1):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.startswith("layers.0"):  # The first layer does not have ReLU applied on its input
            param.data.normal_(0, factor / math.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, math.sqrt(2) * factor / math.sqrt(param.shape[1]))


def get_network(network_name, network_kwargs, initialization_factor=None):
    if network_name == "four_layer_cnn":
        model = four_layer_cnn(**network_kwargs)
    elif network_name == "two_layer_cnn":
        model = two_layer_cnn(**network_kwargs)
    elif network_name == "five_layer_cnn":
        model = five_layer_cnn(**network_kwargs)
    elif network_name == "vgg16":
        _model = models.vgg16(pretrained=network_kwargs.get('pretrained', False))
        num_features = _model.classifier[6].in_features
        _model.classifier[6] = torch.nn.Linear(num_features, network_kwargs['num_classes'])
        seq_model = torch.nn.Sequential(*(list(_model.features) +
                                          [_model.avgpool, torch.nn.Flatten(1)] +
                                          list(_model.classifier)))
        model = seq_model
    elif network_name == "ffn":
        model = ffn(**network_kwargs)
    else:
        raise ValueError(f"Unknown network: {network_name}")

    if initialization_factor:
        kaiming_init(model, initialization_factor)
    return model
