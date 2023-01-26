import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def four_layer_cnn(in_ch, in_dim, width, linear_size, num_classes=2):
    """
    CNN, relatively large 4-layer
    Parameter:
        in_ch: input image channel, 1 for MNIST and 3 for CIFAR
        in_dim: input dimension, 28 for MNIST and 32 for CIFAR
        width: width multiplier
    """
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4 * width, 4 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4 * width, 8 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8 * width, 8 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_classes)
    )
    return model


def two_layer_cnn(in_ch, in_dim, width, linear_size=128, num_classes=2):
    # model = nn.Sequential(Print())

    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_classes)
    )
    return model


def five_layer_cnn(in_ch, in_dim, linear_size=512, num_classes=2):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(361 * 256, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_classes)
    )
    return model


def ffn(in_ch, in_dim, hidden_dim, hidden_lay, num_cls, act_fn_type='relu'):
    _out = hidden_dim if hidden_lay > 0 else num_cls
    layers = [torch.nn.Flatten(), torch.nn.Linear(in_features=in_dim*in_ch, out_features=_out)]
    if act_fn_type == 'relu':
        act_fn = torch.nn.ReLU
    else:
        act_fn = torch.nn.Sigmoid
    if hidden_lay > 0:
        layers.append(act_fn())
        for hi in range(hidden_lay - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
            _out = hidden_dim
        layers.append(torch.nn.Linear(_out, num_cls))

    return torch.nn.Sequential(*layers)
