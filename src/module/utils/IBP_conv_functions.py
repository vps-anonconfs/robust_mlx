# Imports
import logging
import warnings
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.nn import DataParallel
from torch.nn import Sequential, Conv2d, Linear, ReLU, Flatten, AdaptiveAvgPool2d, AdaptiveMaxPool2d, AvgPool2d, MaxPool2d
from itertools import chain

GLOBAL_MODEL_PETURB = 'relative'
SOMEWHAT_BIG_NUMBER = 10

"""
=============================================================
Interval Propagation Code for Learning Certifiably Fair NNs
=============================================================
"""

# Copyright (C) 2020, Jiameng Fan <jmfan@bu.edu>
#
# This program is licenced under the MIT License,
# contained in the LICENCE file in this directory.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def AttributionIBPRegularizer(IBP_model, inp, lab, vec, eps, num_class=2, mode='train'):
    # print("check", inp.min(), inp.max(), vec.min(), vec.max())
    x_L = inp - (vec * eps)
    x_U = inp + (vec * eps)
    
    # storing computed upper and lower bounds after scatter
    logit_u, logit_l = IBP_model.interval_range(norm=np.inf, x_U=x_U, x_L=x_L,
                                                eps=eps, C=None, layer_idx=0, intermediate=False)
    label = lab
    logit_worst_case = F.one_hot(label, num_class)
    # torch.zeros(inp.size(0), num_class).to(torch.long).to(inp.device)
    # logit_worst_case[torch.arange(inp.size(0)), label] = 1
    logit_worst_case = torch.where(logit_worst_case == 1, logit_l, logit_u)

    # IBP Reg only when mask.sum()>0 : we return torch.zero instead of ce loss when no spurious part in specification (mask)
    # [B] of boolean vals that are true when mask is active
    # is_mask_nonzero = vec.reshape(inp.size(0), -1).sum(dim=1) > 1.0
    # logit_worst_case = logit_worst_case[is_mask_nonzero]
    # lab = lab[is_mask_nonzero]

    # diff
    prob_u, prob_l = torch.softmax(logit_u, dim=-1), torch.softmax(logit_l, dim=-1)
    # logit_diff = (logit_u[is_mask_nonzero] - logit_l[is_mask_nonzero]).abs().mean()
    b = len(prob_u)
    logit_diff = (logit_u[torch.arange(b), lab] - logit_l[torch.arange(b), lab]).mean()
    prob_diff = (prob_u[torch.arange(b), lab] - prob_l[torch.arange(b), lab]).mean()
    # if len(logit_worst_case) < 1:
    #     return torch.zeros(1).to(inp.device), torch.zeros(1).to(inp.device)

    return F.cross_entropy(logit_worst_case, lab), logit_diff, prob_diff


class BoundFlatten(nn.Module):
    # TODO: ignores the start and end_dim of Flatten, can lead to runtime failures
    def __init__(self, bound_opts=None):
        super(BoundFlatten, self).__init__()
        self.bound_opts = bound_opts

    def forward(self, x):
        self.shape = x.size()[1:]
        return x.view(x.size(0), -1)

    def interval_propagate(self, norm, h_U, h_L, eps):
        return (
            norm, h_U.view(h_U.size(0), -1), h_L.view(h_L.size(0), -1)
        )


class BoundPooling(nn.Module):
    def __init__(self, pooling_op, bound_opts=None):
        super(BoundPooling, self).__init__()
        self.pooling_op = pooling_op
        self.bound_opts = bound_opts

    def forward(self, x):
        return self.pooling_op(x)

    def interval_propagate(self, norm, h_U, h_L, eps):
        assert norm == np.inf
        return (
            norm, self.pooling_op(h_U), self.pooling_op(h_L)
        )


class BoundLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, bound_opts=None):
        super(BoundLinear, self).__init__(in_features, out_features, bias)
        self.bound_opts = bound_opts

    @staticmethod
    def convert(linear_layer, bound_opts=None):
        layer = BoundLinear(
            linear_layer.in_features,
            linear_layer.out_features,
            linear_layer.bias is not None,
            bound_opts
        )
        layer.weight.data.copy_(linear_layer.weight.data)
        layer.bias.data.copy_(linear_layer.bias.data)
        return layer

    def interval_propagate(self, norm, h_U, h_L, eps, C=None):
        # merge the specification
        if C is not None:
            # after multiplication with C, we have (batch, output_shape, pre_layer_shape)
            # we have batch dimension here because of each example has different C
            weight = C.matmul(self.weight)
            bias = C.matmul(self.bias)
        else:
            # weight dimension (this_layer_shape, prev_layer_shape)
            weight = self.weight
            bias = self.bias

        if norm == np.inf:
            # Linf norm
            mid = (h_U + h_L) / 2.
            diff = (h_U - h_L) / 2.
            weight_abs = weight.abs()
            if C is not None:
                center = weight.matmul(mid.unsqueeze(-1)) + bias.unsqueeze(-1)
                deviation = weight_abs.matmul(diff.unsqueeze(-1))
                # these have an extra (1,) dimension as the last dimension
                center = center.squeeze(-1)
                deviation = deviation.squeeze(-1)
                if len(deviation[deviation < 0]) > 0:
                    print(f"Possible numerical issue (linear), deviations found to be negative for "
                          f"#{len(deviation[deviation < 0])} they are: {deviation[deviation < 0]} {deviation.min()}")
                deviation = torch.clip(deviation, min=0)
            else:
                # fused multiply-add
                center = torch.addmm(bias, mid, weight.t())
                deviation = diff.matmul(weight_abs.t())
        else:
            # L2 norm
            h = h_U  # h_U = h_L, and eps is used
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            if C is not None:
                center = weight.matmul(h.unsqueeze(-1)) + bias.unsqueeze(-1)
                center = center.squeeze(-1)
            else:
                center = torch.addmm(bias, h, weight.t())
            deviation = weight.norm(dual_norm, -1) * eps

        upper = center + deviation
        lower = center - deviation

        return np.inf, upper, lower


class BoundConv2d(Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, bias=True, bound_opts=None
    ):
        super(BoundConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )
        self.bound_opts = bound_opts

    @staticmethod
    def convert(l, bound_opts=None):
        nl = BoundConv2d(
            l.in_channels, l.out_channels, l.kernel_size, l.stride, l.padding,
            l.dilation, l.groups, l.bias is not None, bound_opts,
        )
        nl.weight.data.copy_(l.weight.data)
        nl.bias.data.copy_(l.bias.data)
        return nl

    def forward(self, input):
        output = super(BoundConv2d, self).forward(input)
        self.output_shape = output.size()[1:]
        self.input_shape = input.size()[1:]
        return output

    def interval_propagate(self, norm, h_U, h_L, eps):
        if norm == np.inf:
            mid = (h_U + h_L) / 2.
            diff = (h_U - h_L) / 2.
            # print(f"stats (U): {h_U.abs().mean()}, {h_U.max(), h_U.min()}")
            # print(f"stats (L): {h_L.abs().mean()}, {h_L.max(), h_L.min()}")
            weight_abs = self.weight.abs()
            deviation = F.conv2d(
                diff, weight_abs, None, self.stride,
                self.padding, self.dilation, self.groups
            )
            if len(deviation[deviation < -1e-5]) > 0:
                print(f"Possible numerical issue (conv layer), deviations found to be negative for "
                      f"#{len(deviation[deviation < 0])} they are: {deviation[deviation < 0]} {deviation.min()}")
            deviation = torch.clip(deviation, min=0)
        else:
            # L2 norm
            mid = h_U
            # TODO: consider padding here?
            deviation = torch.mul(self.weight, self.weight).sum((1, 2, 3)).sqrt() * eps
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        center = F.conv2d(
            mid, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )
        upper = center + deviation
        lower = center - deviation
        return np.inf, upper, lower


class BoundReLU(ReLU):
    def __init__(self, prev_layer, inplace=False, bound_opts=None):
        super(BoundReLU, self).__init__(inplace)
        # ReLU needs the previous layer's bound
        self.bound_opts = bound_opts

    @staticmethod
    def convert(act_layer, prev_layer, bound_opts=None):
        layer = BoundReLU(prev_layer, act_layer.inplace, bound_opts)
        return layer

    def interval_propagate(self, norm, h_U, h_L, eps):
        assert norm == np.inf
        # stored upper and lower bounds will be used for backward bound propagation
        self.upper_u = h_U
        self.lower_l = h_L
        return norm, F.relu(h_U), F.relu(h_L)


class BoundSequential(Sequential):
    def __init__(self, *args):
        super(BoundSequential, self).__init__(*args)

    @staticmethod
    def convert_resnet(block_and_sequential_model, bound_opts=None):

        def convert_component(l, layers):
            if isinstance(l, Linear):
                layers.append(BoundLinear.convert(l, bound_opts))
            if isinstance(l, Conv2d):
                layers.append(BoundConv2d.convert(l, bound_opts))
            if isinstance(l, ReLU):
                layers.append(BoundReLU.convert(l, layers[-1], bound_opts))
            if isinstance(l, Flatten):
                layers.append(BoundFlatten(bound_opts))
            return layers

        layers = []
        for bl in block_and_sequential_model:
            if isinstance(sequential_model, Sequential):
                seq_model = sequential_model
                for l in seq_model:
                    convert_component(l, layers)
            else:
                convert_component(bl, layers)
                # seq_model = sequential_model.module

        return BoundSequential(*layers)

    @staticmethod
    def convert(sequential_model, bound_opts=None):
        layers = []
        if isinstance(sequential_model, Sequential):
            seq_model = sequential_model
        else:
            seq_model = sequential_model.module
        for l in seq_model:
            if isinstance(l, Linear):
                layers.append(BoundLinear.convert(l, bound_opts))
            elif isinstance(l, Conv2d):
                layers.append(BoundConv2d.convert(l, bound_opts))
            elif isinstance(l, ReLU):
                layers.append(BoundReLU.convert(l, layers[-1], bound_opts))
            elif isinstance(l, Flatten):
                layers.append(BoundFlatten(bound_opts))
            elif isinstance(l, AdaptiveAvgPool2d) or isinstance(l, AdaptiveMaxPool2d) or isinstance(l, AvgPool2d) or \
                    isinstance(l, MaxPool2d):
                layers.append(BoundPooling(l, bound_opts))
        return BoundSequential(*layers)

    def __call__(self, *input, **kwargs):

        if "method_opt" in kwargs:
            opt = kwargs["method_opt"]
            kwargs.pop("method_opt")
        else:
            raise ValueError(
                "Please specify the 'method_opt' as the last argument."
            )
        if "disable_multi_gpu" in kwargs:
            kwargs.pop("disable_multi_gpu")
        if opt == "interval_range":
            return self.interval_range(*input, **kwargs)
        else:
            if "layer_idx" in kwargs:
                # obtain the output by feeding input from the intermediate
                # layer: layer_idx
                layer_idx = kwargs["layer_idx"]
                kwargs.pop("layer_idx")
                layers = []
                for idxModule, layer in enumerate(
                    list(self._modules.values())
                ):
                    if idxModule >= layer_idx:
                        layers.append(layer)
                intermediate_model = Sequential(*layers)
                return intermediate_model(*input, **kwargs)
            else:
                return super(BoundSequential, self).__call__(*input, **kwargs)

    def interval_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, layer_idx=0, intermediate=False):
        h_U = x_U
        h_L = x_L
        if not intermediate:
            for i, module in enumerate(
                list(self._modules.values())[layer_idx:-1]
            ):
                # all internal layers should have Linf norm, except for the first layer
                norm, h_U, h_L = module.interval_propagate(
                    norm, h_U, h_L, eps
                )
                if (h_U < h_L).any():
                    if (h_U - h_L > -1e-3).all():
                        warnings.warn(
                            'fix numerical issue for IBP computation!'
                        )
                        # h_L[(h_U - h_L) < 0] = (
                        #     h_U[(h_U - h_L) < 0] + h_L[(h_U - h_L) < 0]
                        # ) / 2.
                        # h_U[(h_U - h_L) < 0] = h_L[(h_U - h_L) < 0]
                    else:
                        raise ValueError(
                            'layer: {}, property: {}\n diff: {}'.format(
                                i, module, (h_U - h_L)[(h_U - h_L) < 0]
                            )
                        )
        else:
            for i, module in enumerate(
                list(self._modules.values())[:layer_idx]
            ):
                norm, h_U, h_L = module.interval_propagate(norm, h_U, h_L, eps)
                if (h_U < h_L).any():
                    if (h_U - h_L > -1e-3).all():
                        warnings.warn(
                            'fix numerical issue for IBP computation!'
                        )
                        # h_L[(h_U - h_L) < 0] = (
                        #     h_U[(h_U - h_L) < 0] + h_L[(h_U - h_L) < 0]
                        # ) / 2.
                        # h_U[(h_U - h_L) < 0] = h_L[(h_U - h_L) < 0]
                    else:
                        raise ValueError(
                            'layer: {}, property: {}\n diff: {}'.format(
                                i, module, (h_U - h_L)[(h_U - h_L) < 0]
                            )
                        )
            return h_U, h_L
        # last layer has C to merge
        norm, h_U, h_L = list(
            self._modules.values()
        )[-1].interval_propagate(norm, h_U, h_L, eps, C)

        # assert (h_U + 1e-5 > h_L).all(), f"Numerical issue?! Few entries of {h_U} are smaller than {h_L}!!"
        return h_U, h_L


class BoundDataParallel(DataParallel):
    # This is a customized DataParallel class
    def __init__(self, *inputs, **kwargs):
        super(BoundDataParallel, self).__init__(*inputs, **kwargs)
        self._replicas = None

    # Overide the forward method
    def forward(self, *inputs, **kwargs):
        disable_multi_gpu = False
        if "disable_multi_gpu" in kwargs:
            disable_multi_gpu = kwargs["disable_multi_gpu"]
            kwargs.pop("disable_multi_gpu")

        if not self.device_ids or disable_multi_gpu:
            return self.module(*inputs, **kwargs)

        # Only replicate during forwarding propagation to save communication
        # cost.
        if self._replicas is None or \
                kwargs.get("method_opt", "forward") == "forward":
            self._replicas = self.replicate(self.module, self.device_ids)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) "
                    "but found one of them on device {}".format(
                        self.src_device_obj, t.device
                    )
                )
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        outputs = self.parallel_apply(
            self._replicas[:len(inputs)], inputs, kwargs
        )
        return self.gather(outputs, self.output_device)