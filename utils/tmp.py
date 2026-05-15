import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init

from classifiers.densenet import DenseNet121
from classifiers.resnet import resnet18, resnet50, wrn34_10
from classifiers.vgg import VGG


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class Model(nn.Module):
    def __init__(self, net, data_normalize):
        super().__init__()
        self.net = net
        self.data_normalize = data_normalize

    def forward(self, x, **kwargs):
        if self.data_normalize is not None:
            x = self.data_normalize(x)
        return self.net(x, **kwargs)


def update_ckpt(args):
    cur_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    if "ckpt" not in args.ckpt:
        args.ckpt = "./ckpt/{}/".format(args.model) + args.ckpt
    if (args.dataset not in args.ckpt) and not args.debug:
        tmp = args.ckpt[:-4].split("/")
        tmp[-1] += ".pth"
        tmp[-1] = "-".join([cur_time, tmp[-1]])
        args.ckpt = "/".join(tmp)
        print(args.ckpt)
    args.logfile = args.logfile.format(args.model)
    if not args.debug:
        args.logfile = (
            args.ckpt.replace("ckpt", "log")
            .replace(args.model + "/", args.model + "-")
            .replace(".pth", ".log")
        )
    return args


def get_model(model_name, num_of_classes=10, dataset=None):
    if dataset == "cifar10":
        data_normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )
    elif dataset == "cifar100":
        data_normalize = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2673, 0.2564, 0.2762],
        )
    elif dataset == "no":
        data_normalize = None
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}")

    if model_name in ("ResNet18", "resnet18"):
        net = resnet18(in_dims=3, out_dims=num_of_classes)
    elif model_name in ("ResNet50", "resnet50"):
        net = resnet50(in_dims=3, out_dims=num_of_classes)
    elif model_name in ("WRN34_10", "wrn34_10"):
        net = wrn34_10(in_dims=3, out_dims=num_of_classes)
    elif model_name in ("VGG13", "vgg13"):
        net = VGG("VGG13", num_classes=num_of_classes)
    elif model_name in ("VGG16", "vgg16"):
        net = VGG("VGG16", num_classes=num_of_classes)
    elif model_name in ("VGG19", "vgg19"):
        net = VGG("VGG19", num_classes=num_of_classes)
    elif model_name in ("DenseNet", "DenseNet121", "densenet121"):
        net = DenseNet121(num_classes=num_of_classes)
    else:
        raise NotImplementedError(f"Unsupported model: {model_name}")

    return Model(net, data_normalize)


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key[7:] if key.startswith("module.") else key
        new_state_dict[name] = value
    return new_state_dict


def init_params(net):
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, std=1e-3)
            if module.bias is not None:
                init.constant_(module.bias, 0)
