import math
from typing import Sized, cast

import torch


def param_groups_weight_decay(
    model: torch.nn.Module, weight_decay=1e-5, no_weight_decay_list=()
):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def adjust_learning_rate(optimizer, epoch, warmup_epochs, total_epochs, max_lr, min_lr):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            # This is only used during finetuning, and not yet
            # implemented in our codebase
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def param_groups_lrd(
    model: torch.nn.Module,
    weight_decay=0.05,
    no_weight_decay_list=[],
    layer_decay=0.75,
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/7067d6b4ec0b44fd38e29ab3658765abcd9c7441/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(cast(Sized, model.encoder.blocks)) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_rest_finetuning(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_rest_finetuning(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/7067d6b4ec0b44fd38e29ab3658765abcd9c7441/beit/optim_factory.py#L33
    """
    if "embed" in name:
        return 0
    elif name.startswith("encoder.blocks"):
        return int(name.split(".")[2]) + 1
    else:
        return num_layers
