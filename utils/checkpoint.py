#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import logging
import os
from collections import defaultdict
import torch
import torch.nn as nn

from typing import Any
from typing import Optional, List, Dict, NamedTuple, Tuple, Iterable

from termcolor import colored

def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"


def _named_modules_with_dup(
        model: nn.Module, prefix: str = ""
) -> Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():  # pyre-ignore
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)


def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, io, logger = None):
    if args.local_rank == 0:
        torch.save({
            'model' : base_model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch' : epoch,
            'metrics' : metrics,#.state_dict() if metrics is not None else dict(),
            'best_metrics' : best_metrics,#.state_dict() if best_metrics is not None else dict(),
            }, os.path.join(args.out_path + '/' + args.exp_name, prefix + '.pth'))
        io.cprint(f"Save checkpoint at {os.path.join(args.out_path + '/' + args.exp_name, prefix + '.pth')}")


def resume_optimizer(optimizer, args, io, logger = None):
    ckpt_path = os.path.join(args.out_path + '/' + args.exp_name, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        io.cprint(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...')
        return 0, 0, 0
    io.cprint(f'[RESUME INFO] Loading optimizer from {ckpt_path}...')
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def resume_model(base_model, args, io, name='ckpt-last.pth', logger = None):
    # ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    ckpt_path = os.path.join(args.out_path + '/' + args.exp_name, name)
    if not os.path.exists(ckpt_path):
        io.cprint(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...')
        return 0, 0
    io.cprint(f'[RESUME INFO] Loading model weights from {ckpt_path}...')

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    # base_ckpt = state_dict['model']
    base_model.load_state_dict(base_ckpt, strict = True)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics #.state_dict()
    # print(best_metrics)

    io.cprint(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})')
    return start_epoch, best_metrics