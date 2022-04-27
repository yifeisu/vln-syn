"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax

from .adamw import AdamW
from .rangerlars import RangerLars


def build_optimizer(model, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer
                                                if not any(nd in n for nd in no_decay)],
                                     'weight_decay': args.weight_decay},
                                    {'params': [p for n, p in param_optimizer
                                                if any(nd in n for nd in no_decay)],
                                     'weight_decay': 0.0}]

    # currently Adam only
    if args.optim == 'adam':
        optimcls = Adam
    elif args.optim == 'adamax':
        optimcls = Adamax
    elif args.optim == 'adamw':
        optimcls = AdamW
    elif args.optim == 'rangerlars':
        optimcls = RangerLars
    else:
        raise ValueError('invalid optimizer')

    optimizer = optimcls(optimizer_grouped_parameters,
                         lr=args.lr,
                         betas=args.betas)

    optimizer.zero_grad()
    return optimizer
