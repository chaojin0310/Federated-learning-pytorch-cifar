#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import get_user_groups


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def get_cifar_dataset(args):
    """ Returns train and test datasets
    """
    if args.dataset == 'cifar-10':
        data_dir = './data/cifar-10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
    elif args.dataset == 'cifar-100':
        data_dir = './data/cifar-100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                          transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                         transform=apply_transform)
    return train_dataset, test_dataset


def get_cifar_user_groups(args):
    ''' Returns a user group dict
    '''
    iid = 'iid' if args.iid else 'niid'
    file_name = 'cifar-100_dict_users_%s_%d_users.json' % (iid, args.num_users)
    return get_user_groups(file_name)



def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
        w_avg[key] = torch.true_divide(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
