#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import numpy as np
from torchvision import datasets, transforms
import json
import random


def cifar_iid_json(dataset, num_users, name):
    '''
    Sample I.I.D client data from cifar-10/cifar-100 dataset
    Store the result as json format
    :param dataset:
    :param num_users:
    :return: None
    '''
    # iid divide
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [float(i) for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])

    # write to json
    parent_path = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(parent_path, "sampled_user")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_path = os.path.join(folder_path, '%s_dict_users_iid_%d_users.json' % (name, num_users))
    with open(file_path, 'w') as outfile:
        json.dump(dict_users, outfile)
    return


def cifar_niid_json(dataset, num_users, num_classes, name):
    '''
    Sample N.I.I.D client data from cifar-10/cifar-100 dataset
    Store the result as json format
    :param dataset:
    :param num_users:
    :param num_classes:
    :return: None
    '''
    # niid divide
    dict_users = {i: [] for i in range(num_users)}
    labels = list(dataset.targets)
    idx_per_class = {i: [] for i in range(num_classes)}
    for i in range(len(labels)):
        idx_per_class[labels[i]].append(float(i))
    sample_num = int(len(dataset) / num_classes)
    sample_range = [i for i in range(0, sample_num + 1)] * 100000
    for label in range(num_classes):
        num_presum = random.sample(sample_range, k=num_users-1)
        num_presum.append(0)
        num_presum.append(sample_num)
        num_presum = sorted(num_presum)
        class_per_user = [num_presum[i] - num_presum[i-1] for i in range(1, len(num_presum))]
        for j in range(len(class_per_user)):
            temp_idxs = np.random.choice(idx_per_class[label], class_per_user[j],
                                         replace=False)
            dict_users[j] += list(temp_idxs)
            idx_per_class[label] = list(set(idx_per_class[label]) - set(temp_idxs))
    for i in range(num_users):
        random.shuffle(dict_users[i])

    # write to json
    parent_path = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(parent_path, "sampled_user")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_path = os.path.join(folder_path, '%s_dict_users_niid_%d_users.json' % (name, num_users))
    with open(file_path, 'w') as outfile:
        json.dump(dict_users, outfile)
    return


def get_user_groups(file_name):
    '''
    Get user_groups from json file
    The filepath is '../sampled_user/dict_users_[iid/niid]_%d_users.json' by default
    :param file_name:
    :return: dict of image index
    '''
    parent_path = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(parent_path, "sampled_user")
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        print("No such file for user_groups!")
        raise FileNotFoundError()
    with open(file_path, 'r') as f:
        user_groups = json.load(f)
    for k in user_groups:
        user_groups[k] = list(np.array(user_groups[k]).astype(np.int))
    return user_groups


if __name__ == '__main__':
    # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,),
    #                                                         (0.3081,))
    #                                ]))
    # num = 100
    # d = mnist_noniid(dataset_train, num)
    data_dir = './data/cifar-100/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                      transform=apply_transform)

    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                     transform=apply_transform)

    cifar_niid_json(train_dataset, 100, 100, 'cifar-100')
    print(get_user_groups('cifar-100_dict_users_niid_100_users.json'))
