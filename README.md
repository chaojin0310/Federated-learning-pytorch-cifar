# Pytorch-cifar100-fed

    This is a Federated Learning repository using pytoch based on **pytoch-cifar100**.

    https://github.com/weiaicunzai/pytorch-cifar100

## Guide

1. Preprocess data

   ```
   cd pytorch-cifar100-fed
   python preprocess.py -dataset cifar-100 --iid 1 --num_users 10
   ```

2. Federated learning

   ```
   python fed_train.py -net XXX --epochs 100 --num_users 10 --frac 1.0 --local_ep 1 -lr 0.1 -b 128 -gpu
   ```
