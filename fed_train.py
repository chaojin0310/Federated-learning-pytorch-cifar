# fed_train.py
#!/usr/bin/env	python3

import os
import sys
import argparse
import time
from datetime import datetime
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from data_utils import average_weights, get_cifar_user_groups


# Split train dataset by indexs for clients
class DatasetSplit(Dataset):
    
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# Clients' local training
class LocalTrain(object):
    
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss()
    
    def train_val_test(self, dataset, idxs):
        trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=self.args.b, shuffle=True, num_workers=2)

        return trainloader

    def local_training(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9, weight_decay=5e-4)
        
        # train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40, 60, 90, 120, 200], gamma=0.6) #learning rate decay

        iter_per_epoch = len(self.trainloader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
        train_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9736)
        if global_round <= args.warm:
            warmup_scheduler.step()
        train_scheduler.step(global_round - 1)

        for ep in range(args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 5000 == 0:
                    print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=global_round,
                    trained_samples=batch_idx * args.b + len(images),
                    total_samples=len(cifar100_training_loader.dataset)
                ))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


# Get top-1,3,5 testing accuracy
@torch.no_grad()
def eval_test_acc(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    correct_1 = 0.0
    correct_3 = 0.0
    correct_5 = 0.0
    loss_function = nn.CrossEntropyLoss()

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        # _, preds = outputs.max(1)
        # correct += preds.eq(labels).sum()
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)

        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()
        #compute top 3
        correct_3 += correct[:, :3].sum()
        #compute top1
        correct_1 += correct[:, :1].sum()

    finish = time.time()
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Top1Accuracy: {:.4f}, Top3Accuracy: {:.4f}, Top5Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct_1.float() / len(cifar100_test_loader.dataset),
        correct_3.float() / len(cifar100_test_loader.dataset),
        correct_5.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()
    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct_1.float() / len(cifar100_test_loader.dataset), epoch)

    return correct_1.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # training args
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('--epochs', type=int, default=400, help="number of rounds of training")
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')

    # federated learning args
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--num_users', type=int, default=10, help='Number of clients')
    parser.add_argument('--frac', type=float, default=1.0, help='Fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1, help='Local training epochs for each client')
    
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    
    args = parser.parse_args()

    net = get_network(args)

    #data preprocessing:
    trainset, cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=16,
        shuffle=True
    )

    _, cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    user_groups = get_cifar_user_groups(args)
    

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # best_acc = 0.0
    # if args.resume:
    #     best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    #     if best_weights:
    #         weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
    #         print('found best acc weights file:{}'.format(weights_path))
    #         print('load best training file to test acc...')
    #         net.load_state_dict(torch.load(weights_path))
    #         best_acc = eval_test_acc(tb=False)
    #         print('best acc is {:0.2f}'.format(best_acc))

    #     recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    #     if not recent_weights_file:
    #         raise Exception('no recent weights file were found')
    #     weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
    #     print('loading weights file {} to resume training.....'.format(weights_path))
    #     net.load_state_dict(torch.load(weights_path))

    #     resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    
    # Get initial global weights
    global_weights = net.state_dict()

    
    for epoch in range(1, args.epochs + 1):
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        net.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalTrain(args=args, dataset=trainset,
                                     idxs=user_groups[str(idx)])
            w, loss = local_model.local_training(
                model=copy.deepcopy(net), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        net.load_state_dict(global_weights)
        eval_test_acc(epoch)

    writer.close()
