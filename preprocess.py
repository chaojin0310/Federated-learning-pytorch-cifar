import argparse
from sampling import cifar_iid_json, cifar_niid_json
from conf import settings
from utils import get_training_dataloader

DATASETS = ['cifar-10', 'cifar-100']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # preprocessed arguments
    parser.add_argument('-dataset', help='name of dataset;',
                        type=str, choices=DATASETS, required=True)
    parser.add_argument('--iid', help='Dafault set to IID. Set to 0 for non-IID.',
                        type=int, default=1)
    parser.add_argument('--num_users', help='number of users to simulate',
                        type=int, default=100)

    args = parser.parse_args()

    train_dataset, _ = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=16,
        shuffle=True
    )
    if args.iid:
        cifar_iid_json(train_dataset, args.num_users, args.dataset)
    elif args.dataset == 'cifar-10':
        cifar_niid_json(train_dataset, args.num_users, 10, 'cifar-10')
    elif args.dataset == 'cifar-100':
        cifar_niid_json(train_dataset, args.num_users, 100, 'cifar-100')
