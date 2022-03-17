# Federated-learning-pytorch-cifar

This is a **Federated Learning** repository implemented by pytorch based on **pytorch-cifar100** (https://github.com/weiaicunzai/pytorch-cifar100).

## Preparation

- `conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch`
- Install the packages listed in `requirements.txt`. I.e. with `pip`: run `pip3 install -r requirements.txt`.

## Simulation

### Preprocess data

Enter the repository directory (`cd pytorch-cifar100-fed`) and assign data shards to several clients.

Our experiments use the following instruction:

```shell
python3 preprocess.py -dataset cifar-100 --iid 1 --num_users 10
```

#### Additional notes

`preprocess.py` supports these tags:

- `-dataset`: name of dataset; options are `cifar-100` and `cifar-10`
- `--iid`: 1 to sample in an I.I.D. manner, or 0 to sample in a non-I.I.D. manner
- `--num_users`: total number of clients

### Federated learning instructions

Currently, we only support single-GPU training instead of distributed multi-GPU training for both federated and centralized DNN training. 

To reproduce our experiments under federated settings, the following instructions are usable if GPU is available (we use four P100s, for example).  It is recommended not to run these commands on the same GPU at the same time.

```shell
CUDA_VISIBLE_DEVICES=0 python3 fed_train.py -net mobilenetv2 --epochs 400 --num_users 10 --frac 1.0 --local_ep 1 -lr 0.2 -b 128 -gpu > fed_mobilenetv2_bs128.log
CUDA_VISIBLE_DEVICES=1 python3 fed_train.py -net mobilenetv2 --epochs 400 --num_users 10 --frac 1.0 --local_ep 1 -lr 0.2 -b 32 -gpu > fed_mobilenetv2_bs32.log
CUDA_VISIBLE_DEVICES=2 python3 fed_train.py -net squeezenet --epochs 400 --num_users 10 --frac 1.0 --local_ep 1 -lr 0.3 -b 128 -gpu > fed_squeezenet_bs128.log
CUDA_VISIBLE_DEVICES=3 python3 fed_train.py -net squeezenet --epochs 400 --num_users 10 --frac 1.0 --local_ep 1 -lr 0.3 -b 32 -gpu > fed_squeezenet_bs32.log
```

If there's no GPU available, just run:

```shell
python3 fed_train.py -net mobilenetv2 --epochs 400 --num_users 10 --frac 1.0 --local_ep 1 -lr 0.2 -b 128 > fed_mobilenetv2_bs128.log
python3 fed_train.py -net mobilenetv2 --epochs 400 --num_users 10 --frac 1.0 --local_ep 1 -lr 0.2 -b 32 > fed_mobilenetv2_bs32.log
python3 fed_train.py -net squeezenet --epochs 400 --num_users 10 --frac 1.0 --local_ep 1 -lr 0.3 -b 128 > fed_squeezenet_bs128.log
python3 fed_train.py -net squeezenet --epochs 400 --num_users 10 --frac 1.0 --local_ep 1 -lr 0.3 -b 32 > fed_squeezenet_bs32.log
```

#### Additional notes

`fed_train.py` supports these tags:

- `-net`: name of DNN model; options are listed in the `models` folder; for example `mobilenetv2` for MobileNetV2 and `squeezenet` for SqueezeNet
- `--epochs`: number of rounds to simulate
- `--num_users`: total number of clients; note that it **should** be consistent with `--number_users` set in the pre-processing process
- `--frac`: fraction of clients to be trained per round
- `--local_ep`:  local training epochs per round for each client
- `-lr`: initial learning rate
- `-b`: batch size for training
- `-gpu`: usage of GPU; default to false; set `-gpu` to make it true

### Centralized learning instructions

Similar to federated learning, two versions of instructions of centralized (single machine) learning are provided respectively.

GPU version:

```shell
CUDA_VISIBLE_DEVICES=0 python3 train.py -net mobilenetv2 -lr 0.06 -b 128 -gpu > central_mobilenetv2_bs128.log
CUDA_VISIBLE_DEVICES=1 python3 train.py -net mobilenetv2 -lr 0.06 -b 32 -gpu > central_mobilenetv2_bs32.log
CUDA_VISIBLE_DEVICES=2 python3 train.py -net squeezenet -lr 0.23 -b 128 -gpu > central_squeezenet_bs128.log
CUDA_VISIBLE_DEVICES=3 python3 train.py -net squeezenet -lr 0.23 -b 32 -gpu > central_squeezenet_bs32.log
```

CPU version:

```shell
python3 train.py -net mobilenetv2 -lr 0.06 -b 128 > central_mobilenetv2_bs128.log
python3 train.py -net mobilenetv2 -lr 0.06 -b 32 > central_mobilenetv2_bs32.log
python3 train.py -net squeezenet -lr 0.23 -b 128 > central_squeezenet_bs128.log
python3 train.py -net squeezenet -lr 0.23 -b 32 > central_squeezenet_bs32.log
```

#### Additional notes

`train.py`  supports `-net`, `-lr`, `-b` and `-gpu` tags, which are exactly the same as those under federated settings.

## Visualization

Ensure that the eight log files listed in the simulation instructions above exist in the repo directory (`./Federated-learning-pytorch-cifar100`).

Run `python3 script.py --mode fed` and `python3 script.py --mode central` to get the federated and centralized results respectively in the `experimental_results` folder.

