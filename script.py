from genericpath import exists
import matplotlib.pyplot as plt
import os
import argparse


def converge_round(acc_list, converge_acc):
    for i in range(len(acc_list)):
        if abs(acc_list[i] - converge_acc) < 0.005:
            return i
    return -1  # maximum round


def read_topk_data(file_path, k=1):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    acc = []
    pos = -7
    if k == 1:
        pos = -7
    elif k == 3:
        pos = -5
    elif k == 5:
        pos = -3
    for line in content:
        if line.find("Test set") != -1:
            temp_list = line.split(sep=' ')
            temp_acc = float(temp_list[pos][:-2])
            acc.append(temp_acc)
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='central or fed')
    args = parser.parse_args()
    mode = args.mode

    M_acc128 = read_topk_data(mode + '_mobilenetv2_bs128.log', 1)
    M_acc32 = read_topk_data(mode + '_mobilenetv2_bs32.log', 1)

    S_acc128 = read_topk_data(mode + '_squeezenet_bs128.log', 1)
    S_acc32 = read_topk_data(mode + '_squeezenet_bs32.log', 1)

    acc_list = []
    # Compute convergent accuracy and round
    gram_size = 5
    print('M_acc128', '{:.2f}%'.format(sum(M_acc128[-gram_size:]) / gram_size * 100))
    print('M_acc32', '{:.2f}%'.format(sum(M_acc32[-gram_size:]) / gram_size * 100))
    print('S_acc128', '{:.2f}%'.format(sum(S_acc128[-gram_size:]) / gram_size * 100))
    print('S_acc32', '{:.2f}%'.format(sum(S_acc32[-gram_size:]) / gram_size * 100))

    acc_list.append(sum(M_acc128[-gram_size:]) / gram_size)
    acc_list.append(sum(M_acc32[-gram_size:]) / gram_size)
    acc_list.append(sum(S_acc128[-gram_size:]) / gram_size)
    acc_list.append(sum(S_acc32[-gram_size:]) / gram_size)
    print('M_acc128 converge_round', '%d' % (converge_round(M_acc128, acc_list[0])))
    print('M_acc32 converge_round', '%d' % (converge_round(M_acc32, acc_list[1])))
    print('S_acc128 converge_round', '%d' % (converge_round(S_acc128, acc_list[2])))
    print('S_acc32 converge_round', '%d' % (converge_round(S_acc32, acc_list[3])))

    plt.xlabel('Round')
    plt.ylabel('Test Accuracy')
    # top1 acc
    plt.plot(M_acc128[0:], label='M-Net-BS128', linestyle='solid', color='k')
    plt.plot(M_acc32[0:], label=mode + 'M-Net-BS32', linestyle='dashed', color='k')

    plt.plot(S_acc128[0:], label=mode + 'S-Net-BS128', linestyle='solid', color='r')
    plt.plot(S_acc32[0:], label=mode + 'S-Net-BS32', linestyle='dashed', color='r')

    plt.legend(loc='lower right')
    # plt.show()

    file_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_path, 'experimental_results')
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    file_path = os.path.join(file_path, mode + '.png')
    plt.savefig(file_path,  bbox_inches='tight')
