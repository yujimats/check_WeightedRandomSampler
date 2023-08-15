import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

def gen_bar(path_output, dataloader, batch_size, itr, phase=None):
    # itrまでのlabelの割合を棒グラフにして可視化
    list_label_0 = []
    list_label_1 = []
    iteration = 0
    while iteration < itr:
        for _, labels in dataloader:
            if iteration >= itr:
                break
            labelcount_1 = torch.sum(labels).item()
            labelcount_0 = batch_size - labelcount_1
            list_label_0.extend([labelcount_0])
            list_label_1.extend([labelcount_1])
            iteration += 1
    left = np.arange(itr)
    hight_0 = np.array(list_label_0)
    hight_1 = np.array(list_label_1)
    p1 = plt.bar(left, hight_1, color='orange')
    p0 = plt.bar(left, hight_0, bottom=hight_1, color='green')
    plt.legend((p1[0], p0[0]), ('class 1', 'class 0'))
    plt.title('{}'.format(phase))
    plt.xticks(np.arange(0, itr, step=1))
    plt.xlabel('iteration')
    plt.ylabel('label count')
    plt.savefig(os.path.join(path_output, 'bar_{}.png'.format(phase)))
    plt.close()
    plt.clf()
    return np.sum(hight_1)

def gen_hist(path_output, dataloader, batch_size, itr, phase=None):
    # itrまでのlabel_1の割合を求める
    list_ratio = []
    iteration = 0
    while iteration < itr:
        for _, labels in dataloader:
            if iteration >= itr:
                break
            labelcount_1 = torch.sum(labels).item()
            ratio = labelcount_1 / batch_size * 100
            list_ratio.extend([ratio])
            iteration += 1
    # ヒストグラム
    plt.hist(list_ratio, bins=50)
    plt.title('{}'.format(phase))
    plt.savefig(os.path.join(path_output, 'hist_{}.png'.format(phase)))
    plt.close()
    plt.clf()
