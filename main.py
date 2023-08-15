import os
import numpy as np
# import time
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
import torchvision.models as models

from utils import MyDataset_path
from fix_seed import fix_seed
from get_files import get_files_list_toyota_cars
from view_info import gen_hist, gen_bar

def main():
    random_seed = 1234
    fix_seed(random_seed)
    path_input = os.path.join('dataset')
    path_output = os.path.join('output')
    label_0 = 'camry'
    label_1 = 'crown'
    batch_size = 32
    itr = 1000

    list_files, _, _ = get_files_list_toyota_cars(path_input=path_input, label_0=label_0, label_1=label_1)

    # datasetを用意; 動作を軽くするため、pathとlabelを返す
    dataset = MyDataset_path(list_files, path_input)

    # WeightedRandomSampler適用しない場合
    ## バッチ内のラベルを確認
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_iterator = iter(dataloader)
    path, labels = next(batch_iterator)
    print(labels)

    ## バッチごとのクラス数を棒グラフで表示
    label1_count = gen_bar(path_output=path_output, dataloader=dataloader, batch_size=batch_size, itr=10, phase='normal')
    print(label1_count) # ラベル1のカウント数を表示

    ## itrまでのlabel_1の割合をヒストグラムで表示
    gen_hist(path_output=path_output, dataloader=dataloader, batch_size=batch_size, itr=itr, phase='normal')

    # WeightedRandomSamplerを適用
    ## ラベルカウントを取得し、weightを計算
    _, labelcount_0, labelcount_1 = get_files_list_toyota_cars(path_input=path_input, label_0=label_0, label_1=label_1)
    labelcount = np.array([labelcount_0, labelcount_1])
    class_weight = 1 / labelcount

    ## サンプルにweightを設定
    sample_weight = [class_weight[list_files[i][1]] for i in range(len(list_files))]

    ## samplerを設定
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(list_files), replacement=True)
    dataloader_WRS = data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    ## バッチ内のラベルを確認
    batch_iterator = iter(dataloader_WRS)
    path, labels = next(batch_iterator)
    print(labels)

    ## バッチごとのクラス数を棒グラフで表示
    label1_count = gen_bar(path_output=path_output, dataloader=dataloader_WRS, batch_size=batch_size, itr=10, phase='WeightedRandomSampler')
    print(label1_count) # ラベル1のカウント数を表示

    ## itrまでのlabel_1の割合をヒストグラムで表示
    gen_hist(path_output=path_output, dataloader=dataloader_WRS, batch_size=batch_size, itr=itr, phase='WeightedRandomSampler')




if __name__=='__main__':
    main()

