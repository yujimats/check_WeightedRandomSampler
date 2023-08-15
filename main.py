import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler

from utils import MyDataset_path
from fix_seed import fix_seed
from get_files import get_files_list_toyota_cars
from view_info import gen_hist, gen_bar, transition_data_use

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

    # サンプリング回数を調査
    df_files = pd.DataFrame(list_files, columns=['path', 'label', 'class'])
    list_all_path = df_files['path'].to_list()
    list_label0_path = df_files[df_files['label']==0]['path'].to_list()
    list_label1_path = df_files[df_files['label']==1]['path'].to_list()

    # 通常の場合
    _ = transition_data_use(path_output=path_output, dataloader=dataloader, list_all_path=list_all_path, phase='normal_alldata')
    list_log_label0 = transition_data_use(path_output=path_output, dataloader=dataloader, list_all_path=list_label0_path, phase='normal_alldata_label0')
    list_log_label1 = transition_data_use(path_output=path_output, dataloader=dataloader, list_all_path=list_label1_path, phase='normal_alldata_label1')

    # 重ねて表示
    x_array_label0, y_array_label0 = np.array(list_log_label0).T
    x_array_label1, y_array_label1 = np.array(list_log_label1).T
    # 規格化
    y_array_label0 = y_array_label0 / np.max(y_array_label0) * 100
    y_array_label1 = y_array_label1 / np.max(y_array_label1) * 100
    # 描画
    p0 = plt.plot(x_array_label0, y_array_label0)
    p1 = plt.plot(x_array_label1, y_array_label1)
    plt.legend((p1[0], p0[0]), ('class 1', 'class 0'))
    plt.title('normal_label0&1')
    plt.xlabel('iteration')
    plt.ylabel('file_use ratio [%]')
    plt.savefig(os.path.join(path_output, 'log_normal_label0&1.png'))
    plt.close()
    plt.clf()

    # WRS適応後
    list_log_all = transition_data_use(path_output=path_output, dataloader=dataloader_WRS, list_all_path=list_all_path, phase='WRS_alldata')
    list_log_label0 = transition_data_use(path_output=path_output, dataloader=dataloader_WRS, list_all_path=list_label0_path, phase='WRS_alldata_label0')
    list_log_label1 = transition_data_use(path_output=path_output, dataloader=dataloader_WRS, list_all_path=list_label1_path, phase='WRS_alldata_label1')

    print(list_log_all[-1])
    print(list_log_label0[-1])
    print(list_log_label1[-1])

    # 重ねて表示
    x_array_label0, y_array_label0 = np.array(list_log_label0).T
    x_array_label1, y_array_label1 = np.array(list_log_label1).T
    # 規格化
    y_array_label0 = y_array_label0 / np.max(y_array_label0) * 100
    y_array_label1 = y_array_label1 / np.max(y_array_label1) * 100
    # 描画
    p0 = plt.plot(x_array_label0, y_array_label0)
    p1 = plt.plot(x_array_label1, y_array_label1)
    plt.legend((p1[0], p0[0]), ('class 1', 'class 0'))
    plt.title('WRS_label0&1')
    plt.xlabel('iteration')
    plt.ylabel('file_use ratio [%]')
    plt.savefig(os.path.join(path_output, 'log_WRS_label0&1.png'))
    plt.close()
    plt.clf()

if __name__=='__main__':
    main()

