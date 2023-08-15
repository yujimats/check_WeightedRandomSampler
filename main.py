import os
# import time
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
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

    list_files = get_files_list_toyota_cars(path_input=path_input, label_0=label_0, label_1=label_1)

    # 動作を軽くするため、pathとlabelを返すdatasetを用意
    dataset = MyDataset_path(list_files, path_input)

    # WeightedRandomSampler適用しない場合
    ## バッチ内のラベルを確認
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_iterator = iter(dataloader)
    path, labels = next(batch_iterator)
    print(labels)

    ## バッチごとのクラス数を棒グラフで表示
    label1_count = gen_bar(path_output=path_output, dataloader=dataloader, batch_size=batch_size, itr=10, phase='normal')
    print(label1_count) # ラベル

    ## itrまでのlabel_1の割合をヒストグラムで表示
    gen_hist(path_output=path_output, dataloader=dataloader, batch_size=batch_size, itr=itr, phase='normal')

    # WeightedRandomSamplerを適用
    


if __name__=='__main__':
    main()

