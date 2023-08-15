import os
import torch.utils.data as data

class MyDataset_path(data.Dataset):
    def __init__(self, list_file, path_input):
        self.list_file = list_file
        self.path_input = path_input

    def __len__(self):
        # ファイル数を返す
        return len(self.list_file)

    def __getitem__(self, index):
        # 画像のパスを取得
        path_image = self.list_file[index][0]

        # ラベルを取得
        label_class = self.list_file[index][1]
        label_type = self.list_file[index][2]
        return path_image, label_class
