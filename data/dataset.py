from torch.utils.data import Dataset
import torch
from functools import partial
import json
import numpy as np


class MyDataSet(Dataset):
    def __init__(self, args):
        super(MyDataSet, self).__init__()
        # 数据集初始化如载入数据集等

    def __len__(self):
        # 统计全部数据个数
        return len(self.data)

    def __getitem__(self, item):
        # 返回item对应的数据
        data = self.data[item]
        return data


