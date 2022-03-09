import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data.dataset import MyDataSet


def collect_fn(batch):
    # 实现数据pad，对不一样长度的数据补齐到统一长度

    return pad_data


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """
        完成数据初始化，如定义数据集路径，需要的数据集名称等
        """
        super(MyDataModule, self).__init__()
        # 获取预训练模型名字
        self.config = config
        self.pre_trained_name = config.PRETRAINED.tokenizer
        self.train_path = config.DATASETS.train
        self.test_path = config.DATASETS.test
        self.valid_path = config.DATASETS.valid

    def prepare_data(self):
        """
        如果数据集需要下载可以在这里定义方法
        """
        pass

    def setup(self, stage=None):
        """
        数据集建立方法，框架自动调用
        :param stage: 当前阶段 fit/test
        """
        if stage == 'fit' or stage is None:

            self.train_dataset = MyDataSet(self.train_path)
            self.valid_dataset = MyDataSet(self.test_path)
            pass
        if stage == 'test' or stage is None:
            self.test_dataset = MyDataSet(self.valid_path)
            pass

    # 以下三个函数均需要返回对应的dataloader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.SOLVER.train_batch_size, num_workers=12, collate_fn=collect_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.SOLVER.valid_batch_size, num_workers=12, collate_fn=collect_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.SOLVER.test_batch_size, num_workers=12, collate_fn=collect_fn)
