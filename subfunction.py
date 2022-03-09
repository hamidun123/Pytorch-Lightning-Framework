import os
from data.data import MyDataModule
from lit_model import LitModel
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from yacs.config import CfgNode as CN
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, TQDMProgressBar
import matplotlib.pyplot as plt


class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super(CustomProgressBar, self).__init__()

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def inference(check_point, config_file, inference_data):
    if os.path.exists(config_file):
        print("加载配置文件{}".format(config_file))
        cfgf = open(config_file)
        config = CN().load_cfg(cfgf)
    else:
        print("没有找到目标配置，将加载默认配置")
        from config import _C as config
    config.merge_from_file("config.yml")  # 可以修改测试数据集等
    model = LitModel(config)
    model.load_model(check_point)
    # model = model.cuda(config.INFERENCE.gpu) if config.INFERENCE.use_cuda else model
    wrong_index = model(inference_data)
    wrong_index = (wrong_index[:, 1] - 1).tolist() if wrong_index.size(0) != 0 else []
    wrong_index = [i for i in wrong_index if i < len(inference_data)]
    result = []
    for i in range(len(inference_data)):
        if i in wrong_index:
            result.append('\033[31m{}\033[0m'.format(inference_data[i]))
        else:
            result.append(inference_data[i])
    print("".join(result))


def train(config):
    model = LitModel(config)
    data_module = MyDataModule(config)
    bar = CustomProgressBar()
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{f1:.2f}',
        monitor='f1',
        save_top_k=config.SAVE.save_top_k,
        mode='max',
        every_n_epochs=config.SAVE.every_n_epochs)
    trainer = pl.Trainer(
        gpus=config.SOLVER.gpus,
        strategy=config.SOLVER.accelerator,
        max_epochs=config.SOLVER.max_epochs,
        callbacks=[checkpoint_callback, bar],
        logger=TensorBoardLogger(config.SAVE.logger, name=config.SAVE.tb_log_path))
    trainer.fit(model, data_module)


def test(check_point, config_file):
    bar = CustomProgressBar()
    if os.path.exists(config_file):
        print("加载配置文件{}".format(config_file))
        cfgf = open(config_file)
        config = CN().load_cfg(cfgf)
    else:
        print("没有找到目标配置，将加载默认配置")
        from config import _C as config
    config.merge_from_file("config.yml")  # 可以修改测试数据集等
    trainer = pl.Trainer(gpus=config.SOLVER.gpus, strategy=config.SOLVER.accelerator, logger=False,
                         callbacks=[bar])
    data_module = MyDataModule(config)
    model = LitModel(config)
    model.load_model(check_point)
