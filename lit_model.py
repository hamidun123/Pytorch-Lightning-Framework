import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import yaml
import math


class LitModel(pl.LightningModule):
    """
    pytorch lightning 模型
    """

    def __init__(self, config):
        """
        初始化模型
        Args:
            config: 模型基本参数
        """
        super(LitModel, self).__init__()

        self.config = config
        config_dict = yaml.load(config.dump(), Loader=yaml.FullLoader)
        self.save_hyperparameters(config_dict)

        # if self.config.SOLVER.loss_type == "focal_loss":
        #     self.caculate_loss = FocalLoss(config)

    def share_step(self, batch, batch_idx):
        """
        train 和 validation 的共同步骤
        """
        original_ids, att_mask, wrong_list, correct_ids, len_list = batch
        pred = self.model(original_ids)
        loss = F.binary_cross_entropy(pred.view(-1), wrong_list.view(-1))
        acc, precision, recall, f1 = self.calculateperformance.caculate(pred.clone(), wrong_list.clone())

        return loss, f1, acc, precision, recall

    def training_step(self, batch, batch_idx):
        """
        train 步骤
        Args:
            batch: 输入数据
            batch_idx: batch的索引
        Returns:
        """
        loss, f1, acc, precision, recall = self.share_step(batch, batch_idx)
        self.log("f1", f1, prog_bar=True, logger=False)
        return {"loss": loss, "train_f1": f1}

    def validation_step(self, batch, batch_idx):
        """
        validation 步骤
        """
        loss, f1, acc, precision, recall = self.share_step(batch, batch_idx)
        self.log("f1", f1, prog_bar=True, logger=False)
        return {"val_loss": loss, "val_f1": f1}

    def test_step(self, batch, batch_idx):
        """
        测试步骤
        Args:
            batch:
            batch_idx:
        Returns:
        """
        loss, f1, acc, precision, recall = self.share_step(batch, batch_idx)
        self.log("f1", f1)
        self.log("acc", acc)
        self.log("precision", precision)
        self.log("recall", recall)

    def training_step_end(self, training_output):
        """
        在训练每个step结束自动调用
        :param training_output:
        :return:
        """

    def validation_step_end(self, validation_output):
        """
        在每个validation step结束自动调用
        :param validation_output:
        :return:
        """
        pass

    def validation_epoch_end(self, outputs):
        """
        在一个 validation epoch 结束后被自动调用，用于计算平均loss和cer
        Args:
            outputs: 所有 validation 输出的集合
        """
        avg_loss, avg_f1 = 0, 0
        for i in outputs:
            avg_loss += i['val_loss'].item()
            avg_f1 += i['val_f1']
        avg_loss = avg_loss / len(outputs)
        avg_f1 = avg_f1 / len(outputs)
        self.print('验证集参数: loss: {:.4f} f1: {:.4f}'.format(avg_loss, avg_f1))

    def configure_optimizers(self):
        """
        配置优化器，默认Adam
        """
        opt = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=self.config.OPT.lr)

        return opt

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None,
                       ):
        """
        自定义opt步骤实现warmup
        （这是钩子函数，不可更改函数参数模型）
        Args:
            epoch:
            batch_idx:
            optimizer:
            optimizer_idx:
            optimizer_closure:
            on_tpu:
            using_native_amp:
            using_lbfgs:
        """
        lr = self.config.OPT.lr * math.pow(self.config.OPT.k_lr, epoch)
        for p in optimizer.param_groups:
            p['lr'] = lr
        optimizer.step(closure=optimizer_closure)

    def forward(self, inference_data):
        """
        推理部分
        Args:
            inference_data: 待推理数据
        Returns:
        """

        self.model.eval()
        pred = self.model()

        return pred

    def load_model(self, path):
        """
        加载模型
        Args:
            path: checkpoint保存路径
        """
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt["state_dict"])
