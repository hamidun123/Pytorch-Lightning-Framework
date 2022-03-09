from yacs.config import CfgNode as CN
import os
import time

root_dir = os.getcwd()
date = time.strftime("%Y-%m-%d", time.localtime())
# 这个文件下全是默认参数，可以自行定义yml文件修改默认参数
_C = CN()

_C.PRETRAINED = CN()
_C.PRETRAINED.tokenizer = "tb_logs/PreTrained/tokenizer/chinese-roberta-wwm-ext/"  # 一部分预训练模型已经保存在本地，建议从本地直接加载
_C.PRETRAINED.model = "tb_logs/PreTrained/model/chinese-roberta-wwm-ext/"
_C.PRETRAINED.config = "tb_logs/PreTrained/model/chinese-roberta-wwm-ext/config.json"

# 模型参数
_C.MODEL = CN()
_C.MODEL.window_size = 3
_C.MODEL.embedding_dim = 256
_C.MODEL.lstm_hidden_dim = 512
_C.MODEL.fc_hidden = 256

# 优化器参数
_C.OPT = CN()
_C.OPT.lr = 1e-3
_C.OPT.k_lr = 0.99

# 数据集参数
_C.DATASETS = CN()
_C.DATASETS.train = "dataset/SIGHAN15/train.json"
_C.DATASETS.valid = "dataset/SIGHAN15/dev.json"
_C.DATASETS.test = "dataset/SIGHAN15/test.json"

# 训练参数
_C.SOLVER = CN()
_C.SOLVER.gpus = [0, 1, 2]
_C.SOLVER.accelerator = 'ddp'  # dp
_C.SOLVER.max_epochs = 100
_C.SOLVER.train_batch_size = 32
_C.SOLVER.valid_batch_size = 32
_C.SOLVER.test_batch_size = 32
_C.SOLVER.loss_type = "cross_entropy"  # cross_entropy / focal_loss
_C.SOLVER.threshold = 0.5  # 置信门限
# label 权重
_C.SOLVER.alpha = [1, 1, 1]
_C.SOLVER.gamma = 2
_C.SOLVER.num_classes = 3
_C.SOLVER.size_average = True

# 生成混淆数据参数
_C.CONFUSION = CN()
_C.CONFUSION.max_mask_prob = 0.15             # 总体混淆概率
_C.CONFUSION.mask_lv1_prob = 0.2         # 同音字概率
_C.CONFUSION.mask_lv2_prob = 0.4         # 音调不同概率
_C.CONFUSION.mask_lv3_prob = 0.2         # 前鼻音后鼻音，平舌音卷舌音
_C.CONFUSION.mask_lv4_prob = 0.1        # 拼音编辑距离为1
_C.CONFUSION.mask_lv5_prob = 0.1        # 拼音编辑距离大于1
_C.CONFUSION.pinyin_set_path = "/data/nas/qian/ChineseCorrection/dataset/pinyin/pinyin_set.json"  # 拼音统计信息路径
_C.CONFUSION.initials_distence_path = "/data/nas/qian/ChineseCorrection/dataset/pinyin/initials_distance.json"  # 拼音统计信息路径
_C.CONFUSION.vowel_distence_path = "/data/nas/qian/ChineseCorrection/dataset/pinyin/vowel_distance.json"  # 拼音统计信息路径


# 保存参数
_C.SAVE = CN()
_C.SAVE.dirpath = root_dir + "/tb_logs/" + date
_C.SAVE.tb_log_path = date
_C.SAVE.save_top_k = 2
_C.SAVE.every_n_epochs = 1
_C.SAVE.logger = "tb_logs"

