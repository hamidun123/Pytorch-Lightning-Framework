from subfunction import inference, test, train
from config import _C as config

if __name__ == '__main__':

    stage = "inference"
    # 如果是inference/test，必须提供以下两个文件
    check_point = "tb_logs/2022-03-09/version_0/checkpoints/epoch=7-f1=0.87.ckpt"  # 保存点
    config_file = "tb_logs/2022-03-09/version_0/hparams.yaml"  # 保存点对应配置文件

    if stage == "train":
        config.merge_from_file("config.yml")
        train(config)
    elif stage == "test":
        test(check_point, config_file)
    elif stage == "inference":
        inference(check_point, config_file, "")
    else:
        print("Unsupported {} stage".format(config.stage))
