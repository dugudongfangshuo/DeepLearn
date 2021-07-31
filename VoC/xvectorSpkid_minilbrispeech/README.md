# 数据使用
The data we use is from Mini Librispeech + OpenRIR.

# 训练说明
- 一个命令
```
python train.py train.yaml
```
# 项目结构说明：
speechBrain训练代码只需要四个文件
* `train.py`: 训练模型
* `train.yaml`: 配置参数
* `custom_model.py`: 自定义模型
* `mini_librispeech_prepare.py`: 数据准备
* `inference.py`:模型调用
# Xvector模型说明

<img src="img.png" width="450px" height="450px">

# 其他训练技巧
- 使用了环境增强方法，把一个batch的数据扩大了两倍，这样好。

# Template for Speaker Identification
  
This folder provides a working, well-documented example for training
a speaker identification model from scratch, based on a few hours of
data. The data we use is from Mini Librispeech + OpenRIR.

There are four files here:

* `train.py`: the main code file, outlines the entire training process.
* `train.yaml`: the hyperparameters file, sets all parameters of execution.
* `custom_model.py`: A file containing the definition of a PyTorch module.
* `mini_librispeech_prepare.py`: If necessary, downloads and prepares data manifests.


[For more information, please take a look into the "speaker-id from scratch" tutorial](https://colab.research.google.com/drive/1UwisnAjr8nQF3UnrkIJ4abBMAWzVwBMh?usp=sharing)