# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    """配置参数"""
    def __init__(self,  out_dim, device=None):
        self.model_name = 'TextCNN'

        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.dropout = 0.5                                              # 随机失活
        self.num_classes = out_dim                                      # 类别数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率

        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.fc = nn.Linear(896, config.num_classes)


    def forward(self, x):
        out = self.fc(x)
        return out
