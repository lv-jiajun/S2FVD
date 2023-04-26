# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    """配置参数"""
    def __init__(self, embedding, out_dim, device=None):
        self.model_name = 'TextCNN'
        self.embedding_pretrained = torch.tensor(embedding).float()  # 预训练词向量
        # 预训练词向量
        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.dropout = 0.5                                              # 随机失活
        self.num_classes = out_dim                                      # 类别数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            # self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)   #改变维度
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
