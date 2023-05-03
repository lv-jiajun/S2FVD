# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):

    """配置参数"""

    def __init__(self, embedding, out_dim, device=None):
        self.model_name = 'TextRCNN'
        self.embedding_pretrained = torch.tensor(embedding).float()  # 预训练词向量
        # 预训练词向量
        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # dropout咋为1
        self.dropout = 1.0                                              # 随机失活
        self.num_classes = out_dim                         # 类别数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 20                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            # self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze(-1)
        out = self.fc(out)
        return out
