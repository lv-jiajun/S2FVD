import os
import sys
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl
import numpy as np
import torch as th
from tensorflow_core.python.framework.test_util import use_gpu
from torch.utils.data import DataLoader

# lib_path = os.path.abspath(os.path.join('.'))
# sys.path.append(lib_path)
from BatchProgramClassifier import BatchProgramClassifier
from ParameterConfig import ParameterConfig
from gnnmodel.OptimizedGCNClassifier import OptimizedGCNClassifier
from gnnmodels.GATClassifier import GATClassifier
from gnnmodels.OptimizedGATClassifier import OptimizedGATClassifier
from .utils import NodesDataset, init_network

class sumModel1(nn.Module):
    def __init__(self, in_dim, hidden_dim, HEAD_NUM, n_classes,embeddings_matrixcnn, embeddings_matrixast,db, node_vec_stg='mean', device=None, layer_num=2,
                 activation=F.relu ):
        super(sumModel1, self).__init__()
        self.n_classes = n_classes
        self.device = device
       # self.embedding_matrix = embedding_matrix
        self.layers = nn.ModuleList()  # 通过网络层拼接的方式来完成模型构建
        x = import_module('gnnmodel.' + node_vec_stg)
        x1 = import_module('gnnmodel.' + 'TextCNN')

        self.config1 = x.Config(embeddings_matrixcnn, hidden_dim, device)
        self.model1 = x.Model(self.config1).to(device)
      #  self.model2 = OptimizedGCNClassifier(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM,
      #                              self.n_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg).to(device)
        self.model2 = OptimizedGATClassifier(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM,HEAD_NUM,
                                                                          self.n_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg).to(device)
      #  self.config3 = x1.Config(embeddings_matrixcnn1, hidden_dim, device)
        self.model3 = BatchProgramClassifier(embedding_dim = 128, hidden_dim= 100, vocab_size = 542, encode_dim = 128, label_size = 192, batch_size = 64,
                                             use_gpu = True, pretrained_weight=embeddings_matrixast).to(device)
        self.fc = nn.Linear(576, 192)
        self.fc1 = nn.Linear(192, 26)


    def forward(self, x1, x2,x3):
        out3 =self.model3(x3)
        out1 =self.model1(x1)
        out2 =self.model2(x2)
      #  out3 =self.model3(x3)
        inputs = [out1, out2, out3]
        out = torch.cat(inputs, dim=1)
       # print(out.shape)
       # out = torch.add(out1, out2, out3)
        out = self.fc(out)
        out = self.fc1(out)
        return out

