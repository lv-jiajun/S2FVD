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
from torch.utils.data import DataLoader

# lib_path = os.path.abspath(os.path.join('.'))
# sys.path.append(lib_path)
from ParameterConfig import ParameterConfig
from gnnmodel.OptimizedGCNClassifier import OptimizedGCNClassifier
from gnnmodels.GATClassifier import GATClassifier
from gnnmodels.OptimizedGATClassifier import OptimizedGATClassifier
from .utils import NodesDataset, init_network

class sumModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,embeddings_matrixcnn, db, node_vec_stg='mean', device=None, layer_num=2,
                 activation=F.relu, dp_rate=0.5):
        super(sumModel, self).__init__()
        self.n_classes = n_classes
        self.device = device
       # self.embedding_matrix = embedding_matrix
        self.layers = nn.ModuleList()  # 通过网络层拼接的方式来完成模型构建
        x = import_module('gnnmodel.' + node_vec_stg)
        self.config = x.Config(embeddings_matrixcnn, hidden_dim, device)
        self.model1 = x.Model(self.config).to(device)
        self.model2 = OptimizedGCNClassifier(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM,
                                    self.n_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg).to(device)
    #    self.model2 = OptimizedGATClassifier(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM,
    #                                                                      self.n_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg).to(device)

        self.fc = nn.Linear(384, 26)
    def forward(self, x1, x2):
        out1 =self.model1(x1)
        out2 =self.model2(x2)
        inputs = [out1, out2]
        out = torch.cat(inputs,dim=1)
       # out = torch.add(out1, out2)
        out = self.fc(out)
        return out

