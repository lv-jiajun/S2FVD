from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F

from BatchProgramClassifier import BatchProgramClassifier
from ParameterConfig import ParameterConfig
from gnnmodels.OptimizedGATClassifier import OptimizedGATClassifier


class sumModel1(nn.Module):
    def __init__(self, in_dim, hidden_dim, HEAD_NUM, n_classes, embeddings_matrixcnn, embeddings_matrixast, db,
                 node_vec_stg='mean', device=None, layer_num=2,
                 activation=F.relu):
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
        self.model2 = OptimizedGATClassifier(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM, HEAD_NUM,
                                             self.n_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg).to(device)
        #  self.config3 = x1.Config(embeddings_matrixcnn1, hidden_dim, device)
        self.model3 = BatchProgramClassifier(embedding_dim=128, hidden_dim=100, vocab_size=1167, encode_dim=128,
                                             label_size=192, batch_size=ParameterConfig.BATCH_SIZE,
                                             use_gpu=True, pretrained_weight=embeddings_matrixast).to(device)
        # self.fc = nn.Linear(576, 192)
        # self.fc1 = nn.Linear(192, 2)

        self.fc = nn.Linear(576, 32)
        self.fc1 = nn.Linear(32, 2)

        self.concat = nn.Linear(576, 2)
        self.concat2 = nn.Linear(192, 2)

        self.MLP_fc1 = nn.Linear(192, 1)
        self.MLP_fc2 = nn.Linear(3, 27)

    def forward(self, x1, x2, x3):
        out3 = self.model3(x3) # batch_size * 192
        out1 = self.model1(x1) # batch_size * 192
        out2 = self.model2(x2) # batch_size * 192

        # ---------MLP concat--------------
        # out1 = out1.unsqueeze(dim=1)
        # out2 = out2.unsqueeze(dim=1)
        # out3 = out3.unsqueeze(dim=1)
        # inputs = [out1, out2, out3]
        # out = torch.cat(inputs, dim=1)  # 256 * 3 * 192
        # out = self.MLP_fc1(out)
        # out = out.squeeze(dim=2)
        # out = self.MLP_fc2(out)
        # -------------end------------------

        # ------------mlp--------------
        inputs = [out1, out2, out3]
        out = torch.cat(inputs, dim=1) # 256 * 576
        out = self.fc(out)
        out = self.fc1(out)
        # -------------------------------

        # ------------concat2--------------
        # inputs = [out1, out2, out3]
        # out = torch.cat(inputs, dim=1) # 256 * 576
        # out = self.concat(out)
        # -------------------------------

        # ---------MAX Polling---------------
        # out1 = out1.unsqueeze(dim=1)
        # out2 = out2.unsqueeze(dim=1)
        # out3 = out3.unsqueeze(dim=1)
        # inputs = [out1, out2, out3]
        # out = torch.cat(inputs, dim=1)  # 256 * 3 * 192
        # out = torch.max(out, dim=1).values  # 256 * 192
        # out = self.fc1(out)
        # -----------------------------------

        # out = out1 + out2 + out3
        # out = self.concat2(out)

        return out
