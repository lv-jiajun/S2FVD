import os
import sys
from importlib import import_module

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl
import numpy as np
import torch as th
from torch.utils.data import DataLoader

# lib_path = os.path.abspath(os.path.join('.'))
# sys.path.append(lib_path)
from .utils import NodesDataset, init_network


class NaiveGCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, embedding_matrix, node_vec_stg='mean', device=None):
        super(NaiveGCNClassifier, self).__init__()
        self.node_vec_stg = node_vec_stg
        if node_vec_stg != 'mean':
            x = import_module('gnnmodels.' + node_vec_stg)
            self.config = x.Config(embedding_matrix, hidden_dim, device)
            model = x.Model(self.config).to(device)
            if node_vec_stg != 'Transformer':
                init_network(model)
            self.node_model = model
        self.device = device

        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.embedding_matrix = embedding_matrix
        self.is_train_mode = True

    def generate_node_vecs(self, g):
        nodes_attrs = g.ndata['w']
        torches = []
        if self.node_vec_stg == 'mean':
            for ins_list in nodes_attrs:
                vec_list = []
                for ins in ins_list:
                    vec = self.embedding_matrix[ins.item()]
                    vec_list.append(vec)
                arr = np.array(vec_list)
                vec = arr.mean(axis=0)
                torches.append(th.tensor(vec))
            return th.stack(torches, 0)
        else:
            nodes_db = NodesDataset(nodes_attrs)
            data_loader = DataLoader(nodes_db, batch_size=self.config.batch_size, shuffle=False)
            node_vec_list = []
            if self.is_train_mode:
                self.node_model.train()
            else:
                self.node_model.eval()
            for iter, batch in enumerate(data_loader):
                batch_out = self.node_model(batch)
                # tensor_batch = th.tensor(batch).to(self.device)
                # batch_out = self.node_model(tensor_batch)
                node_vec_list.append(batch_out)
            return th.cat(node_vec_list, dim=0)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        # h = g.in_degrees().view(-1, 1).float()
        h = self.generate_node_vecs(g).float().to(self.device)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        # h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        # hg = dgl.max_nodes(g, 'h')

        return self.classify(hg)