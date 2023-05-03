from importlib import import_module

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from .utils import NodesDataset, init_network


class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, head_num, n_classes, embedding_matrix, node_vec_stg='mean', merge='cat', device=None):
        super(GATClassifier, self).__init__()
        self.merge = merge
        self.node_vec_stg = node_vec_stg
        if node_vec_stg != 'mean':
            x = import_module('gnnmodels.' + node_vec_stg)
            self.config = x.Config(embedding_matrix, hidden_dim, device)
            model = x.Model(self.config).to(device)
            if node_vec_stg != 'Transformer':
                init_network(model)
            self.node_model = model
        self.device = device

        self.conv1 = GATConv(in_dim, hidden_dim, head_num, residual=True)
        if merge == 'cat':
            self.conv2 = GATConv(hidden_dim * head_num, hidden_dim, num_heads=1, residual=True)
        else:
            self.conv2 = GATConv(hidden_dim, hidden_dim, num_heads=1, residual=True)
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
                # tensor_batch = batch.clone().detach().to(self.device)
                # batch_out = self.node_model(tensor_batch)
                batch_out = self.node_model(batch)
                node_vec_list.append(batch_out)
            return th.cat(node_vec_list, dim=0)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        # h = g.in_degrees().view(-1, 1).float()
        h = self.generate_node_vecs(g).float().to(self.device)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        # concat on the output feature dimension (dim=1)
        h = th.transpose(h, 0, 1)
        heads = [hd for hd in h]
        if self.merge == 'cat':
            h = th.cat(heads, dim=1)
        else:
            # merge using average
            h = th.mean(th.stack(heads), dim=0)
        h = F.relu(self.conv2(g, h))
        h = h.squeeze(1)
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        # hg = dgl.max_nodes(g, 'h')
        return self.classify(hg)