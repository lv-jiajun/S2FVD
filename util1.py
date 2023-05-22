import torch.utils.data as Data
import torch.nn as nn


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


class NodesDataset(Data.Dataset):
    """
    Stores all nodes (represented as instruction list) within a single CFG
    """

    def __init__(self, node_list):
        super(NodesDataset, self).__init__()
        self.node_list = node_list

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, index):
        return self.node_list[index]
