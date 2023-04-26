import numpy as np
from sklearn.metrics import accuracy_score
'''
y_true = [[0, 1, 1],[0, 0, 1],[0, 1, 1]]
y_pred = [[0, 1, 0],[0, 1, 1],[0, 1, 1]]
#target_names = ['class 0', 'class 1', 'class 2']https://blog.csdn.net/u012841977/article/details/106309381
#print(classification_report(y_true, y_pred, target_names=target_names))
result = hamming_loss(y_true,y_pred)
accur1 = accuracy_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred,average = 'weighted')
print(result)
print(accur1)
print(f1)
'''
import torch
from torch.utils.data.dataset import ConcatDataset
import torch.utils.data.dataset as Dataset

'''
class MyFirstDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((-torch.ones(50), torch.ones(5)))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]


class MySecondDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((torch.ones(50) * 5, torch.ones(5) * -5))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]

'''


class subDataset(Dataset.Dataset):
    def __init__(self, texts, db,label):
        self.Feature_1 = texts
        self.Feature_2 = db
        self.label = label
    def __len__(self):
        return len(self.Feature_1)

    def __getitem__(self, index):
        Feature_1 = torch.Tensor(self.Feature_1[index].numpy())
        Feature_2 = torch.Tensor(self.Feature_2[index].numpy())
        label =torch.Tensor(self.label[index].numpy())
        return Feature_1, Feature_2,label

texts = torch.cat((torch.ones(50) * 5, torch.ones(5) * -5))
db = torch.cat((torch.ones(50)*10, torch.ones(5)*-10))
label = torch.cat((torch.ones(50)*1, torch.ones(5)*-1))
train_dataset = subDataset(texts, db,label)

batch_size = 10
# basic dataloader
dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True
                                         )

for iter,(input1,inputs,label) in enumerate(dataloader):
    print(input1)
    print(inputs)
    print(label)
    print('\n')



