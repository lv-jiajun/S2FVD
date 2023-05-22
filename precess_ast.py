import pandas as pd
import torch
import numpy as np
from gensim.models.word2vec import Word2Vec

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def get_batch(dataset, idx, bs):
#     tmp = dataset.iloc[idx: idx + bs]
#     data, labels = [], []
#     for _, item in tmp.iterrows():
#         data.append(item[1])
#         labels.append(item[2])
#     # print(labels)
#     return data, torch.LongTensor(labels)

def precess_ast():
    root = '/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/trees'
    train_data = pd.read_pickle(root + '/train_block.pkl')
    texts = []
    labels = []
    print("function is :" + str(len(train_data)))
    for _, item in train_data.iterrows():
        texts.append(item[2])
        labels.append(item[1])
    word2vec = Word2Vec.load(root + "/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    return texts, embeddings
