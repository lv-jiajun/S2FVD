import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
import torch as th
from BatchProgramClassifier import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2])
    # print(labels)
    return data, torch.LongTensor(labels)


if __name__ == '__main__':

    root = 'data/'
    train_data = pd.read_pickle(root + 'train/blocks.pkl')
    val_data = pd.read_pickle(root + 'dev/blocks.pkl')
    test_data = pd.read_pickle(root + 'test/blocks.pkl')
    texts = []
    labels = []
    print("function is :" + str(len(train_data)))
    for _, item in train_data.iterrows():
        texts.append(item[1])
        labels.append(item[2])
    word2vec = Word2Vec.load(root + "train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    print(word2vec.syn0.shape[0])
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0


    hidden_dim = 100
    encode_dim = 128
    labels = 26
    epochs = 35
    batch_size = 64
    use_gpu = True
    max_tokens = word2vec.syn0.shape[0]
    embedding_dim = word2vec.syn0.shape[1]

    model = BatchProgramClassifier(embedding_dim, hidden_dim, max_tokens + 1, encode_dim, labels, batch_size,
                                   use_gpu, embeddings)
    if use_gpu:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=5e-4)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model
    for epoch in range(epochs):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(train_data):
            batch = get_batch(train_data, i, batch_size)
            i += batch_size
            train_inputs, train_labels = batch
            if use_gpu:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
           # model.hidden = model.init_hidden()
            output = model(train_inputs)
           # output = model.forward(train_inputs)
         #   print("out:"+str(output.shape))
            loss = loss_function(output, train_labels)
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item() * len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(val_data):
            batch = get_batch(val_data, i, batch_size)
            i += batch_size
            val_inputs, val_labels = batch
            if use_gpu:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output = model(val_inputs)

            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item() * len(val_inputs)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc / total > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, epochs, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    pred = []
    ground_truth_for_sklearn = []
    while i < len(test_data):
        batch = get_batch(test_data, i, batch_size)
        i += batch_size
        test_inputs, test_labels = batch
        if use_gpu:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        pred.append(predicted)
        ground_truth_for_sklearn.append(test_labels)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
    ground_truth_for_sklearn = th.cat(ground_truth_for_sklearn, dim=0).cpu().numpy()
    ground_truth_for_sklearn = np.stack(ground_truth_for_sklearn)
    pred = pred
    pred = th.cat(pred, dim=0).cpu().numpy()
    pred = np.stack(pred)

    accuracy = accuracy_score(ground_truth_for_sklearn, pred)
    print('accuracy = %.4f' % (accuracy))
    report_str = str(classification_report(ground_truth_for_sklearn, pred, digits=4))
    print(report_str)
    print('------Weighted------')
    print('Weighted precision', precision_score(ground_truth_for_sklearn, pred, average='weighted'))
    print('Weighted recall', recall_score(ground_truth_for_sklearn, pred, average='weighted'))
    print('Weighted f1-score', f1_score(ground_truth_for_sklearn, pred, average='weighted'))
    print('------Macro------')
    print('Macro precision', precision_score(ground_truth_for_sklearn, pred, average='macro'))
    print('Macro recall', recall_score(ground_truth_for_sklearn, pred, average='macro'))
    print('Macro f1-score', f1_score(ground_truth_for_sklearn, pred, average='macro'))
    print('------Micro------')
    print('Micro precision', precision_score(ground_truth_for_sklearn, pred, average='micro'))
    print('Micro recall', recall_score(ground_truth_for_sklearn, pred, average='micro'))
    print('Micro f1-score', f1_score(ground_truth_for_sklearn, pred, average='micro'))
    print("Testing results(Acc):", total_acc.item() / total)
