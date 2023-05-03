import io
import math
import time

import dgl
import torch
import torch as th
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, f1_score, roc_auc_score, \
    precision_recall_curve, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

from ParameterConfig import ParameterConfig


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    # * unpacks a list or tuple into position arguments.
    # ** unpacks a dictionary into keyword arguments.
    texts, texts1, db = map(list, zip(*samples))
    graphs, labels2 = map(list, zip(*db))
    batched_graph = dgl.batch(graphs).to(ParameterConfig.device)
    texts = [aa.tolist() for aa in texts]
    #texts1 = [aa1.tolist() for aa1 in texts1]
    texts1 = [aa1 for aa1 in texts1]

    return batched_graph, th.tensor(texts).to(ParameterConfig.device),\
           texts1, th.tensor(labels2).to(ParameterConfig.device)



def plot_train_validation_acc_loss(fig_prefix, history):
    # acc curve during the training
    fig = plt.figure()
    acc = history['acc']
    val_acc = history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Train accuracy')
    plt.plot(epochs, val_acc, 'b', label='Test accuracy')
    plt.title('Train and Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'test'], loc='lower left')
    # plt.show()
    fig.savefig(fig_prefix + '#accuracy-curve.eps')
    plt.close(fig)

    # loss curve during the training
    fig = plt.figure()
    plt.plot(epochs, history['loss'])
    plt.plot(epochs, history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    fig.savefig(fig_prefix + '#loss-curve.eps')
    plt.close(fig)


def model_train(fig_prefix, trainset, model, loss_func, optimizer, scheduler, validationset=None):
    """
    train the model on the train set with the specified loss function and optimizer,
    and evaluate on the validation set (if provided)
    :param trainset:
    :param model:
    :param loss_func:
    :param optimizer:
    :param scheduler:
    :param validationset:
    :return:
    """

    best_val_loss = float("inf")
    best_val_acc = 0.
    best_model = fig_prefix + '.pt'
    history = {'loss': [], 'acc': [], 'sampled_acc': [], 'val_loss': [], 'val_acc': [],
               'val_sampled_acc': []}
    # Use PyTorch's DataLoader and the collate function defined to batch data
    data_loader = DataLoader(trainset, batch_size=ParameterConfig.BATCH_SIZE, shuffle=True,
                             collate_fn=collate, pin_memory=ParameterConfig.PIN_MEM)

    model.train()
    model.is_train_mode = True
    start = time.time()
    for epoch in range(ParameterConfig.EPOCHES):
        epoch_start = time.time()
        epoch_loss, epoch_argmax_acc, epoch_sampled_acc = train(data_loader, loss_func, model, optimizer, scheduler,
                                                                epoch)
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_argmax_acc)
        history['sampled_acc'].append(epoch_sampled_acc)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
              'train argmax accuracy {:5.2f} | train sampled accuracy {:5.2f}'.format(epoch,
                                                                                      (time.time() - epoch_start),
                                                                                      epoch_loss, epoch_argmax_acc,
                                                                                      epoch_sampled_acc))

        if validationset is not None:
            epoch_start = time.time()
            val_loss, val_argmax_acc, val_sampled_acc = evaluate_loss_and_acc(model, validationset, loss_func)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_argmax_acc)
            history['val_sampled_acc'].append(val_sampled_acc)
            print('| end of epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} | '
                  'validation argmax accuracy {:5.2f} | validation sampled accuracy {:5.2f}'.format(epoch, (
                    time.time() - epoch_start), val_loss, val_argmax_acc, val_sampled_acc))

            # if val_loss < best_val_loss:  # choose model with the least validation loss as the best model
            #     best_val_loss = val_loss
            #     th.save(model.state_dict(), best_model)

            if val_argmax_acc > best_val_acc:   # choose model with the best validation accuracy as the best model
                best_val_acc = val_argmax_acc
                th.save(model.state_dict(), best_model)

        print('-' * 89)
        # adjust the learning rate after each epoch
        scheduler.step()

    end = time.time()
    print()
    print('Time total: {:5.2f} sec'.format(end - start))
    print('Time per epoch: {:5.2f} sec'.format((end - start) / ParameterConfig.EPOCHES))
    print()
    return best_model, history


def train(data_loader, loss_func, model, optimizer, scheduler, epoch):
    """
    train process for an epoch
    :param data_loader:
    :param loss_func:
    :param model:
    :param optimizer:
    :param scheduler:
    :param epoch:
    :return:
    """
    model.train()  # Turn on the train mode
    model.is_train_mode=True
    total_loss = 0.
    period_loss = 0.

    total_correct_sampled = 0
    period_correct_sampled = 0
    total_correct_argmax = 0
    peroid_correct_argmax = 0
    total_sample_num = 0
    period_sample_num = 0

    start_time = time.time()
    for iter, (bg1,bg, bg2,labels) in enumerate(data_loader):
       # texts, batched_graph = map(list, zip(*bg))
       # batched_graph = dgl.batch(batched_graph).to(ParameterConfig.device)
        prediction = model(bg, bg1,bg2)
        loss = loss_func(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        # Note!!! Applying gradient clipping may consume a lot of computation time
        # nn.utils.clip_grad_norm_(model.parameters(), ParameterConfig.clip)
        optimizer.step()
        # calculate loss
        # total_loss += loss.detach.item()
        # period_loss += loss.detach.item()
        total_loss += loss.item()
        period_loss += loss.item()
        # calculate accuracy
        label = labels.float().view(-1, 1).to(ParameterConfig.device)
        probs_Y = th.softmax(prediction, 1)
        sampled_Y = th.multinomial(probs_Y, 1)
        argmax_Y = th.max(probs_Y, 1)[1].view(-1, 1)
        total_correct_sampled += th.sum(label == sampled_Y.float()).item()
        total_correct_argmax += th.sum(label == argmax_Y.float()).item()
        total_sample_num += len(label)
        period_correct_sampled += th.sum(label == sampled_Y.float()).item()
        peroid_correct_argmax += th.sum(label == argmax_Y.float()).item()
        period_sample_num += len(label)

        log_interval = 200
        if iter % log_interval == 0 and iter > 0:
            cur_loss = period_loss / log_interval
            cur_argmax_acc = peroid_correct_argmax / period_sample_num * 100
            cur_sampled_acc = period_correct_sampled / period_sample_num * 100
            elapsed = time.time() - start_time
            # print('| epoch {:3d} | {:5d}/{:5d} batches | '
            #       'lr {:02.5f} | ms/batch {:5.2f} | '
            #       'loss {:5.2f} | ppl {:8.2f} | '
            #       'argmax accuracy {:5.2f}% | sampled accuracy {:5.2f}%'.format(epoch, iter, len(data_loader),
            #                                                                     scheduler.get_lr()[0],
            #                                                                     elapsed * 1000 / log_interval,
            #                                                                     cur_loss, math.exp(cur_loss),
            #                                                                     cur_argmax_acc, cur_sampled_acc))
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | '
                  'argmax accuracy {:5.2f}% | sampled accuracy {:5.2f}%'.format(epoch, iter, len(data_loader),
                                                                                scheduler.get_lr()[0],
                                                                                elapsed * 1000 / log_interval,
                                                                                cur_loss,
                                                                                cur_argmax_acc, cur_sampled_acc))
            period_loss = 0.
            period_correct_sampled = 0
            peroid_correct_argmax = 0
            period_sample_num = 0
            start_time = time.time()

    return total_loss / (
                iter + 1), total_correct_argmax / total_sample_num * 100, total_correct_sampled / total_sample_num * 100


def evaluate_loss_and_acc(model, dataset, loss_func):
    """
    Evaluate on a validation set the performance of a (partially) trained model
    :param model:
    :param dataset:
    :param loss_func:
    :return:
    """
    # Use PyTorch's DataLoader and the collate function defined.
    data_loader = DataLoader(dataset, batch_size=ParameterConfig.BATCH_SIZE, collate_fn=collate, pin_memory=ParameterConfig.PIN_MEM)
    model.eval()  # Turn on the evaluation mode
    model.is_train_mode=False
    total_loss = 0.
    correct_sampled = 0
    correct_argmax = 0
    total_sample_num = 0
    with th.no_grad():
        for iter, (bg1,bg,bg2, labels) in enumerate(data_loader):
            prediction = model(bg, bg1,bg2)
           # loss = loss_func(prediction, labels)
            # calculate loss
            loss = loss_func(prediction, labels)
            total_loss += loss.item()
            # calculate accuracy
            label = labels.float().view(-1, 1).to(ParameterConfig.device)
            probs_Y = th.softmax(prediction, 1)
            sampled_Y = th.multinomial(probs_Y, 1)
            argmax_Y = th.max(probs_Y, 1)[1].view(-1, 1)
            correct_sampled += th.sum(label == sampled_Y.float()).item()
            correct_argmax += th.sum(label == argmax_Y.float()).item()
            total_sample_num += len(label)

    return total_loss / (iter + 1), correct_argmax / total_sample_num * 100, correct_sampled / total_sample_num * 100


def model_evaluate(model, dataset, loss_func, fig_prefix=''):
    """
    When a model is trained, we can use this method to evaluate
    the performance of the model on the test dataset    :param model:
    :param dataset:
    :param loss_func:
    :return:
    """
    # Use PyTorch's DataLoader and the collate function defined.
    data_loader = DataLoader(dataset, batch_size=ParameterConfig.BATCH_SIZE, collate_fn=collate, pin_memory=ParameterConfig.PIN_MEM)
    model.eval()  # Turn on the evaluation mode
    model.is_train_mode=False
    all_preds = []
    all_labels = []
    total_loss = 0.
    correct_sampled = 0
    correct_argmax = 0
    total_sample_num = 0
    flag = 0
    with th.no_grad():
        for iter, (bg1,bg,bg2, labels) in enumerate(data_loader):
            prediction = model(bg, bg1,bg2)
            # calculate loss
            loss = loss_func(prediction, labels)
            total_loss += loss.item()
            # calculate accuracy
            # label = labels.float().view(-1, 1).to(device)
            label = labels.float().view(-1, 1)
            probs_Y = th.softmax(prediction, 1)   #[[0.5295, 0.4705],[0.8905, 0.1095], [0.9143, 0.0857],[0.4138, 0.5862],...]
            if flag == 0:
                flag = 1
                print(probs_Y)
            sampled_Y = th.multinomial(probs_Y, 1)
            argmax_Y = th.max(probs_Y, 1)[1].view(-1, 1)
            correct_sampled += th.sum(label == sampled_Y.float()).item()
            correct_argmax += th.sum(label == argmax_Y.float()).item()
            total_sample_num += len(label)
            all_preds.append(probs_Y)
            all_labels.append(labels)

    preds = th.cat(all_preds, dim=0)
    print('weidupreds' + str(preds.ndim))   #weidupreds2
    ground_truth = th.cat(all_labels, dim=0).cpu().numpy()
    print('weiduground_truth' + str(ground_truth.ndim))  #weiduground_truth1
    rst_acc = 'loss: {:5.2f} | accuracy: {:5.2f} | sampled accuracy {:5.2f}'.format(total_loss / (iter + 1),
                correct_argmax / total_sample_num * 100,
                correct_sampled / total_sample_num * 100)
    print(rst_acc)
    with open(fig_prefix + '#classification_report.txt', 'w') as f:
        f.write(rst_acc+'\n')
        f.close()
    return preds, ground_truth


def metric_predictions(pre, ground_truth, fig_prefix=''):
    '''
    # do transformation so as to utilize sklearn APIs
    class_num = pre.shape[1]
    print("列数是"+str(class_num))  #列数是2
    print("行数是" + str(pre.shape[0]))  #行数是33042
    ground_truth_for_sklearn = []
    for idx in ground_truth:
        #print("idx:"+str(idx))
        label_arr = np.zeros(class_num)
        label_arr[idx] = 1
        ground_truth_for_sklearn.append(label_arr)

    ground_truth_for_sklearn = np.stack(ground_truth_for_sklearn)

    # 计算每一类的ROC Curve和AUC-ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(class_num):
        fpr[i], tpr[i], thresholds_ = roc_curve(ground_truth_for_sklearn[:, i], pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(ground_truth_for_sklearn.ravel(), pre.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= class_num
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['r', 'g', 'b', 'c', 'm', 'y', 'aqua', 'darkorange', 'cornflowerblue', 'pink'])
    for i, color in zip(range(class_num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of compiler family identification')
    plt.legend(loc="lower right")
    # plt.show()
    fig.savefig(fig_prefix + '#ROC-curve.eps')
    plt.close(fig)

    # Test,这是干嘛？

    for i in range(len(pre)):
        max_value = max(pre[i])
        for j in range(len(pre[i])):
            if max_value == pre[i][j]:
                pre[i][j] = 1
            else:
                pre[i][j] = 0

    # 生成分类评估报告
   # report_str = str(classification_report(ground_truth_for_sklearn, pre, digits=4))
   # with open(fig_prefix + '#classification_report.txt', 'a') as f:
   #     f.write(report_str)
   #     f.close()
   # print(report_str)
    pre = pre[:, 1]
    ground_truth_for_sklearn = ground_truth_for_sklearn[:, 1]

    print('weiduone'+str(pre.ndim))  #weiduone1
    print('weidutwo' + str(ground_truth_for_sklearn.ndim)) #weidutwo1
    lr_precision, lr_recall, _ = precision_recall_curve(ground_truth_for_sklearn, pre)

    lr_auc = auc(lr_recall, lr_precision)
    plt.figure(1)  # 创建图表1
    plt.title('PR Curve')  # give plot a title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    plt.plot(lr_recall,lr_precision,label='PR (area = {0:0.3f})'
                               ''.format(lr_auc))
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('p-r.png')
    plt.figure(2)  # 创建图表2

    fpr, tpr, thresholds = roc_curve(ground_truth_for_sklearn, pre)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    lw = 2
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve')  # give plot a title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr, color='darkorange',lw = lw, label='ROC (area = {0:0.3f})'
                               ''.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend(loc="lower right")
    plt.show()
    #plt.savefig('roc.png')


    plt.title('ROC Curve')  # give plot a title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr)
    plt.show()
    plt.savefig('roc.png')

    #lr_auc = auc(lr_recall, lr_precision)
    print('******************************' +'\n')
   # print('PR AUC = %.4f' % (lr_auc))
    lr_auc = roc_auc_score(ground_truth_for_sklearn, pre)  #https://zhuanlan.zhihu.com/p/349516115
   # print('ROC AUC = %.4f' % (lr_auc))

    for i in range(len(pre)):
        if(pre[i]>=0.5):
            pre[i] = 1
        else:
            pre[i] = 0

    f1 = f1_score(ground_truth_for_sklearn, pre)
    print('F1 = %.4f' % (f1))
    accuracy = accuracy_score(ground_truth_for_sklearn, pre)
    print('accuracy = %.4f' % (accuracy))
  #  print('precision = %.4f' % (precision_score(ground_truth_for_sklearn, pre, average='binary', pos_label=0)))
  #  print('recall = %.4f' % (recall_score(ground_truth_for_sklearn, pre, average='binary', pos_label=0)))
    C2 = confusion_matrix(ground_truth_for_sklearn, pre)
    TN, FP, FN, TP = C2.ravel()
    numerator = (TP * TN) - (FP * FN)  # 马修斯相关系数公式分子部分
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # 马修斯相关系数公式分母部分
    result = numerator / denominator
    print('MCC = %.4f' % (result))

    return fpr, tpr,roc_auc
    '''
  # do transformation so as to utilize sklearn APIs
    class_num = pre.shape[1]
    print("列数是"+str(class_num))  #列数是26
    print("行数是" + str(pre.shape[0]))  #行数是33042
    ground_truth_for_sklearn = []
    ground_truth_for_sklearn = ground_truth

    '''
    for idx in ground_truth:
        #print("idx:"+str(idx))
        label_arr = np.zeros(class_num)
        label_arr[idx] = 1
        ground_truth_for_sklearn.append(label_arr)
'''
    ground_truth_for_sklearn = np.stack(ground_truth_for_sklearn)

    # 计算每一类的ROC Curve和AUC-ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    '''
    for i in range(class_num):
        fpr[i], tpr[i], thresholds_ = roc_curve(ground_truth_for_sklearn[:, i], pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(ground_truth_for_sklearn.ravel(), pre.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= class_num
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['r', 'g', 'b', 'c', 'm', 'y', 'aqua', 'darkorange', 'cornflowerblue', 'pink'])
    for i, color in zip(range(class_num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of compiler family identification')
    plt.legend(loc="lower right")
    # plt.show()
    fig.savefig(fig_prefix + '#ROC-curve.eps')
    plt.close(fig)
    '''
    pred = torch.max(pre, 1)[1].cpu().numpy()
    pred = pred
    print(pred)
    '''
    # Test,这是干嘛？
    for i in range(len(pre)):
        max_value = max(pre[i])
        for j in range(len(pre[i])):
            if max_value == pre[i][j]:
                pre[i][j] = 1
            else:
                pre[i][j] = 0
    # 生成分类评估报告
    '''
    f1 = f1_score(ground_truth_for_sklearn, pred, average="macro")
    print('F1 = %.4f' % (f1))
    f1 = f1_score(ground_truth_for_sklearn, pred, average="weighted")
    print('F1 = %.4f' % (f1))
    accuracy = accuracy_score(ground_truth_for_sklearn, pred)
    print('accuracy = %.4f' % (accuracy))
    report_str = str(classification_report(ground_truth_for_sklearn, pred, digits=4))
    with open(fig_prefix + '#classification_report.txt', 'a') as f:
        f.write(report_str)
        f.close()
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
    # pre = pre[:, 1]
    # ground_truth_for_sklearn = ground_truth_for_sklearn[:, 1]
    print('weiduone' + str(pred.ndim))  # weiduone1
    print('weidutwo' + str(ground_truth_for_sklearn.ndim))  # weidutwo1
    #    lr_precision, lr_recall, _ = precision_recall_curve(ground_truth_for_sklearn, pred)
    #   lr_auc = auc(lr_recall, lr_precision)
    print('******************************' + '\n')