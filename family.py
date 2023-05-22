import os
import sys
import time
from torch.utils.data import DataLoader
import torch as th
import torch.nn as nn
import torch.optim as optim
from gnnmodels.sumModel import sumModel
from gnnmodels.sumModel1 import sumModel1

from precess_sequence import precess_sequences
from precess_ast import precess_ast
from precess_cfg import CFGDataset

from ParameterConfig import ParameterConfig
from train_eval import model_train, model_evaluate, plot_train_validation_acc_loss, metric_predictions
from gnnmodels.NaiveGCNClassifier import NaiveGCNClassifier
from train_evalcnn import model_train1, model_evaluate1, plot_train_validation_acc_loss, metric_predictions1
from gnnmodels.OptimizedGCNClassifier import OptimizedGCNClassifier
from gnnmodels.GATClassifier import GATClassifier
from gnnmodels.OptimizedGATClassifier import OptimizedGATClassifier

lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
device = 'cuda' if th.cuda.is_available() else 'cpu'
device = th.device(device)
ParameterConfig.device = device
print(device)


def run_whole_procedure(fig_prefix_org, corpus_path1, dic_file_path, func_rep_stg, unique_funs_file, num_classes,
                        label_maps=None, task_str=None, model_types=[],
                        node_vec_stg_list=[], black_list=[]):
    global model2
    # token sequence
    print('loading  token sequences ...')
    filename_train = '/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/tokens/train.txt'
    texts, embeddings_matrixcnn, fig_prefix = precess_sequences(filename_train, vec=400)

    # ast tree
    print('loading  ast trees ...')
    texts1, embeddings_matrixast = precess_ast()

    print('loading  cfg graph...')
    db = CFGDataset(corpus_path1, dic_file_path, func_rep_stg, unique_funs_file, label_maps, task_str=task_str,
                    black_list=black_list)
    dt_size = len(db)

    points_tuple = list(zip(texts, texts1, db))

    # split into train and test set
    train_size = int((1 - ParameterConfig.dataset_split_ratio) * dt_size)
    test_size = dt_size - train_size
    trainset, testset = th.utils.data.random_split(points_tuple, [train_size, test_size])

    for model_type in model_types:
        for node_vec_stg in node_vec_stg_list:
            fig_prefix = model_type + '#' + node_vec_stg + '#' + fig_prefix_org
            # step 2: construct and train the model
            print('setting up the {} model...'.format(model_type))
            if model_type == 'GAT':  # use the GAT model
                if node_vec_stg == 'mean':
                    model = GATClassifier(ParameterConfig.EMBEDDING_DIM, ParameterConfig.GAT_HIDDEN_DIM,
                                          ParameterConfig.HEAD_NUM,
                                          num_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg, merge='mean',
                                          device=device).to(device)
                else:
                    model = GATClassifier(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM,
                                          ParameterConfig.HEAD_NUM,
                                          num_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg, merge='mean',
                                          device=device).to(device)
            elif model_type == 'OPT_GAT':
                if node_vec_stg == 'mean':
                    model = OptimizedGATClassifier(ParameterConfig.EMBEDDING_DIM, ParameterConfig.GAT_HIDDEN_DIM,
                                                   ParameterConfig.HEAD_NUM, num_classes, db.embeddings_matrix,
                                                   node_vec_stg=node_vec_stg, device=device,
                                                   layer_num=ParameterConfig.GAT_Layer_Num,
                                                   feat_drop=ParameterConfig.GAT_FEAT_DP_RATE,
                                                   attn_drop=ParameterConfig.GAT_ATT_DP_RATE).to(device)
                else:
                    model = sumModel1(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM,
                                      ParameterConfig.HEAD_NUM,
                                      num_classes, embeddings_matrixcnn, embeddings_matrixast, db,
                                      node_vec_stg=node_vec_stg,
                                      device=device, layer_num=ParameterConfig.GAT_Layer_Num
                                      ).to(device)
            elif model_type == 'OPT_GCN':
                if node_vec_stg == 'mean':
                    model = OptimizedGCNClassifier(ParameterConfig.EMBEDDING_DIM, ParameterConfig.GCN_HIDDEN_DIM,
                                                   num_classes, db.embeddings_matrix, device=device,
                                                   layer_num=ParameterConfig.GCN_Layer_Num,
                                                   dp_rate=ParameterConfig.GCN_DP_RATE).to(device)
                else:
                    model = sumModel(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM,
                                     num_classes, embeddings_matrixcnn, db, node_vec_stg=node_vec_stg,
                                     device=device, layer_num=ParameterConfig.GCN_Layer_Num,
                                     dp_rate=ParameterConfig.GCN_DP_RATE).to(device)
                    print("wf,vgel,v,ef,v")
            else:  # use the basic GCN model
                if node_vec_stg == 'mean':
                    model = NaiveGCNClassifier(ParameterConfig.EMBEDDING_DIM, ParameterConfig.GCN_HIDDEN_DIM,
                                               num_classes, db.embeddings_matrix, device=device).to(device)
                else:
                    model = NaiveGCNClassifier(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM,
                                               num_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg,
                                               device=device).to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=ParameterConfig.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=ParameterConfig.lr_decay)
            # train the model
            print('training the {}-{} model'.format(model_type, node_vec_stg))
            model_train(fig_prefix, trainset, model, loss_func, optimizer, scheduler, testset)


def main():
    print("ast+acfg+token")
    # corpus_base_path = './图神经网络/testcfg/input/'
    corpus_base_path = '/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/cfgs/'
    # dic_base_path = './图神经网络/vec26/'
    dic_base_path = '/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/cfgs/'
    unique_fun_base_path = ''
    black_list = []
    classes = 2

    unique_fun_stgs = ['coarse']
    ins_abs_stgs = ['coarse']  # ['coarse', 'medium', 'fine'] instruction abstraction strategies
    # construct the label map
    label_maps_26 = {
        "#0": 0,
        "#1": 1,
        "#2": 2,
        "#3": 3,
        "#4": 4,
        "#5": 5,
        "#6": 6,
        "#7": 7,
        "#8": 8,
        "#9": 9,
        "#10": 10,
        "#11": 11,
        "#12": 12,
        "#13": 13,
        "#14": 14,
        "#15": 15,
        "#16": 16,
        "#17": 17,
        "#18": 18,
        "#19": 19,
        "#20": 20,
        "#21": 21,
        "#22": 22,
        "#23": 23,
        "#24": 24,
        "#25": 25,
        "#26": 26
    }
    label_maps_2 = {
        "#0": 0,
        "#1": 1,
        "#2": 1,
        "#3": 0,
        "#4": 0,
        "#5": 1,
        "#6": 1,
        "#7": 1,
        "#8": 1,
        "#9": 1,
        "#10": 1,
        "#11": 1,
        "#12": 1,
        "#13": 1,
        "#14": 1,
        "#15": 1,
        "#16": 1,
        "#17": 1,
        "#18": 1,
        "#19": 1,
        "#20": 1,
        "#21": 1,
        "#22": 1,
        "#23": 1,
        "#24": 1,
        "#25": 1,
        "#26": 1
    }

    if classes == 2:
        label_maps = label_maps_2
    elif classes == 26:
        label_maps = label_maps_26

    CLASS_NUMBER = classes
    model_types = ['OPT_GAT']  # ['OPT_GAT'] or ['OPT_GCN']
    node_vec_stg_list = [
        'TextCNN']  # ['mean'] ['TextCNN']  ['TextRNN'] ['Transformer','DPCNN'] ['TextRCNN'], 'TextRNN_Att']

    hyper_log_prefix = model_types[0] + '-Family'
    ParameterConfig.log_config(hyper_log_prefix)
    func_rep_stg = 'CFG'
    for i in range(len(ins_abs_stgs)):
        path_stg_suffix = func_rep_stg + '#' + ins_abs_stgs[i]
        for unique_fun_stg in unique_fun_stgs:
            model_name = model_types[0] + '-Family#' + func_rep_stg + '#' + \
                         ins_abs_stgs[i] + '#' + unique_fun_stg + '.pkl'
            print('-------------------' + model_name + '-------------------')
            corpus_path = corpus_base_path + ins_abs_stgs[i]
            dic_file_path = dic_base_path + 'ins2vec_' + ins_abs_stgs[i] + '.dic'
            unique_fun_path = unique_fun_base_path + 'WHOLE#' + unique_fun_stg + '.csv'
            run_whole_procedure(model_name.replace('.pkl', ''), corpus_path, dic_file_path, path_stg_suffix,
                                unique_fun_path, CLASS_NUMBER,
                                label_maps, task_str='family', model_types=model_types,
                                node_vec_stg_list=node_vec_stg_list, black_list=black_list)


if __name__ == '__main__':
    main()
