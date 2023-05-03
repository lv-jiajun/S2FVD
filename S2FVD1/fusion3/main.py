import os
import sys
import time

import dgl
import torch
from torch.utils.data import DataLoader
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.utils.data.dataset as Dataset
#from model1_trun import model1_trun
from gnnmodels.sumModel import sumModel
from gnnmodels.sumModel1 import sumModel1
from trainlv import lv_part
from vuldeepecker import Part1

lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
from ParameterConfig import ParameterConfig
from CFGDataset import CFGDataset
# from models import NaiveGCNClassifier, GATClassifier
from train_eval import model_train, model_evaluate, plot_train_validation_acc_loss, metric_predictions
from gnnmodels.NaiveGCNClassifier import NaiveGCNClassifier
from train_evalcnn import model_train1, model_evaluate1, plot_train_validation_acc_loss, metric_predictions1
from gnnmodels.OptimizedGCNClassifier import OptimizedGCNClassifier
from gnnmodels.GATClassifier import GATClassifier
from gnnmodels.OptimizedGATClassifier import OptimizedGATClassifier


device = 'cuda' if th.cuda.is_available() else 'cpu'
#device ='cpu'
device = th.device(device)
ParameterConfig.device = device
print(device)


def run_whole_procedure(fig_prefix_org, corpus_path1, dic_file_path, func_rep_stg, unique_funs_file, num_classes,
                        label_maps=None, task_str=None, family=None, version_list=None, model_types=[],
                        node_vec_stg_list=[], gat_multihead_merge_stg='cat', black_list=[]):
    """
    execute the whole procedure
    :param fig_prefix: specifies where the plotted figures will be stored
    :param corpus_path: path of the dataset directory
    :param dic_file_path:
    :param func_rep_stg:
    :param unique_funs_file:
    :param num_classes:
    :param label_maps:
    :param task_str:
    :param family:
    :param version_list:
    :param model_types:
    :param node_vec_stg:
    :param gat_multihead_merge_stg:
    :return:
    """
    global model2
   # filename_train= 'D:/JiajunLv/code3/sell_finally.txt'
  #  filename_train = 'D:/JiajunLv/jie/sell_finally.txt'
    filename_train = './writelabel1.txt'
  #  filename_train1 = 'D:/JiajunLv/CNNDUO/ast-label26.txt'

    texts, embeddings_matrixcnn, fig_prefix= Part1(filename_train, vec=400)
    print(type(texts))
   # texts1, embeddings_matrixcnn1, fig_prefix1= Part1(filename_train1,vec=1400)
    texts1, embeddings_matrixast = lv_part()
    print(type(texts1))
    dt_size = len(texts)
    print("dataset的长度：" + str(dt_size))
    dt_size1 = len(texts1)
    print("dataset的长度1：" + str(dt_size1))
    '''
    optimizer = optim.Adam(model1.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)
    start = time.time()
    model_stats, history = model_train1(fig_prefix, dataset1_train, model1, loss_func, optimizer, scheduler, dataset1_test)
    end = time.time()
    print('evaluating on the test-set the trained textcnn model')
    # model = init_model.load_state_dict(th.load(model_stats)).to(device)
    model1.load_state_dict(th.load(model_stats))
    model1.eval()

    preds, ground_truth = model_evaluate1(model1, dataset1_test, loss_func, fig_prefix)
    metric_predictions1(preds, ground_truth, fig_prefix)
    th.cuda.empty_cache()
    '''

    # step 1: load data
    print('loading cfg data...')
    db = CFGDataset(corpus_path1, dic_file_path, func_rep_stg, unique_funs_file, label_maps, task_str=task_str, black_list=black_list)
    dt_size = len(db)
    print("db的长度：" + str(dt_size))
    points_tulpe = list(zip(texts, texts1, db))



  #  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # split into train and test set
    train_size = int((1 - ParameterConfig.dataset_split_ratio) * dt_size)
    test_size = dt_size - train_size
    trainset, testset = th.utils.data.random_split(points_tulpe, [train_size, test_size])
   # dataset_train, db_train = map(list, zip(*trainset))
   # dataset_test, db_test = map(list, zip(*testset))
   # preds1, ground_truth = model1_trun(fig_prefix, model1, dataset_train, dataset_test)
    print('dataset size: {:d} CFGs, {:d} for training and {:d} for testing'.format(dt_size, len(trainset), len(testset)))





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
                    print("liusiyu")
                    model = sumModel1(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM,
                                     ParameterConfig.HEAD_NUM,
                                     num_classes, embeddings_matrixcnn, embeddings_matrixast, db,
                                      node_vec_stg=node_vec_stg,
                                     device=device, layer_num=ParameterConfig.GAT_Layer_Num
                                     ).to(device)
                    '''
                    model = OptimizedGATClassifier(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM,
                                                   ParameterConfig.HEAD_NUM, num_classes, db.embeddings_matrix,
                                                   node_vec_stg=node_vec_stg, device=device,
                                                   layer_num=ParameterConfig.GAT_Layer_Num,
                                                   feat_drop=ParameterConfig.GAT_FEAT_DP_RATE,
                                                   attn_drop=ParameterConfig.GAT_ATT_DP_RATE).to(device)
                    '''
            elif model_type == 'OPT_GCN':
                if node_vec_stg == 'mean':
                    model = OptimizedGCNClassifier(ParameterConfig.EMBEDDING_DIM, ParameterConfig.GCN_HIDDEN_DIM,
                                                   num_classes, db.embeddings_matrix, device=device,
                                                   layer_num=ParameterConfig.GCN_Layer_Num,
                                                   dp_rate=ParameterConfig.GCN_DP_RATE).to(device)
                else:
                    model = sumModel(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM,
                                                   num_classes,embeddings_matrixcnn, db, node_vec_stg=node_vec_stg,
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
            '''
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model2.parameters(), lr=ParameterConfig.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=ParameterConfig.lr_decay)

            # train the model
            print('training the {}-{} model'.format(model_type, node_vec_stg))
            start = time.time()
            model_train(fig_prefix, trainset, model1, model2, loss_func, testset)
            model_stats, history = model_train(fig_prefix, db_train, model2, loss_func, optimizer, scheduler, testset)

            plot_train_validation_acc_loss(fig_prefix, history)
            end = time.time()
            print('neural-net training takes: %s seconds' % (end - start))
            # model is got and stored
            # th.save(model, model_store_path)

            # step 3: evaluate the trained model on the test data
            print('evaluating on the test-set the trained {}-{} model'.format(model_type, node_vec_stg))
            # model = init_model.load_state_dict(th.load(model_stats)).to(device)
            model2.load_state_dict(th.load(model_stats))
            model2.eval()
            preds, ground_truth = model_evaluate(model2, testset, loss_func, fig_prefix)
            '''
            #loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.5, 1]), reduction="mean").to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=ParameterConfig.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=ParameterConfig.lr_decay)
            # train the model
            print('training the {}-{} model'.format(model_type, node_vec_stg))
            start = time.time()
            model_stats, history = model_train(fig_prefix, trainset, model, loss_func, optimizer, scheduler, testset)
            plot_train_validation_acc_loss(fig_prefix, history)
            end = time.time()
            print('neural-net training takes: %s seconds' % (end - start))
            # model is got and stored
            # th.save(model, model_store_path)

            # step 3: evaluate the trained model on the test data
            print('evaluating on the test-set the trained {}-{} model'.format(model_type, node_vec_stg))
            # model = init_model.load_state_dict(th.load(model_stats)).to(device)
            model.load_state_dict(th.load(model_stats))
            model.eval()
            preds, ground_truth = model_evaluate(model, testset, loss_func, fig_prefix)
            fpr, tpr,roc_auc = metric_predictions(preds ,ground_truth, fig_prefix)
            th.cuda.empty_cache()
            return  fpr, tpr,roc_auc



def lvpart():
    print("ast+acfg+token")
    corpus_base_path = './GNN/testcfg/input/'
    dic_base_path = './GNN/vec26/'
    model_store_base_path = './GNN/testcfg/trained/'
    unique_fun_base_path = './GNN/UniqueFunList/'
    nd_stg_type = '1'
    md_type = '1'
    black_list = []


    # do some hyper-parameter configuration

    # more functions are considered identical and got removed as the stg varies from 'fine' to 'medium' to 'coarse'
    # That is, the "coarse" stg should indicates the most conservative and lower bound performance of trained models
    # 'Intact' means do not remove any functions
    # unique_fun_stgs = ['coarse', 'Intact']
    unique_fun_stgs = ['coarse']
    ins_abs_stgs = ['coarse']  # instruction abstraction strategies
    # ins_abs_stgs = ['coarse', 'medium', 'fine']     # instruction abstraction strategies

    # construct the label map

    label_maps = {
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
        "10": 10,
        "11": 11,
        "12": 12,
        "13": 13,
        "14": 14,
        "15": 15,
        "16": 16,
        "17": 17,
        "18": 18,
        "19": 19,
        "20": 20,
        "21": 21,
        "22": 22,
        "23": 23,
        "24": 24,
        "25": 25
    }
    #CLASS_NUMBER = len(label_maps)
    CLASS_NUMBER = 26
    if md_type == '2':
        model_types = ['OPT_GCN']
    else:
        model_types = ['OPT_GAT']

    if nd_stg_type == '0':
        node_vec_stg_list = ['mean']
    elif nd_stg_type == '1':
        ParameterConfig.BATCH_SIZE = 64
        node_vec_stg_list = ['TextCNN']
    elif nd_stg_type == '2':
        ParameterConfig.BATCH_SIZE = 32
        node_vec_stg_list = ['TextRNN']
    elif nd_stg_type == '3':
        ParameterConfig.BATCH_SIZE = 24
        # node_vec_stg_list = ['Transformer', 'DPCNN']
        node_vec_stg_list = ['DPCNN']
    else:
        ParameterConfig.BATCH_SIZE = 32
        node_vec_stg_list = ['TextRCNN', 'TextRNN_Att']

    # ParameterConfig.BATCH_SIZE = 128
    hyper_log_prefix = model_types[0] + '-Family'
    ParameterConfig.log_config(hyper_log_prefix)

    func_rep_stg = 'CFG'
    for i in range(len(ins_abs_stgs)):
        # consisted of the 'function representation strategy # instruction abstraction strategy'
        # For example, 'CFG#coarse.csv' specifies the function representation strategy is 'CFG' and
        # the 'instruction abstraction strategy' is 'coarse'.
        #path_stg_suffix = func_rep_stg + '#' + ins_abs_stgs[i] + ".csv"
        path_stg_suffix = func_rep_stg + '#' + ins_abs_stgs[i]
        for unique_fun_stg in unique_fun_stgs:
            # form: 'the task # the instruction abstraction strategy # the unique function extraction strategy #
            # the compiler family to be analyzed on'
            # such as 'cp-optimization#fine#coarse#clang'
            model_name = model_types[0] + '-Family#' + func_rep_stg + '#' + \
                         ins_abs_stgs[i] + '#' + unique_fun_stg + '.pkl'
            print('-------------------' + model_name + '-------------------')
            corpus_path = corpus_base_path + ins_abs_stgs[i]
            dic_file_path = dic_base_path + 'ins2vec_' + ins_abs_stgs[i] + '.dic'
            model_store_path = model_store_base_path + model_name
            unique_fun_path = unique_fun_base_path + 'WHOLE#' + unique_fun_stg + '.csv'
            run_whole_procedure(model_name.replace('.pkl', ''), corpus_path, dic_file_path, path_stg_suffix,
                                unique_fun_path, CLASS_NUMBER,
                                label_maps, task_str='family', model_types=model_types,
                                node_vec_stg_list=node_vec_stg_list, black_list=black_list)


if __name__ == '__main__':
    '''
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # data_location = sys.argv[1]
    # nd_stg_type = sys.argv[2]
    # md_type = sys.argv[3]
    #
    # if data_location == '1':
    #     # restricted dataset on IST GPU Server
    #     corpus_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/RestrictedDB-CFG/'
    #     dic_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseII-ins2vec/'
    #     model_store_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/Trained-Models/'
    #     unique_fun_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/UniqueFunList-Strict/'
    # elif data_location == '2':
    #     # on IST GPU Server
    #     corpus_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseI-CFGs/'
    #     dic_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseII-ins2vec/'
    #     model_store_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/Trained-Models/'
    #     unique_fun_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/UniqueFunList-Strict/'
    # elif data_location == '3':
    #     # on GPU Server 2
    #     corpus_base_path = '/data/zpt5059/compilerprovenance/data/PhaseI-CFGs/'
    #     # corpus_base_path = '/data/zpt5059/compilerprovenance/data/RestrictedDB-CFG/'
    #     # corpus_base_path = '/data/zpt5059/compilerprovenance/data/testcfg/'
    #     dic_base_path = '/data/zpt5059/compilerprovenance/data/PhaseII-ins2vec/'
    #     model_store_base_path = '/data/zpt5059/compilerprovenance/data/Trained-Models/'
    #     unique_fun_base_path = '/data/zpt5059/compilerprovenance/data/UniqueFunList-Strict/'
    # else:
    #     # test on IST GPU Server
    #     corpus_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/testcfg/'
    #     dic_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseII-ins2vec/'
    #     model_store_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/Trained-Models/'
    #     unique_fun_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/UniqueFunList-Strict/'

    corpus_base_path = 'F:/tmp/compilerprovenance/testcfg/input/'
    dic_base_path = 'F:/tmp/compilerprovenance/PhaseII-ins2vec/'
    model_store_base_path = 'F:/tmp/compilerprovenance/testcfg/trained/'
    unique_fun_base_path = 'F:/tmp/compilerprovenance/UniqueFunList/'
    '''
    lvpart()