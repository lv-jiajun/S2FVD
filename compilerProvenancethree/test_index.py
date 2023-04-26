import os
import sys
import time

from torch.utils.data import DataLoader
import torch as th
import torch.nn as nn
import torch.optim as optim

lib_path = os.path.abspath(os.path.join('gnnmodels'))
sys.path.append(lib_path)
from ParameterConfig import ParameterConfig
from CFGDataset import CFGDataset
# from models import NaiveGCNClassifier, GATClassifier
from train_eval import collate, model_train, model_evaluate, plot_train_validation_acc_loss, metric_predictions
from gnnmodels.NaiveGCNClassifier import NaiveGCNClassifier
from gnnmodels.GATClassifier import GATClassifier

device = 'cuda' if th.cuda.is_available() else 'cpu'
device = th.device(device)
ParameterConfig.device = device
print(device)


def run_whole_procedure(fig_prefix_org, corpus_path, dic_file_path, func_rep_stg, unique_funs_file, num_classes,
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
    # step 1: load data
    print('loading cfg data...')
    db = CFGDataset(corpus_path, dic_file_path, func_rep_stg, unique_funs_file, label_maps, task_str=task_str,
                    black_list=black_list)
    # split into train and test set
    db_size = len(db)
    train_size = int((1 - ParameterConfig.dataset_split_ratio) * db_size)
    test_size = db_size - train_size
    trainset, testset = th.utils.data.random_split(db, [train_size, test_size])
    print(
        'dataset size: {:d} CFGs, {:d} for training and {:d} for testing'.format(db_size, len(trainset), len(testset)))

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
            else:  # use the basic GCN model
                if node_vec_stg == 'mean':
                    model = NaiveGCNClassifier(ParameterConfig.EMBEDDING_DIM, ParameterConfig.GCN_HIDDEN_DIM,
                                               num_classes, db.embeddings_matrix, device=device).to(device)
                else:
                    model = NaiveGCNClassifier(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM,
                                               num_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg,
                                               device=device).to(device)
            # if model_type == 'GAT':  # use the GAT model
            #     if node_vec_stg == 'mean':
            #         model = GATClassifier(ParameterConfig.EMBEDDING_DIM, ParameterConfig.GAT_HIDDEN_DIM,
            #                               ParameterConfig.HEAD_NUM,
            #                               num_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg, merge='mean',
            #                               device=device)
            #     else:
            #         model = GATClassifier(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM,
            #                               ParameterConfig.HEAD_NUM,
            #                               num_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg, merge='mean',
            #                               device=device)
            # else:  # use the basic GCN model
            #     if node_vec_stg == 'mean':
            #         model = NaiveGCNClassifier(ParameterConfig.EMBEDDING_DIM, ParameterConfig.GCN_HIDDEN_DIM,
            #                                    num_classes, db.embeddings_matrix, device=device)
            #     else:
            #         model = NaiveGCNClassifier(ParameterConfig.GCN_HIDDEN_DIM, ParameterConfig.GCN_HIDDEN_DIM,
            #                                    num_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg,
            #                                    device=device)
            # if th.cuda.device_count() > 1:
            #     model = nn.DataParallel(model)
            # model.to(device)

            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=ParameterConfig.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=ParameterConfig.lr_decay)
            # train the model
            print('training the {}-{} model'.format(model_type, node_vec_stg))
            start = time.time()
            model, history = model_train(trainset, model, loss_func, optimizer, scheduler, testset)
            plot_train_validation_acc_loss(fig_prefix, history)
            end = time.time()
            print('neural-net training takes: %s seconds' % (end - start))
            # model is got and stored
            # th.save(model, model_store_path)

            # step 3: evaluate the trained model on the test data
            print('evaluating on the test-set the trained {}-{} model'.format(model_type, node_vec_stg))
            preds, ground_truth = model_evaluate(model, testset, loss_func, fig_prefix)
            metric_predictions(preds, ground_truth, fig_prefix)

            th.cuda.empty_cache()


if __name__ == '__main__':
    data_location = sys.argv[1]
    nd_stg_type = sys.argv[2]

    if data_location == '1':
        # restricted dataset on IST GPU Server
        corpus_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/RestrictedDB-CFG/'
        dic_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseII-ins2vec/'
        model_store_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/Trained-Models/'
        unique_fun_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/UniqueFunList-Strict/'
    elif data_location == '2':
        # on IST GPU Server
        corpus_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseI-CFGs/'
        dic_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseII-ins2vec/'
        model_store_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/Trained-Models/'
        unique_fun_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/UniqueFunList-Strict/'
    elif data_location == '3':
        # on GPU Server 2
        # corpus_base_path = '/data/zpt5059/compilerprovenance/data/PhaseI-CFGs/'
        # corpus_base_path = '/data/zpt5059/compilerprovenance/data/RestrictedDB-CFG/'
        corpus_base_path = '/data/zpt5059/compilerprovenance/data/testcfg/'
        dic_base_path = '/data/zpt5059/compilerprovenance/data/PhaseII-ins2vec/'
        model_store_base_path = '/data/zpt5059/compilerprovenance/data/Trained-Models/'
        unique_fun_base_path = '/data/zpt5059/compilerprovenance/data/UniqueFunList-Strict/'
    else:
        # test on IST GPU Server
        corpus_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/testcfg/'
        dic_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/PhaseII-ins2vec/'
        model_store_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/Trained-Models/'
        unique_fun_base_path = '/dgxhome/zpt5059/projects/compilerprovenance/data/UniqueFunList-Strict/'

    # corpus_base_path = 'F:/tmp/compilerprovenance/testcfg/input/'
    # dic_base_path = 'F:/tmp/compilerprovenance/PhaseII-ins2vec/'
    # model_store_base_path = 'F:/tmp/compilerprovenance/testcfg/trained/'
    # unique_fun_base_path = 'F:/tmp/compilerprovenance/UniqueFunList/'
    nd_stg_type = '2'

    # black_list = ['valgrind3.15']
    # black_list = ['valgrind3.15', 'gcc-4.8', 'gcc-4.9', 'gcc-5', 'gcc-7']
    black_list = []

    # do some hyper-parameter configuration
    ParameterConfig.EPOCHES = 1
    ParameterConfig.BATCH_SIZE = 32

    # more functions are considered identical and got removed as the stg varies from 'fine' to 'medium' to 'coarse'
    # That is, the "coarse" stg should indicates the most conservative and lower bound performance of trained models
    # 'Intact' means do not remove any functions
    # unique_fun_stgs = ['coarse', 'Intact']
    unique_fun_stgs = ['Intact']
    ins_abs_stgs = ['coarse']  # instruction abstraction strategies
    cp_families = ['clang', 'gcc', 'icc']
    cp_optimization_options = ['O0', 'O1', 'O2', 'O3']

    # construct the label map
    label_maps = {}
    i = 0
    label_count = 0
    for family in cp_families:
        for opt in cp_optimization_options:
            label_maps[family + "-" + opt] = i
            if opt == 'O2':
                continue
            else:
                i += 1
                label_count += 1
    print(label_count)
    CLASS_NUMBER = label_count
    hyper_log_prefix = 'test-GCN-FamilyandTripleOptimization'
    # hyper_log_prefix = 'binutils2.32-GCN-FamilyandTripleOptimization'
    ParameterConfig.log_config(hyper_log_prefix)

    model_types = ['GCN']
    if nd_stg_type == '1':
        node_vec_stg_list = ['mean']
    elif nd_stg_type == '2':
        node_vec_stg_list = ['TextCNN']
    elif nd_stg_type == '3':
        node_vec_stg_list = ['Transformer', 'DPCNN']
    else:
        node_vec_stg_list = ['TextRCNN', 'TextRNN_Att']
    # node_vec_stg_list = []
    # node_vec_stg_list.append(specified_node_stg)
    func_rep_stg = 'CFG'
    for i in range(len(ins_abs_stgs)):
        # consisted of the 'function representation strategy # instruction abstraction strategy'
        # For example, 'CFG#coarse.csv' specifies the function representation strategy is 'CFG' and
        # the 'instruction abstraction strategy' is 'coarse'.
        path_stg_suffix = func_rep_stg + '#' + ins_abs_stgs[i] + ".csv"
        for unique_fun_stg in unique_fun_stgs:
            # form: 'the task # the instruction abstraction strategy # the unique function extraction strategy #
            # the compiler family to be analyzed on'
            # such as 'cp-optimization#fine#coarse#clang'
            model_name = 'test-GCN-FamilyandTripleOptimization#' + func_rep_stg + '#' + \
                         ins_abs_stgs[
                             i] + '#' + unique_fun_stg + '.pkl'
            # model_name = 'binutils2.32-GCN-FamilyandTripleOptimization#' + func_rep_stg + '#' + ins_abs_stgs[
            #     i] + '#' + unique_fun_stg + '.pkl'
            print('-------------------' + model_name + '-------------------')
            corpus_path = corpus_base_path + ins_abs_stgs[i]
            dic_file_path = dic_base_path + 'ins2vec_' + ins_abs_stgs[i] + '.dic'
            model_store_path = model_store_base_path + model_name
            unique_fun_path = unique_fun_base_path + 'WHOLE#' + unique_fun_stg + '.csv'
            run_whole_procedure(model_name.replace('.pkl', ''), corpus_path, dic_file_path, path_stg_suffix,
                                unique_fun_path, CLASS_NUMBER,
                                label_maps, task_str='family-tripleopt', model_types=model_types,
                                node_vec_stg_list=node_vec_stg_list, black_list=black_list)

