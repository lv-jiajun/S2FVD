import numpy as np


class ParameterConfig(object):
    """
     Some Configurations
     """
    # 设置一些参数
    # GNN Parameters
   # GCN_HIDDEN_DIM = 128    # hidden dimension of node state
    GCN_HIDDEN_DIM = 192  # hidden dimension of node state
    GCN_Layer_Num = 1   # the number of hidden layers
    GCN_DP_RATE = 0.

    # GAT
   # GAT_HIDDEN_DIM = 128    # hidden dimension of node state
    GAT_HIDDEN_DIM = 192  # hidden dimension of node state
    HEAD_NUM = 5
    GAT_Layer_Num = 1
    GAT_FEAT_DP_RATE = 0.
    GAT_ATT_DP_RATE = 0.

    # General
    device = None

    EMBEDDING_DIM = 100  # dimension of pre-trained word vector
    MAX_SEQUENCE_LENGTH = 20  # max length of a sentence (basic block)
    EPOCHES = 50  # 设置轮数
    BATCH_SIZE = 64  # 最好是2的倍数
    dataset_split_ratio = 0.2


    OCCUPY_ALL = False  # occupy all GPU or not
    PRINT_PER_BATCH = 100  # print result every xxx batches
    PRE_TRAINING = True  # use vectors trained by word2vec or not
    SEED = np.random.seed()
    lr = 2e-3 # learning rate
    lr_decay = 0.9  # learning rate decay
    # lr_decay = 0.1  # learning rate decay
    clip = 1.0  # gradient clipping threshold
    l2_reg_lambda = 0.01  # l2 regularization lambda

    # TextCNN-related
    NUM_FILTERS = 128  # number of convolution kernel
    FILTER_SIZES = [2, 3, 4]  # size of convolution kernel
    DROP_OUT = 0.5 # drop out rate

    # other parameters
    CFG_MIN_EDGE_NUM = 1

    PIN_MEM = False
    NUM_WORKERS = 4

    def log_config(prefix):
        with open(prefix + '#config', 'w') as f:
            f.write('EMBEDDING_DIM =' + str(ParameterConfig.EMBEDDING_DIM) + '\n')
            f.write('MAX_SEQUENCE_LENGTH=' + str(ParameterConfig.MAX_SEQUENCE_LENGTH) + '\n')
            f.write('GCN_HIDDEN_DIM='+str(ParameterConfig.GCN_HIDDEN_DIM) + '\n')
            f.write('GAT_HIDDEN_DIM='+str(ParameterConfig.GAT_HIDDEN_DIM) + '\n')
            f.write('HEAD_NUM='+str(ParameterConfig.HEAD_NUM) + '\n')
            f.write('CFG_MIN_EDGE_NUM='+str(ParameterConfig.CFG_MIN_EDGE_NUM) + '\n')

            f.write('NUM_FILTERS=' + str(ParameterConfig.NUM_FILTERS) + '\n')
            f.write('FILTER_SIZES=' + str(ParameterConfig.FILTER_SIZES) + '\n')
            f.write('DROP_OUT=' + str(ParameterConfig.DROP_OUT) + '\n')
            f.write('BATCH_SIZE=' + str(ParameterConfig.BATCH_SIZE) + '\n')
            f.write('EPOCHES=' + str(ParameterConfig.EPOCHES) + '\n')
            f.write('LEARNING_RATE=' + str(ParameterConfig.lr) + '\n')
            f.write('DATASET_SPLIT_RATIO=' + str(ParameterConfig.dataset_split_ratio) + '\n')
            f.write('HIDDEN_DIM=' + str(ParameterConfig.HEAD_NUM) + '\n')
            f.write('HEAD_NUM=' + str(ParameterConfig.HEAD_NUM) + '\n')
            f.close()