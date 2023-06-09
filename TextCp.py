from gensim.models import Word2Vec
import torch
import numpy as np
dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""
Some Configurations
"""
# 设置一些参数
OCCUPY_ALL = False   # occupy all GPU or not
EMBEDDING_DIM = 100  # dimension of pre-trained word vector
MAX_SEQUENCE_LENGTH = 450   # max length of a sentence
CLASS_NUMBER = 26  # 设置分类数

NUM_FILTERS = 128  # number of convolution kernel
FILTER_SIZES = [2, 3, 4]  # size of convolution kernel
DROP_OUT = 0.3  # drop out rate

BATCH_SIZE = 128  # 最好是2的倍数
EPOCHES = 5  # 设置轮数
PRINT_PER_BATCH = 100  # print result every xxx batches

PRE_TRAINING = True  # use vectors trained by word2vec or not
dataset_split_ratio = 0.2

SEED = 7
SEED = np.random.seed(SEED)
lr = 1e-4  # learning rate
lr_decay = 0.9  # learning rate decay
clip = 6.0  # gradient clipping threshold
l2_reg_lambda = 0.01  # l2 regularization lambda


def log_config(prefix):
    with open(prefix + '#config', 'w') as f:
        f.write('EMBEDDING_DIM ='+str(EMBEDDING_DIM)+'\n')
        f.write('MAX_SEQUENCE_LENGTH='+str(MAX_SEQUENCE_LENGTH)+'\n')
        f.write('NUM_FILTERS='+str(NUM_FILTERS)+'\n')
        f.write('FILTER_SIZES='+str(FILTER_SIZES)+'\n')
        f.write('DROP_OUT='+str(DROP_OUT)+'\n')
        f.write('BATCH_SIZE='+str(BATCH_SIZE)+'\n')
        f.write('EPOCHES='+str(EPOCHES)+'\n')
        f.write('LEARNING_RATE='+str(lr)+'\n')
        f.write('DATASET_SPLIT_RATIO='+str(dataset_split_ratio)+'\n')
        f.close()

def construct_ins_embedding():
    """
    construct a vector matrix from pre-trained word2vec models
    :param ins2vec_model:
    :return:
    """
    vocab_size = len(ins2vec_model.wv.index2word)
    print(vocab_size)

    index = 0
    # 存储所有的词语及其索引
    # 初始化 [word : index]
    word_index = {"PAD": index}
    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于padding补零。
    # 行数为所有单词数+1；比如10000+1；列数为词向量“维度”，比如100。
    if ins2vec_model.vector_size != EMBEDDING_DIM:
        print("W2V vector dimension not equal to the configured dimension")
    embeddings_matrix = np.zeros((vocab_size + 2, ins2vec_model.vector_size))

    # 填充上述的字典和矩阵
    for word in ins2vec_model.wv.index2word:
        index = index + 1
        word_index[word] = index
        embeddings_matrix[index] = ins2vec_model.wv[word]

    # OOV词随机初始化为同一向量
    index = index + 1
    word_index["UNKNOWN"] = index
    embeddings_matrix[index] = np.random.rand(ins2vec_model.vector_size) / 10
    return word_index, embeddings_matrix


def prepare_data(df,veclength):
    """
    Prepare data for compiler family identification task
    Data preparation for other tasks can be easily implemented with slight modifications to this method
    :param data_dir:
    :param func_rep_stg: the function representation strategy, such as 'RSP#coarse.csv', 'WHOLE#fine.csv', 'EDGE#medium.csv'
    :return:
    """
    texts = []
    labels = []
    for i in range(0, len(df)):
        indices = []
        for word in df.iloc[i]['gadget']:
            word = word.rstrip()
            try:
                indices.append(word_index[word])  # 把句子中的词语转化为index
            except:
                indices.append(word_index['UNKNOWN'])
        indices = add_pad(indices, veclength, 0)
        texts.append(indices)
        if str("10") in df.iloc[i]['val1']:
            labels.append(10)
        elif str("11") in df.iloc[i]['val1']:
            labels.append(11)
        elif str("12") in df.iloc[i]['val1']:
            labels.append(12)
        elif str("13") in df.iloc[i]['val1']:
            labels.append(13)
        elif str("14") in df.iloc[i]['val1']:
            labels.append(14)
        elif str("15") in df.iloc[i]['val1']:
            labels.append(15)
        elif str("16") in df.iloc[i]['val1']:
            labels.append(16)
        elif str("17") in df.iloc[i]['val1']:
            labels.append(17)
        elif str("18") in df.iloc[i]['val1']:
            labels.append(18)
        elif str("19") in df.iloc[i]['val1']:
            labels.append(19)
        elif str("20") in df.iloc[i]['val1']:
            labels.append(20)
        elif str("21") in df.iloc[i]['val1']:
            labels.append(21)
        elif str("22") in df.iloc[i]['val1']:
            labels.append(22)
        elif str("23") in df.iloc[i]['val1']:
            labels.append(23)
        elif str("24") in df.iloc[i]['val1']:
            labels.append(24)
        elif str("25") in df.iloc[i]['val1']:
            labels.append(25)
        elif str("0") in df.iloc[i]['val1']:
            labels.append(0)
        elif str("1") in df.iloc[i]['val1']:
            labels.append(1)
        elif str("2") in df.iloc[i]['val1']:
            labels.append(2)
        elif str("3") in df.iloc[i]['val1']:
            labels.append(3)
        elif str("4") in df.iloc[i]['val1']:
            labels.append(4)
        elif str("5") in df.iloc[i]['val1']:
            labels.append(5)
        elif str("6") in df.iloc[i]['val1']:
            labels.append(6)
        elif str("7") in df.iloc[i]['val1']:
            labels.append(7)
        elif str("8") in df.iloc[i]['val1']:
            labels.append(8)
        elif str("9") in df.iloc[i]['val1']:
            labels.append(9)

    print('transforming labels to array')
    print('padding the sequences')
    return texts, labels
'''
def prepare_data(df):

    
    texts = []
    labels = []
    print("function is :"+str(len(df)))
    for i in range(0, len(df)):
        indices = []
        for word in df.iloc[i]['gadget']:
            word = word.rstrip()
           # for word in line.split(','):
            try:
                indices.append(word_index[word])  # 把句子中的词语转化为index
            except:
                indices.append(word_index['UNKNOWN'])
        indices = add_pad(indices, MAX_SEQUENCE_LENGTH, 0)
        texts.append(indices)
        
        if str("CWE-119  True") in df.iloc[i]['val1']:
            labels.append(1)
        elif str("CWE-120  True") in df.iloc[i]['val2']:
            labels.append(1)
        elif str("CWE-469  True") in df.iloc[i]['val3']:
            labels.append(1)
        elif str("CWE-476  True") in df.iloc[i]['val4']:
            labels.append(1)
        elif str("CWE-other  True") in df.iloc[i]['val5']:
            labels.append(1)
        else:
            labels.append(0)
        
        if str("0") in df.iloc[i]['val1']:
            labels.append(0)
        else:
            labels.append(1)
    print('transforming labels to array')
    #labels =to_categorical(np.asarray(labels), CLASS_NUMBER) # 将标签转换为数组形式
    # 使用keras的内置函数padding对齐句子，好处是输出numpy数组，不用自己转化了
    print('padding the sequences')
    #padded_data = sequence.pad_sequences(texts, maxlen=MAX_SEQUENCE_LENGTH)
    return texts, labels
 '''
def add_pad(text,max_seq,pad_tag):
    if(len(text)>max_seq):
        text = text[:max_seq]
    if (len(text)<=max_seq):
        text.extend([pad_tag]*(max_seq-len(text)))
    return text




def run_whole_procedure(fig_prefix,df,veclength):
    ins_abs_stgs = ['coarse']
    dic_base_path = '/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/tokens/'
    dic_file_path = dic_base_path + 'ins2vec_' + ins_abs_stgs[0] + '.dic'
    global ins2vec_model, word_index, x_test, y_test

    # step 1: load pre-trained ins2vec model
    print('loading pre-trained ins2vec model...')
    ins2vec_model = Word2Vec.load(dic_file_path)
    # step 2: construct ins2vec embeddings
    print('constructing pre-trained ins2vec embeddings...')
    word_index, embeddings_matrix = construct_ins_embedding()

    # step 3: prepare data
    texts, labels = prepare_data(df,veclength)
    texts, labels = torch.LongTensor(texts), torch.LongTensor(labels)
    return texts,embeddings_matrix,fig_prefix



def TextCnn(df,veclength):
    model_store_base_path = '/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/tokens/'
    unique_fun_stgs = ['coarse']

    ins_abs_stgs = ['coarse']  # instruction abstraction strategies

    global label_strs
    label_strs = ['true', 'false']
    hyper_log_prefix = 'family'
    log_config(hyper_log_prefix)

    for i in range(len(ins_abs_stgs)):
        # consisted of the 'function representation strategy # instruction abstraction strategy'
        # For example, 'WHOLE#coarse.csv' specifies the function representation strategy is 'WHOLE' and
        # the 'instruction abstraction strategy' is 'coarse'.
        # For the textCNN model defined in this file, the 'function representation strategy' should always be 'WHOLE'
        # path_stg_suffix = 'WHOLE#' + ins_abs_stgs[i] + ".csv"
        for unique_fun_stg in unique_fun_stgs:
            # form: 'the task # the instruction abstraction strategy # the unique function extraction strategy'
            # such as 'cp-family#fine#medium'
            model_name = 'cp-family#' + ins_abs_stgs[i] + '#' + unique_fun_stg + '.h5'
            global model_store_path
            model_store_path = model_store_base_path + model_name

            texts ,embeddings_matrix,fig_prefix = run_whole_procedure(model_name.replace('.h5', ''),df,veclength)
            return texts,embeddings_matrix,fig_prefix


