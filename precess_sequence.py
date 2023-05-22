"""
Interface to VulDeePecker project
"""
import sys
import os
import pandas
from TextCp import TextCnn
from vectorize_gadget import GadgetVectorizer
from gensim.models import Word2Vec, FastText



def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        flag = 0
        count = 0
        sum = 0
        for line in file:
            stripped = line.strip()  # 去除首尾的空格
            if not stripped:  # 判断是否是空，如果是则continue
                continue
            gadget = stripped[:-2]
            gadget_val1 = stripped[-1]
            gadget_val2 = stripped[-2]
            if (gadget_val1.startswith("0")) and gadget_val2.startswith("#"):
                gadget_val1 = "0"
            elif (gadget_val1.startswith("1")) and gadget_val2.startswith("#"):
                gadget_val1 = "1"
            elif (gadget_val1.startswith("2")) and gadget_val2.startswith("#"):
                gadget_val1 = "2"
            elif (gadget_val1.startswith("3")) and gadget_val2.startswith("#"):
                gadget_val1 = "3"
            elif (gadget_val1.startswith("4")) and gadget_val2.startswith("#"):
                gadget_val1 = "4"
            elif (gadget_val1.startswith("5")) and gadget_val2.startswith("#"):
                gadget_val1 = "5"
            elif (gadget_val1.startswith("6")) and gadget_val2.startswith("#"):
                gadget_val1 = "6"
            elif (gadget_val1.startswith("7")) and gadget_val2.startswith("#"):
                gadget_val1 = "7"
            elif (gadget_val1.startswith("8")) and gadget_val2.startswith("#"):
                gadget_val1 = "8"
            elif (gadget_val1.startswith("9")) and gadget_val2.startswith("#"):
                gadget_val1 = "9"
            elif (gadget_val1.startswith("0")) and gadget_val2.startswith("1"):
                gadget_val1 = "10"
            elif (gadget_val1.startswith("1")) and gadget_val2.startswith("1"):
                gadget_val1 = "11"
            elif (gadget_val1.startswith("2")) and gadget_val2.startswith("1"):
                gadget_val1 = "12"
            elif (gadget_val1.startswith("3")) and gadget_val2.startswith("1"):
                gadget_val1 = "13"
            elif (gadget_val1.startswith("4")) and gadget_val2.startswith("1"):
                gadget_val1 = "14"
            elif (gadget_val1.startswith("5")) and gadget_val2.startswith("1"):
                gadget_val1 = "15"
            elif (gadget_val1.startswith("6")) and gadget_val2.startswith("1"):
                gadget_val1 = "16"
            elif (gadget_val1.startswith("7")) and gadget_val2.startswith("1"):
                gadget_val1 = "17"
            elif (gadget_val1.startswith("8")) and gadget_val2.startswith("1"):
                gadget_val1 = "18"
            elif (gadget_val1.startswith("9")) and gadget_val2.startswith("1"):
                gadget_val1 = "19"
            elif (gadget_val1.startswith("0")) and gadget_val2.startswith("2"):
                gadget_val1 = "20"
            elif (gadget_val1.startswith("1")) and gadget_val2.startswith("2"):
                gadget_val1 = "21"
            elif (gadget_val1.startswith("2")) and gadget_val2.startswith("2"):
                gadget_val1 = "22"
            elif (gadget_val1.startswith("3")) and gadget_val2.startswith("2"):
                gadget_val1 = "23"
            elif (gadget_val1.startswith("4")) and gadget_val2.startswith("2"):
                gadget_val1 = "24"
            elif (gadget_val1.startswith("5")) and gadget_val2.startswith("2"):
                gadget_val1 = "25"
            yield gadget, gadget_val1


def get_vectors_df(filename, vector_length=100):
    gadgets = []
    count = 0
    vectorizer = GadgetVectorizer(vector_length)
    for gadget, val1 in parse_file(filename):
        count += 1
        # print(gadget)
        print("Collecting gadgets...", count, end="\r")
        gadget = vectorizer.add_gadget(gadget)  # 判断是前向切片还是后向切片
        row = {"gadget": gadget, "val1": val1}
        gadgets.append(row)

    df = pandas.DataFrame(gadgets)
    outputpath = '/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/tokens/lvtoken2.csv'
    df.to_csv(outputpath, sep=',', index=True, header=True)
    return df


class MyCorpus(object):
    def __init__(self, df, suffix):
        self.df = df
        self.suffix = suffix

    def __iter__(self):
        for i in range(0, len(self.df)):
            for line in self.df.iloc[i]['gadget']:
                yield line.split(' ')


def trainWord2Vec(df, dic_file_path, suffix, save_whole_model=True):
    """
    obtain a phaseII dictionary with skip-gram model
    :param corpus_path:
    :param dic_file_path:
    :param suffix:
    :param save_whole_model: default True, save the whole model. otherwise just save the standalone keyed vectors
    :return:
    """
    texts = MyCorpus(df, suffix)
    model = Word2Vec(texts, size=100, window=5, min_count=3, workers=96, sg=1)
    # model = FastText(texts, size=100, window=5, min_count=5,  sg=1)
    if save_whole_model:
        model.save(dic_file_path)
        model.wv.save_word2vec_format('/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/tokens/word2vec.vector')
    else:
        model.wv.save_word2vec_format(dic_file_path, binary=False)


def precess_sequences(filename, vec):
    parse_file(filename)
    vector_length = 50
    df = get_vectors_df(filename, vector_length)
    dic_file_path = '/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/tokens/ins2vec_coarse.dic'
    suffix = 'lvtoken2.csv'
    if not os.path.exists(dic_file_path):
        trainWord2Vec(df, dic_file_path, suffix)
    texts, embeddings_matrix, fig_prefix = TextCnn(df, vec)
    return texts, embeddings_matrix, fig_prefix


