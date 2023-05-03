"""A mini synthetic dataset for graph classification benchmark."""
import math
import os
import sys

import dgl
import networkx as nx
import numpy as np
import torch as th

from dgl import DGLGraph
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
from ParameterConfig import ParameterConfig

__all__ = ['CFGDataset']

class CFGDataset(object):
    """The dataset class.

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Parameters
    ----------
    unique_fun_file: str
        path of the file storing all the unique functions
    func_rep_stg:
        can be 'CFG#coarse.csv', 'CFG#medium.csv' or 'CFG#fine.csv'
    label_maps: dict
        a map consisting of label-str:int-val
    task_str: str
        determines how to generate the labels, could be 'family', 'version', 'family-binaryopt', 'family-tripleopt',
        'family-opt', 'opt', 'binaryopt' and 'tripleopt'
    family: str
        determines which compiler family to focus on, can leave unspecified or 'gcc' or 'clang'
    version_list: list
        which compiler versions to be analyzed
    """
    def __init__(self, corpus_path, dic_file_path, func_rep_stg, unique_funs_file, label_maps=None, task_str=None,
                 family=None, version_list=None, black_list=[]):
        super(CFGDataset, self).__init__()
        self.corpus_path = corpus_path
        self.func_rep_stg = func_rep_stg
        self.unique_funs_file = unique_funs_file    #type: str
        self.dic_file_path = dic_file_path
        self.label_maps = label_maps
        self.version_list = version_list
        self.task_str = task_str
        self.family = family
        self.black_list = black_list

        self.graphs = []
        self.labels = []
        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]

    @property
    def num_classes(self):
        """Number of classes."""
        class_set = set(self.labels)
        return len(class_set)

    # @num_classes.setter
    # def num_classes(self, num_classes):
    #     self.num_classes = num_classes

    # @property
    # def word_index(self):
    #     return self.word_index

    # @word_index.setter
    # def word_index(self, wd_idx):
    #     self.word_index = wd_idx
    #
    # @property
    # def embeddings_matrix(self):
    #     return self.embeddings_matrix
    #
    # @embeddings_matrix.setter
    # def embeddings_matrix(self, emb_matrix):
    #     self.embeddings_matrix = emb_matrix

    def _generate(self):
        self.word_index, self.embeddings_matrix = self.construct_ins_embedding()

      #  if self.unique_funs_file.endswith('Intact.csv'):
        texts, cfgs, labels = self.prepare_data(self.corpus_path, self.func_rep_stg, self.word_index, self.label_maps, self.task_str, self.family, self.version_list)
     #   else:
        #texts, cfgs, labels = self.prepare_data_with_unique(self.corpus_path, self.func_rep_stg, self.unique_funs_file, self.word_index, self.label_maps, self.task_str, self.family, self.version_list)

        self.labels = labels
        self.generate_dglgraph(texts, cfgs)

        # preprocess
        # for i in range(len(self.graphs)):
        #     self.graphs[i] = DGLGraph(self.graphs[i])
        #     # add self edges
        #     nodes = self.graphs[i].nodes()
        #     self.graphs[i].add_edges(nodes, nodes)

    def construct_ins_embedding(self):
        """
        construct a vector matrix from pre-trained word2vec models
        :param ins2vec_model:
        :return:
        """
        ins2vec_model = Word2Vec.load(self.dic_file_path)
        vocab_size = len(ins2vec_model.wv.index2word)
        print('vocabulary size: {:d}'.format(vocab_size))

        index = 0
        # 存储所有的词语及其索引
        # 初始化 [word : index]
        word_index = {"PAD": index}
        # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于padding补零。
        # 行数为所有单词数+1；比如10000+1；列数为词向量“维度”，比如100。
        if ins2vec_model.vector_size != ParameterConfig.EMBEDDING_DIM:
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

        # with open('tmp.csv', 'w') as f:
        #     for key in word_index.keys():
        #         idx = word_index[key]
        #         f.write(key+','+str(idx)+'\n')


        return word_index, embeddings_matrix

    def is_in_black_list(self, fname):
        for blk in self.black_list:
            if blk in fname:
                return True
        return False

    def prepare_data_with_unique(self, data_dir, func_rep_stg, unique_funs_file, word_index, label_maps, task_str, family='', ver_list=[]):
        """
        Prepare data for compiler family identification task
        Data preparation for other tasks can be easily implemented with slight modifications to this method
        :param ver_list:
        :param family:
        :param task_str: determines how to generate the labels, could be 'family', 'version', 'family-binaryopt',
        'family-tripleopt', 'family-opt', 'opt', 'binaryopt', 'tripleopt', 'allsettings-binary', 'allsettings-triple' and 'family-version'
        :param data_dir:
        :param func_rep_stg: the function representation strategy, such as 'RSP#coarse.csv', 'WHOLE#fine.csv', 'EDGE#medium.csv'
        :param unique_funs_file: ignore functions not listed in unique_funs_file
        :return:
        """
        print("2333")
        # organize as a set the unique functions
        unique_funs = set()
        with open(unique_funs_file) as f:
            line = f.readline()
            while line:
                unique_funs.add(line)
                line = f.readline()
            f.close()

        fun_rep_file_list = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(func_rep_stg):
                    fun_rep_file_list.append(file)
        texts = []
        labels = []
        cfgs = []
        processed = 0
        for fname in fun_rep_file_list:
            if self.is_in_black_list(fname):
                continue

            processed += 1
            if processed % 100 == 0:
                print('{:.2f} percent files parsed......'.format(processed / len(fun_rep_file_list)*100))
            # print('Processing '+fname)
            fpath = os.path.join(data_dir, fname)
            compiler_setting_str = fname.split('#')[2]
            compiler_options = compiler_setting_str.split('-')
            compiler_family = compiler_options[0]
            compiler_version = str(compiler_options[1])
            compiler_opt_level = str(compiler_options[2])
            label_str = ''
            if task_str == 'family':
                label_str = compiler_family
            elif task_str == 'family-binaryopt' or task_str == 'family-tripleopt' or task_str == 'family-opt':
                label_str = compiler_family + '-' + compiler_opt_level
            elif task_str == 'version':
                if compiler_family != family:  # skip irrelevant files
                    continue
                if compiler_version not in ver_list:  # skip irrelevant files
                    continue
                label_str = compiler_version
            elif task_str == 'opt' or task_str == 'binaryopt' or task_str == 'tripleopt':
                if compiler_family != family:   # skip irrelevant files
                    continue
                label_str = compiler_opt_level
            elif task_str == 'allsettings-binary' or task_str == 'allsettings-triple':
                if compiler_version not in ver_list:  # skip irrelevant files
                    continue
                label_str = compiler_family + '-' + compiler_version + '-' + compiler_opt_level
            elif task_str == 'family-version':
                if compiler_version not in ver_list:  # skip irrelevant files
                    continue
                label_str = compiler_family + '-' + compiler_version

            with open(fpath) as f:
                lines = f.readlines()
                indices_set = []  # indices set representation for a function
                for line in lines:
                    line = line.strip()
                    if line.startswith('>>>'):
                        if line.startswith('>>>CFG'):
                            edges = line.split(' ')[1:]
                        if line.startswith('>>>Func'):
                            f_label = fname.replace(func_rep_stg, '') + '>' + line.split('&')[1] + '\n'
                            if f_label in unique_funs:
                                has_cfg = line.split('&')[-1]
                                if has_cfg != '-1':  # -1 indicates there exists no cfg for the function
                                    if len(edges) >= ParameterConfig.CFG_MIN_EDGE_NUM:
                                        texts.append(indices_set)
                                        labels.append(label_maps[label_str])
                                        cfgs.append(edges)
                            indices_set = []
                        continue
                    # 序号化文本，tokenizer句子，并返回每个句子所对应的词语索引
                    indices = []
                    for word in line.split(' '):
                        try:
                            indices.append(word_index[word])  # 把句子中的词语转化为index
                        except:
                            indices.append(word_index['UNKNOWN'])  # OOV词统一用'UNKNOWN'对应的向量表示
                    indices_set.append(indices)
                f.close()
        return texts, cfgs, labels

    def prepare_data(self, data_dir, func_rep_stg, word_index, label_maps, task_str, family='', ver_list=[]):
        """
        Prepare data for compiler family identification task
        Data preparation for other tasks can be easily implemented with slight modifications to this method
        :param ver_list:
        :param family:
        :param task_str: determines how to generate the labels, could be 'family', 'version', 'family-binaryopt',
        'family-tripleopt', 'family-opt', 'opt', 'binaryopt', 'tripleopt', 'allsettings-binary', 'allsettings-triple' and 'family-version'
        :param data_dir:
        :param func_rep_stg: the function representation strategy, such as 'RSP#coarse.csv', 'WHOLE#fine.csv', 'EDGE#medium.csv'
        :return:
        """
        sum = 0
        fun_rep_file_list = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(func_rep_stg+'.csv'):
                    sum = sum + 1
                    fun_rep_file_list.append(file)
                print(sum)
        texts = []
        labels = []
        cfgs = []
        processed = 0
        for fname in fun_rep_file_list:
            if self.is_in_black_list(fname):
                continue
            processed += 1
            if processed % 100 == 0:
                print('{:.2f} percent files parsed......'.format(processed / len(fun_rep_file_list) * 100))
            print('Processing '+fname)
            #fpath = os.path.join(data_dir+'/input/coarse', fname)
            fpath = os.path.join(data_dir, fname)
            '''
            compiler_setting_str = fname.split('#')[2]
            compiler_options = compiler_setting_str.split('-')
            compiler_family = compiler_options[0]
            compiler_version = str(compiler_options[1])
            compiler_opt_level = str(compiler_options[2])
            '''
            '''
            label_str = ''
            if task_str == 'family':
                label_str = compiler_family
            elif task_str == 'family-binaryopt' or task_str == 'family-tripleopt' or task_str == 'family-opt':
                label_str = compiler_family + '-' + compiler_opt_level
            elif task_str == 'version':
                if compiler_family != family:  # skip irrelevant files
                    continue
                if compiler_version not in ver_list:  # skip irrelevant files
                    continue
                label_str = compiler_version
            elif task_str == 'opt' or task_str == 'binaryopt' or task_str == 'tripleopt':
                if compiler_family != family:   # skip irrelevant files
                    continue
                label_str = compiler_opt_level
            elif task_str == 'allsettings-binary' or task_str == 'allsettings-triple':
                if compiler_version not in ver_list:  # skip irrelevant files
                    continue
                label_str = compiler_family + '-' + compiler_version + '-' + compiler_opt_level
            elif task_str == 'family-version':
                if compiler_version not in ver_list:  # skip irrelevant files
                    continue
                label_str = compiler_family + '-' + compiler_version
            '''
            count = 0
            count1 = 0
            with open(fpath) as f:
                lines = f.readlines()
                indices_set = []  # indices set representation for a function
                for line in lines:
                    line = line.strip()
                    if line.startswith('>>>'):
                        if line.startswith('>>>cfg&'):
                            edges = line.split(' ')[1:]
                            flag = line.split(' ')[0].split('&')[1]
                            count = count + 1
                        if line.startswith('>>>func&label&'):
                            lab = line.split('&')
                            label_str = lab[2]
                            count1 = count1 + 1
                            if(count!=count1):
                                print("butongdian:"+str(flag))
                           # has_cfg = line.split('&')[-1]
                            has_cfg = 1
                            if has_cfg != '-1':  # -1 indicates there exists no cfg for the function
                                if len(edges) >= ParameterConfig.CFG_MIN_EDGE_NUM:
                                    texts.append(indices_set)
                                    labels.append(label_maps[label_str])
                                    cfgs.append(edges)
                                else:
                                    print("weizhi"+str(flag))
                            indices_set = []
                        continue
                    # 序号化文本，tokenizer句子，并返回每个句子所对应的词语索引
                    indices = []
                    for word in line.split(' '):
                        try:
                            indices.append(word_index[word])  # 把句子中的词语转化为index
                        except:
                            indices.append(word_index['UNKNOWN'])  # OOV词统一用'UNKNOWN'对应的向量表示
                    indices_set.append(indices)
                f.close()
        print(count)
        print(count1)
        return texts, cfgs, labels

    def generate_dglgraph(self, texts, cfgs):
        '''
        organize all cfgs together with their nodes attributes into dglgraphs
        :param texts: a list consisting of all nodes attributes of the functions
        :param cfgs: a list consisting of all cfgs of all the functions
        :return:
        '''
        assert len(texts) == len(cfgs)
        f_num = len(cfgs)
        print('数目'+str(f_num))
        # Create the graph from a list of integer pairs.
        elist = []
        for i in range(f_num):
            # create a dgl graph for each cfg
            cfg = cfgs[i]
            for edg in cfg:
                nd_ids = edg.split('->')
                nd_ids = [int(j) for j in nd_ids]
                edg = tuple(nd_ids)
                elist.append(edg)
            dgl_graph = dgl.graph(elist)
            dgl_graph = dgl.add_self_loop(dgl_graph)
            elist = []

            # assign node attributes
            nodes_attrs = texts[i]
            # convert attribute list to torch tensor
            nodes_attrs = self.pad_nodes(nodes_attrs)
            dgl_graph.ndata['w'] = nodes_attrs
            # append to the graph list
            self.graphs.append(dgl_graph)
            # Visualize the graph.
            #nx.draw(dgl_graph.to_networkx(), with_labels=True)
            #plt.show()

    def pad_nodes(self, nodes_attrs):
        '''
        pad/truncate all sequences to the same length (specified by ParameterConfig.MAX_SEQUENCE_LENGTH)
        :param nodes_attrs:
        :return: torch tensor
        '''
        torches = []
        for node_atts in nodes_attrs:
            tmp_torch = th.tensor(node_atts)
            actual_len = len(tmp_torch)
            if actual_len > ParameterConfig.MAX_SEQUENCE_LENGTH:
                tmp_torch = tmp_torch[:ParameterConfig.MAX_SEQUENCE_LENGTH]
            else:
                tmp_torch = th.cat([tmp_torch, tmp_torch.new_zeros(ParameterConfig.MAX_SEQUENCE_LENGTH - actual_len)], 0)
            torches.append(tmp_torch)
        return th.stack(torches, 0)


if __name__ == '__main__':
    corpus_path = 'E:/图神经网络/testinputs/testcfg'
    dic_file_path = 'E:/图神经网络/testinputs/PhaseII-ins2vec/ins2vec_coarse.dic'
    unique_fun_stg = 'coarse'
    #unique_fun_stg = 'Intact'
    versions = {
        'clang': ['3.8.0', '5.0']
    }
    label_maps = {}
    i = 0
    # construct the label map
    for ver in versions['clang']:
        label_maps[ver] = i
        i += 1
    print(str(label_maps['5.0']))   #label_maps['5.0'] == 1   label_maps['3.8.0'] == 0
    db = CFGDataset(corpus_path, dic_file_path, unique_fun_stg, 2, label_maps, task_str='version', family='clang', version_list=versions['clang'])
    print(len(db))