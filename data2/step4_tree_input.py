import argparse
import copy

import pandas as pd
import os
import tree_sitter
from sklearn.utils import shuffle
from tree_sitter import Language, Parser

from prepare_data import get_root_paths
from clean_gadget import clean_gadget
import re

def normalization(source):
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            code += line + ' '
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget([code])
        nor_code.append(code[0])
        print(code[0])
    return nor_code

def cannot_gen_cfg():
    with open("/data/bhtian2/win_linux_mapping/three-fusion/data2/ours/no_train.txt", 'r') as f:
        file = f.readlines()
        path = file[0].split("0-cfg.dot")
        names = [p.split("/")[-2] for p in path if len(p) > 3]

        # print(len([name.split("_")[0] for name in names]))
        # return [int(name.split("_")[0]) for name in names]
        return names


def parse_ast(source):
    # C_LANGUAGE = Language('build_languages/my-languages.so', 'c')
    CPP_LANGUAGE = Language('/data/bhtian2/win_linux_mapping/three-fusion/data2/build_languages/my-languages.so', 'cpp')
    # JAVA_LANGUAGE = Language('../build_languages/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)  # set the parser for certain language
    tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
    return tree


# args = parse_options()


class Pipeline:
    def __init__(self):
        self.train = None
        self.train_keep = None
        self.train_block = None
        self.dev = None
        self.dev_keep = None
        self.dev_block = None
        self.test = None
        self.test_keep = None
        self.test_block = None
        self.size = None
        self.w2v_path = None

    # parse source code
    def parse_source(self):
        # train = pd.read_csv(
        #     '/data/bhtian2/win_linux_mapping/devign_revealrecurrent/code-slicer/datasets/d2a/d2a_lbv1_function_train.csv',
        #     encoding='utf-8')
        # dev = pd.read_csv(
        #     '/data/bhtian2/win_linux_mapping/devign_revealrecurrent/code-slicer/datasets/d2a/d2a_lbv1_function_dev.csv',
        #     encoding='utf-8')
        # train = pd.concat([train, dev], axis=0)
        # train['code'] = normalization(train)

        # train = pd.read_pickle("/data/bhtian2/win_linux_mapping/three-fusion/data2/ours/ours.pkl")
        # listType = train['label'].unique()
        # data0 = train[train['label'].isin([listType[0]])]
        # data1 = train[train['label'].isin([listType[1]])]
        # data0 = data0[:13541]
        # data1 = data1[:11792]
        # train = shuffle(pd.concat([data0, data1]), random_state=42)
        # train.insert(0, 'id', range(len(train)), allow_duplicates=False)
        # print(train['label'].value_counts())

        train = pd.read_csv('/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/data2.csv', encoding='utf-8',
                            delimiter='#')

        print(train['label'].value_counts())
        train['code'] = normalization(train)

        # no_cfg = cannot_gen_cfg()
        # for no in no_cfg:
        #     id_label = no.split("_")
        #     train = train[train.id != int(id_label[0])]

        train['code'] = train['code'].apply(parse_ast)
        self.train = train
        self.train_keep = copy.deepcopy(train)


    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, size):
        self.size = size
        trees = self.train
        self.w2v_path = '/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/trees/node_w2v_' + str(size)
        # if not os.path.exists('pretrain/'+args.input):
        #     os.makedirs('pretrain/'+args.input)
        from prepare_data import get_sequences
        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            # collect all root-leaf paths
            # paths = []
            # get_root_paths(ast, paths, [])
            # add root to leaf path as corpus
            # paths.append(sequence)
            # return paths
            return sequence

        # train word2vec embedding if not exists
        if not os.path.exists(self.w2v_path):
            corpus = trees['code'].apply(trans_to_sequences)
            paths = []
            for path in corpus:
                path = [token.decode('utf-8') if type(token) is bytes else token for token in path]
                paths.append(path)
            corpus = paths
            # training word2vec model
            from gensim.models.word2vec import Word2Vec
            print('corpus size: ', len(corpus))
            w2v = Word2Vec(corpus, size=size, workers=96, sg=1, min_count=3)
            print('word2vec : ', w2v)
            w2v.save(self.w2v_path)

    # generate block sequences with index representations
    def generate_block_seqs(self, data):
        blocks_path = '/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/trees/train_block.pkl'

        from gensim.models.word2vec import Word2Vec
        word2vec = Word2Vec.load(self.w2v_path).wv
        vocab = word2vec.vocab
        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):

            if not isinstance(node, tree_sitter.Tree):
                token = node.type
                if len(node.children) == 0:
                    token = node.text
                children = node.children
            else:
                token = node.root_node.type
                if len(node.root_node.children) == 0:
                    token = node.root_node.text
                children = node.root_node.children

            if type(token) is bytes:
                token = token.decode('utf-8')

            result = [vocab[token].index if token in vocab else max_token]
            for child in children:
                result.append(tree_to_index(child))
            return result

        def tree_to_token(node):
            token = node.token
            if type(token) is bytes:
                token = token.decode('utf-8')
            result = [token]
            children = node.children
            for child in children:
                result.append(tree_to_token(child))
            return result

        def trans2seq(r):
            btree = tree_to_index(r)
            return btree

        trees = data
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(blocks_path)
        return trees

    # run for processing data_raw to train
    def run(self):
        print('parse source code...')
        self.parse_source()
        print('train word2vec model...')
        self.dictionary_and_embedding(size=128)
        print('generate block sequences...')
        self.train_block = self.generate_block_seqs(self.train_keep)


if __name__ == '__main__':
    ppl = Pipeline()
    ppl.run()
