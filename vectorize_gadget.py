import re
import sys

import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
import numpy

# Sets for operators
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', 
    '!~', '<<', '>>', '<=', '>=', 
    '==', '!=', '&&', '||', '+=', 
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
operators1 = { 
    '(', ')', '[', ']', '.', 
    '+', '-', '*', '&', '/', 
    '%', '<', '>', '^', '|', 
    '=', ',', '?', ':' , ';',
    '{', '}'
    }

"""
Functionality to train Word2Vec model and vectorize gadgets
Buffers list of tokenized gadgets in memory
Trains Word2Vec model using list of tokenized gadgets
Uses trained model embeddings to create 2D gadget vectors
训练Word2Vec模型和矢量化小工具的功能
在内存中缓冲标记化小工具的列表
使用标记化小工具列表训练Word2Vec模型
使用训练有素的模型嵌入来创建2D小工具向量
"""
class GadgetVectorizer:

    def __init__(self, vector_length):
        self.gadgets = []
        self.vector_length = vector_length
        self.forward_slices = 0
        self.backward_slices = 0

    """
    Takes a line of C++ code (string) as input
    Tokenizes C++ code (breaks down into identifier, variables, keywords, operators)
    Returns a list of tokens, preserving order in which they appear
    将一行C ++代码（字符串）作为输入
     标记化C ++代码（分为标识符，变量，关键字，运算符）
     返回令牌列表，保留它们出现的顺序
    """
    @staticmethod
    def tokenize(line):
        tmp, w = [], []
        i = 0
        while i < len(line):
            # Ignore spaces and combine previously collected chars to form words
            if line[i] == ' ':
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            # Check operators and append to final list
            elif line[i:i+3] in operators3:
                tmp.append(''.join(w))
                tmp.append(line[i:i+3])
                w = []
                i += 3
            elif line[i:i+2] in operators2:
                tmp.append(''.join(w))
                tmp.append(line[i:i+2])
                w = []
                i += 2
            elif line[i] in operators1:
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            # Character appended to word list
            else:
                w.append(line[i])
                i += 1
        # Filter out irrelevant strings
        res = list(filter(lambda c: c != '', tmp))
        return list(filter(lambda c: c != ' ', res))

    """
    Tokenize entire gadget
    Tokenize each line and concatenate to one long list
    标记整个小工具
     标记每一行并连接到一个长列表
    """
    @staticmethod
    def tokenize_gadget(gadget):
        tokenized = []
        function_regex = re.compile('FUN(\d)+')
        backwards_slice = False
        '''
        for line in gadget:
            tokens = GadgetVectorizer.tokenize(line)
            tokenized += tokens
        '''
        tokens = GadgetVectorizer.tokenize(gadget)
        tokenized += tokens
           # if len(list(filter(function_regex.match, tokens))) > 0:
              #  backwards_slice = True
          #  else:
           #     backwards_slice = False
        return tokenized, backwards_slice

    """
    Add input gadget to model
    Tokenize gadget and buffer it to list
    将输入小工具添加到模型
    标记化小工具并将其缓冲以列出
    """
    def add_gadget(self, gadget):
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(gadget)
        self.gadgets.append(tokenized_gadget)
        return tokenized_gadget

       # if backwards_slice:
         #   self.backward_slices += 1
       # else:
         #   self.forward_slices += 1

    """
    Uses Word2Vec to create a vector for each gadget
    Gets a vector for the gadget by combining token embeddings
    Number of tokens used is min of number_of_tokens and 50
    使用Word2Vec为每个小工具创建一个向量
     通过组合标记嵌入来获取小工具的向量
     使用的令牌数量是number_of_tokens和50的最小值
    """
    '''
        def vectorize(self, gadget):
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(gadget)
        vectors = numpy.zeros(shape=(50, self.vector_length))
        if backwards_slice:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[50 - 1 - i] = self.embeddings[tokenized_gadget[len(tokenized_gadget) - 1 - i]]
        else:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[i] = self.embeddings[tokenized_gadget[i]]
        return vectors
        '''

    """
    Done adding gadgets, now train Word2Vec model
    Only keep list of embeddings, delete model and list of gadgets
    完成添加小工具，现在可以训练Word2Vec模型
     只保留嵌入清单，删除模型和小工具清单
    """
    '''
    def train_model(self):
        # Set min_count to 1 to prevent out-of-vocabulary errors
        model = Word2Vec(self.gadgets, min_count=1, size=self.vector_length, sg=1)
        self.embeddings = model.wv
        del model
        del self.gadgets
     '''