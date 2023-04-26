import pycparser.c_ast
from pycparser import c_parser, c_ast
import pandas as pd
import os
import re
import sys
from gensim.models.word2vec import Word2Vec
import pickle
from tree import ASTNode, SingleNode
import numpy as np


def selectDef(node):
    for _, child in node.children():
        if isinstance(child, pycparser.c_ast.FuncDef):
            return child


def selectFunDecl(node):
    for _, child in node.children():
        if isinstance(child, pycparser.c_ast.FuncDecl):
            return child


def get_sequences(node, sequence):
    current = SingleNode(node)
    tokens = current.get_token()
    sequence.append(current.get_token())
    # print(type(node))
    for _, child in node.children():
        # if child.type is 'FunDef':
        get_sequences(child, sequence)
    if current.get_token().lower() == 'compound':
        sequence.append('End')


def get_blocks(node, block_seq):
    children = node.children()
    name = node.__class__.__name__
    if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
        block_seq.append(ASTNode(node))
        if name is not 'For':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i][1]
            if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
    elif name is 'Compound':
        block_seq.append(ASTNode(name))
        for _, child in node.children():
            if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
        block_seq.append(ASTNode('End'))
    else:
        for _, child in node.children():
            get_blocks(child, block_seq)


def trans_to_sequences(ast):
    sequence = []
    get_sequences(ast, sequence)
    return sequence
