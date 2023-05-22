import pandas as pd
import re

from sklearn.utils import shuffle

from clean_gadget import clean_gadget


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


def construct_ours():
    train = pd.read_pickle("/data/bhtian2/win_linux_mapping/three-fusion/data2/ours/ours.pkl")
    listType = train['label'].unique()
    data0 = train[train['label'].isin([listType[0]])]
    data1 = train[train['label'].isin([listType[1]])]
    data0 = data0[:13541]
    data1 = data1[:11792]
    train = shuffle(pd.concat([data0, data1]), random_state=42)
    train.insert(0, 'id', range(len(train)), allow_duplicates=False)
    print(train['label'].value_counts())

    for i, file in enumerate(train.values):
        file_name = str(file[0]) + "_" + str(file[1])
        code_text = file[2]
        path = "/data/bhtian2/win_linux_mapping/three-fusion/data2/ours/data/" + file_name + ".c"
        with open(path, 'w') as f:
            f.write(code_text)
            f.flush()


def construct_ours_26():
    train = pd.read_csv('/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/data2.csv', encoding='utf-8',
                        delimiter='#')

    print(train['label'].value_counts())
    train['code'] = normalization(train)
    for i, file in enumerate(train.values):
        file_name = str(file[0]) + "_" + str(file[2])
        code_text = file[1]
        path = "/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/data/" + file_name + ".c"
        with open(path, 'w') as f:
            f.write(code_text)
            f.flush()


def csv_gen_c_files():
    train = pd.read_csv(
        '/data/bhtian2/win_linux_mapping/devign_revealrecurrent/code-slicer/datasets/d2a/d2a_lbv1_function_train.csv',
        encoding='utf-8')
    dev = pd.read_csv(
        '/data/bhtian2/win_linux_mapping/devign_revealrecurrent/code-slicer/datasets/d2a/d2a_lbv1_function_dev.csv',
        encoding='utf-8')

    train = pd.concat([train, dev], axis=0)
    train['code'] = normalization(train)

    # train
    for i, file in enumerate(train.values):
        file_name = str(file[0]) + "_" + str(file[1])
        code_text = file[2]
        path = "/data/bhtian2/win_linux_mapping/three-fusion/data2/d2a/data/" + file_name + ".c"
        with open(path, 'w') as f:
            f.write(code_text)
            f.flush()


if __name__ == '__main__':
    construct_ours_26()
    # construct_ours()
    # csv_gen_c_files()
