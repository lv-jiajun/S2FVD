import pandas as pd
from sklearn.utils import shuffle

from step0_preprocess import normalization


def cannot_gen_cfg():
    with open("/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/no_train.txt", 'r') as f:
        file = f.readlines()
        path = file[0].split("0-cfg.dot")
        names = [p.split("/")[-2] for p in path if len(p) > 3]

        # print(len([name.split("_")[0] for name in names]))
        # return [int(name.split("_")[0]) for name in names]
        return names


def filter_no_cfg():
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


    no_cfg = cannot_gen_cfg()
    # train_n = train.drop(no_cfg)
    for no in no_cfg:
        id_label = no.split("_")
        train = train[train.id != int(id_label[0])]

        # train.drop(train[(train['id'] == int(id_label[0])) and (train['label'] == int(id_label[1]))].index, inplace=True)

    with open("/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/tokens/train.txt", 'w') as f:
        for i, file in enumerate(train.values):
            label = str(file[2])
            # label = file[1]
            code_text = file[1]
            f.write(code_text + "\t#" + label + "\n")
            f.flush()
    return train

    # df_clear = df.drop(df[df['x'] < 0.01].index)
    # 也可以使用多个条件
    # df_clear = df.drop(df[(df['x'] < 0.01) | (df['x'] > 10)].index)  # 删除x小于0.01或大于10的行


if __name__ == '__main__':
    filter_no_cfg()
