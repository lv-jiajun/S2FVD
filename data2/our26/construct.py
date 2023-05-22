import pandas as pd
from sklearn.utils import shuffle


def add(source):
    labels = []
    for label in source['label']:
        labels.append(label + 1)
    return labels


def select0():
    train = pd.read_pickle('./ours26.pkl')
    listType = train['label'].unique()
    data0 = train[train['label'].isin([listType[0]])]
    with open("./1.csv", 'w') as f:
        for i, file in enumerate(data0.values):
            label = file[0]
            code = file[1]
            if len(code) > 400:
                f.write(code+"\t#"+str(label) +"\n")
                f.flush()


data = pd.read_csv('./data.csv', delimiter='#')
data['label'] = add(data)


data0 = pd.read_csv('./1.csv', delimiter='#')
train = pd.concat([data, data0], axis=0)
data = shuffle(train, random_state=42)

with open("./data2.csv", 'w') as f:
    for i, file in enumerate(data.values):
        label = int(file[1])
        code = file[0]
        f.write(str(i+1)+"\t#"+code + "\t#" + str(label) + "\n")
        f.flush()


print(data['label'].value_counts())
