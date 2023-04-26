import numpy as np
from gmpy2 import to_binary
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.utils.np_utils import to_categorical
'''
labels = []


labels.append(6)
labels.append(5)
labels.append(4)
labels.append(3)
labels.append(2)
labels.append(1)
labels.append(0)

labels = to_categorical(np.asarray(labels))
'''
flag = 1
mlb = MultiLabelBinarizer(classes=[1,2,3,4])
l = np.zeros((30, 4), dtype=int)
for i in range(len(l)):
      if(flag==1):
            l[i][0]=1
            l[i][1]=2
#print(l)
lab = mlb.fit_transform(l)
print(lab)


'''
l.append(mlb.fit_transform([(1,2)]))
l.append(mlb.fit_transform([(1,2,3)]))
print(l)
print(np.array(l))
'''
'''
for i in range(len(labels)):
    max_value = max(labels[i])
    print(max_value)
    for j in range(len(labels[i])):
        if max_value == labels[i][j]:
            labels[i][j] = 1
        else:
            labels[i][j] = 0
'''