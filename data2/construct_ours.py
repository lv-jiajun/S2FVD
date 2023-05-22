import pandas as pd

data = pd.read_csv('our26/data.csv', delimiter='#')

ours = pd.read_pickle('ours/ours.pkl')

print(data['label'].value_counts())

