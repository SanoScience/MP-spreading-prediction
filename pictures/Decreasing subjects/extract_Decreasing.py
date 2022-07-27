import pandas as pd
import json
import numpy as np

df = pd.read_csv('list_MAR.csv', delimiter=',')

with open('decreasing_list.json', 'r') as f:
    dataset_all = json.load(f)

ids_all =[]
for sub in dataset_all.keys():
    ids_all.append(sub.replace('\n', ' '))

with open('decreasing_list.json', 'r') as f:
    dataset = json.load(f)

ids =[]
for sub in dataset.keys():
    ids.append(sub.replace('\n', ' '))

#print('Decreasing subjects: ', ids)
print(df.head())

avg_mse = []
avg_pcc = []
for id in range(len(ids_all)):
    if ids_all[id] in ids:
        print(ids_all[id])
        avg_mse.append(df.loc[id]['MSE'])
        avg_pcc.append(df.loc[id]['PCC'])

std_mse = np.std(avg_mse)
std_pcc = np.std(avg_pcc)
avg_mse = np.mean(avg_mse)
avg_pcc = np.mean(avg_pcc)

print("decreasing subjects: ", len(ids))
print('avg mse: ', avg_mse)
print('std mse', std_mse)
print('avg pcc: ', avg_pcc)
print('std pcc', std_pcc)