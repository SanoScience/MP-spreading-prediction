import pandas as pd
import numpy as np

subjects_ESM = pd.read_csv('ESM_ALL.csv', delimiter=',', index_col=0)
subjects_GCN = pd.read_csv('GCN_ALL.csv', delimiter=',', index_col=0)
subjects_MAR = pd.read_csv('MAR_ALL.csv', delimiter=',', index_col=0)
subjects_NDM = pd.read_csv('NDM_ALL.csv', delimiter=',', index_col=0)

for i in subjects_ESM.index:
    if subjects_MAR.loc[i, 'PCC'] - subjects_ESM.loc[i, 'PCC'] > 0.001 and subjects_MAR.loc[i, 'PCC'] - subjects_NDM.loc[i, 'PCC'] > 0.001:
        print(f"Subjects {i} is a good one!")