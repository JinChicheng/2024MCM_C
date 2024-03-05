import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Wimbledon_featured_matches.csv')
df.reset_index(drop=True, inplace=True)

x1_ls, x2_ls, x3_ls, x4_ls, x5_ls = [], [], [], [], []
label_ls = []

for i in range(len(df)):
    win1 = 0
    win = df.loc[i, 'p1_points_won'] - df.loc[i, 'p2_points_won']
    for k in range(1, 10):
        if i > k:
            if df.loc[i, 'point_victor'] == df.loc[i - k, 'point_victor']:
                win1 += 1
            if df.loc[i, 'point_victor'] != df.loc[i - k, 'point_victor']:
                break
    x1 = 1 if df.loc[i, 'p2_double_fault'] == 1 else 0  # 是否双误
    x2 = 1 if df.loc[i, 'p1_ace'] == 1 else 0  # 一发得分
    x3 = 1 if df.loc[i, 'p1_break_pt_won'] == 1 else 0  # 是否破发
    x4 = 1 if win1 > 0 else 0  # 是否连续得分
    x5 = 1 if df.loc[i, 'server'] == 1 and df.loc[i, 'point_victor'] == 1 else 0  # 发球得分

    label = 1 if df.loc[i, 'point_victor'] == 1 else 0
    label_ls.append(label)

    x1_ls.append(x1)
    x2_ls.append(x2)
    x3_ls.append(x3)
    x4_ls.append(x4)
    x5_ls.append(x5)

dataset = pd.DataFrame(
    {'Opponent double fault': x1_ls, 'Ace': x2_ls, 'Break point won': x3_ls, 'Continuous score': x4_ls,
     'Server': x5_ls, 'label': label_ls})

# scaler = MinMaxScaler()
# columns = [col for col in dataset.columns[:-1]]
# scaler.fit(dataset[columns].values)
# dataset[columns] = scaler.transform(dataset[columns].values)
dataset.to_excel('斯皮尔曼相关系数.xlsx', index=False)
