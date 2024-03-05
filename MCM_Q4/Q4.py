import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('2023-wimbledon-points-mixed.csv')

x1_ls, x2_ls, x3_ls, x4_ls, x5_ls, x6_ls, x7_ls, x8_ls, x9_ls, x10_ls, x11_ls = [], [], [], [], [], [], [], [], [], [], []
label_ls = []

for i in range(3, len(df)):
    x1 = df.loc[i, 'P1GamesWon']  # 赢得的game数
    x2 = df.loc[i, 'RallyCount']  # 连续回球的次数
    x3 = 1 if df.loc[i, 'ServeIndicator'] == 1 else 0  # 是否发球
    x4 = 1 if df.loc[i - 1, 'PointWinner'] == 1 else 0  # 上一个回合是否得分
    x5 = 1 if df.loc[i, 'P1Ace'] == 1 else 0  # 是否发球直接得分
    x6 = 1 if df.loc[i, 'P1Winner'] == 1 else 0  # 制胜球是否得分
    x7 = 1 if df.loc[i, 'P1DoubleFault'] == 1 else 0  # 是否出现双误
    x8 = 1 if df.loc[i, 'P1UnfErr'] == 1 else 0  # 是否出现非受迫性失误
    x9 = 1 if df.loc[i, 'P1NetPointWon'] == 1 else 0  # 是否上网拿分
    x10 = 1 if df.loc[i, 'P1BreakPointWon'] == 1 else 0  # 破发点得分
    x11 = 1 if df.loc[i, 'ServeNumber'] == 1 else 0  # 一发是否成功

    label = 1 if df.loc[i, 'PointWinner'] == 1 else 0
    label_ls.append(label)
    x1_ls.append(x1)
    x2_ls.append(x2)
    x3_ls.append(x3)
    x4_ls.append(x4)
    x5_ls.append(x5)
    x6_ls.append(x6)
    x7_ls.append(x7)
    x8_ls.append(x8)
    x9_ls.append(x9)
    x10_ls.append(x10)
    x11_ls.append(x11)


dataset = pd.DataFrame(
    {'P1GamesWon': x1_ls, 'RallyCount': x2_ls, 'Is server': x3_ls, 'PointWinner': x4_ls, 'P1Ace': x5_ls, 'P1Winner': x6_ls, 'P1DoubleFault': x7_ls, 'P1UnfErr': x8_ls,
     'P1NetPointWon': x9_ls, 'P1BreakPointWon': x10_ls, 'ServeNumber': x11_ls,
     'label': label_ls, 'match_id': df['match_id'][3:]})

scaler = MinMaxScaler()
columns = dataset.columns[:-1]
scaler.fit(dataset[columns].values)
dataset[columns] = scaler.transform(dataset[columns].values)

dataset.to_excel('test_wim_mixed.xlsx', index=False)

# def function(model):
#     auc = round(
#         cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='roc_auc').mean(), 2)
#     acc = round(
#         cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='accuracy').mean(), 2)
#     recall = round(
#         cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='recall').mean(), 2)
#     precision = round(
#         cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='precision').mean(), 2)
#     f1 = round(cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='f1').mean(),
#                2)
#     return acc, recall, precision, f1, auc
#
#
# model1 = LGBMClassifier(random_state=30, force_col_wise=True)
# # model2 = DecisionTreeClassifier(random_state=42)
# # model3 = SVC(probability=True,random_state=50)
# # model4 = MLPClassifier(random_state=60)
# # model5 = XGBClassifier(random_state=50)
#
#
# # print(f'LGBMClassifier acc, recall, precision, f1, auc :{function(model1)}')
# # print(f'DecisionTreeClassifier acc, recall, precision, f1, auc :{function(model2)}')
# # print(f'SVC acc, recall, precision, f1, auc :{function(model3)}')
# # print(f'MLPClassifier acc, recall, precision, f1, auc :{function(model4)}')
# # print(f'XGBClassifier acc, recall, precision, f1, auc :{function(model5)}')
#
#
# def f(model_list, name_list, types='train'):
#     plt.figure(figsize=(8, 7), dpi=80, facecolor='w')
#     plt.xlim((-0.01, 1.02))
#     plt.ylim((-0.01, 1.02))
#
#     if types == 'test':
#         for model, name in zip(model_list, name_list):
#             ytest_prob = model.predict_proba(xvalid)[:, 1]
#             fpr, tpr, _ = metrics.roc_curve(yvalid, ytest_prob)
#             auc = metrics.auc(fpr, tpr)
#             plt.plot(fpr, tpr, '-', lw=2, label=f'{name} AUC:%.4f' % auc)
#     else:
#         for model, name in zip(model_list, name_list):
#             ytest_prob = model.predict_proba(xtrain)[:, 1]
#             fpr, tpr, _ = metrics.roc_curve(ytrain, ytest_prob)
#             auc = metrics.auc(fpr, tpr)
#             plt.plot(fpr, tpr, '-', lw=2, label=f'{name} AUC:%.4f' % auc)
#     plt.xlabel('False Positive Rate', fontsize=14)
#     plt.ylabel('Ture Positive Rate', fontsize=14)
#     plt.tick_params(labelsize=23)
#     plt.show()
#
#
# #
# #     plt.savefig(f'Question1.png', dpi=500)
# #     plt.show()
# #
# #
#
#
# xtrain, xvalid, ytrain, yvalid = train_test_split(dataset[columns].values, dataset['label'].values, random_state=620,
#                                                   test_size=0.2)
# model1.fit(xtrain, ytrain)
# model2.fit(xtrain, ytrain)
# model3.fit(xtrain, ytrain)
# model4.fit(xtrain, ytrain)
# model5.fit(xtrain, ytrain)
#
# f([model1, model2, model3, model4, model5], ['LGBM', 'Decision Tree','SVC','MLPC','XGBC'], 'test')
# f([model1, model3], ['LGBM', 'Decision Tree','SVC','MLPC','XGBC'], 'train')
# print(f'LGBMClassifier acc, recall, precision, f1, auc :{function(model1)}')
# print(f'LGBMClassifier acc, recall, precision, f1, auc :{function(model2)}')
# print(f'LGBMClassifier acc, recall, precision, f1, auc :{function(model3)}')
# print(f'LGBMClassifier acc, recall, precision, f1, auc :{function(model4)}')
# print(f'DecisionTreeClassifier acc, recall, precision, f1, auc :{function(model5)}')

# # % config InlineBackend.figure.format = "retina"
# # %matplotlib inline
#
# index = df[df.match_id == '2023-wimbledon-1701'].reset_index(drop=True).index
# test = dataset.iloc[index]
# train = dataset.drop(index, axis=0)
# model = LGBMClassifier(random_state=30)
# model.fit(train[columns].values, train['label'].values)
# pred = model.predict_proba(test[columns].values)
# pred = pd.DataFrame({'实时得分': pred[:, 0]})
# import seaborn as sns
#
# #
# sns.set(font="simhei", style="whitegrid", font_scale=1.6)
#
# import matplotlib
#
# matplotlib.rcParams['axes.unicode_minus'] = False
# import plotly.express as px
#
# f = pd.DataFrame({'col': list(columns), 'score': model1.feature_importances_}).sort_values(by='score', ascending=False)
# f.index = f['col']
# f['score'].plot(kind='bar', figsize=(12, 6))
# plt.xlabel('feature')
# plt.ylabel('importance')
# plt.show()
# #
# # 对于某一场比赛的预测
# pred.plot(kind='line', figsize=(12, 6))
# pred.to_excel('predictions.xlsx', index=True)
# plt.show()
