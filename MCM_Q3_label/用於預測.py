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

df = pd.read_csv('Wimbledon_featured_matches_withlabel.csv')
df['speed_mph'].fillna(112, inplace=True)

x1_ls, x2_ls, x3_ls, x4_ls, x5_ls, x6_ls, x7_ls, x8_ls, x9_ls, x10_ls, x11_ls, x12_ls, x13_ls, x14_ls, x15_ls = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
label_ls = []

for i in range(1, len(df)):
    x1 = df.loc[i, 'p1_games']  # 赢得的game数
    x2 = df.loc[i, 'rally_count']  # 连续回球的次数
    # 在专业比赛中，长时间的回球可以构成激烈的拉力战，需要运动员有出色的身体素质、技术和耐心。记录和分析回球次数可以帮助评估比赛的激烈程度和球员的体能水平。
    win1 = 0
    win = df.loc[i, 'p1_points_won'] - df.loc[i, 'p2_points_won']
    for k in range(1, 10):
        if i > k:
            if df.loc[i, 'point_victor'] == df.loc[i - k, 'point_victor']:
                win1 += 1
            if df.loc[i, 'point_victor'] != df.loc[i - k, 'point_victor']:
                break
    x3 = 1 if df.loc[i, 'server'] == 1 else 0  # 是否发球
    x4 = 1 if df.loc[i - 1, 'point_victor'] == 1 else 0  # 上一个回合是否得分
    x5 = 1 if win1 > 0 else 0  # 是否连续得分
    x6 = abs(df.loc[i, 'p1_sets'] - df.loc[i, 'p2_sets'])  # 局数差
    x7 = 1 if df.loc[i, 'p1_ace'] == 1 else 0  # 是否发球直接得分
    x8 = 1 if df.loc[i, 'p1_winner'] == 1 else 0  # 制胜球是否得分
    x9 = 1 if df.loc[i, 'p1_double_fault'] == 1 else 0  # 是否出现双误
    x10 = 1 if df.loc[i, 'p1_unf_err'] == 1 else 0  # 是否出现非受迫性失误
    x11 = 1 if df.loc[i, 'p1_net_pt_won'] == 1 else 0  # 是否上网拿分
    x12 = 1 if df.loc[i, 'p1_break_pt_won'] == 1 else 0  # 是否破发得分
    x13 = df.loc[i, 'p1_distance_run']
    speed = df.loc[i, 'speed_mph']  # 球速
    x14 = speed * x3
    x15 = 1 if df.loc[i, 'serve_no'] == 1 else 0  # 一发是否成功

    label = 1 if df.loc[i, 'point_victor'] == 1 else 0
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
    x12_ls.append(x12)
    x13_ls.append(x13)
    x14_ls.append(x14)
    x15_ls.append(x15)

dataset = pd.DataFrame(
    {'Game win': x1_ls, 'Rally count': x2_ls, 'Server': x3_ls, 'Last game win': x4_ls,
     'Continuous score': x5_ls, 'difference of set won': x6_ls, 'Ace': x7_ls, 'Winner ball': x8_ls,
     'Double fault': x9_ls, 'Unf err': x10_ls, 'Net point won': x11_ls, 'Break point won': x12_ls,
     'Distance run': x13_ls, 'Speed': x14_ls, 'First serve success': x15_ls,
     'label': label_ls})

# scaler = MinMaxScaler()
# columns = dataset.columns[:-1]
# scaler.fit(dataset[columns].values)
# dataset[columns] = scaler.transform(dataset[columns].values)

dataset.to_excel('test_Q3.xlsx', index=False)

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
# model2 = DecisionTreeClassifier(random_state=42)
# model3 = SVC(probability=True,random_state=50)
# model4 = MLPClassifier(random_state=60)
# model5 = XGBClassifier(random_state=50)

#
# print(f'LGBMClassifier acc, recall, precision, f1, auc :{function(model1)}')


# print(f'DecisionTreeClassifier acc, recall, precision, f1, auc :{function(model2)}')
# print(f'SVC acc, recall, precision, f1, auc :{function(model3)}')
# print(f'MLPClassifier acc, recall, precision, f1, auc :{function(model4)}')
# print(f'XGBClassifier acc, recall, precision, f1, auc :{function(model5)}')


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
#
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
# pred = pd.DataFrame({'Point_Win': pred[:, 0]})
# import seaborn as sns
#
# pre = pred['Point_Win']
# x = range(len(pre))
# print(pred)
# print(x)

# sns.set(font="simhei", style="whitegrid", font_scale=1.6)
#
# import matplotlib
#
# matplotlib.rcParams['axes.unicode_minus'] = False
# import plotly.express as px
#
# # f = pd.DataFrame({'col': list(columns), 'score': model1.feature_importances_}).sort_values(by='score', ascending=False)
# # f.index = f['col']
# # f['score'].plot(kind='bar', figsize=(12, 6))
# # plt.xlabel('feature')
# # plt.ylabel('importance')
# # plt.show()
# # #
# # 对于某一场比赛的预测
# pred.plot(kind='scatter', x=pred.index, y=pred.Point_Win, figsize=(12, 6))
# plt.show()
# for i in range(len(pre)):
#     if pre[i] > 0.5:
#         plt.scatter(x[i], pre[i], color='green')
#     else:
#         plt.scatter(x[i], pre[i], color='red')
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 12
# plt.xlabel('Point_number')
# plt.ylabel('Point_Win')
# plt.title('Prediction of Win in Match ID=1701')
# plt.show()


# index = df[df.match_id == '2023-wimbledon-1601'].reset_index(drop=True).index
# test = dataset.iloc[index]
# train = dataset.drop(index, axis=0)
# model = LGBMClassifier(random_state=30)
# model.fit(train[columns].values, train['label'].values)
# pred = model.predict_proba(test[columns].values)
# pred = pd.DataFrame({'Point_Win': pred[:, 0]})
import seaborn as sns

# pre = pred['Point_Win']
# x = range(len(pre))
# print(pred)
# print(x)
#
#
# for i in range(len(pre)):
#     if pre[i] > 0.5:
#         plt.scatter(x[i], pre[i], color='green')
#     else:
#         plt.scatter(x[i], pre[i], color='red')
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 12
# plt.xlabel('Point_number')
# plt.ylabel('Point_Win')
# plt.title('Prediction of Win in Match ID=1601')
# plt.show()