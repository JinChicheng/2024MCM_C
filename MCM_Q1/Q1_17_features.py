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
import warnings

warnings.filterwarnings("ignore")


def function(model):
    auc = round(
        cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='roc_auc').mean(), 2)
    acc = round(
        cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='accuracy').mean(), 2)
    recall = round(
        cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='recall').mean(), 2)
    precision = round(
        cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='precision').mean(), 2)
    f1 = round(cross_val_score(model, dataset[columns].values, dataset['label'].values, cv=5, scoring='f1').mean(),
               2)
    return acc, recall, precision, f1, auc


df = pd.read_csv('Wimbledon_featured_matches.csv')
df.loc[(df.p1_score == 'AD'), 'p1_score'] = 50
df.loc[(df.p2_score == 'AD'), 'p2_score'] = 50
df['p1_score'] = df['p1_score'].astype(int)
df['p2_score'] = df['p2_score'].astype(int)
df.dropna(subset=['speed_mph'], inplace=True)

x1_ls, x2_ls, x3_ls, x4_ls, x5_ls, x6_ls, x7_ls, x8_ls, x9_ls, x10_ls, x11_ls, x12_ls, x13_ls, x14_ls, x15_ls, x16_ls, x17_ls = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
label_ls = []
for match_id, set_no, game_no, point_no in zip(df.match_id, df.set_no, df.game_no, df.point_no):
    match = df[df.match_id == match_id]
    set_ = match[match.set_no == set_no]
    game_ = set_[set_.game_no == game_no]
    point_ = game_[game_.point_no == point_no]
    x1 = point_['p1_games'].values[0]  # 赢得的game数
    x2 = point_['rally_count'].values[0]  # 连续回球的次数
    # 在专业比赛中，长时间的回球可以构成激烈的拉力战，需要运动员有出色的身体素质、技术和耐心。记录和分析回球次数可以帮助评估比赛的激烈程度和球员的体能水平。
    points = point_['p1_points_won'].values[0] - point_['p2_points_won'].values[0]
    x3 = 1 if point_['serve_no'].values[0] == 1 else 0  # 是否发球
    x4 = 0 if points < 0 else 1  # 上一个回合是否得分
    x5 = 1 if points >= 2 else 0  # 是否连续得分
    x6 = point_['p1_sets'].values[0] - point_['p2_sets'].values[0]  # 领先的局数
    x7 = 1 if 1 in game_['p1_ace'].values else 0  # 是否发球得分
    x8 = 1 if 1 in game_['p1_winner'].values else 0  # 制胜球是否得分
    x9 = 1 if 0 in game_['p1_double_fault'].values else 1  # 是否出现双误
    x10 = 1 if 0 in game_['p1_unf_err'].values else 1  # 是否出现非受迫性失误
    x11 = game_['p1_net_pt_won'].sum() / game_['p1_net_pt'].sum() if game_['p1_net_pt'].sum() != 0 else 0  # 上网拿分的比例
    x12 = set_['p1_break_pt_won'].sum() / set_['p1_break_pt'].sum() if game_['p1_break_pt'].sum() != 0 else 0  # 破发的比例
    index = match.index.tolist().index(point_.index.tolist()[0])
    x13 = match.iloc[index - 2:index + 1]['p1_distance_run'].sum()  # 最近三个point中的总计跑图里程
    x14 = point_['p1_distance_run'].values[0]
    speed = point_['speed_mph'].values[0]  # 实时球速
    x15 = speed * x3
    x16 = 1 if 1 in game_['serve_no'].values else 0  # 一发是否成功
    x17 = point_['p2_distance_run'].values[0] - point_['p1_distance_run'].values[0]  # 调动对手的程度 p1-p2

    label = 1 if point_['point_victor'].values[0] == 1 else 0
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
    x16_ls.append(x16)
    x17_ls.append(x17)
dataset = pd.DataFrame(
    {'x1': x1_ls, 'x2': x2_ls, 'x3': x3_ls, 'x4': x4_ls, 'x5': x5_ls, 'x6': x6_ls, 'x7': x7_ls, 'x8': x8_ls,
     'x9': x9_ls, 'x10': x10_ls, 'x11': x11_ls, 'x12': x12_ls, 'x13': x13_ls, 'x14': x14_ls, 'x15': x15_ls,
     'x16': x16_ls, 'x17': x17_ls, 'label': label_ls})

scaler = MinMaxScaler()
columns = [col for col in dataset.columns[:-1]]
scaler.fit(dataset[columns].values)
dataset[columns] = scaler.transform(dataset[columns].values)
# dataset.to_excel('标准化.xlsx', index=False)
model = LGBMClassifier(random_state=30, force_col_wise=True)


def f(model_list, name_list, types='train'):
    plt.figure(figsize=(8, 7), dpi=80, facecolor='w')
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))

    if types == 'test':
        for model, name in zip(model_list, name_list):
            ytest_prob = model.predict_proba(xvalid)[:, 1]
            fpr, tpr, _ = metrics.roc_curve(yvalid, ytest_prob)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, '-', lw=2, label=f'{name} AUC:%.4f' % auc)
    else:
        for model, name in zip(model_list, name_list):
            ytest_prob = model.predict_proba(xtrain)[:, 1]
            fpr, tpr, _ = metrics.roc_curve(ytrain, ytest_prob)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, '-', lw=2, label=f'{name} AUC:%.4f' % auc)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('Ture Positive Rate', fontsize=14)
    plt.tick_params(labelsize=23)

    plt.savefig(f'Question1.png', dpi=500)
    plt.show()


xtrain, xvalid, ytrain, yvalid = train_test_split(dataset[columns].values, dataset['label'].values, random_state=620,
                                                  test_size=0.2)
model1 = LGBMClassifier(random_state=30)
model1.fit(xtrain, ytrain)
model2 = DecisionTreeClassifier(random_state=42)
model2.fit(xtrain, ytrain)

f([model1, model2], ['LGBM', 'Decision Tree'], 'test')
f([model1, model2], ['LGBM', 'Decision Tree'], 'train')
print(f'LGBMClassifier acc, recall, precision, f1, auc :{function(model1)}')
print(f'DecisionTreeClassifier acc, recall, precision, f1, auc :{function(model2)}')

# % config InlineBackend.figure.format = "retina"
# %matplotlib inline

# index = df[df.match_id == '2023-wimbledon-1701'].reset_index(drop=True).index
# test = dataset.iloc[index]
# train = dataset.drop(index, axis=0)
# model = LGBMClassifier(random_state=30)
# model.fit(train[columns].values, train['label'].values)
# pred = model.predict_proba(test[columns].values)
# pred = pd.DataFrame({'实时得分': pred[:, 0]})
import seaborn as sns

sns.set(font="simhei", style="whitegrid", font_scale=1.6)

import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False
import plotly.express as px
f = pd.DataFrame({'col':list(columns),'score':model1.feature_importances_}).sort_values(by='score',ascending=False)
f.index = f['col']
f['score'].plot(kind='bar',figsize=(12,6))
plt.xlabel('feature')
plt.ylabel('importance')
plt.show()

# 对于某一场比赛的预测
# pred.plot(kind='line', figsize=(12, 6))
# pred.to_excel('predictions.xlsx', index=True)
# plt.show()
