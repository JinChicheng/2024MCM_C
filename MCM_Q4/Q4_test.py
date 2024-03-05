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
from 五条Q4 import columns
import warnings

warnings.filterwarnings("ignore")

dataset = pd.read_excel('test_wim_double.xlsx')


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


model1 = LGBMClassifier(random_state=30, force_col_wise=True)


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
    plt.show()


#
#     plt.savefig(f'Question1.png', dpi=500)
#     plt.show()


# xtrain, xvalid, ytrain, yvalid = train_test_split(dataset[columns].values, dataset['label'].values, random_state=620,
#                                                   test_size=0.2)
# model1.fit(xtrain, ytrain)
#
index = dataset[dataset['match_id'] == '2023-wimbledon-1701'].reset_index(drop=True).index
test = dataset.iloc[index]
train = dataset.drop(index, axis=0)
model = LGBMClassifier(random_state=30)
model.fit(train[columns].values, train['label'].values)
pred = model.predict_proba(test[columns].values)
pred = pd.DataFrame({'Point_Win': pred[:, 0]})
pre = pred['Point_Win']
y = pre.index
print(y)
print(pre)
# import seaborn as sns
#
# #
# sns.set(font="simhei", style="whitegrid", font_scale=1.6)
#
# import matplotlib
#
# matplotlib.rcParams['axes.unicode_minus'] = False
# #
# 对于某一场比赛的预测




