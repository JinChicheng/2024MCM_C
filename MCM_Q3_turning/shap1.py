# 引入需要的包
import pandas as pd
import numpy as np
import warnings
import shap
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

all_data = pd.read_excel('test_Q3.xlsx')
all_data.info()
# 拆分下训练集和测试集
train, test = train_test_split(all_data, test_size=0.2)

# 利用LightGBM训练模型
import lightgbm as lgb
from sklearn import metrics

params = {'objective': 'binary',
          'metric': 'binary_logloss',
          'num_round': 80,
          'verbose': 1
          }
num_round = params.pop('num_round', 1000)
xtrain = lgb.Dataset(train.drop(columns=['label']), train['label'], free_raw_data=False)
xeval = lgb.Dataset(test.drop(columns=['label']), test['label'], free_raw_data=False)
evallist = [xtrain, xeval]
clf = lgb.train(params, xtrain, num_round, valid_sets=evallist)
ytrain = np.where(clf.predict(train.drop(columns=['label'])) >= 0.5, 1, 0)
ytest = np.where(clf.predict(test.drop(columns=['label'])) >= 0.5, 1, 0)
print("train classification report")
print(metrics.classification_report(train['label'], ytrain))
print('*' * 60)
print("test classification report")
print(metrics.classification_report(test['label'], ytest))
# 如果数据量大，这个运行的会非常慢
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(train.drop(columns=['label']))  # 获取shap value
train.drop(columns=['label']).iloc[0].T


#查看单个样本的特征贡献的第二种方法
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][0],train.drop(columns=['label']).iloc[0])

shap.summary_plot(shap_values, train.drop(columns=['label']),plot_type="bar")

shap.summary_plot(shap_values[1], train.drop(columns=['label']))

# shap.force_plot(explainer.expected_value[1], shap_values[1],train.drop(columns=['Survived']))#我们看下训练集所有样本的特征贡献情况
