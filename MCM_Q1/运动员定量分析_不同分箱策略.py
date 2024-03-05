import pandas as pd
import numpy as np
from sklearn import cluster

df = pd.read_excel('diff_momentum.xlsx')
momentum_diff_1 = np.array(df["momentum_diff_1"].tolist()).reshape(-1, 1)
from sklearn.preprocessing import KBinsDiscretizer

# 等宽分箱
dis = KBinsDiscretizer(n_bins=2,
                       encode="ordinal",
                       strategy="uniform"
                       )
label_uniform = dis.fit_transform(momentum_diff_1)  # 转换器
df["label_uniform"] = label_uniform

# 等频分箱
# 1、先排序
sort_df = sorted(df["momentum_diff_1"])
dis = KBinsDiscretizer(n_bins=3,
                       encode="ordinal",
                       strategy="quantile"
                       )

label_quantile = dis.fit_transform(momentum_diff_1)  # 转换器


df["label_quantile"] = label_quantile

#
dis = KBinsDiscretizer(n_bins=3,
                       encode="ordinal",
                       strategy="kmeans"
                       )

label_kmeans = dis.fit_transform(momentum_diff_1)  # 转换器
df["label_kmeans"] = label_kmeans

df.to_excel('1701_p1_state_classify.xlsx', index=False)
