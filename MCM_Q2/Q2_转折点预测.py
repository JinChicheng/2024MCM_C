import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_path = 'momentum.xlsx'

df = pd.read_excel(data_path)
match_id = '2023-wimbledon-1701'
df = df[df['match_id'] == match_id]
df.reset_index(drop=True, inplace=True)
window_size = 4
variance_threshold = 25
x_value = []
y_value = []
df['is_Turning'] = 0
temp = 0

for i in range(1, len(df) - window_size):
    if df.loc[i, 'set_no'] == df.loc[i + window_size, 'set_no']:
        window_data = df.loc[i:i + window_size, 'abs_diff']
        variance = np.var(window_data)
        if variance > variance_threshold and i > temp + 4:
            df.loc[i, 'is_Turning'] = 1
            x_value.append(df.loc[i, 'point_no'])
            temp = i
            y_value.append((df.loc[i, 'p1_momentum'] + df.loc[i, 'p2_momentum']) / 2)
        else:
            df.loc[i, 'is_Turing'] = 0

df1 = pd.DataFrame({'Turning_Point': df['is_Turning']})
df1.to_excel('turning.xlsx', index=False)


plt.plot(df.index, df['p1_momentum'], linestyle='-', label='Line 1')
plt.plot(df.index, df['p2_momentum'], linestyle='-', label='Line 2')
plt.scatter(x_value, y_value, color='blue', marker='o')
plt.show()
