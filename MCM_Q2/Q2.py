import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Wimbledon_featured_matches.csv'

df = pd.read_csv(data_path)
# match_id = '2023-wimbledon-1701'
# df = df[df['match_id'] == match_id]
df.reset_index(drop=True, inplace=True)

df['p1_momentum'] = 0
df['p2_momentum'] = 0
final_ls = []

p1_momentum = 0
p2_momentum = 0

# 统计player1的势能变化
for i in range(1, len(df)):
    win1 = 0
    if df.loc[i - 1, 'set_no'] != df.loc[i, 'set_no']:
        p1_momentum = 0
    win = df.loc[i, 'p1_points_won'] - df.loc[i, 'p2_points_won']
    for k in range(1, 10):
        if i > k:
            if df.loc[i, 'point_victor'] == df.loc[i - k, 'point_victor']:
                win1 += 1
            if df.loc[i, 'point_victor'] == df.loc[i - k, 'point_victor']:
                break
    if df.loc[i, 'p2_double_fault'] == 1:
        p1_momentum += 5.89
    if df.loc[i, 'point_victor'] == 1:
        p1_momentum += 1

        if df.loc[i, 'server'] == 1:
            p1_momentum += 1.62

        if df.loc[i, 'p1_ace'] == 1:
            p1_momentum += 4.51

        if df.loc[i, 'p1_break_pt_won'] == 1:
            p1_momentum += 6.26
        if win1 > 0:
            p1_momentum += win1 * 1.08
    df.loc[i, 'p1_momentum'] = p1_momentum

for i in range(1, len(df)):
    win2 = 0
    if df.loc[i - 1, 'set_no'] != df.loc[i, 'set_no']:
        p2_momentum = 0
    win = df.loc[i, 'p2_points_won'] - df.loc[i, 'p1_points_won']
    for k in range(1, 10):
        if i > k:
            if df.loc[i, 'point_victor'] == df.loc[i - k, 'point_victor']:
                win2 += 1
            if df.loc[i, 'point_victor'] == df.loc[i - k, 'point_victor']:
                break
    if df.loc[i, 'p1_double_fault'] == 1:
        p2_momentum += 5.89
    if df.loc[i, 'point_victor'] == 2:
        p2_momentum += 1

        if df.loc[i, 'server'] == 2:
            p2_momentum += 1.62

        if df.loc[i, 'p2_ace'] == 1:
            p2_momentum += 4.51

        if df.loc[i, 'p2_break_pt_won'] == 1:
            p2_momentum += 6.26

        if win2 > 0:
            p2_momentum += win2 * 1.08
    df.loc[i, 'p2_momentum'] = p2_momentum
selected_columns = ['match_id', 'player1', 'player2', 'point_no', 'p1_momentum', 'p2_momentum']
df2 = pd.DataFrame()
df2 = df[selected_columns]
df2.to_excel('momentum.xlsx', index=False)

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['p1_momentum'], label="Player1 Momentum")
plt.plot(df.index, df['p2_momentum'], label="Player2 Momentum")
plt.xlabel('Point Index')
plt.ylabel('Momentum Score')
plt.legend()
plt.show()

for i in range(len(df)):

    if df.loc[i, 'set_victor'] != 0:
        diff = df.loc[i, 'p1_momentum'] - df.loc[i, 'p2_momentum']
        print(diff)
        print(df.loc[i, 'set_victor'])
        if diff > 0 and df.loc[i, 'set_victor'] == 1:
            final = 1
        elif diff < 0 and df.loc[i, 'set_victor'] == 2:
            final = 1
        else:
            final = 0
        print(final)
        final_ls.append(final)
df1 = pd.DataFrame({'Final': final_ls})
df1.to_excel('势差与局输赢的关系.xlsx', index=False)

