import pandas as pd
import numpy as np

data = pd.read_csv('data/bwip.csv')
data = data.set_index('BWIP')

start = 2004
end = 2017

d = data.loc[2004]
months = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]

new = [[[str(i)+"-"+str(k), j] for j, k in zip(data.loc[i], months) if k > 6: j -= 1] for i in data.index if i >= start and i <= end]

df = pd.DataFrame(new)
df = df.stack()
# print(df.stack())
print(df.shape)
# df.reset_index(drop=True, inplace=True)
df.to_csv("reformatted.csv", index=False)

# for i in data.index:
#     if(i >= start and i <= end):
#         d = data.loc[i]
#         new.append(zip([i for i in d.name], [d[i] for i in d.index]))
#         # for x in d.index:
#         #     print(d[x], d.name)
#         #     new.append((d.name+"-"+1, d[x]))

# print(new)