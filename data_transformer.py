import pandas as pd
import numpy as np

data = pd.read_csv('bwip.csv')
data = data.set_index('BWIP')

start = 2004
end = 2017

new = pd.DataFrame()

for i in data.index:
    if(i > start and i < end):
        d = data.loc[i]
        for x in d.index:
            new.append()