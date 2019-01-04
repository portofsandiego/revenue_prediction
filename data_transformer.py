import pandas as pd
import numpy as np

def reformat(filename):
    data = pd.read_csv(filename)
    if data.shape != (36, 2): # (36, 2) is good data 
                              # ~(14, 13) need reformatting
        filename = filename[filename.rfind('/')+1:-4]
        data = data.set_index(filename)

        months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]

        # new = np.array([[[str(i)+"-"+str(k), j] for j, k in zip(data.loc[i], months)] for i in data.index if i >= start and i <= end]).reshape(168,2)

        a = pd.DataFrame(columns=["Date", "Revenue"])
        year = 0
        for i in data.index:
            for j, k in zip(data.loc[i], months):
                year = i
                if k > 6:
                    year = i - 1
                a.loc[len(a)] = [str(year)+'-'+str(k), j]

        a = a.iloc[-48:]

        a.set_index('Date', inplace=True)
        a.to_csv("reformatted_{filename}_FY{start}-{end}.csv".format(filename=filename, start=year - 4, end=year))
        
        return a