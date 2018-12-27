import pandas as pd
import numpy as np

def reformat():
    filename = "california_yacht_marina"
    data = pd.read_csv("data/{filename}.csv".format(filename=filename))
    data = data.set_index("{filename}".format(filename=filename.upper()))

    start = 2004
    end = 2017

    d = data.loc[2004]
    months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]

    # new = np.array([[[str(i)+"-"+str(k), j] for j, k in zip(data.loc[i], months)] for i in data.index if i >= start and i <= end]).reshape(168,2)

    a = pd.DataFrame(columns=["Date", "Revenue"])

    for i in data.index:
        if i >= start and i <= end:
            for j, k in zip(data.loc[i], months):
                year = i
                if k > 6:
                    year = i - 1
                a.loc[len(a)] = [str(year)+'-'+str(k), j]

    a.set_index('Date', inplace=True)
    a.to_csv("reformatted_{filename}_FY{start}-{end}.csv".format(filename=filename, start=start, end=end))