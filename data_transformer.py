import pandas as pd
import numpy as np
import re

def reformat(filename):
    data = pd.read_csv(filename)
    if data.shape != (48, 2): # (48, 2) is good data 
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

        # get last four years data
        a = a.iloc[-48:]

        a.set_index('Date', inplace=True)
        a.to_csv("reformatted_{filename}_FY{start}-{end}.csv".format(filename=filename, start=year - 4, end=year))
        
        return a
    return data

def print_forecast(preds, filename, last_date):
    filename = filename[filename.rfind('/')+1:-4]
    preds = np.around(preds, decimals=2)
    date = re.split('-',last_date)
    year = int(date[0])
    month = int(date[1])+1
    if month == 13:
        year+=1
        month=1
    date.clear()

    for i in preds:
        if month == 12:
            date.append(str(year)+'-'+str(month))
            year+=1
            month=1
        else:
            date.append(str(year)+'-'+str(month))
            month+=1
    p = pd.DataFrame({'Date':date[:], 'Forecast':preds[:]})
    p.set_index('Date',inplace=True,drop=True)
    p.to_csv("forecast-{filename_}.csv".format(filename_=filename))
    print("**FORECASTING COMPLETE**")