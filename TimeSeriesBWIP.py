#!/usr/bin/env python
# coding: utf-8

# # Time Series Forecasting Models

# ## Import Statements

# In[1]:


# Install tools and keep up to date
import sys
get_ipython().system('{sys.executable} -m pip install tqdm')
get_ipython().system('{sys.executable} -m pip install msgpack')
get_ipython().system('{sys.executable} -m pip install --upgrade pip ')
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

import warnings                                  # `do not disturb` mode
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import Data

# In[2]:


rev = pd.read_csv('bwip_reformatted_FY2003-2017.csv', index_col=['Date'], parse_dates=['Date'])


# ## Plot Data

# In[3]:


plt.figure(figsize=(15, 7))
plt.plot(rev.Revenue)
plt.title('Revenue (yearly data)')
plt.grid(True)
plt.show()


# ## Import Sklearn and MAPE Function

# In[4]:


# Importing everything from above

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ## Define Moving Average Function

# In[5]:


def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])

moving_average(rev, 12) # prediction for the last observed fiscal year (past 12 months)


# ## Plot Moving Average

# In[6]:


def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
        
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)


# In[7]:


plotMovingAverage(rev, 12)


# ## Yearly Rolling Average

# In[8]:


plotMovingAverage(rev, 12, plot_intervals=True)


# ## Seasonal Rolling Average

# In[9]:


plotMovingAverage(rev, 3, plot_intervals=True, plot_anomalies=True)


# ## Define Weighted Average function

# In[10]:


def weighted_average(series, weights):
    """
        Calculate weighted average on series
    """
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series.iloc[-n-1] * weights[n]
    return float(result)


# In[11]:


weighted_average(rev, [0.6, 0.3, 0.1])


# ## Define Exponential Smoothing Function
# For every value in the dataset, we multipy it with an alpha smoothing parameter 

# In[12]:


def exponential_smoothing(series, alpha):
    """
        series - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result


# ## Define the Plotter for Exponential Smoothing

# In[13]:


def plotExponentialSmoothing(series, alphas):
    """
        Plots exponential smoothing with different alphas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters
        
    """
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(15, 7))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);


# In[14]:


plotExponentialSmoothing(rev.Revenue, [0.3, 0.05])


# ## Define Double Exponential Smoothing function
# For every value in the dataset, we multipy it with an alpha smoothing parameter 

# In[15]:


def double_exponential_smoothing(series, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

def plotDoubleExponentialSmoothing(series, alphas, betas):
    """
        Plots double exponential smoothing with different alphas and betas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters for level
        betas - list of floats, smoothing parameters for trend
    """
    
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(20, 8))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series.values, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)


# In[16]:


plotDoubleExponentialSmoothing(rev.Revenue, alphas=[0.9, 0.02], betas=[0.9, 0.02])


# ## Define Holt Winters Model

# In[17]:


class HoltWinters:
    
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    
    # series - initial time series
    # slen - length of a season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    
    """
    
    
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen  
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])
                
                self.PredictedDeviation.append(0)
                
                self.UpperBond.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBond.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])
                
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 
                                               + (1-self.gamma)*self.PredictedDeviation[-1])
                     
            self.UpperBond.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])


# ## Define Time Series Scorer

# In[18]:


from sklearn.model_selection import TimeSeriesSplit # you have everything done for you

def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=12):
    """
        Returns error on CV  
        
        params - vector of parameters for optimization
        series - dataset with timeseries
        slen - season length for Holt-Winters model
    """
    # errors array
    errors = []
    
    values = series.values
    alpha, beta, gamma = params
    
    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=3) 
    
    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):

        model = HoltWinters(series=values[train], slen=slen, 
                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()
        
        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
        
    return np.mean(np.array(errors))


# ## Train Holt Winters Model on Loss Function MSLE

# In[19]:


get_ipython().run_cell_magic('time', '', 'data = rev.Revenue[:-20] # leave some data for testing\n\n# initializing model parameters alpha, beta and gamma\nx = [0, 0, 0] \n\n# Minimizing the loss function \nopt = minimize(timeseriesCVscore, x0=x, \n               args=(data, mean_squared_log_error), \n               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))\n              )\n\n# Take optimal values...\nalpha_final, beta_final, gamma_final = opt.x\nprint("Alpha final: {} Beta Final: {} Gamma Final: {}".format(alpha_final, beta_final, gamma_final))\n\n# ...and train the model with them, forecasting for the next 12 months\nmodel = HoltWinters(data, slen = 12, \n                    alpha = alpha_final, \n                    beta = beta_final, \n                    gamma = gamma_final, \n                    n_preds = 36, scaling_factor = 3)\nmodel.triple_exponential_smoothing()')


# ## Define Holt Winters Plotter

# In[20]:


def plotHoltWinters(series, plot_intervals=False, plot_anomalies=False):
    """
        series - dataset with timeseries
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    
    plt.figure(figsize=(20, 10))
    plt.plot(model.result,  label = "Model")
    plt.plot(series.values, label = "Actual")
    print("Length: ", len(series.values))
    print("Length: ", len(model.result))
    error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    
    if plot_anomalies:
        anomalies = np.array([np.NaN]*len(series))
        anomalies[series.values<model.LowerBond[:len(series)]] =             series.values[series.values<model.LowerBond[:len(series)]]
        anomalies[series.values>model.UpperBond[:len(series)]] =             series.values[series.values>model.UpperBond[:len(series)]]
        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    if plot_intervals:
        plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(model.LowerBond, "r--", alpha=0.5)
        plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, 
                         y2=model.LowerBond, alpha=0.2, color = "grey")    
        
    plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
    plt.axvspan(len(series)-20, len(model.result), alpha=0.3, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13);


# In[21]:


plotHoltWinters(rev.Revenue)


# In[22]:


plotHoltWinters(rev.Revenue, plot_intervals=True, plot_anomalies=True)


# ## Plot Brutlag's Predicted Deviation
# Used for anomaly detection

# In[23]:


plt.figure(figsize=(25,5))
plt.plot(model.PredictedDeviation)
plt.grid(True)
plt.axis('tight')
plt.title("Brutlag's Predicted Deviation")


# ## Train Holt Winters Model on Loss Function MAPE

# In[24]:


get_ipython().run_cell_magic('time', '', 'data = rev.Revenue[:-50]\nslen = 12\n\nx = [0, 0, 0]\n\nopt = minimize(timeseriesCVscore, x0=x,\n              args=(data, mean_absolute_percentage_error, slen),\n              method="TNC", bounds=((0,1), (0,1), (0,1))\n              )\n\nalpha_final, beta_final, gamma_final = opt.x\nprint("Alpha final: {} Beta Final: {} Gamma Final: {}".format(alpha_final, beta_final, gamma_final))\n\nmodel = HoltWinters(data, slen=slen,\n                   alpha=alpha_final,\n                   beta=beta_final,\n                   gamma=gamma_final,\n                   n_preds=100, scaling_factor=3)\nmodel.triple_exponential_smoothing()')


# ## Plot Prediction 

# In[25]:


plotHoltWinters(rev.Revenue)


# In[26]:


plotHoltWinters(rev.Revenue, plot_intervals=True, plot_anomalies=True)


# ## Plot Brutlag's Predicted Deviation 

# In[27]:


plt.figure(figsize=(20,5))
plt.plot(model.PredictedDeviation)
plt.grid(True)
plt.axis('tight')
plt.title("Burtlag's Predicted Deviation")


# ## Example of Dickey Fuller Test 

# In[28]:


white_noise = np.random.normal(size=1000)
with plt.style.context('bmh'):
    plt.figure(figsize=(15, 5))
    plt.plot(white_noise)


# In[29]:


def plotProcess(n_samples=1000, rho=0):
    x = w = np.random.normal(size=n_samples)
    for t in range(n_samples):
        x[t] = rho * x[t-1] + w[t]
        
    with plt.style.context('bmh'):
        plt.figure(figsize=(10,3))
        plt.plot(x)
        plt.title("Rho {}\n Dickey-Fuller p-value: {}".format(rho, round(sm.tsa.stattools.adfuller(x)[1], 3)))
        
for rho in [0, 0.6, 0.9, 1]:
    plotProcess(rho=rho)


# ## Plot Time Series Analysis with Dickey Fuller 

# In[30]:


def tsplot(y, lags=None, figsize=(12,7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# In[31]:


tsplot(rev.Revenue, lags=60)


# ## Seasonal Differences
# ### Shift data by 12

# In[32]:


rev_diff = rev.Revenue - rev.Revenue.shift(12)
tsplot(rev_diff[12:], lags=60)


# ### Seasonal Difference with extra lag

# In[33]:


rev_diff = rev_diff - rev_diff.shift(1)
tsplot(rev_diff[12+1:], lags=60)


# ## Define SARMIAX parameters list

# In[34]:


ps = range(0,7) #
d=1
qs = range(0,7) 
Ps = range(0,2) 
D=1
Qs = range(0,2) 
s = 12

parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# ## Define SARIMAX model optimizer

# In[35]:


def optimizeSARIMA(parameters_list, d, D, s):
    results = []
    best_aic = float('inf')
    
    for param in tqdm_notebook(parameters_list):
        try:
            model = sm.tsa.statespace.SARIMAX(rev.Revenue, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)   
        except:
            continue
            
        aic = model.aic
        
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
        
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table


# In[36]:


get_ipython().run_cell_magic('time', '', 'result_table = optimizeSARIMA(parameters_list, d, D, s)')


# In[37]:


result_table.head()


# ## Summary of the Best Model
# Selects best model based on lowest AIC Akaike’s Information
# Criteria

# In[38]:


p, q, P, Q = result_table.parameters[0]

best_model = sm.tsa.statespace.SARIMAX(rev.Revenue, order=(p,d,q),
                                              seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())


# ## Plot Time Series Analysis with Best Model Residual and Dickey Fuller

# In[39]:


tsplot(best_model.resid[12+1:], lags=60)


# ## Plot SARIMAX predictions

# In[40]:


def plotSARIMA(series, model, n_steps):
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    data['arima_model'][:s+d] = np.NaN
    
    forecast = model.predict(start=data.shape[0], end=data.shape[0]+n_steps)
    forecast = data.arima_model.append(forecast)
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])
    
    plt.figure(figsize=(15,7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label='model')
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label='actual')
    plt.legend()
    plt.grid(True)
    print(forecast.iloc[156:168])


# In[41]:


print(rev[156:168])
plotSARIMA(rev, best_model, 50)


# In[42]:


data = pd.DataFrame(rev.Revenue.copy())
data.columns = ['y']


# ## Create 12 lags

# In[43]:


for i in range(1, 13):
    data['lag_{}'.format(i)] = data.y.shift(i)


# In[44]:


data.tail(12)


# In[45]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

tscv = TimeSeriesSplit(n_splits=5)


# ## Define train and testing data

# In[46]:


def timeseries_train_test_split(X, y, test_size):
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, y_train, X_test, y_test


# In[47]:


y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

X_train, y_train, X_test, y_test = timeseries_train_test_split(X, y, test_size=0.3)


# ## Fit the Linear Regression Model

# In[48]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# ## Plot Model Results and Coefficients

# In[49]:


def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15,7))
    plt.plot(prediction, "g", label='prediction', linewidth=2.0)
    plt.plot(y_test.values, label='actual', linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                                     cv=tscv,
                                     scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")
            
        error = mean_absolute_percentage_error(prediction, y_test)
        plt.title("Mean absolute percentage error {0:.2f}%".format(error))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)
        
def plotCoefficients(model):
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ['coef']
    coefs['abs'] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by='abs', ascending=False).drop(['abs'], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')


# In[50]:


plotModelResults(lr, plot_intervals=True)
plotCoefficients(lr)


# ## Append more Data Points to dataset

# In[51]:


data.index = pd.to_datetime(data.index)
data['month'] = data.index.month
data['year'] = data.index.year
data['is_spring'] = data.month.isin([3, 4, 5])*1
data['is_summer'] = data.month.isin([6, 7, 8])*1
data['is_fall'] = data.month.isin([9, 10, 11])*1
data['is_winter'] = data.month.isin([12, 1, 2])*1
data.tail()


# In[52]:


plt.figure(figsize=(16, 5))
plt.title('Encoding features')
data.month.plot()
# data.year.plot()
data.is_spring.plot()
data.is_summer.plot()
data.is_fall.plot()
data.is_winter.plot()
plt.grid(True)


# In[53]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# ## Transform data with Scaler, fit and plot

# In[54]:


y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

X_train, y_train, X_test, y_test = timeseries_train_test_split(X, y, test_size=0.3)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
plotCoefficients(lr)


# ## Define data averager

# In[55]:


def code_mean(data, cat_feature, real_feature):
    return dict(data.groupby(cat_feature)[real_feature].mean())


# ## Get month averages

# In[56]:


average_month = code_mean(data, 'month', 'y')
plt.figure(figsize=(7,5))
plt.title("Month Averages")
pd.DataFrame.from_dict(average_month, orient='index')[0].plot()
plt.grid(True)


# ## Get year averages

# In[57]:


average_year = code_mean(data, 'year', 'y')
plt.figure(figsize=(7,5))
plt.title('Year Averages')
pd.DataFrame.from_dict(average_year, orient='index')[0].plot()
plt.grid(True)


# ## Do everything in repeat from above
# ### Set lags, create more data points, add target encoding, return train and test data

# In[58]:


def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
    data = pd.DataFrame(series.copy())
    data.columns = ['y']
    
    for i in range(lag_start, lag_end):
        data['lag_{}'.format(i)] = data.y.shift(i)
        
    data.index = pd.to_datetime(data.index)
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['is_spring'] = data.month.isin([3, 4, 5])*1
    data['is_summer'] = data.month.isin([6, 7, 8])*1
    data['is_fall'] = data.month.isin([9, 10, 11])*1
    data['is_winter'] = data.month.isin([12, 1, 2])*1
    
    if target_encoding:
        test_index = int(len(data.dropna())*(1-test_size))
        data['year_average'] = list(map(code_mean(data[:test_index], 'year', 'y').get, data.year))
        data['month_average'] = list(map(code_mean(data[:test_index], 'month', 'y').get, data.month))
        
        data.drop(['year', 'month'], axis=1, inplace=True)
        
    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)
    X_train, y_train, X_test, y_test = timeseries_train_test_split(X, y, test_size=test_size)
    
    return X_train, y_train, X_test, y_test


# ## Plot it

# In[59]:


X_train, y_train, X_test, y_test =prepareData(rev.Revenue, lag_start=6, lag_end=13, test_size=0.3, target_encoding=True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)
plotCoefficients(lr)


# In[60]:


X_train, y_train, X_test, y_test =prepareData(rev.Revenue, lag_start=1, lag_end=13, test_size=0.3, target_encoding=False)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Plot Heat Map of the correlations

# In[61]:


plt.figure(figsize=(10,8))
sns.heatmap(X_train.corr())


# ## Plot RidgeCV Model Results and Coefficients

# In[62]:


from sklearn.linear_model import LassoCV, RidgeCV

ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_scaled, y_train)

plotModelResults(ridge,
                 X_train=X_train_scaled,
                 X_test=X_test_scaled,
                 plot_intervals=True, plot_anomalies=True)
plotCoefficients(ridge)


# ## Plot LassoCV Model Results and Coefficients

# In[63]:


lasso = LassoCV(cv=tscv)
lasso.fit(X_train_scaled, y_train)

plotModelResults(lasso,
                X_train=X_train_scaled,
                X_test=X_test_scaled,
                plot_intervals=True, plot_anomalies=True)
plotCoefficients(lasso)


# In[ ]:




