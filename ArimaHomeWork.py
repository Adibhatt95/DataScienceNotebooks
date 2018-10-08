import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from statsmodels.graphics.api import qqplot

df = pd.read_csv('cases.csv', index_col=0)
print(df)
print(df[df.columns[0]].values)

df.index.name=None 
df.reset_index(inplace=True)
#df.drop(df.index[309], inplace=True)

start = datetime.datetime.strptime("1700-01-01", "%Y-%m-%d")
date_list = [start + relativedelta(years=x) for x in range(0,309)]
df['index'] =date_list
df.set_index(['index'], inplace=True)
df.index.name=None

df.columns= ['riders']
df['riders'] = df.riders.apply(lambda x: int(x)*100)

df.riders.plot(figsize=(12,8), title= 'Monthly Ridership', fontsize=14)



from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(center=False, window=12).mean()
    rolstd = timeseries.rolling(center=False, window=12).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Durbin Watson test:
    dfoutput=sm.stats.durbin_watson(timeseries)

    print(dfoutput)

test_stationarity(df.riders)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df, lags=40, ax=ax2)
plt.show()

