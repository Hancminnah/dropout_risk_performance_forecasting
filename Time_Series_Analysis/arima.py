# This code demonstrates how to use the ARIMA model for time series analysis and forecasting.
# From https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# Additional references: https://machinelearningmastery.com/make-sample-forecasts-arima-python/

from pandas import read_csv
from pandas import DataFrame
import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def parser(x):
	return datetime.datetime.strptime(x, '%Y-%m')

series = read_csv('./data/shampoo.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
print(series.head())
#series.plot()

#autocorrelation_plot(series)
#pyplot.show()


# fit an ARIMA model and plot residual errors
series.index = series.index.to_period('M')

# Find the best parameters p,d,q
import statsmodels.api as sm
result = {}
for p in range(5):
    for q in range(5):
        arma = sm.tsa.ARIMA(series, order=(p,0,q))
        arma_fit = arma.fit()
        result[(p,q)] = arma_fit.aic

p,q = min(result, key=result.get)


# fit model
model = ARIMA(series, order=(4,1,4))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
#residuals.plot()
#pyplot.show()

##### density plot of residuals
#residuals.plot(kind='kde')
#pyplot.show()

# summary stats of residuals
print(residuals.describe())

# evaluate an ARIMA model using a walk-forward validation
# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(4,1,4))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# References:
# 1. https://www.datacamp.com/tutorial/arima
# 2. https://stackoverflow.com/questions/62783633/how-to-interpret-plots-of-autocorrelation-and-partial-autocorrelation-using-pyth
# 3. https://people.duke.edu/%7Ernau/411arim3.htm
# 4. https://people.duke.edu/%7Ernau/411arim3.htm