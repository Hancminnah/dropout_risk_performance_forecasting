# This code demonstrates how to use the ARIMA model for time series analysis and forecasting.
# From https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# Additional references: https://machinelearningmastery.com/make-sample-forecasts-arima-python/


import pandas as pd
import copy
from sklearn.linear_model import LinearRegression
import numpy as np
from plotnine import ggplot, aes, geom_line, geom_point, facet_wrap, labs, theme_bw
from plotnine import ggsave
from plotnine import scale_x_datetime
from sklearn.metrics import mean_squared_error


# Process Data
perf_data = pd.read_csv('./data/course performance data.csv')
dropout_data = pd.read_csv('./data/student enrollment data.csv')
df = copy.deepcopy(perf_data)#pd.read_csv('./data/temp_data.csv')

# Convert semester to numerical format for sorting
df['semester_num'] = df['semester'].copy()
df['semester_num'] = df['semester_num'].replace({'Fall 2020': '2020-01-01', 'Spring 2021': '2020-01-02', 'Fall 2021': '2020-01-03', 'Spring 2022': '2020-01-04','Fall 2022': '2020-01-05','Spring 2023': '2020-01-06'})
df['semester_num'] = pd.to_datetime(df['semester_num'], format='%Y-%m-%d')
# Sort by student_id and semester to ensure proper time sequence
df = df.sort_values(['student_id', 'semester_num'])
df = df.merge(dropout_data[['student_id','department']],on='student_id',how='left')
df = pd.get_dummies(df, columns=['completion_status'],drop_first=False)
times_series_df = df.groupby(['course_name','semester_num'])['completion_status_failed'].apply(lambda x: (x.sum() / x.count())*100).reset_index()
times_series_df['semester'] = df['semester'].replace({'2020-01-01':'Fall 2020', '2020-01-02':'Spring 2021', '2020-01-03':'Fall 2021', '2020-01-04':'Spring 2022','2020-01-05':'Fall 2022','2020-01-06':'Spring 2023'})

plot_df = times_series_df.copy()
# Ensure 'semester' is ordered categorically for proper plotting
semester_order = ['Fall 2020', 'Spring 2021', 'Fall 2021', 'Spring 2022', 'Fall 2022', 'Spring 2023']
plot_df['semester'] = pd.Categorical(plot_df['semester'], categories=semester_order, ordered=True)

p = (
    ggplot(plot_df, aes(x='semester', y='completion_status_failed', group='course_name', color='course_name'))
    + geom_line()
    + geom_point()
    + facet_wrap('~course_name', nrow=len(plot_df['course_name'].unique()), ncol=1)
    + labs(x='Semester', y='Percentage of students who failed', title='Failure Rate by Course and Semester', color='Course Name')
    + theme_bw()
)

ggsave(p, filename='results/failure_rate_by_course_and_semester.png', verbose=False)
print(p)

times_series_df.to_csv('./results/time_series_data.csv', index=False)

# Check for stationarity using Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from plotnine import geom_ribbon
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('Dickey-Fuller Test:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")

check_stationarity(times_series_df[times_series_df['course_name']=='Design Thinking']['completion_status_failed'])
# Since the series is not stationary, difference it to make stationary
design_thinking_series = times_series_df[times_series_df['course_name']=='Design Thinking']['completion_status_failed']
design_thinking_diff = design_thinking_series.diff().dropna()
check_stationarity(design_thinking_diff)

# Since the series is not stationary, difference it to make stationary
engineering_basic_series = times_series_df[times_series_df['course_name']=='Engineering Basics']['completion_status_failed']
engineering_basic_diff = engineering_basic_series.diff().dropna()
check_stationarity(engineering_basic_diff)

# Since the series is not stationary, difference it to make stationary
intro_cs_series = times_series_df[times_series_df['course_name']=='Intro to CS']['completion_status_failed']
intro_cs_diff = intro_cs_series.diff().dropna()
check_stationarity(intro_cs_diff)

# Since the series is not stationary, difference it to make stationary
prin_business_series = times_series_df[times_series_df['course_name']=='Principles of Business']['completion_status_failed']
prin_business_diff = prin_business_series.diff().dropna()
check_stationarity(prin_business_diff)

# Plot ACF and PACF
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(design_thinking_diff, ax=plt.gca(), lags=4)
plt.subplot(122)
plot_pacf(design_thinking_diff, ax=plt.gca())
plt.tight_layout()


X = np.array(range(len(design_thinking_series))).reshape(-1, 1)
y = design_thinking_series.values
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)
plt.figure(figsize=(8, 6))
plt.plot(plot_df['semester'].iloc[:6], y, label='Original')
plt.plot(plot_df['semester'].iloc[:6], trend, color='red', linestyle = '--', label='Trend')
plt.xlabel('Semester')
plt.ylabel('Percentage of students who failed')
plt.title('Design Thinking')
plt.legend()
plt.savefig('Time_Series_Analysis/design_thinking_trend.png')
plt.show()

X = np.array(range(len(engineering_basic_series))).reshape(-1, 1)
y = engineering_basic_series.values
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)
plt.figure(figsize=(8, 6))
plt.plot(plot_df['semester'].iloc[:6], y, label='Original')
plt.plot(plot_df['semester'].iloc[:6], trend, color='red', linestyle = '--', label='Trend')
plt.xlabel('Semester')
plt.ylabel('Percentage of students who failed')
plt.title('Engineering Basics')
plt.legend()
plt.savefig('Time_Series_Analysis/engineering_basic_trend.png')
plt.show()

X = np.array(range(len(intro_cs_series))).reshape(-1, 1)
y = intro_cs_series.values
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)
plt.figure(figsize=(8, 6))
plt.plot(plot_df['semester'].iloc[:6], y, label='Original')
plt.plot(plot_df['semester'].iloc[:6], trend, color='red', linestyle = '--', label='Trend')
plt.xlabel('Semester')
plt.ylabel('Percentage of students who failed')
plt.title('Intro to CS')
plt.legend()
plt.savefig('Time_Series_Analysis/intro_cs_trend.png')
plt.show()

X = np.array(range(len(prin_business_series))).reshape(-1, 1)
y = prin_business_series.values
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)
plt.figure(figsize=(8, 6))
plt.plot(plot_df['semester'].iloc[:6], y, label='Original')
plt.plot(plot_df['semester'].iloc[:6], trend, color='red', linestyle = '--', label='Trend')
plt.xlabel('Semester')
plt.ylabel('Percentage of students who failed')
plt.title('Principles of Business')
plt.legend()
plt.savefig('Time_Series_Analysis/prin_business_trend.png')
plt.show()
# Note: The ARIMA model fitting and forecasting code is not included here as the dataset is small and primarily for demonstration.
# References:
# 1. https://www.datacamp.com/tutorial/arima
# 2. https://stackoverflow.com/questions/62783633/how-to-interpret-plots-of-autocorrelation-and-partial-autocorrelation-using-pyth
# 3. https://people.duke.edu/%7Ernau/411arim3.htm
# 4. https://people.duke.edu/%7Ernau/411arim3.htm
# 5. https://www.geeksforgeeks.org/machine-learning/time-series-analysis-and-forecasting/
# 6. https://www.geeksforgeeks.org/machine-learning/python-arima-model-for-time-series-forecasting/
# 7. https://statpowers.com/timeSeries.html