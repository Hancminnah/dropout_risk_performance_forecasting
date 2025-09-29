# This code demonstrates how to use the ARIMA model for time series analysis and forecasting.
# From https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# Additional references: https://machinelearningmastery.com/make-sample-forecasts-arima-python/


import pandas as pd
import copy
from plotnine import ggplot, aes, geom_line, geom_point, facet_wrap, labs, theme_bw
from plotnine import ggsave


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

# fit an ARIMA model and plot residual errors
design_thinking = times_series_df[times_series_df['course_name']=='Design Thinking'][['semester','completion_status_failed']]
design_thinking = design_thinking.set_index('semester')
design_thinking.index.freq = 'semester'

engineering_basics = times_series_df[times_series_df['course_name']=='Engineering Basics'][['semester','completion_status_failed']]
engineering_basics = engineering_basics.set_index('semester')
engineering_basics.index.freq = 'semester'

intro_cs = times_series_df[times_series_df['course_name']=='Intro to CS'][['semester','completion_status_failed']]
intro_cs = intro_cs.set_index('semester')
intro_cs.index.freq = 'semester'

prin_business = times_series_df[times_series_df['course_name']=='Principles of Business'][['semester','completion_status_failed']]
prin_business = prin_business.set_index('semester')
prin_business.index.freq = 'semester'

plot_df = times_series_df.copy()
plot_df['semester'] = plot_df['semester'].astype(str)

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


# pyplot.plot(design_thinking.index,design_thinking, linestyle='--', marker='o', color='k')
# pyplot.plot(engineering_basics.index,engineering_basics, linestyle='--', marker='o', color='b')
# pyplot.plot(intro_cs.index,intro_cs, linestyle='--', marker='o', color='r')
# pyplot.plot(prin_business.index,prin_business, linestyle='--', marker='o', color='g')
# pyplot.xlabel('Semester')
# pyplot.ylabel('Percentage of students who failed')
# # Add legend
# pyplot.legend(['Design Thinking','Engineering Basics','Intro to CS','Principles of Business'])
# pyplot.show()

# References:
# 1. https://www.datacamp.com/tutorial/arima
# 2. https://stackoverflow.com/questions/62783633/how-to-interpret-plots-of-autocorrelation-and-partial-autocorrelation-using-pyth
# 3. https://people.duke.edu/%7Ernau/411arim3.htm
# 4. https://people.duke.edu/%7Ernau/411arim3.htm
# 5. https://www.geeksforgeeks.org/machine-learning/time-series-analysis-and-forecasting/
# 6. https://www.geeksforgeeks.org/machine-learning/python-arima-model-for-time-series-forecasting/