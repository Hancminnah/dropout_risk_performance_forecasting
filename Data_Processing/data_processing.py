import pandas as pd
import numpy as np
perf_data = pd.read_csv('./data/course performance data.csv')
dropout_data = pd.read_csv('./data/student enrollment data.csv')
merged_data = pd.merge(perf_data, dropout_data, on='student_id',how='outer')
merged_data = merged_data.replace({'semester': {'Fall 2020': 0, 'Spring 2021': 6, 'Fall 2021': 12, 'Spring 2022': 18,'Fall 2022': 24,'Spring 2023': 30}})

# Computing percentage of completion, failed, in progress for each student through the whole period 
completion_status_data = merged_data.groupby('student_id')['completion_status'].value_counts().reset_index()
completion_status_data['compl_percentage'] = np.nan
completion_status_data['compl_percentage'] = completion_status_data.groupby('student_id')['count'].transform(lambda x: x / float(x.sum()))
# pivot the table to have completion status as columns
completion_status_pivot = completion_status_data.pivot(index='student_id', columns='completion_status', values='compl_percentage').fillna(0)
completion_status_pivot = completion_status_pivot.reset_index()
completion_status_pivot.columns.name = None
completion_status_pivot

# Computing percentage of grades A, B, C, D, F for each student through the whole period
grade_data = merged_data.groupby('student_id')['grade'].value_counts().reset_index()
grade_data['grade_percentage'] = np.nan
grade_data['grade_percentage'] = grade_data.groupby('student_id')['count'].transform(lambda x: x / float(x.sum()))
# pivot the table to have grades as columns
grade_pivot = grade_data.pivot(index='student_id', columns='grade', values='grade_percentage').fillna(0)
grade_pivot = grade_pivot.reset_index()
grade_pivot.columns.name = None

final_data = dropout_data.merge(completion_status_pivot,on='student_id',how='left')
final_data = final_data.merge(grade_pivot,on='student_id',how='left')
final_data['entry_year'] = final_data['entry_year'].astype(str)
# one hot encoding for categorical variables
final_data = pd.get_dummies(final_data, columns=['gender','department','enrollment_status','entry_year'], drop_first=False)
final_data = final_data.drop(columns=['gender_M','department_Business','enrollment_status_enrolled','enrollment_status_graduated','entry_year_2018'])
# Convert boolean columns to integers
final_data = final_data.astype({'gender_F': int, 'department_Computer Science': int, 'department_Design': int,'department_Engineering': int, 'enrollment_status_dropped out': int, 'entry_year_2019': int, 'entry_year_2020': int, 'entry_year_2021': int, 'entry_year_2022': int})

# Percentage of missing values per column
missing_percentage = (final_data.isnull().sum() / len(final_data)) * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)

final_data = final_data.drop(columns=['university_gpa','student_id'])
# Analyze each subject group separately
grouped_cs = merged_data[merged_data['department']=='Computer Science']
grouped_ds = merged_data[merged_data['department']=='Design']
grouped_eg = merged_data[merged_data['department']=='Engineering']
grouped_bs = merged_data[merged_data['department']=='Business']


val_cs = grouped_cs['completion_status'].value_counts()
val_cs = val_cs/val_cs.sum()

# References:
# 1. https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-017-0442-1#:~:text=The%20MAR%20and%20MNAR%20conditions,analyses%20to%20handle%20MNAR%20data.