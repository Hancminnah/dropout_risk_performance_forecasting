import pandas as pd
perf_data = pd.read_csv('./data/course performance data.csv')
dropout_data = pd.read_csv('./data/student enrollment data.csv')
merged_data = pd.merge(perf_data, dropout_data, on='student_id',how='outer')

# Analyze each subject group separately
grouped_cs = merged_data[merged_data['department']=='Computer Science'].groupby('student_id')
grouped_ds = merged_data[merged_data['department']=='Design'].groupby('student_id')
grouped_eg = merged_data[merged_data['department']=='Engineering'].groupby('student_id')
grouped_bs = merged_data[merged_data['department']=='Business'].groupby('student_id')

for name, group in grouped_cs:
    print(f"Group: {name}")
    print(group)
    print("-" * 20)

# How to extract a list of keys and values
# How to study trends and anomalies in each group for each subject
for name, group in grouped_cs:
    print(grouped_cs.get_group(name)['course_name'].value_counts())
    input('pause')

# What are the assumptions
# All students started in Fall 2020, but not sure why their entry_year is different
grp = grouped_cs.get_group(1072)
grp = grp.replace({'semester': {'Fall 2020': 0, 'Spring 2021': 6, 'Fall 2021': 12, 'Spring 2022': 18,'Fall 2022': 24,'Spring 2023': 30}})
grp = grp.groupby('course_name') 

# Check Dropout Distribution
# Check Missing Data
# Check Outlier
# Check Correlation between Features
# Descriptives
# Statistical Testings