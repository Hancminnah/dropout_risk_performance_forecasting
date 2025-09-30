import pandas as pd
import numpy as np
import pickle
import os
import copy
import plotnine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from plotnine import ggplot, aes, geom_boxplot, labs, theme_bw
from plotnine import facet_wrap
from lib.generate_descriptives_adapted import generate_descriptives_edited


final_data = pd.read_csv('./data/final_data.csv')
completion_grade_count = pd.read_csv('./data/completion_grade_count.csv')
grade_pivot_count = pd.read_csv('./data/grade_pivot_count.csv')

# Percentage of missing values per column
missing_percentage = (final_data.isnull().sum() / len(final_data)) * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)

# ==== Outlier Analysis ==== #

# Combine age and high_school_gpa into long format for faceting
plot_data = final_data[['age', 'high_school_gpa']].melt(var_name='variable', value_name='value')
# Faceted boxplot: age and high_school_gpa in separate panels
p = (
        ggplot(plot_data, aes(x="variable", y="value")) +
        geom_boxplot() +
        labs(title="Box Plots of Age and High School GPA", x="", y="Value") +
        facet_wrap('~variable', scales='free') +
        theme_bw()
)
p.save('./results/boxplot_age_gpa.png', dpi=300)
# ============================== #

# ===== Perform Descriptives ===== #
df_descriptives = copy.deepcopy(final_data)
df_descriptives = df_descriptives.merge(completion_grade_count,on='student_id',how='left')
descriptives_table = generate_descriptives_edited(dataframe=df_descriptives,use_cols=[x for x in df_descriptives.columns if x not in ['student_id','university_gpa']], continuous_stat_agg=['mean','median_range','min_max'],float_precision=1)
descriptives_table.to_csv("./results/descriptives_table.csv",index=False)
# ================================ #

# ====== Statistical Comparisons ====== #
df_comparatives = copy.deepcopy(final_data)
df_comparatives = df_comparatives.merge(completion_grade_count,on='student_id',how='left')
df_comparatives['Exposure'] = np.nan
df_comparatives.loc[df_comparatives['enrollment_status']=='dropped out','Exposure'] = '1'
df_comparatives['Exposure'] = df_comparatives['Exposure'].fillna('0')
comparatives_table = generate_descriptives_edited(dataframe=df_comparatives,cohort_col = 'Exposure',use_cols=[x for x in df_comparatives.columns if x not in ['student_id','Exposure','enrollment_status']],continuous_stat_agg =['mean','median_range','min_max'],float_precision=1)
comparatives_table.to_csv("./results/comparatives_table.csv",index=False)