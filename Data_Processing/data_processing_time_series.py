# Python code to predict dropout using time series data with Random Forest Classifier.
# Created lagged variable using the time column semester with features as completion status and grade and the outcome is dropout.

import pandas as pd
import numpy as np
import copy
import os
import pickle
from group_lasso import LogisticGroupLasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from lib.modelling_libraries import train_evaluate_LR, \
                                    train_randomsearch_evaluate_RF, \
                                    train_gridsearch_evaluate_RF, \
                                    train_randomsearch_evaluate_XGB, \
                                    train_gridsearch_evaluate_XGB


random_seed_nb = 42
n_folds = 5

perf_data = pd.read_csv('./data/course performance data.csv')

dropout_data = pd.read_csv('./data/student enrollment data.csv')
dropout_data['dropout'] = np.nan
dropout_data.loc[dropout_data['enrollment_status']=='dropped out','dropout'] = 1
dropout_data = dropout_data.fillna(0)
dropout_data['dropout'] = dropout_data['dropout'].astype(int)
dropout_data = dropout_data.drop(columns=['enrollment_status','university_gpa','entry_year'])
# one-hot encoding for categorical variables
dropout_data = pd.get_dummies(dropout_data, columns=['gender','department'],drop_first=False).drop(columns=['gender_M','department_Design'])
dropout_data['gender_F'] = dropout_data['gender_F'].astype(int)
dropout_data['department_Business'] = dropout_data['department_Business'].astype(int)
dropout_data['department_Computer Science'] = dropout_data['department_Computer Science'].astype(int)
dropout_data['department_Engineering'] = dropout_data['department_Engineering'].astype(int)

df = copy.deepcopy(perf_data)#pd.read_csv('./data/temp_data.csv')

# Convert semester to numerical format for sorting
df['semester_num'] = df['semester'].apply(lambda x: int(x.split()[-1]) * 10 + 
                                         (1 if 'Fall' in x else 2))

# Sort by student_id and semester to ensure proper time sequence
df = df.sort_values(['student_id', 'semester_num'])

# Encode categorical variables
label_encoders = {}
categorical_cols = ['grade', 'completion_status']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create lagged features function
def create_lagged_features(group, lag_periods=5):
    """Create lagged features for each student up to 5 semesters"""
    features = []
    
    for lag in range(1, lag_periods + 1):
        # Lagged grade
        group[f'grade_lag_{lag}'] = group['grade_encoded'].shift(lag)
        
        # Lagged completion status
        group[f'status_lag_{lag}'] = group['completion_status_encoded'].shift(lag)
        
        # Grade trend features
        if lag > 1:
            group[f'grade_trend_lag_{lag}'] = group['grade_encoded'].shift(lag-1) - group['grade_encoded'].shift(lag)
            group[f'status_trend_lag_{lag}'] = group['completion_status_encoded'].shift(lag-1) - group['completion_status_encoded'].shift(lag)
    
    return group

# Apply lagged features creation
df = df.groupby('student_id').apply(create_lagged_features).reset_index(drop=True)

# Filter to only include students who have completed exactly 6 semesters
student_semester_counts = df.groupby('student_id')['semester'].count()
students_with_6_semesters = student_semester_counts[student_semester_counts == 6].index

df_6_semesters = df[df['student_id'].isin(students_with_6_semesters)]

# Use only the 6th semester record for each student with all historical data
df_final = df_6_semesters.groupby('student_id').tail(1).copy()

df_final = df_final.drop(columns=['course_id','course_name','grade','completion_status','semester','semester_num'])

# The target is the dropout status at the end of 6 semesters
df_final = df_final.merge(dropout_data,on='student_id',how='left')
df_final = df_final.drop(columns=['student_id'])

X, y = df_final.drop(columns=['dropout']), df_final['dropout']
grade_lag_cols = [col for col in X.columns if 'grade_lag' in col]
grade_trend_lag_cols = [col for col in X.columns if 'grade_trend_' in col]
status_lag_cols = [col for col in X.columns if 'status_lag' in col]
status_trend_lag_cols = [col for col in X.columns if 'status_trend_' in col]
other_cols = [col for col in X.columns if col not in grade_lag_cols + grade_trend_lag_cols + status_lag_cols + status_trend_lag_cols]
col_groups = ([size * [i] for i, size in enumerate([len(grade_lag_cols+grade_trend_lag_cols), len(status_lag_cols+status_trend_lag_cols)])])
col_groups.append(list(range(2,len(other_cols)+2)))
col_groups = np.concatenate(col_groups)
X = X[grade_lag_cols + grade_trend_lag_cols + status_lag_cols + status_trend_lag_cols+other_cols]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed_nb, stratify=y)

# # Feature Selection
# gl = LogisticGroupLasso(
#     groups=col_groups,
#     group_reg=0.05,
#     l1_reg=0,
#     scale_reg="inverse_group_size",
#     subsampling_scheme=1,
#     supress_warning=True,
# )
# gl.fit(X_train, np.array(y_train).reshape(len(y_train),1))
# selected_features = X_train.columns[gl.sparsity_mask_]
# X_train = X_train[selected_features]
# X_test = X_test[selected_features]

output_random_rf = train_randomsearch_evaluate_RF(random_seed_nb, n_folds, X_train, y_train, X_test, y_test)
param_rf_gs = {"n_estimators" : [99,100,101],
            "min_samples_split" : [2,3,4],
            "min_samples_leaf" : [3,4],
           "max_features" : ["sqrt"],
          "max_depth" : [19,20,21,None],
          "criterion" :['gini'],
          "bootstrap" : [True]}   
rf_model_outcome, rf_results_outcome, calibrated_rf_model_outcome, calibrated_rf_results_outcome = train_gridsearch_evaluate_RF(random_seed_nb, param_rf_gs, X_train, y_train, X_test, y_test)


output_random_xgb = train_randomsearch_evaluate_XGB(random_seed_nb, n_folds, X_train, y_train, X_test, y_test)
param_xgb_gs = {"n_estimators":[99,100,101],
        "scale_pos_weight":[1,2],
        "max_depth":[15,16,17],
        "gamma":[0.001],
        "colsample_bytree":[0.1],
        "learning_rate":[0.01]
}
xgb_model_outcome, xgb_results_outcome, calibrated_xgb_model_outcome, calibrated_xgb_results_outcome = train_gridsearch_evaluate_XGB(random_seed_nb, param_xgb_gs, X_train, y_train, X_test, y_test)

# ===== Results ===== #
# Save models and results
# Save non-calibrated models
os.makedirs('./results/best_models/time_series',exist_ok=True)
pickle.dump(rf_model_outcome,open('./results/best_models/time_series/random_forest_model_outcome.pkl','wb'))
pickle.dump(xgb_model_outcome,open('./results/best_models/time_series/xgboost_model_outcome.pkl','wb'))


# Save calibrated models
pickle.dump(calibrated_rf_model_outcome,open('./results/best_models/time_series/calibrated_random_forest_model_outcome.pkl','wb'))
pickle.dump(calibrated_xgb_model_outcome,open('./results/best_models/time_series/calibrated_xgboost_model_outcome.pkl','wb'))


# results will be formated and saved in .csv soon. For now just save in pickle.
pickle.dump(rf_results_outcome,open('./results/best_models/time_series/random_forest_results_outcome.pkl','wb'))
pickle.dump(xgb_results_outcome,open('./results/best_models/time_series/xgboost_results_outcome.pkl','wb'))


# Save calibrated results
pickle.dump(calibrated_rf_results_outcome,open('./results/best_models/time_series/calibrated_random_forest_results_outcome.pkl','wb'))
pickle.dump(calibrated_xgb_results_outcome,open('./results/best_models/time_series/calibrated_xgboost_results_outcome.pkl','wb'))

# Save output from randomized search
pickle.dump(output_random_rf,open('./results/best_models/time_series/output_random_rf.pkl','wb'))
pickle.dump(output_random_xgb,open('./results/best_models/time_series/output_random_xgb.pkl','wb'))

# Save train and test splits
pickle.dump(X_train,open('./results/best_models/time_series/X_train.pkl','wb'))
pickle.dump(X_test,open('./results/best_models/time_series/X_test.pkl','wb'))
pickle.dump(y_train,open('./results/best_models/time_series/y_train.pkl','wb'))
pickle.dump(y_test,open('./results/best_models/time_series/y_test.pkl','wb'))
print("All models and results saved")

result_colnames = ["auc", "acc", "prec", "recall", "f1", "tpr_final", "fpr"]
result_index_names = ["rf","calibrated_rf","xgb","calibrated_xgb"]
list_data = [rf_results_outcome, calibrated_rf_results_outcome, xgb_results_outcome, calibrated_xgb_results_outcome]
results_out = pd.DataFrame(list_data, index=result_index_names, columns=result_colnames)
results_out.to_csv('./results/best_models/time_series/model_results_summary.csv')