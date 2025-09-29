import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from lib.modelling_libraries import train_evaluate_LR, \
                                    train_randomsearch_evaluate_RF, \
                                    train_gridsearch_evaluate_RF, \
                                    train_randomsearch_evaluate_XGB, \
                                    train_gridsearch_evaluate_XGB

random_seed_nb = 42
n_folds = 5
perf_data = pd.read_csv('./data/course performance data.csv')
dropout_data = pd.read_csv('./data/student enrollment data.csv')
merged_data = pd.merge(perf_data, dropout_data, on='student_id',how='outer')
# merged_data = merged_data.replace({'semester': {'Fall 2020': 0, 'Spring 2021': 6, 'Fall 2021': 12, 'Spring 2022': 18,'Fall 2022': 24,'Spring 2023': 30}})

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
final_data = final_data.drop(columns=['gender_M','department_Design','enrollment_status_enrolled','enrollment_status_graduated','entry_year_2018'])
# Convert boolean columns to integers
final_data = final_data.astype({'gender_F': int, 'department_Computer Science': int, 'department_Business': int,'department_Engineering': int, 'enrollment_status_dropped out': int, 'entry_year_2019': int, 'entry_year_2020': int, 'entry_year_2021': int, 'entry_year_2022': int})

# Percentage of missing values per column
missing_percentage = (final_data.isnull().sum() / len(final_data)) * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)

final_data = final_data.drop(columns=['university_gpa','student_id'])

X, y = final_data.drop(columns=['enrollment_status_dropped out']), final_data['enrollment_status_dropped out']
X = X.drop(columns=['entry_year_2019','entry_year_2020','entry_year_2021','entry_year_2022'])  # to avoid dummy variable trap
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed_nb,stratify=y)

# Feature Selection Method 1: L1 Regression
l1_clf = LogisticRegressionCV(cv=n_folds, penalty='l1',
                        refit=True,
                        solver='liblinear', random_state=random_seed_nb,
                        class_weight='balanced',max_iter = 2000)
l1_model = l1_clf.fit(X_train, y_train)
retained_coefs_ind = (np.squeeze(abs(l1_model.coef_))>0)     

# Feature Selectin Method 2: Boruta
rf_boruta = RandomForestClassifier(n_estimators=100, random_state = random_seed_nb, max_depth=5)
boruta_selector = BorutaPy(estimator = rf_boruta, n_estimators='auto', random_state=random_seed_nb)
boruta_selector.fit(X_train.values, y_train.values)
selected_features_indices = np.where(boruta_selector.support_)[0]
selected_feature_names = X.columns[selected_features_indices].tolist()

# Random Forest for Outcome 1 and Outcome 2
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
os.makedirs('./results/best_models/non_time_series',exist_ok=True)
pickle.dump(rf_model_outcome,open('./results/best_models/non_time_series/random_forest_model_outcome.pkl','wb'))
pickle.dump(xgb_model_outcome,open('./results/best_models/non_time_series/xgboost_model_outcome.pkl','wb'))


# Save calibrated models
pickle.dump(calibrated_rf_model_outcome,open('./results/best_models/non_time_series/calibrated_random_forest_model_outcome.pkl','wb'))
pickle.dump(calibrated_xgb_model_outcome,open('./results/best_models/non_time_series/calibrated_xgboost_model_outcome.pkl','wb'))


# results will be formated and saved in .csv soon. For now just save in pickle.
pickle.dump(rf_results_outcome,open('./results/best_models/non_time_series/random_forest_results_outcome.pkl','wb'))
pickle.dump(xgb_results_outcome,open('./results/best_models/non_time_series/xgboost_results_outcome.pkl','wb'))


# Save calibrated results
pickle.dump(calibrated_rf_results_outcome,open('./results/best_models/non_time_series/calibrated_random_forest_results_outcome.pkl','wb'))
pickle.dump(calibrated_xgb_results_outcome,open('./results/best_models/non_time_series/calibrated_xgboost_results_outcome.pkl','wb'))

# Save output from randomized search
pickle.dump(output_random_rf,open('./results/best_models/non_time_series/output_random_rf.pkl','wb'))
pickle.dump(output_random_xgb,open('./results/best_models/non_time_series/output_random_xgb.pkl','wb'))

# Save train and test splits
pickle.dump(X_train,open('./results/best_models/non_time_series/X_train.pkl','wb'))
pickle.dump(X_test,open('./results/best_models/non_time_series/X_test.pkl','wb'))
pickle.dump(y_train,open('./results/best_models/non_time_series/y_train.pkl','wb'))
pickle.dump(y_test,open('./results/best_models/non_time_series/y_test.pkl','wb'))
print("All models and results saved")

result_colnames = ["auc", "acc", "prec", "recall", "f1", "tpr_final", "fpr"]
result_index_names = ["rf","calibrated_rf","xgb","calibrated_xgb"]
list_data = [rf_results_outcome, calibrated_rf_results_outcome, xgb_results_outcome, calibrated_xgb_results_outcome]
results_out = pd.DataFrame(list_data, index=result_index_names, columns=result_colnames)
results_out.to_csv('./results/best_models/non_time_series/model_results_summary.csv')
# References:
# 1. https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-017-0442-1#:~:text=The%20MAR%20and%20MNAR%20conditions,analyses%20to%20handle%20MNAR%20data.