# Python code to predict dropout using time series data with Random Forest Classifier.
# Created lagged variable using the time column semester with features as completion status and grade and the outcome is dropout.

import pandas as pd
import numpy as np
import copy
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
dropout_data = pd.get_dummies(dropout_data, columns=['gender','department'],drop_first=False).drop(columns=['gender_M','department_Computer Science'])
dropout_data['gender_F'] = dropout_data['gender_F'].astype(int)
dropout_data['department_Business'] = dropout_data['department_Business'].astype(int)
dropout_data['department_Design'] = dropout_data['department_Design'].astype(int)
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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed_nb, stratify=y)

output_random_rf = train_randomsearch_evaluate_RF(random_seed_nb, n_folds, X_train, y_train, X_test, y_test)
output_random_xgb = train_randomsearch_evaluate_XGB(random_seed_nb, n_folds, X_train, y_train, X_test, y_test)
# # Feature importance
# feature_importance = pd.DataFrame({
#     'feature': features,
#     'importance': rf_classifier.feature_importances_
# }).sort_values('importance', ascending=False)

# print("\nTop 10 Most Important Features:")
# print(feature_importance.head(10))

# # Display some predictions with probabilities
# results_df = pd.DataFrame({
#     'student_id': df_final.loc[X_test.index, 'student_id'].values,
#     'actual_dropout': y_test.values,
#     'predicted_dropout': y_pred,
#     'dropout_probability': y_pred_proba
# })

# print("\nSample Predictions (First 10 students):")
# print(results_df.head(10))

# # Function to predict dropout for new students after 6 semesters
# def predict_dropout_after_6_semesters(student_data, model, feature_names):
#     """
#     Predict dropout probability for a student after completing 6 semesters
    
#     Parameters:
#     student_data: DataFrame with 6 semesters of data for one student
#     model: Trained Random Forest model
#     feature_names: List of feature names used in training
    
#     Returns:
#     prediction: 0 or 1 (not dropout vs dropout)
#     probability: Probability of dropout
#     """
#     # Ensure data is sorted by semester
#     student_data = student_data.sort_values('semester_num')
    
#     # Encode categorical variables
#     for col in categorical_cols:
#         student_data[col + '_encoded'] = label_encoders[col].transform(student_data[col])
    
#     # Create lagged features (5 lags for 6 semesters of data)
#     student_data = create_lagged_features(student_data, lag_periods=5)
    
#     # Use only the last semester (6th semester)
#     latest_record = student_data.iloc[[-1]]
    
#     # Prepare features
#     X_new = latest_record[feature_names].fillna(-1)  # Fill missing values
    
#     # Make prediction
#     prediction = model.predict(X_new)[0]
#     probability = model.predict_proba(X_new)[0][1]
    
#     return prediction, probability

# # Example usage with sample student data
# print("\n" + "="*50)
# print("EXAMPLE: PREDICTING FOR A SAMPLE STUDENT")
# print("="*50)

# # Get a sample student from the test set
# sample_student_id = results_df.iloc[0]['student_id']
# sample_student_data = df[df['student_id'] == sample_student_id]

# print(f"Student ID: {sample_student_id}")
# print(f"Actual dropout status: {results_df.iloc[0]['actual_dropout']}")

# prediction, probability = predict_dropout_after_6_semesters(
#     sample_student_data, rf_classifier, features
# )

# print(f"Predicted dropout: {prediction}")
# print(f"Dropout probability: {probability:.4f}")

# # Summary statistics
# print("\n" + "="*50)
# print("SUMMARY STATISTICS")
# print("="*50)
# print(f"Total students in dataset: {df['student_id'].nunique()}")
# print(f"Students with 6 semesters: {len(students_with_6_semesters)}")
# print(f"Dropout rate among 6-semester students: {y.mean():.2%}")

# # Comments:


# # Single Prediction Point: Only predicts dropout at the end of 6 semesters

# # Data Filtering: Uses only students who have exactly 6 semesters of data

# # Single Record per Student: Uses only the 6th semester record for prediction

# # Historical Features: Includes lagged features from all previous 5 semesters

# # Final Status: Uses the dropout status from the 6th semester as the target

# # How it works:

# # For each student with 6 semesters of data, we use their 6th semester record

# # The features include current semester data + lagged data from semesters 1-5

# # The target is whether the student dropped out by the end of semester 6

# # This creates a single prediction point per student at the completion of their 6th semester

# # This approach is more realistic for educational institutions that want to predict which students are at risk of dropping out after completing a specific program duration (6 semesters in this case).