# SHAP Visualization
import copy
import matplotlib
import matplotlib.pyplot as plt
import jsonref
import numpy as np
import pickle
import pandas as pd
import shap
import os

rf_model_outcome1 = pickle.load(open('./results/best_models/non_time_series/random_forest_model_outcome.pkl','rb')).best_estimator_
xgb_model_outcome1 = pickle.load(open('./results/best_models/non_time_series/xgboost_model_outcome.pkl','rb')).best_estimator_


df_train_imputed1 = pickle.load(open('./results/best_models/non_time_series/X_train.pkl','rb'))
df_test_imputed1 = pickle.load(open('./results/best_models/non_time_series/X_test.pkl','rb'))
y1_train = pickle.load(open('./results/best_models/non_time_series/y_train.pkl','rb'))
y1_test = pickle.load(open('./results/best_models/non_time_series/y_test.pkl','rb'))


X1_transform = pd.concat([df_train_imputed1, df_test_imputed1],axis=0)
y1_overall = pd.concat([y1_train,y1_test],axis=0)

current_variables = X1_transform.columns.tolist()
renamed_variables = ['Age', 'High School GPA', 'Percentage of failed modules', 'Percentage of Grade B', 'Percentage of Grade C', 'Percentage of Grade D', 'Percentage of Grade F', 'Female', 'Business', 'Computer Science', 'Engineering', 'Entry Year 2019', 'Entry Year 2020', 'Entry Year 2021', 'Entry Year 2022']
rename_dict = dict(zip(current_variables, renamed_variables))


print("performing shap on random forest models...")
X1_transform_rf = X1_transform.copy()
X1_transform_rf_display = X1_transform.rename(columns=rename_dict)                                           
explainer_rf1 = shap.TreeExplainer(rf_model_outcome1)
shap_values_rf1 = explainer_rf1.shap_values(X1_transform_rf)[1]


print("performing shap on xgboost models...")
# With shap version 0.42.1 and xgboost version 1.7.6
# Please see github closed issue on TreeExplainer error: 'utf-8' codec can't decode byte 0xff for xgboost model
# And resolution within the thread by bhishanpdl on Sep 30, 2020
# https://github.com/shap/shap/issues/1215
# Ideally we want to run TreeSHAP on XGBoost model but there was an issue where SHAP could not read the model data from XGBoost
# Seems like we need to read the booster output from the 5th index onwards for TreeSHAP to work on XGBoost model output
X1_transform_xgb = X1_transform.copy()
X1_transform_xgb_display = X1_transform.rename(columns=rename_dict)  
booster1 = xgb_model_outcome1.get_booster()
model_bytearray1 = booster1.save_raw()[4:]
booster1.save_raw = lambda: model_bytearray1                           
explainer_xgb1 = shap.TreeExplainer(booster1)
shap_values_xgb1 = explainer_xgb1.shap_values(X1_transform_xgb)

os.makedirs('./results/best_models/non_time_series/outcome1/shap_outputs',exist_ok=True)
os.makedirs('./results/best_models/non_time_series/outcome1/shap_outputs/business_dropouts',exist_ok=True)
os.makedirs('./results/best_models/non_time_series/outcome1/shap_outputs/business_students',exist_ok=True)

business_array = np.where(X1_transform['department_Business'] == 1)[0]
b_array = np.where((X1_transform['department_Business'] == 1) & (y1_overall == 1))[0]
# Force Plot
for b in b_array:
    plt.figure()
    shap.force_plot(explainer_rf1.expected_value[1], shap_values_rf1[b,:], X1_transform.iloc[b, :], matplotlib=True,show=False)
    plt.savefig('./results/best_models/non_time_series/outcome1/shap_outputs/business_dropouts/shap_force_plot_rf_model_'+str(b)+'.png')
    plt.close()

for bus in business_array:
    plt.figure()
    shap.force_plot(explainer_rf1.expected_value[1], shap_values_rf1[bus,:], X1_transform.iloc[bus, :], matplotlib=True,show=False)
    plt.savefig('./results/best_models/non_time_series/outcome1/shap_outputs/business_students/shap_force_plot_rf_model_'+str(bus)+'.png')
    plt.close()

for outcome_type in [1]:
    for model_str in ['rf','xgb']:
        for calibration_str in ['']:
            print('visualization for '+model_str+' in outcome '+str(outcome_type))
            shap_values_i = copy.deepcopy(locals()['shap_values_'+model_str+str(outcome_type)+calibration_str])
            if model_str == 'll':
                if calibration_str == '_calibrated':
                    X1_transform_i = copy.deepcopy(locals()['X'+str(outcome_type)+'_transform_'+model_str])
                    X1_transform_i_display = copy.deepcopy(locals()['X'+str(outcome_type)+'_transform_'+model_str+'_display'])
                else:
                    X1_transform_i = copy.deepcopy(add_constant(locals()['X'+str(outcome_type)+'_transform_'+model_str]))
                    X1_transform_i_display = copy.deepcopy(add_constant(locals()['X'+str(outcome_type)+'_transform_'+model_str+'_display']))                    
            else:
                X1_transform_i = copy.deepcopy(locals()['X'+str(outcome_type)+'_transform_'+model_str])
                X1_transform_i_display = copy.deepcopy(locals()['X'+str(outcome_type)+'_transform_'+model_str+'_display'])
            # Summary Plot (Bar)
            plt.figure()
            shap.summary_plot(shap_values_i, X1_transform_i, plot_type="bar",show=False,feature_names=X1_transform_i_display.columns)
            # Add this code
            print(f'Original size: {plt.gcf().get_size_inches()}')
            w, _ = plt.gcf().get_size_inches()
            plt.gcf().set_size_inches(w*6/4, w)
            plt.tight_layout()
            print(f'New size: {plt.gcf().get_size_inches()}')
            plt.savefig('./results/best_models/non_time_series/outcome'+str(outcome_type)+'/shap_outputs'+calibration_str+'/shap_summary_plot_bar_'+model_str+'_model.png', bbox_inches='tight',dpi=100)
            plt.close()

            # Summary Plot
            plt.figure()
            shap.summary_plot(shap_values_i, X1_transform_i, show=False, feature_names=X1_transform_i_display.columns)
            print(f'Original size: {plt.gcf().get_size_inches()}')
            w, _ = plt.gcf().get_size_inches()
            plt.gcf().set_size_inches(w*6/4, w)
            plt.tight_layout()
            print(f'New size: {plt.gcf().get_size_inches()}')
            plt.savefig('./results/best_models/non_time_series/outcome'+str(outcome_type)+'/shap_outputs'+calibration_str+'/shap_summary_plot_'+model_str+'_model.png')
            plt.close()
            print("SAVED all plots")
            # Save shap values
            shap_values_df = pd.DataFrame(shap_values_i, columns=X1_transform_i_display.columns)
            shap_values_sorted = pd.DataFrame(abs(shap_values_df).mean(axis=0).sort_values(ascending=False)).reset_index().rename(columns={'index':'Feature',0:'mean_abs_shap_value'})
            pickle.dump(shap_values_df,open('./results/best_models/non_time_series/outcome'+str(outcome_type)+'/shap_outputs'+calibration_str+'/shap_values_df_'+model_str+'.pkl','wb'))
            shap_values_sorted.to_csv('./results/best_models/non_time_series/outcome'+str(outcome_type)+'/shap_outputs'+calibration_str+'/shap_values_sorted_'+model_str+'.csv',index=False)