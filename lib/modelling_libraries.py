import numpy as np
import pandas as pd
import jsonref
from sklearn.metrics import accuracy_score, \
                           confusion_matrix, \
                           precision_score, \
                           recall_score, \
                           f1_score, \
                           roc_curve, \
                           auc
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
from statsmodels.tools import add_constant
import statsmodels.api as sm
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

modelling_config = jsonref.load(open('./config/modelling_config.json'))

def evaluate(model, X_test, y_test, y_pred, metric, model_type):
  # y_pred refers to the predicted labels (instead of probabilities)
    if model_type == "sklearn":
        y_proba_pred = model.predict_proba(X_test)[:,1]
    elif model_type == "statsmodel":
        y_proba_pred = model.predict(X_test)
    if metric == "auc":
        fpr, tpr, thresholds = roc_curve(y_test, y_proba_pred)
        auc_score = auc(fpr,tpr)
        out = (fpr,tpr,thresholds,auc_score)
    elif metric == "accuracy":
        out = accuracy_score(y_test, y_pred)
    elif metric == "precision":
        out = precision_score(y_test, y_pred)
    elif metric == "recall":
        out = recall_score(y_test, y_pred) 
    elif metric == "f1":
        out = f1_score(y_test, y_pred)
    elif metric == "confusion_matrix":
        out = confusion_matrix(y_test, y_pred).ravel()
    return out   

def assess_model_performance(model,X,y,model_type,cutoff_val=None):
    fpr, tpr, thresholds, auc = evaluate(model,X,y,"","auc",model_type)
    i = np.arange(len(tpr)) # index for df
    # Reference for the following to identify cut-off threshold is found here: https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    J = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    if model_type == "statsmodel":
        y_proba_pred = model.predict(X)
    elif model_type == "sklearn":
        y_proba_pred = model.predict_proba(X)[:,1]
    if cutoff_val == None:    
        cutoff = J.iloc[(J.tf-0).abs().argsort()].iloc[0]['thresholds']
        tpr_final = J.iloc[(J.tf-0).abs().argsort()].iloc[0]['tpr']
        fpr_final = J.iloc[(J.tf-0).abs().argsort()].iloc[0]['fpr']
        y_pred = (y_proba_pred>=cutoff).astype(int)
    else:
        cutoff = cutoff_val
        y_pred = (y_proba_pred>=cutoff).astype(int)
    acc = evaluate(model, X, y, y_pred, "accuracy",model_type)
    prec = evaluate(model, X, y, y_pred, "precision",model_type)
    recall = evaluate(model, X, y, y_pred, "recall",model_type)
    f1 = evaluate(model, X, y, y_pred, "f1",model_type)
    return auc, acc, prec, recall, f1, tpr_final, fpr_final

# This function is not used but kept here for future reference. Using mice instead. 
def impute_missing_data(X_to_be_fitted, X_to_be_transformed, input_strategy='mean'):
    if input_strategy == 'mean':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X_to_be_fitted)
        X_transformed = imp.transform(X_to_be_transformed)
    # =================================== #
    return X_transformed

def train_evaluate_LR(random_seed_nb, n_folds, X_train, y_train, X_test, y_test):
    print("Training Lasso Logistic Regression")
    # When refit is True, scores are averaged across all folds, 
    # and the coefs and the hyperparameter C that corresponds to the best score is taken, 
    # and a final refit is done using these parameters.
    # class_weight is set to balanced and an imbalanced dataset is handled within the algorithm. 
    # penalty set to l1 to execute lasso.
    # solver is set to liblinear as it is a good choice for smaller datasets and supports L1 regularization. 
    lr_clf = LogisticRegressionCV(cv=n_folds, penalty='l1',
                            refit=True,
                            solver='liblinear', random_state=random_seed_nb,
                            class_weight='balanced',max_iter = 2000)
    lr_model = lr_clf.fit(X_train, y_train)
    # Extract confidence intervals
    # https://stats.stackexchange.com/questions/471901/how-to-get-a-confidence-interval-around-the-output-of-logistic-regression
    col_names = X_train.columns
    retained_coefs_ind = (np.squeeze(abs(lr_model.coef_))>0)
    retained_cols = list(col_names[retained_coefs_ind])
    X_train_retained = X_train.loc[:,retained_cols]
    X_train_retained_beta0 = add_constant(X_train_retained) # This appends a column of ones to the left of X_train_retained as we want to obtain an estimate of the coefficient for the intercept
    X_test_retained = X_test.loc[:,retained_cols]
    X_test_retained_beta0 = add_constant(X_test_retained) # This appends a column of ones to the left of X_test_retained as we want to obtain an estimate of the coefficient for the intercept

    X_train_retained_beta0_new = pd.DataFrame(X_train_retained_beta0)
    X_test_retained_beta0_new = pd.DataFrame(X_test_retained_beta0)
    y_train_new = np.expand_dims(y_train,axis=1)
    y_test_new = np.expand_dims(y_test,axis=1)
    print("Fitting statsmodel on reduced features now")
    lr_model_sm = sm.Logit(y_train_new, X_train_retained_beta0_new).fit(method='lbfgs',maxiter=5000)
    lr_model_sm_results = assess_model_performance(lr_model_sm,X_test_retained_beta0_new,y_test_new,"statsmodel")

    selected_features = pd.concat([lr_model_sm.params[lr_model_sm.pvalues<modelling_config['lr_siglevel']],lr_model_sm.pvalues[lr_model_sm.pvalues<modelling_config['lr_siglevel']],lr_model_sm.conf_int()[lr_model_sm.pvalues<modelling_config['lr_siglevel']]],axis=1)
    selected_features.columns = ['coef','p-values','0.025','0.975']
    # As we could not run calibration on statsmodel, we model the data using LogisticRegression from sklearn instead.
    # Coefficients between statsmodel and sklearn output is similar.
    # https://stackoverflow.com/questions/62005911/coefficients-for-logistic-regression-scikit-learn-vs-statsmodels
    print("Fitting Calibrated LogisticRegression model on reduced features now!!!!!!!!!!!!!!!!!!!!!!")
    res_sk = LogisticRegression(solver='lbfgs',max_iter=2000,fit_intercept=True,penalty=None)
    res_sk.fit(X_train_retained_beta0_new.drop(columns='const'),y_train)
    calibrated_lr = CalibratedClassifierCV(res_sk, method = 'sigmoid')
    calibrated_lr.fit(X_train_retained_beta0_new.drop(columns='const'), y_train)
    calibrated_lr_results = assess_model_performance(calibrated_lr,X_test_retained_beta0_new.drop(columns='const'),y_test,"sklearn")

    return lr_model_sm, selected_features, lr_model_sm_results, calibrated_lr, calibrated_lr_results

def train_randomsearch_evaluate_RF(random_seed_nb, n_folds, X_train, y_train, X_test, y_test):
    # Hyperparameter Tuning (RandomizedSearchCV)
    # Create the random grid
    param_dist = modelling_config['rf_random_param_grid']
    model_rf = RandomForestClassifier(random_state=random_seed_nb, class_weight = 'balanced')
    model_rf_random_cv = RandomizedSearchCV(model_rf, param_dist, cv=n_folds, random_state = random_seed_nb, scoring = 'neg_log_loss') # scoring is neg_log_loss as requested, else the default will be accuracy_score for the RandomForestClassifier
    print("fitting rf model now")
    model_rf_random_cv.fit(X_train, y_train) # Fit it to the data
    print("Random Forest RandomizedSearchCV Parameters: {}".format(model_rf_random_cv.best_params_)) # Print the tuned parameters and score
    # rs_eval_out = evaluate(model_rf_random_cv.best_estimator_, X_test, y_test, "", "auc","sklearn")
    rs_eval_out = assess_model_performance(model_rf_random_cv.best_estimator_,X_test,y_test,"sklearn")
    print("Best Test AUC with random search is {}".format(rs_eval_out[0]))

    calibrated_random_rf = CalibratedClassifierCV(model_rf_random_cv.best_estimator_, method = 'sigmoid')
    calibrated_random_rf.fit(X_train, y_train)
    calibrated_random_rf_results = assess_model_performance(calibrated_random_rf,X_test,y_test,"sklearn")

    return model_rf_random_cv,rs_eval_out,calibrated_random_rf, calibrated_random_rf_results,model_rf_random_cv.best_params_

def train_gridsearch_evaluate_RF(random_seed_nb, param_rf_gs, X_train, y_train, X_test, y_test):
    # grid search for randomsearch
    model = RandomForestClassifier(random_state=random_seed_nb,class_weight='balanced') #class_weight = 'balanced_subsample'

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_seed_nb)
    grid = GridSearchCV(estimator=model, param_grid=param_rf_gs, n_jobs=-1, cv=cv,scoring='neg_log_loss') # scoring is neg_log_loss as requested, else the default will be accuracy_score for the RandomForestClassifier
    grid_result = grid.fit(X_train, y_train)
    gs_eval_out = evaluate(grid_result.best_estimator_, X_test, y_test, "","auc","sklearn")
    print("AUC from Best Trained Model: %f using %s" % (gs_eval_out[3], grid_result.best_params_))
    rf_model_results = assess_model_performance(grid_result.best_estimator_,X_test,y_test,"sklearn")
    print("Best Test AUC with grid search is {}".format(rf_model_results[0]))

    calibrated_grid_rf = CalibratedClassifierCV(grid_result.best_estimator_, method = 'sigmoid')
    calibrated_grid_rf.fit(X_train, y_train)
    calibrated_grid_rf_results = assess_model_performance(calibrated_grid_rf,X_test,y_test,"sklearn")

    return grid_result, rf_model_results, calibrated_grid_rf, calibrated_grid_rf_results

def train_randomsearch_evaluate_XGB(random_seed_nb, n_folds, X_train, y_train, X_test, y_test):
    # Hyperparameter Tuning (RandomizedSearchCV)
    # Guide to hyperparameter tuning in xgboost: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    model_xgb = XGBClassifier(random_state = random_seed_nb)
    model_xgb_random_cv = RandomizedSearchCV(model_xgb, modelling_config['xgb_random_param_grid'], cv=n_folds, random_state = random_seed_nb,scoring = 'neg_log_loss')
    model_xgb_random_cv.fit(X_train, y_train) # Fit it to the data
    print("XGB RandomizedSearchCV Parameters: {}".format(model_xgb_random_cv.best_params_)) # Print the tuned parameters and score
    xgb_eval_out = assess_model_performance(model_xgb_random_cv.best_estimator_,X_test,y_test,"sklearn")
    print("Best Test AUC with random search is {}".format(xgb_eval_out[0]))

    calibrated_random_xgb = CalibratedClassifierCV(model_xgb_random_cv.best_estimator_, method = 'sigmoid')
    calibrated_random_xgb.fit(X_train, y_train)
    calibrated_random_xgb_results = assess_model_performance(calibrated_random_xgb,X_test,y_test,"sklearn")

    return model_xgb_random_cv, xgb_eval_out,calibrated_random_xgb, calibrated_random_xgb_results, model_xgb_random_cv.best_params_

def train_gridsearch_evaluate_XGB(random_seed_nb, param_xgb_gs, X_train, y_train, X_test, y_test):
    model = XGBClassifier(random_state = random_seed_nb)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_seed_nb)
    grid_xgb = GridSearchCV(estimator=model, param_grid=param_xgb_gs, n_jobs=-1, cv=cv, scoring='neg_log_loss')
    print("Fitting XGB gridsearch model now")
    grid_result_xgb = grid_xgb.fit(X_train, y_train)
    print("Fitting of XGB gridsearch model done")
    #gs_eval_out_xgb = evaluate(grid_result_xgb.best_estimator_, X_test, y_test, "","auc","sklearn")
    xgb_model_results = assess_model_performance(grid_result_xgb.best_estimator_,X_test,y_test,"sklearn")
    print("Best Test AUC from grid search: %f using %s" % (xgb_model_results[0], grid_result_xgb.best_params_))
    #print("Best Test AUC with grid search is {}".format(xgb_model_results[0]))

    calibrated_grid_xgb = CalibratedClassifierCV(grid_result_xgb.best_estimator_, method = 'sigmoid')
    print("Fitting calibrated XGB model now")
    calibrated_grid_xgb.fit(X_train, y_train)
    print("Fitting of calibrated XGB model done")
    calibrated_grid_xgb_results = assess_model_performance(calibrated_grid_xgb,X_test,y_test,"sklearn")

    return grid_result_xgb, xgb_model_results, calibrated_grid_xgb, calibrated_grid_xgb_results