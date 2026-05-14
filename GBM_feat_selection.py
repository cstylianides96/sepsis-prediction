# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import joblib
from model_evaluation import evaluate
import os
import itertools
from plot import plot_feat_importances
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def run_gbm(n_feat_imp):
    results = pd.DataFrame(columns=['model', 'best_params', 'n_feat', 'train_auc_mean', 'train_auc_sd', 'val_auc', 
                                    'val_sen_90', 'val_spec_90', 'val_precision_90','val_npv_90', 
                                    'val_sen_yuden', 'val_spec_yuden', 'val_precision_yuden', 'val_npv_yuden', 
                                    'thres_90', 'thres_yuden'])

    df_train = pd.read_csv('/data_processed/train_selected.csv')
    X_df_train = df_train.iloc[:, :-1]
    y_df_train = df_train.iloc[:, -1]
    if 'stay_id' in X_df_train.columns:
        X_df_train = X_df_train.drop('stay_id', axis=1)

    # Edit to filter features included
    X_df_train = X_df_train[['220051_diff_0', 'hosp_to_icu', '223900_max', '220546_range', '220621_range', '225624_diff_1', '220621_diff_1', '223900_23', '220545_diff_2', '220050_range', '220052_14', '225612_diff_18', '225624_diff_0', '220621_diff_0', '223901_0', '220615_mean', '220645_diff_1', '220050_diff_0', '220045_diff_0', '225677_range', '220050_diff_20', '220052_diff_10', '220051_diff_2', '225624_diff_2', '223762_min', '220645_mean', '220052_diff_19', '220052_diff_0', '220210_range', '227468_14', '225641_diff_0', '220050_21', '220545_4', '220632_diff_5', '225641_diff_1', '227444_diff_12', '220635_3', '225641_diff_22', '220045_diff_1', '220632_min', '220632_diff_0', '225624_diff_4', '227468_diff_10', '220045_diff_18', '227467_max', '220045_diff_13', '220621_diff_2', '220545_diff_0', '220045_diff_7', '225612_diff_20']]
    
    # Running final model
    # model = joblib.load('/models/GBM_feat_selection_feat50(40).pkl')
    # model_feats = model.feature_names_in_.tolist()
    # print(model_feats)
    # X_df_train = X_df_train[model_feats]

    print(y_df_train.value_counts())
    n_feat = len(X_df_train.columns)
    print(n_feat)
   
    model_name = 'GBM_feat_selection'

    param_grid = [
        {'learning_rate': [0.1],
            'n_estimators': [270],
            'subsample': [0.9],
            'max_depth': [3],
            'max_features': [1.0]}]
    model = GradientBoostingClassifier(random_state=123)

    cv = StratifiedKFold(5)
    grid_search = RandomizedSearchCV(model, param_grid[0], cv=cv, scoring='roc_auc', random_state=123, n_iter=100, n_jobs=-1)
    grid_search.fit(X_df_train, y_df_train)
    best_params = str(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    # print(best_params, best_model)
    cvres = grid_search.cv_results_

    print(' features: ', n_feat)
    for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
        print(mean_score, params)
    train_auc_mean = cvres['mean_test_score'][grid_search.best_index_]
    train_auc_sd = cvres['std_test_score'][grid_search.best_index_]

    if n_feat_imp: # fit again based on feature importance, use top n_feat_imp features
        importances = best_model.feature_importances_
        feature_importances = pd.DataFrame({'Feature': X_df_train.columns, 'Importance': importances})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        print(feature_importances)
        top_features = feature_importances.head(n_feat_imp)['Feature'].tolist()
        print(len(top_features))

        X_df_train = X_df_train[top_features]
        best_model.fit(X_df_train, y_df_train)
        train_prob = best_model.predict_proba(X_df_train)[:, 1]
        train_auc_mean = np.round(roc_auc_score(y_df_train, train_prob), 5)
        train_auc_sd = '-'
        n_feat = str(n_feat) + '(' + str(n_feat_imp) + ')'

    # save model
    joblib.dump(best_model, '/models/GBM_feat_selection_feat' + str(n_feat) + '.pkl')

    # val set evaluation
    usecols = list(X_df_train.columns) + ['label']
    df_val = pd.read_csv('/data_processed/val_selected.csv', usecols=usecols)
    X_df_val = df_val.iloc[:, :-1]
    y_df_val = df_val.iloc[:, -1]
    print(y_df_val.value_counts())
    X_df_val = X_df_val[X_df_train.columns]
    prob = best_model.predict_proba(X_df_val)[:, 1]
    val_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90, thres_yuden = evaluate(prob, y_df_val, acc=False)
    print(val_auc)

    # save results_set5
    results.loc[len(results)] = [model_name, best_params, n_feat, train_auc_mean, train_auc_sd, 
                                 val_auc, sen_90, spec_90, precision_90, npv_90, 
                                 sen_yuden, spec_yuden, precision_yuden, npv_yuden, 
                                 thres_90, thres_yuden]
    
    if not os.path.isfile('/results/GBM_feat_selection_results.csv'):
        results.to_csv('/results/GBM_feat_selection_results.csv', index=False)
    else:
        results.to_csv('/results/GBM_feat_selection_results.csv', mode='a',header=False, index=False)

    #plot most important features (in training set)
    plot_feat_importances(model_name, n_feat, best_model)



def gbm_feat_selection():
    model = joblib.load('/models/GBM_feat_selection_feat50(40).pkl')
    features = model.feature_names_in_.tolist()
    print(len(features))
    train_df_new = pd.read_csv('/data_processed/train_selected.csv', usecols=features + ['label'])
    train_df_new.to_csv('/data_processed/train_selected_feat40.csv', index=False)
    val_df_new = pd.read_csv('/data_processed/val_selected.csv', usecols=features + ['label'])
    val_df_new.to_csv('/data_processed/val_selected_feat40.csv', index=False)
    test_df_new = pd.read_csv('/data_processed/test_selected.csv', usecols=features + ['label'])
    test_df_new.to_csv('/data_processed/test_selected_feat40.csv', index=False)
