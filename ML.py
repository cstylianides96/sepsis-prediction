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


def generate_hidden_layer_combinations(sizes, max_layers, min_layers):
    """
    Generate all possible combinations of hidden layer sizes.

    Args:
        sizes (list): List of possible sizes for each layer.
        max_layers (int): Maximum number of layers.

    Returns:
        list: All combinations of hidden layer sizes.
    """
    combinations = []
    for num_layers in range(min_layers, max_layers + 1):
        for combo in itertools.product(sizes, repeat=num_layers):
            combinations.append(combo)
    return combinations


def run_ml(model_name, obs_win, pred_win, n_splits, n_feat_imp):
    results = pd.DataFrame(columns=['model', 'obs_win', 'pred_win', 'best_params', 'n_feat', 'train_auc_mean',
                                    'train_auc_sd', 'test_auc', 'test_sen_90', 'test_spec_90', 'test_precision_90',
                                    'test_npv_90', 'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden',
                                    'test_npv_yuden', 'thres_90', 'thres_yuden'])

    df_train = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_3.csv')
    df_train = df_train.loc[:, df_train.columns != 'adm_to_pred'] # exclude 'adm_to_pred'
    print(df_train.iloc[:, -1])
    X_df_train = df_train.iloc[:, :-1]

    # Edit to filter features included
    #X_df_train = X_df_train[['223901_0', '220546_1', '223830_1', '220052_16_diff', '221906_17', '223762_6', '224728_0', '220052_12_diff', '224755_range', '229374_14', '225624_5_diff', '228351_1', '220546_8_diff', '223762_6_diff', '225935_12', '224697_17', '223830_1_diff', '229761_15', '229011_0', '227529_1', '220052_15_diff', '229241_8', '220632_11', '220050_6_diff', '226329_range', '229295_0', '225152_1', '220045_10_diff', '223830_13', '228154_min', '223834_1', '220546_10_diff', '220546_0', '227525_1', '224697_15', '229240_max', '223834_17', '220645_max', '220546_4', 'J9601', '229014_3', '221906_3', 'B9561', '229297_0', '228154_17', '220050_5_diff', '220052_3', '229009_3', '223901_2_diff', 'age', '227547_7', '223762_2_diff', '223834_0', '227547_3', '220632_13', '223762_1_diff', '220050_2', '225810_min', '229010_14', '225936_0', '228198_4', '226534_17', '226272_14', '229374_12', '227066_12', '224422_max', '223830_0', '226774_mean', 'N390', '220052_1_diff']]

    y_df_train = df_train.iloc[:, -1]
    n_feat = len(X_df_train.columns)
    print(n_feat)

    scale_pos_weight = y_df_train.value_counts()[0]/y_df_train.value_counts()[1]

    if model_name == 'RF':
        param_grid = [
            {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 500, 600],
             'max_features': [1],
             'max_samples': [0.9]}]
        model = RandomForestClassifier(random_state=123, class_weight='balanced')

    elif model_name == 'XGB':
        param_grid = [
            {'n_estimators': [80, 100, 150, 200, 250, 300, 350, 400],
             'eta': [0.1], # increase to red overfitting
             'gamma': [0], # increase to red overfitting
             'max_depth': [3],  # decrease to red overfitting
             'subsample': [0.8, 0.9, 1],
             'colsample_bytree': [0.9, 1],
             'lambda':[0.0001],
            'alpha': [0.0001]}]
        model = XGBClassifier(random_state=123, scale_pos_weight=scale_pos_weight) #device="cuda"

    elif model_name == 'GBM':

        param_grid = [
            {'learning_rate': [0.1],
             'n_estimators': [200],
             'subsample': [ 1],
             'max_depth': [3],
             'max_features': [0.9]}]
        model = GradientBoostingClassifier(random_state=123)

    elif model_name == 'LGBM':
        param_grid = [
            {'n_estimators': [50, 100],
             'learning_rate': [0.0001, 0.001, 0.01, 0.1],
             'max_depth': [4, 5],
             'subsample': [0.9, 1],
             'colsample_bytree': [0.9, 1],
             'reg_alpha': [0.0001, 0.001, 0.01, 0.1],
             'reg_lambda': [0.0001, 0.001, 0.01, 0.1]}]
        model = LGBMClassifier(objective='binary', random_state=123, n_jobs=-1, verbosity=-1, scale_pos_weight=scale_pos_weight) # device='gpu'

    elif model_name == 'ADABOOST':
        param_grid = [
            {'n_estimators': [150, 200, 250, 300, 350, 400],
             'learning_rate': [0.1, 0.2, 0.3, 0.4]}]
        model = AdaBoostClassifier(random_state=123)  # default estimator DecisionTreeClassifier

    elif model_name == 'MLP':
        sizes = [16, 32]
        max_layers = 4
        min_layers = 1
        hidden_layer_combinations = generate_hidden_layer_combinations(sizes, max_layers, min_layers)
        param_grid = [{
            'hidden_layer_sizes': hidden_layer_combinations,
            'alpha': [0, 0.0001, 0.001],
            'learning_rate_init': [0.01],
            'batch_size': [600],
            'max_iter': [35]}] #(epochs)
        model = MLPClassifier(random_state=123)

    cv = StratifiedKFold(n_splits)
    grid_search = RandomizedSearchCV(model, param_grid[0], cv=cv, scoring='roc_auc', random_state=123, n_iter=100, n_jobs=-1)
    grid_search.fit(X_df_train, y_df_train)
    best_params = str(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    # print(best_params, best_model)
    cvres = grid_search.cv_results_

    print(model_name, ' observation window: ', obs_win, ' prediction window: ', pred_win, ' features: ', n_feat)
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
    joblib.dump(best_model, 'models/' + model_name + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_3.pkl')

    # test set
    usecols = list(X_df_train.columns) + ['label']
    df_test = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_3.csv', usecols=usecols)
    X_df_test = df_test.iloc[:, :-1]
    y_df_test = df_test.iloc[:, -1]
    X_df_test = X_df_test[X_df_train.columns]

    # predict on test set
    # pred = best_model.predict(X_df_test)
    prob = best_model.predict_proba(X_df_test)[:, 1]
    test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90, thres_yuden = evaluate(prob, y_df_test, model_name, obs_win, pred_win, n_feat, acc=False)
    print(test_auc)

    # save results_set5
    results.loc[len(results)] = [model_name, obs_win, pred_win, best_params, n_feat, train_auc_mean,
                                 train_auc_sd, test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden,
                                 precision_yuden, npv_yuden, thres_90, thres_yuden]
    prob = pd.DataFrame(prob)
    prob.to_csv('predictions/' + str(model_name) + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat'
                + str(n_feat) + '_prob_3.csv', index=False)

    if not os.path.isfile('results/obs' + str(obs_win) + '_pred' + str(pred_win) + '_results_3.csv'):
        results.to_csv('results/obs' + str(obs_win) + '_pred' + str(pred_win) + '_results_3.csv', index=False)
    else:
        results.to_csv('results/obs' + str(obs_win) + '_pred' + str(pred_win) + '_results_3.csv', mode='a',
                       header=False, index=False)

    #plot most important features (in training set)
    if model_name != 'MLP':
        plot_feat_importances(model_name, obs_win, pred_win, n_feat, best_model)


# run_ml(model_name='GBM', obs_win = 24, pred_win = 12, n_splits = 5, n_feat_imp = 60)

# def create_ml_models():
#     models = ['RF', 'XGB', 'GBM', 'ADABOOST', 'LGBM', 'MLP']
#     obs_wins = [4, 8, 12, 18, 24]
#     pred_wins = [12]
#     feats = [500]
#     n_splits = 5
#
#     for model in models:
#         for obs_win in obs_wins:
#             for pred_win in pred_wins:
#                 for f in feats:
#                     params = {'model_name': model, 'obs_win': obs_win, 'pred_win': pred_win, 'n_splits': n_splits, 'n_feat_imp': 60}
#                     run_ml(**params)
