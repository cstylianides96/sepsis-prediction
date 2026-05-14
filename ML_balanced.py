# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from model_evaluation import evaluate


def run_ml_balanced(encoded=False):

    model_name = 'GBM'
    results = pd.DataFrame(columns=['model', 'best_params', 'n_feat', 'train_auc_mean', 'train_auc_sd', 'test_auc', 
                                    'test_sen_90', 'test_spec_90', 'test_precision_90','test_npv_90', 
                                    'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden', 'test_npv_yuden', 
                                    'thres_90', 'thres_yuden', 'acc_90', 'acc_yuden'])

    for idx in range(0, 40):
        print(idx+1, '/', 40)
        if encoded:
            df_train = pd.read_csv('/data_processed/train_' + str(idx+1) + '_encoded.csv').iloc[:, :-1] #remove index
        else:
            df_train = pd.read_csv('/data_processed/train_' + str(idx+1) + '.csv').iloc[:, :-1] #remove index
        X_df_train = df_train.iloc[:, :-1]
        y_df_train = df_train.iloc[:, -1]
        print(y_df_train.value_counts())
        n_feat = len(X_df_train.columns)

        param_grid = [
            {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2],
             'n_estimators': [80, 100, 150, 200, 250, 300],
             'subsample': [0.8, 0.9, 1],
             'max_depth': [3, 4, 5, 6],
             'max_features': [0.8, 0.9, 1]}]
        model = GradientBoostingClassifier(random_state=123)

        cv = StratifiedKFold(5)
        grid_search = RandomizedSearchCV(model, param_grid[0], cv=cv, scoring='roc_auc', random_state=123, n_iter=150, n_jobs=-1)
        grid_search.fit(X_df_train, y_df_train)
        best_params = str(grid_search.best_params_)
        best_model = grid_search.best_estimator_
        cvres = grid_search.cv_results_

        for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
            print(mean_score, params)
        train_auc_mean = cvres['mean_test_score'][grid_search.best_index_]
        train_auc_sd = cvres['std_test_score'][grid_search.best_index_]

        # test set
        if encoded:
            df_test = pd.read_csv('/data_processed/test_' + str(idx+1) + '_encoded.csv').iloc[:, :-1] #remove index
        else:
            df_test = pd.read_csv('/data_processed/test_' + str(idx+1) + '.csv').iloc[:, :-1] #remove index
        X_df_test = df_test.iloc[:, :-1]
        y_df_test = df_test.iloc[:, -1]
        print(y_df_test.value_counts())

        # predict on test set
        prob = best_model.predict_proba(X_df_test)[:, 1]
        test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90, thres_yuden, acc_90, acc_yuden = evaluate(prob, y_df_test, acc=True)
        print(test_auc)

        # save results
        results.loc[len(results)] = [model_name, best_params, n_feat, train_auc_mean, train_auc_sd, 
                                 test_auc, sen_90, spec_90, precision_90, npv_90, 
                                 sen_yuden, spec_yuden, precision_yuden, npv_yuden, 
                                 thres_90, thres_yuden, acc_90, acc_yuden]

        if encoded:
            results.to_csv('/results/ML_results_balanced_encoded.csv', index=False)
        else:
            results.to_csv('/results/ML_results_balanced.csv', index=False)

        # save probs for each model
        prob = pd.DataFrame(prob)

        if encoded:
            prob.to_csv('/predictions/ML_prob_balanced_' + str(idx + 1) + '_encoded.csv', index=False)
        else:
            prob.to_csv('/predictions/ML_prob_balanced_' + str(idx + 1) + '.csv', index=False)


def run_ml_average(encoded=False):

    if encoded:
        results = pd.read_csv('/results/ML_results_balanced_encoded.csv')
    else:
        results = pd.read_csv('/results/ML_results_balanced.csv')

    results_mean = results[['train_auc_mean', 'test_auc', 
                                    'test_sen_90', 'test_spec_90', 'test_precision_90','test_npv_90', 
                                    'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden', 'test_npv_yuden', 
                                    'acc_90', 'acc_yuden']].mean()
    results_sd = results[['train_auc_mean', 'test_auc', 
                                    'test_sen_90', 'test_spec_90', 'test_precision_90','test_npv_90', 
                                    'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden', 'test_npv_yuden', 
                                    'acc_90', 'acc_yuden']].std()
    print(results_mean)
    #print(results_sd)


# ML BALANCED
# train_auc_mean          0.868304
# test_auc                0.871052
# test_sen_90             0.905505
# test_spec_90            0.580504
# test_precision_90       0.684593
# test_npv_90             0.859088

# test_sen_yuden          0.766972
# test_spec_yuden         0.824886
# test_precision_yuden    0.817580
# test_npv_yuden          0.782454
# acc_90                  0.743004
# acc_yuden               0.795929


# ENSEMBLE 
# test_auc                0.873292
# test_sen_90             0.905161
# test_spec_90            0.593120
# test_precision_90       0.691125
# test_npv_90             0.861287

# test_sen_yuden          0.792547
# test_spec_yuden         0.808716
# test_precision_yuden    0.808249
# test_npv_yuden          0.798001
# acc_90                  0.749140
# acc_yuden               0.800631
