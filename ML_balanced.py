import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from model_evaluation import evaluate
from pathlib import Path


def count_files_with_string(directory, search_string):
    return sum(1 for file in Path(directory).iterdir() if file.is_file() and search_string in file.name)


def run_ml_balanced():

    obs_win=24
    pred_win=12
    model_name = 'GBM'
    results = pd.DataFrame(columns=['model', 'obs_win', 'pred_win', 'best_params', 'n_feat', 'train_auc_mean',
                                    'train_auc_sd', 'test_auc', 'test_sen_90', 'test_spec_90', 'test_precision_90',
                                    'test_npv_90', 'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden',
                                    'test_npv_yuden', 'acc_90', 'acc_yuden'])

    thres_df = pd.DataFrame(columns=['thres_90', 'thres_yuden'])

    indices = count_files_with_string('data_processed', 'train')
    for idx in range(0, indices):
        print(idx+1, '/', indices)
        df_train = pd.read_csv('data_processed/train_' + str(idx+1) + '.csv').iloc[:, :-1] #remove index
        X_df_train = df_train.iloc[:, :-1]
        y_df_train = df_train.iloc[:, -1]
        n_feat = len(X_df_train.columns)

        param_grid = [
            {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2],
             'n_estimators': [80, 100, 150, 200, 250, 300, 350, 400],
             'subsample': [0.8, 0.9, 1],
             'max_depth': [3, 4, 5],
             'max_features': [0.8, 0.9, 1]}]
        model = GradientBoostingClassifier(random_state=123)

        cv = StratifiedKFold(5)
        grid_search = RandomizedSearchCV(model, param_grid[0], cv=cv, scoring='roc_auc', random_state=123, n_iter=150, n_jobs=-1)
        grid_search.fit(X_df_train, y_df_train)
        best_params = str(grid_search.best_params_)
        best_model = grid_search.best_estimator_
        cvres = grid_search.cv_results_

        print(model_name, ' observation window: ', obs_win, ' prediction window: ', pred_win, ' features: ', n_feat)
        for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
            print(mean_score, params)
        train_auc_mean = cvres['mean_test_score'][grid_search.best_index_]
        train_auc_sd = cvres['std_test_score'][grid_search.best_index_]

        # test set
        df_test = pd.read_csv('data_processed/test_' + str(idx+1) + '.csv').iloc[:, :-1] #remove index
        X_df_test = df_test.iloc[:, :-1]
        y_df_test = df_test.iloc[:, -1]

        # predict on test set
        prob = best_model.predict_proba(X_df_test)[:, 1]
        (test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90,
         thres_yuden, acc_90, acc_yuden) = evaluate(prob, y_df_test, model_name, obs_win, pred_win, n_feat, acc=True)
        print(test_auc)

        # save results
        results.loc[len(results)] = [model_name, obs_win, pred_win, best_params, n_feat, train_auc_mean,
                                     train_auc_sd, test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden,
                                     precision_yuden, npv_yuden, acc_90, acc_yuden]

        results.to_csv('results/obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) +  '_results_balanced_3.csv', index=False)

        # save probs for each model
        prob = pd.DataFrame(prob)
        prob.to_csv(
            'predictions/' + str(model_name) + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat'
            + str(n_feat) + '_balanced_prob' + str(idx + 1) + '_3.csv', index=False)

        # save thres_90 and thres_yuden for each model
        thres_df.loc[len(thres_df)] = [thres_90, thres_yuden]
        thres_df.to_csv('thresholds/THRESROC_' + str(model_name) + '_obs' + str(obs_win) +
                        '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_balanced_3.csv')



def run_ml_balanced_encoded(model_name, obs_win, pred_win, n_splits):
    results = pd.DataFrame(columns=['model', 'obs_win', 'pred_win', 'best_params', 'n_feat', 'train_auc_mean',
                                    'train_auc_sd', 'test_auc', 'test_sen_90', 'test_spec_90', 'test_precision_90',
                                    'test_npv_90', 'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden',
                                    'test_npv_yuden', 'acc_90', 'acc_yuden'])

    thres_df = pd.DataFrame(columns=['thres_90', 'thres_yuden'])

    indices = count_files_with_string('data_processed', 'train')
    for idx in range(0, indices):
        print(idx+1, '/', indices)
        df_train = pd.read_csv('data_processed/train_' + str(idx+1) + '_encoded.csv').iloc[:, :-1] #remove index
        X_df_train = df_train.iloc[:, :-1]
        y_df_train = df_train.iloc[:, -1]
        n_feat = len(X_df_train.columns)

        param_grid = [
            {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2],
             'n_estimators': [80, 100, 150, 200, 250, 300, 350, 400],
             'subsample': [0.8, 0.9, 1],
             'max_depth': [3, 4, 5],
             'max_features': [0.8, 0.9, 1]}]
        model = GradientBoostingClassifier(random_state=123)

        cv = StratifiedKFold(n_splits)
        grid_search = RandomizedSearchCV(model, param_grid[0], cv=cv, scoring='roc_auc', random_state=123, n_iter=150, n_jobs=-1)
        grid_search.fit(X_df_train, y_df_train)
        best_params = str(grid_search.best_params_)
        best_model = grid_search.best_estimator_
        cvres = grid_search.cv_results_

        print(model_name, ' observation window: ', obs_win, ' prediction window: ', pred_win, ' features: ', n_feat)
        for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
            print(mean_score, params)
        train_auc_mean = cvres['mean_test_score'][grid_search.best_index_]
        train_auc_sd = cvres['std_test_score'][grid_search.best_index_]

        # test set
        df_test = pd.read_csv('data_processed/test_' + str(idx+1) + '_encoded.csv').iloc[:, :-1] #remove index
        X_df_test = df_test.iloc[:, :-1]
        y_df_test = df_test.iloc[:, -1]

        # predict on test set
        prob = best_model.predict_proba(X_df_test)[:, 1]
        (test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90,
         thres_yuden, acc_90, acc_yuden) = evaluate(prob, y_df_test, model_name, obs_win, pred_win, n_feat, acc=True)
        print(test_auc)

        # save results
        results.loc[len(results)] = [model_name, obs_win, pred_win, best_params, n_feat, train_auc_mean,
                                     train_auc_sd, test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden,
                                     precision_yuden, npv_yuden, acc_90, acc_yuden]

        results.to_csv('results/obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) +  '_results_balanced_3_encoded.csv', index=False)


        # save probs for each model
        prob = pd.DataFrame(prob)
        prob.to_csv(
            'predictions/' + str(model_name) + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat'
            + str(n_feat) + '_balanced_prob' + str(idx + 1) + '_3_encoded.csv', index=False)

        # save thres_90 and thres_yuden for each model
        thres_df.loc[len(thres_df)] = [thres_90, thres_yuden]
        thres_df.to_csv('thresholds/THRESROC_' + str(model_name) + '_obs' + str(obs_win) +
                            '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_balanced_3_encoded.csv')


def run_ml_average():
    results = pd.read_csv('results/obs24_pred12_feat70_results_balanced_3_encoded.csv')
    results_mean = results[['test_auc', 'test_sen_90', 'test_spec_90', 'test_precision_90',
                                    'test_npv_90', 'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden',
                                    'test_npv_yuden', 'acc_90', 'acc_yuden']].mean()
    results_sd = results[['test_auc', 'test_sen_90', 'test_spec_90', 'test_precision_90',
                                    'test_npv_90', 'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden',
                                    'test_npv_yuden', 'acc_90', 'acc_yuden']].std()
    print(results_mean)
    #print(results_sd)
