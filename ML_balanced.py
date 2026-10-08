# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from model_evaluation import evaluate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


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


def run_ml_balanced_smote():
    model_name = 'GBM'
    results = pd.DataFrame(columns=['model', 'best_params', 'n_feat', 'train_auc_mean', 'train_auc_sd', 'test_auc', 
                                        'test_sen_90', 'test_spec_90', 'test_precision_90','test_npv_90', 
                                        'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden', 'test_npv_yuden', 
                                        'thres_90', 'thres_yuden', 'acc_90', 'acc_yuden'])

    # load training data (unbalanced)
    df_train = pd.read_csv('/data_processed/train_selected_feat40.csv')
    X_df_train = df_train.iloc[:, :-1]
    y_df_train = df_train.iloc[:, -1]
    print(y_df_train.value_counts())
    n_feat = len(X_df_train.columns)

    # apply SMOTE inside cross-validation via an imblearn Pipeline
    sampler = SMOTE(random_state=123)
    clf = GradientBoostingClassifier(random_state=123)
    pipeline = ImbPipeline([('smote', sampler), ('clf', clf)])

    param_grid = [{
        'clf__learning_rate': [0.1],
        'clf__n_estimators': [170, 200, 220],
        'clf__subsample': [0.9, 1],
        'clf__max_depth': [5, 6, 7],
        'clf__max_features': [0.7, 0.8]
    }]

    cv = StratifiedKFold(5)
    grid_search = RandomizedSearchCV(pipeline, param_grid[0], cv=cv, scoring='roc_auc', random_state=123, n_iter=20, n_jobs=-1)
    grid_search.fit(X_df_train, y_df_train) 
    best_params = str(grid_search.best_params_)
    best_model = grid_search.best_estimator_ #best estimator already fit on the training data with SMOTE applied
    cvres = grid_search.cv_results_

    for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
        print(mean_score, params)
    train_auc_mean = cvres['mean_test_score'][grid_search.best_index_]
    train_auc_sd = cvres['std_test_score'][grid_search.best_index_]

    # test set
    df_test = pd.read_csv('/data_processed/test_selected_feat40.csv')
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

    results.to_csv('/results/ML_results_balanced_smote.csv', index=False, mode='a', header=False)

    # save probs for each model
    prob = pd.DataFrame(prob)
    prob.to_csv('/predictions/ML_prob_balanced_smote.csv', index=False)


def features_list():
    model = joblib.load('models/GBM_balanced_smote.pkl')
    items = model.feature_names_in_.tolist()
    itemids = pd.read_csv('data_raw/d_items.csv')[['itemid', 'label', 'linksto']]
    icd10_codes = pd.read_csv('icd10cm_codes_2024.csv')
    print(items)

    # get feature labels
    labels = []
    for item in items:
        if ('_' in item) and (item[-1].isdigit() and ('diff' not in item)):  # ending in window number
            itemid = item.rsplit('_', 1)[0]
            if itemid.isdigit():  # itemid (chartevent)
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 1)[1]
                labels.append(label)
            else: #ratios
                labels.append(item)
        elif ('_' in item) and (item.rsplit('_', 1)[1] in ['mean', 'min', 'max','range']): #stats for temporal features
            itemid = item.rsplit('_', 1)[0]
            if itemid.isdigit(): #itemid (chartevent)
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 1)[1]
                labels.append(label)
            else: #ratios
                labels.append(item)
        elif item in icd10_codes['icd10_code'].tolist():  # diagnosis
            label = icd10_codes.loc[icd10_codes['icd10_code'] == item]['label'].values[0]
            labels.append(label)
        elif 'diff' in item:
            itemid = item.rsplit('_', 2)[0]
            if itemid.isdigit():
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 2)[1] + '_' + item.rsplit('_', 2)[2]
                labels.append(label)
            else: #gcs_sum
                labels.append(item)
        else:  # gender/age/hosp_to_icu
            labels.append(item)
    labels = pd.DataFrame(labels).reset_index(drop=True)
    labels.to_csv('features_GBM_balanced_smote.csv', index=False)

