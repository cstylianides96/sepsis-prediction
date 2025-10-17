# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
import joblib
import numpy as np

def itemid_to_name_dataset(model_name, obs_win, pred_win, n_feat, model_path):
    model = joblib.load(model_path)
    model_feats = model.feature_names_in_.tolist()
    print(model_feats)
    X_train = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_3.csv')[model_feats]
    X_test = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_3.csv')[model_feats]
    print(X_test)
    y_train = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_3.csv').iloc[:, -1]
    y_test = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_3.csv').iloc[:, -1]
    X_cols = model_feats

    itemids = pd.read_csv('data_raw/d_items.csv')[['itemid', 'label', 'linksto']]
    icd10_codes = pd.read_csv('data_raw/icd10cm_codes_2024.csv')
    importances = model.feature_importances_
    indices = np.argsort(importances)
    items = [X_cols[i] for i in indices]
    #print(items)
    labels = []
    for item in items:
        if ('_' in item) and (item[-1].isdigit()):  # ending in window number
            itemid = item.rsplit('_', 1)[0]
            if itemid.isdigit():  # itemid (chartevent/inputevent/outputevent/procedureevent)
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 1)[1]
                labels.append(label)
            else:  # ratios
                labels.append(item)
        elif ('_' in item) and (item.rsplit('_', 1)[1] in ['mean', 'median', 'min', 'max', 'sd', 'range',
                                                           'sum']):  # stats for temporal features
            itemid = item.rsplit('_', 1)[0]
            if itemid.isdigit():  # itemid (chartevent/inputevent/outputevent/procedureevent)
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 1)[1]
                labels.append(label)
            else:  # ratios
                labels.append(item)
        elif item in icd10_codes['icd10_code'].tolist():  # diagnosis
            label = icd10_codes.loc[icd10_codes['icd10_code'] == item]['label'].values[0]
            labels.append(label)
        elif 'diff' in item:
            itemid = item.rsplit('_', 2)[0]
            if itemid.isdigit():
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 2)[-2] + '_' + item.rsplit('_', 2)[-1]
                labels.append(label)
            else:  # gcs_sum
                labels.append(item)
        else:  # gender/ethnicity/age/adm_to_pred/hosp_to_icu
            labels.append(item)
    labels = pd.DataFrame(labels)

    # train and test dataframes with label columns
    for X, y, name in [(X_train, y_train, 'train'), (X_test, y_test, 'test')]:
        X = X[items]
        X.columns = labels.iloc[:, 0].tolist()
        df = pd.concat([X,y], axis=1)
        print(df)
        df.to_csv('DF' + name + '_' + model_name + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_clean_3.csv', index=False)

    unique_labels = []
    categories = []
    for l in labels.iloc[:, 0].tolist():
        if '_' in l:
            l = l.rsplit('_', 2)[0]
        unique_labels.append(l)
        cat = itemids.loc[itemids['label']==l, 'linksto'].to_list()
        if not cat:
            cat = ' '
        else:
            cat = cat[0]
        categories.append(cat)

    unique_labels = pd.DataFrame(zip(unique_labels, categories))[::-1].reset_index(drop=True)
    unique_labels.columns = ['label', 'category']
    unique_labels = unique_labels.drop_duplicates().reset_index(drop=True) #most imp to least imp
    labels = labels[::-1].reset_index(drop=True) #most imp to least imp

    # unique labels with categories of events (empty category for feature that is not an event) (most imp to least imp)
    unique_labels.to_csv('BEST_set5_' + model_name + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_unique_clean_3.csv', index=False)
    # all labels in dataset (most imp to least imp)
    labels.to_csv('BEST_set5_' + model_name + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_clean_3.csv', index=False)

#itemid_to_name_dataset('GBM', 24, 12, '70', 'models/GBM_obs24_pred12_feat80(70)_3.pkl')
