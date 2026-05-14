# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def itemid_to_name_dataset():
    model = joblib.load('PIPELINE2/models/GBM_feat_selection_feat50(40).pkl')
    model_feats = model.feature_names_in_.tolist()
    print(model_feats)
    X_train = pd.read_csv('PIPELINE2/data_processed/train_selected_feat40.csv')
    X_val = pd.read_csv('PIPELINE2/data_processed/val_selected_feat40.csv')
    X_test = pd.read_csv('PIPELINE2/data_processed/test_selected_feat40.csv')
    print(X_test)
    y_train = pd.read_csv('PIPELINE2/data_processed/train_selected_feat40.csv').iloc[:, -1]
    y_val = pd.read_csv('PIPELINE2/data_processed/val_selected_feat40.csv').iloc[:, -1]
    y_test = pd.read_csv('PIPELINE2/data_processed/test_selected_feat40.csv').iloc[:, -1]
    X_cols = model_feats

    itemids = pd.read_csv('PIPELINE2/data_raw/d_items.csv')[['itemid', 'label', 'linksto']]
    icd10_codes = pd.read_csv('PIPELINE2/data_raw/icd10cm_codes_2024.csv')
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # most important to least important
    items = [X_cols[i] for i in indices]
    #print(items)

    # get feature labels (most imp to least imp)
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
    labels.to_csv('PIPELINE2/data_processed/features_ranked.csv', index=False)

    for X, y, name in [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]:
        X = X[items]
        X.columns = labels.iloc[:, 0].tolist()
        df = pd.concat([X,y], axis=1)
        print(df)
        df.to_csv('PIPELINE2/data_processed/' + name + '_selected_feat40_names.csv', index=False)

    # get unique labels and their categories (most imp to least imp)
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

    unique_labels = pd.DataFrame(zip(unique_labels, categories)).reset_index(drop=True)
    unique_labels.columns = ['label', 'category']
    unique_labels = unique_labels.drop_duplicates().reset_index(drop=True) 
    unique_labels.to_csv('PIPELINE2/data_processed/unique_features_ranked.csv', index=False)


def categorize():
    df_train = pd.read_csv('PIPELINE2/data_processed/train_selected_feat40_names.csv')
    print(df_train.columns)
    df_val = pd.read_csv('PIPELINE2/data_processed/val_selected_feat40_names.csv')
    df_test = pd.read_csv('PIPELINE2/data_processed/test_selected_feat40_names.csv')

    for df, name in [(df_train, 'train'), (df_val, 'val'), (df_test, 'test')]:
        temporal_cols_categories = {
            
            'GCS - Verbal Response': {
                'bins': [0, 1, 2, 3, 4, 5],
                'labels': ['None', 'Incomprehensible', 'Inappropriate', 'Confused', 'Oriented']
            },

            'Arterial Blood Pressure systolic': {
                'bins': [0, 90, 100, 110, 219, float('inf')],
                'labels': ['Very Low', 'Low', 'Pre-Normal', 'Normal', 'High']
            },

            'GCS - Motor Response': {
                'bins': [0, 1, 2, 3, 4, 5, float('inf')],
                'labels': ['None', 'Abnormal extension to pain', 'Abnormal flexion to pain', 'Withdraws from pain', 'Localizes pain', 'Obeys commands']
            },

           'Sodium (serum)': {
                'bins': [0, 135, 145, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },
            'Arterial Blood Pressure mean': {
                'bins': [0, 69, 100, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },
            'Magnesium': {
                'bins': [0, 1.6, 2.4, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },

            'Creatinine (serum)': {
                'bins': [0, 0.6, 1.3, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },

            'Fibrinogen': {
                'bins': [-float('inf'), 99, 199, 400, 600, float('inf')],
                'labels': ['Very Low', 'Low', 'Normal', 'High', 'Very High']
            },

            'LDH': {
                'bins': [-float('inf'), 99, 139, 280, 500, 1000, float('inf')],
                'labels': ['Very Low', 'Low', 'Normal', 'High', 'Very High', 'Critical']
            },

            'Hematocrit (serum)': {
                'bins': [0, 35, 51, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            }
            
        }
  
        for col, rules in temporal_cols_categories.items():
            for var in df.columns:
                if col in var and all(x not in var for x in ['range', 'diff']):
                    print(var)
                    df[var] = pd.cut(
                        df[var],
                        bins=rules['bins'],
                        labels=rules['labels'],
                        right=True,
                        include_lowest=True)

                    oe = OrdinalEncoder(categories=[rules['labels']])
                    df[var] = oe.fit_transform(df[[var]]).astype(int)

        df.to_csv(f'PIPELINE2/data_processed/{name}_selected_feat40_names_encoded.csv', index=False)


            # 'gcs_sum': {
            #     'bins': [0, 8, 12, 15],
            #     'labels': ['comatose', 'confused/lethargic', 'alert/minimally confused']
            # },

        
            # 'Potassium (serum)': {
            #     'bins': [0, 3.4, 5.2, float('inf')],
            #     'labels': ['Low', 'Normal', 'High']
            # },


            # 'WBC': {
            #     'bins': [0, 3.6, 10.7, float('inf')],
            #     'labels': ['Low', 'Normal', 'High']
            # },
         
            # 'Total Bilirubin': {
            #     'bins': [0, 1.1, 1.9, 5.9, 11.9, float('inf')],
            #     'labels': ['Normal', 'Above Normal', 'High', 'Very High', 'Extremely High']
            # },
           
            # 'INR': {
            #     'bins': [0, 0.7, 1.2, float('inf')],
            #     'labels': ['Low', 'Normal', 'High']
