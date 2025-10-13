import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder


def concat(obs_win, pred_win):
    sel_cols = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) +
                      '_imputed_train_fs2_1000.csv').columns
    # sel = pd.read_csv('data_processed_set5_missing/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed2.csv.gz',
    #                  compression='gzip', usecols=sel_cols)
    sel = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv', usecols=sel_cols)
    print(sel.shape)
    sugg = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_all_changes.csv')
    print(sugg.shape)

    age = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv',
                      usecols=['age'])
    combo = pd.concat([age, sel, sugg], axis=1)
    print(combo.shape)
    combo = combo.loc[:, ~combo.columns.duplicated()]
    print(combo.shape)
    y = combo.pop('label')
    combo['label'] = y
    combo.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv', index=False)
    print(combo.columns)



def features_to_encode(obs_win, pred_win):
    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_processed.csv')
    itemids = pd.read_csv('data_raw/d_items.csv')[['itemid', 'linksto', 'category']]

    #columns with 0 percentage (more than 60%)
    zero_counts = (df == 0).sum()
    total_counts = df.shape[0]
    zero_percentages = (zero_counts / total_counts) * 100
    high_zero_cols = zero_percentages[zero_percentages > 60].index.tolist()
    high_zero_cols_unique = []

    for col in high_zero_cols:
        col = col.rsplit('_', 1)[0]
        high_zero_cols_unique.append(col)
    high_zero_cols_unique = set(high_zero_cols_unique)

    device_mnar_categories = {
        'Impella',
        'IABP',
        'PiCCO',
        'NICOM',
        'ECMO',
        'Centrimag',
        'Cardiovascular (Pacer Data)',
        'Dialysis',
        'Access Lines - Invasive',
        'Hemodynamics',
        'Skin - Impairment',
        'Treatments',
        'Pain / Sedation',
        'Alarms'}

    cols_to_encode = []
    # columns with missing values more than 60% that are chartevents
    for col in high_zero_cols_unique:
        if col.isdigit():
            event = itemids.loc[itemids['itemid']==int(col)]['linksto'].values[0]
            if event=='chartevents':
                #print(col)
                category = itemids.loc[itemids['itemid']==int(col)]['category'].values[0]
                if category in device_mnar_categories:
                    cols_to_encode.append(col)
    df = df[[col for col in df.columns if any(r in col for r in cols_to_encode)]]
    print(cols_to_encode)
    return cols_to_encode, df


def clean(obs_win, pred_win):

    # Remove any features containing pf_ratio_b, resp rate, resprate (spontaneous), daily weight, solution
    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv')
    label = df.iloc[:, -1]
    print(df)
    rem = ['pf_ratio_b', '220210', '224689', '224639', '225943', '228872'] #resp rate, resprate (spontaneous), daily weight, solution, HM II-Mean BP
    df = df[[col for col in df.columns if not any(r in col for r in rem)]]
    print(df)

    # Remove stats for inputevents
    cols = df.columns
    itemids = pd.read_csv('data_raw/d_items.csv')[['itemid', 'linksto']]
    inputevents = itemids.loc[itemids['linksto'] == 'inputevents']['itemid'].astype(str).tolist()
    stat_keywords = ['mean', 'median', 'min', 'max', 'sd', 'range', 'sum', 'diff']

    for col in cols:
        if any(event in col for event in inputevents) and any(stat in col for stat in stat_keywords):
            df = df.drop(columns=[col])
            print(f"Dropped column: {col}")

    # Encode inputevent variables as binary (0 as it is, any amount to 1)
    for col in df.columns:
        if any(event in col for event in inputevents):
            df[col] = df[col].apply(lambda x: 0 if x == 0 else 1)
            print(f"Converted to binary: {col}")

    rem2 = ['220179', '220180', '220181',  #non-invasive bp systolic, non-invasive bp diastolic, non-invasive bp mean,
            '228872', '228869', '228870', '228871', '228873', '228874', '228875', '228876',  #HM II-Mean BP, HM II...
            '224702', '229364', '229365',#PCV Level, P2 (ECMO), P1-P2 (ECMO)
            '228195', '228874', '229262', '229263', '229277', '229303', '29829', '229836', '229845', '229895'] #Speed ralated
    df = df[[col for col in df.columns if not any(r in col for r in rem2)]]
    print(df)


    # Remove KNN-imputed features coming from devices (MNAR)
    feats_to_encode, df_orig = features_to_encode(obs_win, pred_win)
    print(len(feats_to_encode))
    columns1 = set(df.columns.tolist())
    df = df[[col for col in df.columns if not any(r in col for r in feats_to_encode)]]
    print(df)
    columns2 = set(df.columns.tolist())
    feats_to_encode_selected = list(columns1-columns2)
    print(feats_to_encode_selected)
    subst = [col for col in feats_to_encode_selected if col in df_orig]
    print(subst)
    df_orig = df_orig[subst]

    # Add the 0-nonzero features coming from devices
    df = pd.concat([df, df_orig], axis=1)
    print(df)

    # Encode the features coming from devices (0/1)
    for col in df.columns:
        if any(event in col for event in feats_to_encode_selected):
            df[col] = df[col].apply(lambda x: 0 if (x == 0 or np.isnan(x)) else 1)
            print(f"Converted to binary: {col}")
    df = df[[col for col in df.columns if '224418' not in col]] #cuff volume
    print(df)
    label = df.pop('label')
    df['label'] = label
    print(df)
    df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_3.csv', index=False)


def create_train_test_splits(obs_win, pred_win):
    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_3.csv')
    #print(df['label'].value_counts())

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df['label']):  # same proportions of label1:label0 for each split
        df_train = df.loc[train_index]
        df_test = df.loc[test_index]

        df_train.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_3.csv', index=False)
        df_test.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_3.csv', index=False)
        # print(df_train['label'].value_counts(normalize=True))
        # print(df_test['label'].value_counts(normalize=True))


# use for XAI
def categorize():
    df_train = pd.read_csv('DFtrain_GBM_obs24_pred12_feat70_clean_3.csv')
    df_test = pd.read_csv('DFtest_GBM_obs24_pred12_feat70_clean_3.csv')

    for df, name in [(df_train, 'DFtrain'), (df_test, 'DFtest')]:
        temporal_cols_categories = {
            'Fspn High': {
                'bins': [0, 8, 11, 20, 24, float('inf')],
                'labels': ['Very Low', 'Low', 'Normal', 'High', 'Very High']
            },
            'PH (Arterial)': {  # pH arterial
                'bins': [0, 7.37, 7.44, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },
            'Tandem Heart Flow': {
                'bins': [0, 2.999999, 4, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },

            'Arterial Blood Pressure systolic': {
                'bins': [0, 90, 100, 110, 219, float('inf')],
                'labels': ['Very Low', 'Low', 'Pre-Normal', 'Normal', 'High']
            },

            'O2 Flow': {
                'bins': [0, 1, 5, 10, 15, float('inf')],
                'labels': ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            },

            'gcs_sum': {
                'bins': [0, 8, 12, 15],
                'labels': ['comatose', 'confused/lethargic', 'alert/minimally confused']
            },

            'Arterial Blood Pressure mean': {
                'bins': [0, 69, 100, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },

            'Negative Insp. Force': {
                'bins': [-float('inf'), -30, float('inf')],
                'labels': ['Normal', 'High']

            },
            'Temperature Celsius': {
                'bins': [0, 35, 36, 38, 39, float('inf')],
                'labels': ['Very Low', 'Low', 'Normal', 'High', 'Very High']
            },

            'SV (Arterial)': {
                'bins': [0, 59, 100, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },

            'Potassium (whole blood)': {
                'bins': [0, 3.4, 5.2, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },

            'Sodium (whole blood)': {
                'bins': [0, 135, 145, float('inf')],
                'labels': ['Low', 'Normal', 'High']
            },
            'RRApacheIIValue': {
                'bins': [0, 8, 11, 20, 24, float('inf')],
                'labels': ['Very Low', 'Low', 'Normal', 'High', 'Very High']
            }}

        for col, rules in temporal_cols_categories.items():
            for var in df.columns:
                if col in var and var.rsplit('_', 2)[-1] not in ['range', 'diff']:
                    print(var)
                    df[var] = pd.cut(
                        df[var],
                        bins=rules['bins'],
                        labels=rules['labels'],
                        right=True,
                        include_lowest=True)

                    oe = OrdinalEncoder(categories=[rules['labels']])
                    df[var] = oe.fit_transform(df[[var]]).astype(int)

        df.to_csv(name + '_GBM_obs24_pred12_feat70_clean_3_encoded.csv', index=False)


def create_datasets_clean():
    obs_win = 24
    pred_win = 12
    concat(obs_win, pred_win)
    clean(obs_win, pred_win)
    create_train_test_splits(obs_win, pred_win)
