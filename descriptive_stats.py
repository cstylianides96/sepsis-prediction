# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd


def dataset_stats():
    train = pd.read_csv('/data_processed/train_selected_feat40.csv')
    val = pd.read_csv('/data_processed/val_selected_feat40.csv')
    test = pd.read_csv('/data_processed/test_selected_feat40.csv')
    df = pd.concat([train, val, test], axis=0).reset_index(drop=True)
    print(df.columns)
    for col in df.columns[:-1]:
        print('----' + col + '-----')
        median = df[[col, 'label']].groupby('label').median() # median, quartiles
        q1 = df[[col, 'label']].groupby('label').quantile(0.25)
        q3 = df[[col, 'label']].groupby('label').quantile(0.75)
        print(median, q1, q3)

        # mean = df[[col, 'label']].groupby('label').mean()
        # min = df[[col, 'label']].groupby('label').min()
        # max = df[[col, 'label']].groupby('label').max()
        # print(mean, min, max)

        # counts = df[[col, 'label']].groupby('label').value_counts() #(normalize=True)
        # print(counts)


def demo_stats():
    stayids_train = pd.read_csv('/data_processed/train_selected.csv')[['stay_id', 'label']]
    stayids_val = pd.read_csv('/data_processed/val_selected.csv')[['stay_id', 'label']]
    stayids_test = pd.read_csv('/data_processed/test_selected.csv')[['stay_id', 'label']]
    stayids = pd.concat([stayids_train, stayids_val, stayids_test], axis=0).reset_index(drop=True)
    stayids.columns = ['stay_id', 'label']
    demo = pd.read_csv('/data_processed/sepsis3_processed.csv')
    demo = pd.merge(stayids, demo, how='left', on=['stay_id', 'label'])
    demo = demo[['stay_id','label', 'age', 'gender', 'race', 'hours_after_adm',  'los', 'hosp_to_icu']]
    demo.to_csv('/data_processed/demographic_labels.csv', index=False)

    # binary, categorical variables
    gender_label = demo[['gender', 'label']].groupby('label').value_counts() #(normalize=True)
    race_label = demo[['race', 'label']].groupby('label').value_counts() #(normalize=True)
    print(gender_label, race_label)

    # continuous, ordinal variables
    for var in ['age', 'hours_after_adm',  'los', 'hosp_to_icu']:
        median = demo[[var, 'label']].groupby('label').median() # median, quartiles
        q1 = demo[[var, 'label']].groupby('label').quantile(0.25)
        q3 = demo[[var, 'label']].groupby('label').quantile(0.75)
        print(median, q1, q3)
