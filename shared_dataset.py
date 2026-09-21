import pandas as pd

def mimiciv():
    # full time series and all feature engineering, before any feature selection 

    df_train = pd.read_csv('data_processed/train_merged_imputed_flattened_aggregated_binary.csv')
    label = df_train.pop('label')
    df_train['label'] = label
    print(df_train.shape)
    print(df_train.head())

    df_val = pd.read_csv('data_processed/val_merged_imputed_flattened_aggregated_binary.csv')
    label = df_val.pop('label')
    df_val['label'] = label
    print(df_val.shape)
    print(df_val.head())

    df_test = pd.read_csv('data_processed/test_merged_imputed_flattened_aggregated_binary.csv')
    label = df_test.pop('label')
    df_test['label'] = label
    print(df_test.shape)
    print(df_test.head())

    shared_dataset_mimic = pd.concat([df_train, df_val, df_test], axis=0)
    print(shared_dataset_mimic.shape)
    print(shared_dataset_mimic.label.value_counts())
    shared_dataset_mimic.to_csv('shared_dataset_mimiciv.csv', index=False)


def eICU():
    shared_dataset_eicu = pd.read_csv('data_processed_eicu/df_imputed_flattened_agg.csv')
    print(shared_dataset_eicu.shape)
    print(shared_dataset_eicu.label.value_counts())
    shared_dataset_eicu.to_csv('shared_dataset_eicu.csv', index=False)
