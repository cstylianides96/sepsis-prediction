# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd

# Split TRAIN, VAL, TEST sets into balanced sets of cases and controls (40 splits for each set)
def df_train_sets_balanced(encoded=False):
    if encoded:
        df_train = pd.read_csv('/data_processed/train_selected_feat40_names_encoded.csv')
    else:
        df_train = pd.read_csv('/data_processed/train_selected_feat40.csv')
    df_train['index'] = range(0, len(df_train))
    print(df_train['label'].value_counts())

    cases_train = df_train.loc[df_train['label']==1]
    controls_train = df_train.loc[df_train['label']==0]

    splits_size =  len(cases_train)
    splits = [controls_train[i:i + splits_size] for i in range(0, len(controls_train), splits_size)]

    for idx, split in enumerate(splits):
        split = split.reset_index(drop=True)
        df_train = pd.concat([split, cases_train], axis=0).reset_index(drop=True)
        if encoded:
            df_train.to_csv('/data_processed/train_' + str(idx+1) + '_encoded.csv', index=False)
        else:
            df_train.to_csv('/data_processed/train_' + str(idx+1) + '.csv', index=False)


def df_val_sets_balanced(encoded=False):
    if encoded:
        df_val = pd.read_csv('/data_processed/val_selected_feat40_names_encoded.csv')
    else:
        df_val = pd.read_csv('/data_processed/val_selected_feat40.csv')
    df_val['index'] = range(0, len(df_val))
    print(df_val['label'].value_counts())

    cases_val = df_val.loc[df_val['label']==1]
    controls_val = df_val.loc[df_val['label']==0]

    splits_size =  len(cases_val)
    splits = [controls_val[i:i + splits_size] for i in range(0, len(controls_val), splits_size)]

    for idx, split in enumerate(splits):
        split = split.reset_index(drop=True)
        df_val = pd.concat([split, cases_val], axis=0).reset_index(drop=True)
        # print(df_val)
        if encoded:
            df_val.to_csv('/data_processed/val_' + str(idx+1) + '_encoded.csv', index=False)
        else:
            df_val.to_csv('/data_processed/val_' + str(idx+1) + '.csv', index=False)


def df_test_sets_balanced(encoded=False):
    if encoded:
        df_test = pd.read_csv('/data_processed/test_selected_feat40_names_encoded.csv')
    else:   
        df_test = pd.read_csv('/data_processed/test_selected_feat40.csv')
    df_test['index'] = range(0, len(df_test))
    print(df_test['label'].value_counts())

    cases_test = df_test.loc[df_test['label']==1]
    controls_test = df_test.loc[df_test['label']==0]

    splits_size =  len(cases_test)
    splits = [controls_test[i:i + splits_size] for i in range(0, len(controls_test), splits_size)]

    for idx, split in enumerate(splits):
        split = split.reset_index(drop=True)
        df_test = pd.concat([split, cases_test], axis=0).reset_index(drop=True)
        # print(df_test)
        if encoded:
            df_test.to_csv('/data_processed/test_' +str(idx+1)+'_encoded.csv', index=False)
        else:
            df_test.to_csv('/data_processed/test_' +str(idx+1)+'.csv', index=False)

def create_balanced_datasets(encoded=False):
    df_train_sets_balanced(encoded=encoded)
    df_val_sets_balanced(encoded=encoded)
    df_test_sets_balanced(encoded=encoded)
