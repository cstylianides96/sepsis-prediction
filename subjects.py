# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd


def data_subjects():

    data_subj = pd.DataFrame(
        columns=['train', 'val', 'test', 'sepsis_train', 'no_sepsis_train', 'sepsis_val', 'no_sepsis_val', 'sepsis_test', 'no_sepsis_test'])
 
    train = pd.read_csv('PIPELINE2/data_processed/train_selected_feat40.csv')
    val = pd.read_csv('PIPELINE2/data_processed/val_selected_feat40.csv')
    test = pd.read_csv('PIPELINE2/data_processed/test_selected_feat40.csv')

    n_train = train.shape[0]
    nosepsis_train = train['label'].value_counts(normalize=True)[0]
    sepsis_train = train['label'].value_counts(normalize=True)[1]

    n_val = val.shape[0]
    nosepsis_val = val['label'].value_counts(normalize=True)[0]
    sepsis_val = val['label'].value_counts(normalize=True)[1]

    n_test = test.shape[0]
    nosepsis_test = test['label'].value_counts(normalize=True)[0]
    sepsis_test = test['label'].value_counts(normalize=True)[1]
    data_subj.loc[len(data_subj)] = [n_train, n_val, n_test, sepsis_train, nosepsis_train, sepsis_val, nosepsis_val, sepsis_test, nosepsis_test]
    data_subj.to_csv('PIPELINE2/data_processed/data_subjects_percentages.csv', index=False)

# train_eicu = pd.read_csv('PIPELINE2/data_processed_eicu/train_df.csv')
# test_eicu = pd.read_csv('PIPELINE2/data_processed_eicu/test_df.csv')
# print(train_eicu.label.value_counts(normalize=True))
# print(test_eicu.label.value_counts(normalize=True))
# print(len(train_eicu))
# print(len(test_eicu))