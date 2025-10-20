# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd


# SPLIT TRAIN SET INTO CASES AND EQUAL SIZED CONTROLS
def df_train_sets_balanced():
    #df_train = pd.read_csv('DFtrain_GBM_obs24_pred12_feat70_clean_3_encoded.csv')
    df_train = pd.read_csv('DFtrain_GBM_obs24_pred12_feat70_clean_3.csv')
    df_train['index'] = range(0, len(df_train))
    #print(df_train['label'].value_counts())

    cases_train = df_train.loc[df_train['label']==1]
    controls_train = df_train.loc[df_train['label']==0]

    # splits_n = len(controls_train) // len(cases_train) #35248 // 870 =40
    splits_size =  len(cases_train)
    splits = [controls_train[i:i + splits_size] for i in range(0, len(controls_train), splits_size)]

    for idx, split in enumerate(splits):
        print(f"Part {idx + 1}: {split}")
        split = split.reset_index(drop=True)
        df_train = pd.concat([split, cases_train], axis=0).reset_index(drop=True)
        print(df_train)
        #df_train.to_csv('data_processed/train_' + str(idx+1) + '_encoded.csv', index=False)
        df_train.to_csv('data_processed/train_' + str(idx+1) + '.csv', index=False)


# SPLIT TEST SET INTO CASES AND EQUAL SIZED CONTROLS
def df_test_sets_balanced():
    #df_test = pd.read_csv('DFtest_GBM_obs24_pred12_feat70_clean_3_encoded.csv')
    df_test = pd.read_csv('DFtest_GBM_obs24_pred12_feat70_clean_3.csv')
    df_test['index'] = range(0, len(df_test))
    #print(df_test['label'].value_counts())

    cases_test = df_test.loc[df_test['label']==1]
    controls_test = df_test.loc[df_test['label']==0]

    # splits_n = len(controls_test) // len(cases_test) # 8812 // 218 =40
    splits_size =  len(cases_test)
    splits = [controls_test[i:i + splits_size] for i in range(0, len(controls_test), splits_size)]

    for idx, split in enumerate(splits):
        print(f"Part {idx + 1}: {split}")
        split = split.reset_index(drop=True)
        df_test = pd.concat([split, cases_test], axis=0).reset_index(drop=True)
        print(df_test)
        #df_test.to_csv('data_processed/test_' +str(idx+1)+'_encoded.csv', index=False)
        df_test.to_csv('data_processed/test_' +str(idx+1)+'.csv', index=False)


# LABELS OF PREDICTIONS (41 BALANCED SETS) ACCORDING TO THRES_90 OF EACH MODEL
def probs_to_pred_thres_90(model):
    if model=='GBM':
        for idx in range(1, 41):
            probs = pd.read_csv('predictions/GBM_obs24_pred12_feat70_balanced_prob' + str(idx) + '_3_encoded.csv')
            thres = pd.read_csv('thresholds/THRESROC_GBM_obs24_pred12_feat70_balanced_3_encoded.csv')
            for i in range(len(probs)):
                if probs.iloc[i, 0]>= thres.iloc[idx-1, 1]:
                    probs.iloc[i, 0] = 1
                else:
                    probs.iloc[i, 0] = 0

            test_index = pd.read_csv('data_processed/test_' + str(idx) + '_encoded.csv')[
                'index']
            df_out = pd.DataFrame({'pred': probs, 'index': test_index})
            df_out.to_csv('predictions/GBM_obs24_pred12_feat70_balanced_pred' + str(idx) + '_3_encoded.csv',
                         index=False)

    if model=='LSTM':
        for idx in range(1, 41):
            probs = pd.read_csv('predictions/LSTM_1_obs24_pred12_balanced_prob_3.csv').iloc[
                :, idx - 1].copy()

            # Apply threshold
            thres_value = pd.read_csv('results/obs24_pred12_results_balanced_LSTM_1_v2.csv')['thres_90'][idx - 1]
            probs = (probs >= thres_value).astype(int)

            test_index = pd.read_csv('data_processed/test_' + str(idx) + '_encoded.csv')[
                'index']

            df_out = pd.DataFrame({'pred': probs, 'index': test_index})
            df_out.to_csv('predictions/LSTM_1_obs24_pred12_balanced_pred' + str(idx) + '_3.csv',
                          index=False)

    if model=='ENSEMBLE':
        for idx in range(1, 41):
            probs = pd.read_csv('predictions/GBM-LSTM_obs24_pred12_prob_balanced_3.csv').iloc[:, idx - 1].copy()
            thres_value = pd.read_csv('results/GBM-LSTM_obs24_pred12_results_balanced_3.csv')['thres_90'][idx-1]
            probs = (probs >= thres_value).astype(int)

            test_index = pd.read_csv('data_processed/test_' + str(idx) + '_encoded.csv')[
                'index']

            df_out = pd.DataFrame({'pred': probs, 'index': test_index})
            df_out.to_csv('predictions/GBM-LSTM_obs24_pred12_balanced_pred' + str(idx) + '_3.csv',
                     index=False)
