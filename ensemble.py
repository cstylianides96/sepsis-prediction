import pandas as pd
import numpy as np
from model_evaluation import evaluate

def run_ensemble():
    results = pd.DataFrame(columns=['model', 'obs_win', 'pred_win', 'test_auc', 'test_sen_90', 'test_spec_90',
                                    'test_precision_90', 'test_npv_90', 'test_sen_yuden', 'test_spec_yuden',
                                    'test_precision_yuden', 'test_npv_yuden', 'thres_90', 'thres_yuden', 'acc_90', 'acc_yuden'])

    # predictions from balanced datasets (GBM, LSTM)
    prob_gbm_df= pd.DataFrame()
    for idx in range(0, 41):
        prob_gbm = pd.read_csv(
            'predictions_set5_clean/GBM_obs24_pred12_feat70_balanced_prob' + str(idx + 1) + '_3.csv')
        prob_gbm_df = pd.concat([prob_gbm_df, prob_gbm], axis=1)
    prob_gbm_df.columns = list(range(1, 42))
    print(prob_gbm_df)

    prob_lstm_df = pd.read_csv(
        'predictions_set5_clean3_DL2/LSTM_1_obs24_pred12_balanced_prob_3.csv')
    print(prob_lstm_df)


    prob_gbm_df.columns = prob_gbm_df.columns.astype(str)
    prob_lstm_df.columns = prob_lstm_df.columns.astype(str)

    prob_gbm_df.index = prob_gbm_df.index.astype(int)
    prob_lstm_df.index = prob_lstm_df.index.astype(int)

    # average of predictions
    prob_avg_df = pd.DataFrame(
        np.nanmean([prob_gbm_df.values, prob_lstm_df.values], axis=0),
        columns=prob_gbm_df.columns,
        index=prob_gbm_df.index
    )
    prob_avg_df.to_csv('predictions_set5_clean_ENSEMBLE2/GBM-LSTM_obs24_pred12_prob_balanced_3.csv', index=False)
    print(prob_avg_df)
    # average of predictions
    # prob_avg_df = (prob_gbm_df.astype(float)+prob_lstm_df.astype(float))/2
    # print(prob_avg_df)

    #y true
    y_test_df = pd.DataFrame()
    for idx in range(0, 41):
        df_test = pd.read_csv('data_processed_set5_clean3_balanced/test_' + str(idx + 1) + '.csv').iloc[:, :-1]  # remove index
        y_test = df_test.iloc[:, -1]
        y_test_df = pd.concat([y_test_df, y_test], axis=1)
    y_test_df.columns = list(range(1, 42))
    print(y_test_df)

    for idx in range(0, 41):
        prob = prob_avg_df.iloc[:, idx]
        prob = prob.dropna()
        y = y_test_df.iloc[:, idx]
        y = y.dropna()

        (test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90,
         thres_yuden, acc_90, acc_yuden) = evaluate(prob, y, 'ENSEMBLE2', 24, 12, 70, acc=True)
        print(test_auc)

        results.loc[len(results)] = ['ENSEMBLE', 24, 12, test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden,
                                     precision_yuden, npv_yuden, thres_90, thres_yuden, acc_90, acc_yuden]

    results.to_csv('results_set5_clean_ENSEMBLE2/GBM-LSTM_obs24_pred12_results_balanced_3.csv', index=False)

    results_mean = results[['test_auc', 'test_sen_90', 'test_spec_90', 'test_precision_90',
                                    'test_npv_90', 'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden',
                                    'test_npv_yuden', 'acc_90', 'acc_yuden']].mean()
    print(results_mean)
