import pandas as pd

def probs_to_pred(model): # edit 'thrs_yuden' to 'thres_90'
    if model=='GBM':
        for idx in range(1, 41):
            probs = pd.read_csv('/predictions/ML_prob_balanced_' + str(idx) + '.csv').iloc[:, 0].copy()
            thres_value = pd.read_csv('/results/ML_results_balanced.csv')['thres_yuden'][idx-1]
            probs = (probs >= thres_value).astype(int)

            test_index = pd.read_csv('/data_processed/test_' + str(idx) + '.csv')['index']
            df_out = pd.DataFrame({'pred': probs, 'index': test_index})
            df_out.to_csv('/predictions/ML_pred_balanced_' + str(idx) + '_yuden.csv', index=False)

    if model=='LSTM':
        for idx in range(1, 41):
            probs = pd.read_csv('/predictions/LSTM_15_balanced_prob.csv').iloc[:, idx - 1].copy()
            thres_value = pd.read_csv('/results/DL_results_balanced_LSTM_15_allmetrics.csv')['thres_yuden'][idx - 1]
            probs = (probs >= thres_value).astype(int)

            test_index = pd.read_csv('/data_processed/test_' + str(idx) + '.csv')['index']
            df_out = pd.DataFrame({'pred': probs, 'index': test_index})
            df_out.to_csv('/predictions/LSTM_15_balanced_pred' + str(idx) + '_yuden.csv', index=False)

    if model=='ENSEMBLE':
        for idx in range(1, 41):
            probs = pd.read_csv('/predictions/GBM-LSTM_balanced_prob.csv').iloc[:, idx - 1].copy()
            thres_value = pd.read_csv('/results/ENSEMBLE_results_balanced.csv')['thres_yuden'][idx-1]
            probs = (probs >= thres_value).astype(int)

            test_index = pd.read_csv('/data_processed/test_' + str(idx) + '.csv')['index']
            df_out = pd.DataFrame({'pred': probs, 'index': test_index})
            df_out.to_csv('/predictions/GBM-LSTM_balanced_pred' + str(idx) + '_yuden.csv', index=False)
