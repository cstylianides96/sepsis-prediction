# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
import numpy as np
from model_evaluation import evaluate

def run_ensemble():
    results = pd.DataFrame(columns=['model', 'obs_win', 'pred_win', 'test_auc', 'test_sen_90', 'test_spec_90',
                                    'test_precision_90', 'test_npv_90', 'test_sen_yuden', 'test_spec_yuden',
                                    'test_precision_yuden', 'test_npv_yuden', 'thres_90', 'thres_yuden', 'acc_90', 'acc_yuden'])

    # predictions from balanced datasets (GBM, 1DCNN)
    prob_ml_df= pd.DataFrame()
    for idx in range(0, 40):
        prob_ml = pd.read_csv('PIPELINE2/predictions/ML_prob_balanced_' + str(idx + 1) + '.csv')
        prob_ml_df = pd.concat([prob_ml_df, prob_ml], axis=1)
    prob_ml_df.columns = list(range(1, 41))
    print(prob_ml_df)

    prob_dl_df = pd.read_csv('PIPELINE2/predictions/LSTM_15_balanced_prob.csv')
    print(prob_dl_df)

    prob_ml_df.columns = prob_ml_df.columns.astype(str)
    prob_dl_df.columns = prob_dl_df.columns.astype(str)
    prob_ml_df.index = prob_ml_df.index.astype(int)
    prob_dl_df.index = prob_dl_df.index.astype(int)

    # average of predictions
    prob_avg_df = pd.DataFrame(
        np.nanmean([prob_ml_df.values, prob_dl_df.values], axis=0),
        columns=prob_ml_df.columns,
        index=prob_ml_df.index
    )
    prob_avg_df.to_csv('PIPELINE2/predictions/GBM-LSTM_balanced_prob.csv', index=False)
    print(prob_avg_df)

    #y true
    y_test_df = pd.DataFrame()
    for idx in range(0, 40):
        df_test = pd.read_csv('PIPELINE2/data_processed/test_' + str(idx + 1) + '.csv').iloc[:, :-1]  # remove index
        y_test = df_test.iloc[:, -1]
        y_test_df = pd.concat([y_test_df, y_test], axis=1)
    y_test_df.columns = list(range(1, 41))
    print(y_test_df)

    for idx in range(0, 40):
        prob = prob_avg_df.iloc[:, idx]
        prob = prob.dropna()
        y = y_test_df.iloc[:, idx]
        y = y.dropna()

        test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90, thres_yuden, acc_90, acc_yuden = evaluate(prob, y, acc=True)
        print(test_auc)

        results.loc[len(results)] = ['ENSEMBLE', 24, 12, test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden,
                                     precision_yuden, npv_yuden, thres_90, thres_yuden, acc_90, acc_yuden]

    results.to_csv('PIPELINE2/results/ENSEMBLE_results_balanced.csv', index=False)

    results_mean = results[['test_auc', 'test_sen_90', 'test_spec_90', 'test_precision_90','test_npv_90', 
                            'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden','test_npv_yuden', 
                            'acc_90', 'acc_yuden']].mean()
    print(results_mean)
