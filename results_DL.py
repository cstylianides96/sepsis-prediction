# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import os
import pandas as pd
from model_evaluation import evaluate

def overall_results_DL():

    directory = 'results'
    obs_win = 24
    pred_win = 12

    # Result container
    compiled_data = []

    # Loop through files in the directory
    for file in os.listdir(directory):
        if file.startswith(f'obs{obs_win}_pred{pred_win}_results_balanced_') and file.endswith('.csv'):
            filepath = os.path.join(directory, file)

            df = pd.read_csv(filepath)
            model_name = filepath.rsplit('_', 5)[4]
            model_ver = int(filepath.rsplit('_', 5)[-1].rsplit('.', 1)[0])

            # Extract values:
            last_row_first4 = [round(num, 2) for num in df.iloc[-1, 0:4].tolist()]
            mean_last2 = [round(num, 2) for num in df.iloc[:, -2:].mean(axis=0).tolist()]

            # Add to results
            compiled_data.append([model_name] + [model_ver] + last_row_first4 + mean_last2)

    # Create the new dataframe
    result_df = pd.DataFrame(compiled_data, columns=['model_name', 'model_ver', 'train_loss', 'train_auc', 'val_loss', 'val_auc', 'test_loss', 'test_auc'])
    result_df = result_df.sort_values(by=['model_name', 'model_ver'])
    result_df = result_df.reset_index(drop=True)
    result_df.to_csv('results_DL.csv', index=False)
    print(result_df)


def overall_results_DL_updated(model_list, ver_list):

    results_new = pd.DataFrame(columns=['test_auc', 'test_sen_90', 'test_spec_90', 'test_precision_90',
                                        'test_npv_90', 'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden',
                                        'test_npv_yuden', 'acc_90', 'acc_yuden'])

    for model, ver in zip(model_list, ver_list):
        print(model, ver)
        results_model = pd.DataFrame(columns=['test_auc', 'test_sen_90', 'test_spec_90', 'test_precision_90',
                                            'test_npv_90', 'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden',
                                            'test_npv_yuden', 'thres_90', 'thres_yuden', 'acc_90', 'acc_yuden'])
        for idx in range(40):
            probs = pd.read_csv(
                'predictions/' + str(model) + '_' + str(ver) + '_obs24_pred12_balanced_prob' + '_3.csv').iloc[
                    :, idx]
            probs = probs.dropna()
            #print(probs)
            y_test = pd.read_csv('data_processed/obs24_pred12_test_X_' + str(idx+1) + '.csv')['label']

            (test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90,
             thres_yuden, acc_90, acc_yuden) = evaluate(probs, y_test, model, 24, 12, '-', acc=True)
            print(test_auc)

            # save results
            results_model.loc[len(results_model)] = [test_auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden,
                                         precision_yuden, npv_yuden, thres_90, thres_yuden, acc_90, acc_yuden]

        results_model.to_csv('results/obs24_pred12_results_balanced_' + model + '_' + str(ver) + '_v2.csv', index=False)
        results_mean = results_model.mean(axis=0)
        results_mean = results_mean.drop(['thres_90', 'thres_yuden'])
        # results_sd = results_model.std()

        results_new.loc[len(results_new)] = results_mean
    print(results_new)

    results_new['model_name'] = model_list
    results_new['ver'] = ver_list
    results_new.to_csv('results_DL_new.csv', index=False)
