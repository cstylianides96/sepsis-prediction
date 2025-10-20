# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd


def data_subjects():

    data_subj = pd.DataFrame(
        columns=['obs_win', 'pred_win', 'train', 'test', 'sepsis_train', 'no_sepsis_train', 'sepsis_test', 'no_sepsis_test'])
    obs_wins = [24]
    pred_wins = [12]
    # obs_wins = [4, 8, 12, 18, 24]
    # pred_wins = [3, 6, 12]

    for obs_win in obs_wins:
        for pred_win in pred_wins:
            print(obs_win, pred_win)

            # train
            df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_3.csv')
            n_train = df.shape[0]
            nosepsis_train = df['label'].value_counts()[0]
            sepsis_train = df['label'].value_counts()[1]

            #test
            df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_3.csv')
            n_test = df.shape[0]
            nosepsis_test = df['label'].value_counts()[0]
            sepsis_test = df['label'].value_counts()[1]

            data_subj.loc[len(data_subj)] = [obs_win, pred_win, n_train, n_test, sepsis_train, nosepsis_train, sepsis_test, nosepsis_test]
    data_subj.to_csv('data_subjects.csv')
    print(data_subjects)
