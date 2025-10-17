# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
#from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif, f_classif
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pointbiserialr as pbs
from tqdm import tqdm
import warnings
from sklearn.impute import KNNImputer
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_dataset(obs_win, pred_win):
    chunksize = 5000
    X_df = pd.DataFrame()
    y_df = pd.DataFrame()

    labels_chunks = pd.read_csv('labels_sepsis.csv', chunksize=chunksize) #patients with sepsis onset at 7 hours or more
    data = pd.read_csv("data_processed/sepsis3_processed.csv.gz", compression='gzip', header=0, index_col=None)

    for labels_chunk in tqdm(labels_chunks):
        label_stay_ids = labels_chunk['stay_id']

        for stay_id in tqdm(list(set(label_stay_ids))): #stayids of cases who experienced sepsis at 7 hours or more
            stay_id = int(stay_id)
            print(stay_id, obs_win, pred_win)
            y = labels_chunk[labels_chunk['stay_id'] == stay_id]['label']

            # cases who experienced sepsis at any point during their ICU stay with enough observation data of 24 hours
            # in the timeframe between their admission in the ICU and 12 hours before their sepsis onset
            if y.values[0]==1:
                sepsis_onset_hour = data[data['stay_id']==stay_id]['hours_after_adm']
                sepsis_onset_hour = int(sepsis_onset_hour)
                time_series = sepsis_onset_hour-pred_win
                dyn = pd.read_csv('data_per_patient/'+str(stay_id) + '/dynamic.csv', header=[0, 1])
                dyn = dyn.iloc[:time_series, :]

                if obs_win<=time_series:
                    dyn.columns = dyn.columns.droplevel(0)
                    dyn = dyn.iloc[-obs_win:, :]

                else:
                    continue

            else:
                #controls did not experience sepsis throughout their ICU course and stayed in the ICU longer than 24 hours
                dyn = pd.read_csv('data_per_patient/' + str(stay_id) + '/dynamic.csv', header=[0, 1])
                dyn.columns = dyn.columns.droplevel(0)
                dyn = dyn.iloc[:24, :]
                dyn = dyn.iloc[-obs_win:, :]


            cols = dyn.columns
            concat_cols = []
            for t in range(obs_win):
                cols_t = [x + "_" + str(t) for x in cols]
                concat_cols.extend(cols_t)

            dyn = dyn.to_numpy()
            dyn = dyn.reshape(1, -1)
            dyn_df = pd.DataFrame(data=dyn, columns=concat_cols)

            stat = pd.read_csv('data_per_patient/'+str(stay_id) + '/static.csv', header=[0, 1])
            stat = stat['COND']

            demo = pd.read_csv('data_per_patient/'+str(stay_id) + '/demo.csv', header=0)

            if X_df.empty:
                X_df = pd.concat([dyn_df, stat], axis=1)
                X_df = pd.concat([X_df, demo], axis=1)

            else:
                X_df = pd.concat([X_df, pd.concat([pd.concat([dyn_df, stat], axis=1), demo], axis=1)], axis=0)

            if y_df.empty:
                y_df = y
            else:
                y_df = pd.concat([y_df, y], axis=0)

            print(X_df, y_df)

    X_df = X_df.reset_index(drop=True)
    y_df = y_df.reset_index(drop=True)
    dataset = pd.concat([X_df, y_df], axis=1)
    dataset.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv', index=False)
    print('SINGLE DATASET CREATED')


def preproc_feat(obs_win, pred_win):

    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv')
    #df = df.fillna(-1)  # for missing ratio values

    # OHE
    ohe = OneHotEncoder()  # or get_dummies()
    gender_ohe = ohe.fit_transform(df['gender'].to_numpy().reshape(-1, 1))
    gender_ohe_features = ohe.get_feature_names_out(['gender'])
    gender_df = pd.DataFrame(gender_ohe.toarray())
    race_ohe = ohe.fit_transform(df['race'].to_numpy().reshape(-1, 1))
    race_ohe_features = ohe.get_feature_names_out(['race'])
    race_df = pd.DataFrame(race_ohe.toarray())

    ohe_features = []
    for f_list in [gender_ohe_features, race_ohe_features]:
        for f in f_list:
            ohe_features.append(f)

    ohe_data = pd.concat([gender_df, race_df], axis=1)
    ohe_data.columns = ohe_features
    df = df.drop(['gender', 'race', 'insurance'], axis=1)
    df = pd.concat([df.iloc[:, :-1], ohe_data, df.iloc[:, -1]], axis=1)
    #print(df.columns)

    df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_processed.csv', index=False)


#KNN imputation - 'missing' files
def KNN_imputation(obs_win, pred_win):
    #OR impute with median
    print('HELLO')
    chartids = pd.read_csv('data_raw/d_items.csv')[['itemid', 'linksto']]
    chartids = chartids.loc[chartids['linksto']=='chartevents']['itemid'].tolist()

    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_processed.csv')
    chartdata = df[df.columns[df.columns.str.contains('|'.join(map(str, chartids)))]]

    # print(len(chartdata.columns))
    chartdata = chartdata.dropna(axis=1, how='all')
    # print(len(chartdata.columns))

    imputer = KNNImputer(n_neighbors=3)
    chartdata_imputed = imputer.fit_transform(chartdata)
    print('yes')
    chartdata_imputed = pd.DataFrame(chartdata_imputed, columns=chartdata.columns)

    columns_not_in_chartdata_imputed = [col for col in df.columns if col not in chartdata_imputed.columns]
    concat_data = pd.concat([df[columns_not_in_chartdata_imputed], chartdata_imputed], axis=1)
    label_column = concat_data.pop('label')  # Remove 'label' column
    concat_data['label'] = label_column  # Add it back as the last column

    concat_data = concat_data.dropna(axis=1, how='all')  ## new line

    concat_data.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv', index=False)


def generate_stayids_demo(obs_win, pred_win):  # , concat_cols,label_enc, one_hot_enc, norm, var_thr, vif, oversam_method):
    chunksize = 5000
    ids = []
    labels = []
    adm_to_pred = []
    hosp_to_icu_list = []

    labels_chunks = pd.read_csv('labels_sepsis.csv',
                                chunksize=chunksize)  # patients with sepsis onset at 7 hours or more
    data = pd.read_csv("data_processed/sepsis3_processed.csv.gz", compression='gzip', header=0, index_col=None)

    for labels_chunk in tqdm(labels_chunks):
        label_stay_ids = labels_chunk['stay_id']

        for stay_id in tqdm(list(set(label_stay_ids))):  # stayids of cases who experienced sepsis at 7 hours or more
            stay_id = int(stay_id)
            print(stay_id, obs_win, pred_win)
            y = labels_chunk[labels_chunk['stay_id'] == stay_id]['label']
            hosp_to_icu = data[data['stay_id'] == stay_id]['hosp_to_icu']

            if y.values[0] == 1:
                sepsis_onset_hour = data[data['stay_id'] == stay_id]['hours_after_adm']
                sepsis_onset_hour = int(sepsis_onset_hour)
                time_series = sepsis_onset_hour - pred_win

                if obs_win <= time_series:
                    ids.append(stay_id)
                    labels.append(y.values[0])
                    adm_to_pred.append(time_series)
                    hosp_to_icu_list.append(hosp_to_icu.values[0])
                else:
                    continue

            else:  # controls
                ids.append(stay_id)
                labels.append(y.values[0])
                adm_to_pred.append(24)
                hosp_to_icu_list.append(hosp_to_icu.values[0])

    stayids_demo = pd.DataFrame({'stayid': ids, 'labels': labels, 'adm_to_pred': adm_to_pred, 'hosp_to_icu': hosp_to_icu_list})
    stayids_demo.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_stayids.csv', index=False)  #normalize


def generate_ratios(obs_win, pred_win):
    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv')

    ratios_df = pd.DataFrame()
    for w in range(obs_win):
        pf_ratio_a = df['220224_'+ str(w)]/df['229280_'+str(w)] # PO2(Arterial) / FIO2(ECMO)
        pf_ratio_b = df['220224_'+ str(w)]/df['229841_'+str(w)] # PO2(Arterial) / FIO2(CH)
        shock_index = df['220045_' + str(w)]/ df['220050_'+str(w)]  # heart rate/ arterial blood pressure systolic
        ratios = pd.concat([pf_ratio_a, pf_ratio_b, shock_index], axis=1)
        ratios_df = pd.concat([ratios_df, ratios], axis=1)

    columns = []
    for w in range(obs_win):
        colpfa = 'pf_ratio_a_' + str(w)
        colpfb = 'pf_ratio_b_' + str(w)
        colsi = 'shock_index_' + str(w)
        columns.append(colpfa)
        columns.append(colpfb)
        columns.append(colsi)

    ratios_df.columns = columns
    ratios_df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_ratios.csv', index=False)  #normalize


def generate_GCS(obs_win, pred_win):
    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv')
    gcs_df = pd.DataFrame()
    for w in range(obs_win):
        gcs_sum = df['220739_' + str(w)] + df['223900_' + str(w)] + df['223901_' + str(w)]
        gcs_df = pd.concat([gcs_df, gcs_sum], axis=1)

    columns = []
    for w in range(obs_win):
        col = 'gcs_sum_' + str(w)
        columns.append(col)
    gcs_df.columns = columns
    gcs_df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_gcs.csv', index=False)  #normalize


def generate_stats(obs_win, pred_win): #(for original, ratios, gcs)
    # concat original, demo, ratios, gcs
    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv')
    demo = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_stayids.csv').iloc[:, -2:]
    ratios = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_ratios.csv')
    gcs = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_gcs.csv')
    df = pd.concat([df.iloc[:, :-1], demo, ratios, gcs, df.iloc[:, -1]], axis=1)

    cols = df.columns
    temporal = []
    for col in cols:
        if '_0' in col:
            temporal.append(col.rsplit('_', 1)[0])
    print(temporal)

    stats_df = pd.DataFrame()
    for col in tqdm(temporal):
        subset = []
        for w in range(obs_win):
            subset.append(col+ '_' + str(w))
        mean = df[subset].mean(axis=1)
        #median = df[subset].median(axis=1)
        #sd = df[subset].std(axis=1)
        max = df[subset].max(axis=1)
        min = df[subset].min(axis=1)
        ran = max-min

        stats = pd.concat([mean, max, min, ran], axis=1) #median,sd
        stats_df = pd.concat([stats_df, stats], axis=1)

    columns = []
    stats = ['mean', 'max', 'min', 'range']  #'median', 'sd'
    for col in temporal:
        for stat in stats:
            column = col + '_' + stat
            columns.append(column)
    stats_df.columns = columns
    stats_df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_stats.csv', index=False) #normalize
    #concat to previous
    df = pd.concat([df.iloc[:, :-1], stats_df, df.iloc[:, -1]], axis=1)
    df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv', index=False)


# create stratified train-test sets
def create_train_test_splits(obs_win, pred_win):

    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed2.csv.gz', compression='gzip')
    print(df['label'].value_counts())
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df['label']):  # same proportions of label1:label0 for each split
        df_train = df.loc[train_index]
        df_test = df.loc[test_index]

        df_train.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed_train.csv', index=False)
        df_test.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed_test.csv', index=False)
        # print(df_train['label'].value_counts(normalize=True))
        # print(df_test['label'].value_counts(normalize=True))


# Feature selection on training set (before modelling) according to variance, Mutual Information and Point Biserial Correlation
def feat_sel(obs_win, pred_win):
    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed_train.csv')
    X_df = df.iloc[:, :-1]
    print('No. of features before selection: ', len(X_df.columns))  # 2621

    # Remove features with variance less than 0.05
    sel = VarianceThreshold(threshold=0.05)
    X_df = pd.DataFrame(sel.fit_transform(X_df))
    print('No. of features after selection (variance): ', len(X_df.columns))  # 990
    X_df.columns = sel.get_feature_names_out()
    df = pd.concat([X_df, df.iloc[:, -1]], axis=1)

    # Mutual Information and Point Biserial Correlation
    feat_selected = pd.DataFrame(columns=['obs_win', 'pred_win', 'Features', 'Names'])
    X_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1]
    corr = []
    features = []

    for f in range(X_df.shape[1]):
        if all(i in [0, 1] for i in X_df.iloc[:, f]):  # binary features
            mi_corr = mutual_info_classif(X_df.iloc[:, f].to_numpy().reshape(-1, 1), y_df.to_numpy().reshape(-1, 1))
            # chi2_corr = chi2(X_df.iloc[:, f], y_df)[0] #returns p-value
            corr.append(mi_corr)
            features.append(X_df.iloc[:, f].name)

        else:  # numerical features
            # f_corr = f_classif(X_df.iloc[:, f].array.reshape(-1, 1), y_df)[0] #returns p-value
            pbs_corr = pbs(X_df.iloc[:, f], y_df)
            if pbs_corr[1] < 0.05:
                corr.append(pbs_corr[0])
                features.append(X_df.iloc[:, f].name)

    corr = pd.DataFrame(zip(corr, features))
    corr.columns = ['correlation', 'feature']
    corr['correlation'] = abs(corr['correlation'])
    corr = corr.sort_values(by='correlation', ascending=False)
    corr = corr.reset_index(drop=True)
    print('No. of features after selection (correlation): ', len(corr))

    feats = corr['feature'].tolist()
    X_df_sel = X_df.loc[:, feats]
    df = pd.concat([X_df_sel, y_df], axis=1)
    df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed_train_fs2.csv', index=False)

    X_df_sel_1000 = X_df_sel.iloc[:, :1000]
    df_1000 = pd.concat([X_df_sel_1000, y_df], axis=1)
    df_1000.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed_train_fs2_1000.csv',
              index=False)

def create_datasets_imputed():
    obs_win = 24
    pred_win = 12
    create_dataset(obs_win, pred_win)
    preproc_feat(obs_win, pred_win)
    KNN_imputation(obs_win, pred_win)
    generate_stayids_demo(obs_win, pred_win)
    generate_ratios(obs_win, pred_win)
    generate_GCS(obs_win, pred_win)
    generate_stats(obs_win, pred_win)
    create_train_test_splits(obs_win, pred_win)
    feat_sel(obs_win, pred_win)
