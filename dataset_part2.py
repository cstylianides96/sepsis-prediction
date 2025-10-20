# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def choose_scorsys_variables(obs_win, pred_win):
    scorsys_vars = ['220210', '224689', '223762', '220050', '220045', '220739', '223900', '223901', '229841',
                '225170', '225690', '220052','221662', '221906', '221289', '229761', '226631', '223830',
                '226534', '227464', '226540', '220546', '225624','225690', '224639', '227444', '220632']
    scorsys_vars_list = []
    for w in range(obs_win):
        for var in scorsys_vars:
            scorsys_vars_list.append(var + '_' + str(w))
    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv', usecols=scorsys_vars_list)
    # df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed2.csv.gz',
    #                  compression='gzip', usecols=scorsys_vars_list)
    df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv', index=False)


def choose_pf_ratio(obs_win, pred_win):
    pf_ratio_cols = []
    for w in range(obs_win):
        pf_ratio_cols.append('pf_ratio_a_' + str(w)) ##pf_ratio_a or b?
    pf_ratio = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_ratios.csv', usecols=pf_ratio_cols)
    pf_ratio.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_pfratio.csv', index=False)


def generate_shock_index_more_than_1(obs_win, pred_win):
    shock_index_cols = []
    for w in range(obs_win):
        shock_index_cols.append('shock_index_' + str(w))
    shock_index = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_ratios.csv', usecols=shock_index_cols)

    #shock index more than 1
    shock_index_morethan1 = (shock_index[shock_index_cols]>1).astype('int')
    shock_index_morethan1.columns = ['shockindex_morethan1_' + str(w) for w in range(obs_win)]
    shock_index = pd.concat([shock_index, shock_index_morethan1], axis=1)
    shock_index.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_shockindex.csv', index=False)


def generate_temp_more_than_39(obs_win, pred_win):
    temp_cols = []
    for w in range(obs_win):
        temp_cols.append('223762_' + str(w))
    temp = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv', usecols=temp_cols)

    temp_morethan39 = (temp[temp_cols] >= 39).astype(int)
    temp_morethan39.columns = ['temp_morethan39_' + str(w) for w in range(obs_win)]
    temp_morethan39.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_temp.csv', index=False)


def generate_map_less_than_65(obs_win, pred_win):
    map_cols = []
    for w in range(obs_win):
        map_cols.append('220052_' + str(w))
    map = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv', usecols=map_cols)

    map_lessthan65 = (map[map_cols]<65).astype('int')
    map_lessthan65 = map_lessthan65.sum(axis=1)
    map_lessthan65 = (map_lessthan65>=1).astype('int') #MAP less than 65 for 1 hour or more
    map_lessthan65 = map_lessthan65.to_frame(name='map_lessthan65')
    map_lessthan65.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_map.csv', index=False) ####column name not saved, it is '0', should be 'map_less_than_65'


def generate_fspn(obs_win, pred_win):
    fspn_cols = []
    for w in range(obs_win):
        fspn_cols.append('224689_' + str(w))
    fspn = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv',
                      usecols=fspn_cols)
    age = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv',
                      usecols=['age'])
    fspn = pd.concat([fspn, age], axis=1)

    fspn_lessthan64 = fspn.loc[fspn['age']<=64]
    fspn_lessthan64 = (fspn_lessthan64[fspn_cols] > 25).astype(int)
    fspn_lessthan64 = fspn_lessthan64.sum(axis=1)
    fspn_lessthan64 = (fspn_lessthan64>1).astype('int') #more than 1 hour

    fspn_morethan64 = fspn.loc[fspn['age']>64]
    fspn_morethan64 = (fspn_morethan64[fspn_cols] > 27).astype(int)
    fspn_morethan64 = fspn_morethan64.sum(axis=1)
    fspn_morethan64 = (fspn_morethan64>1).astype('int') # more than 1 hour

    fspn = pd.concat([fspn_lessthan64, fspn_morethan64], axis=0).sort_index()
    fspn = fspn.to_frame(name='fspn_morethan25or27')
    fspn.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_fspn.csv', index=False)


def generate_qsofa(obs_win, pred_win):
    resp_rate_cols = []
    sysbp_cols = []
    for w in range(obs_win):
        resp_rate_cols.append('220210_' + str(w)) #respiratory rate
        sysbp_cols.append('220050_' + str(w))

    resp_rate = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv',
                       usecols=resp_rate_cols)
    sysbp = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv',
                       usecols=sysbp_cols)
    gcs = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_gcs.csv')
    gcs_cols = gcs.columns

    resp_rate = (resp_rate[resp_rate_cols]>=22).astype('int')
    resp_rate.columns = [''] * resp_rate.shape[1]

    sysbp = (sysbp[sysbp_cols]<=100).astype('int')
    sysbp.columns = [''] * sysbp.shape[1]

    gcs = (gcs[gcs_cols]<15).astype('int')
    gcs.columns = [''] * gcs.shape[1]

    qsofa = resp_rate.add(sysbp).add(gcs)
    qsofa.columns = ['qsofa_' + str(w) for w in range(obs_win)]
    qsofa_cols = qsofa.columns
    qsofa = (qsofa[qsofa_cols]==3).astype('int') #binary, column number = obs win

    qsofa.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_qsofa.csv', index=False)


# itemids show no records
# conditions in SOFA score
def generate_dop_epineph_norepineph(obs_win, pred_win):
    dop_cols = []
    for w in range(obs_win):
        dop_cols.append('221662_' + str(w))
    dop = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv',
                       usecols=dop_cols)

    epineph_cols = []
    for w in range(obs_win):
        epineph_cols.append('221289_' + str(w))
    epineph = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv',
                       usecols=epineph_cols)

    norepineph_cols = []
    for w in range(obs_win):
        norepineph_cols.append('221906_' + str(w))
    norepineph = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv',
                       usecols=norepineph_cols)

    dop_lessthan5 = (dop[dop_cols]<=5).astype('int')
    dop_lessthan5.columns =  ['dop_lessthan5_' + str(w) for w in range(obs_win)]

    dop_epineph_nor_a_df = pd.DataFrame()
    dop_epineph_nor_b_df = pd.DataFrame()
    for w in range(obs_win):
        dop_epineph_nor_a = ((dop['221662_' + str(w)] > 5) |
                       (epineph['221289_' + str(w)] <= 0.1) |
                       (norepineph['221906_' + str(w)] <= 0.1)).astype('int')
        dop_epineph_nor_a_df = pd.concat([dop_epineph_nor_a_df, dop_epineph_nor_a], axis=1)

        dop_epineph_nor_b = ((dop['221662_' + str(w)] > 15) |
                           (epineph['221289_' + str(w)] > 0.1) |
                           (norepineph['221906_' + str(w)] > 0.1)).astype('int')
        dop_epineph_nor_b_df = pd.concat([dop_epineph_nor_b_df, dop_epineph_nor_b], axis=1)
    dop_epineph_nor_a_df.columns = ['dop_epineph_nor_a_' + str(w) for w in range(obs_win)]
    dop_epineph_nor_b_df.columns = ['dop_epineph_nor_b_' + str(w) for w in range(obs_win)]

    dop_lessthan5.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_dop.csv', index=False)
    dop_epineph_nor_a_df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_dopepinephnor_a.csv', index=False)
    dop_epineph_nor_b_df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_dopepinephnor_b.csv', index=False)


def generate_stats(obs_win, pred_win): # specific columns for min / max (saps-ii)
    cols_min_names = ['220050', '226534', '227464', '220546'] #+gcs, pfratio_a
    cols_min_toget = []
    cols_max_names = ['223762', '227464', '225690']
    cols_max_toget = []

    for col in cols_min_names:
        for w in range(obs_win):
            cols_min_toget.append(col + '_' + str(w))
    for col in cols_max_names:
        for w in range(obs_win):
            cols_max_toget.append(col + '_' + str(w))
    cols_min = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv', usecols=cols_min_toget)
    cols_max = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv', usecols=cols_max_toget)

    df_min = pd.DataFrame()
    df_max = pd.DataFrame()

    for col in cols_min_names:
        subset = []
        for w in range(obs_win):
            subset.append(col + '_' + str(w))
        min = cols_min[subset].min(axis=1)
        df_min = pd.concat([df_min, min], axis=1)

    for col in cols_max_names:
        subset = []
        for w in range(obs_win):
            subset.append(col + '_' + str(w))
        max = cols_max[subset].max(axis=1)
        df_max = pd.concat([df_max, max], axis=1)

    df_min.columns = [col + '_min' for col in cols_min_names]
    df_max.columns = [col + '_max' for col in cols_max_names]

    gcs = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_gcs.csv')
    gcs_min = gcs.min(axis=1)
    pfratio = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_pfratio.csv')
    pfratio_min = pfratio.min(axis=1)

    df_stats = pd.concat([df_min, df_max, gcs_min, pfratio_min], axis=1)
    df_stats.columns = list(df_min.columns) + list(df_max.columns) + ['gcs_min', 'pfratio_min']
    df_stats.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_stats.csv', index=False)


def choose_diagnoses(obs_win, pred_win):

    # found in scoring systems
    keywords = [
        'respiratory failure', 'heart failure', 'cirrhosis', 'lung disease', 'dialysis',
        'renal failure', 'infection', 'organ dysfunction', 'hypotension', 'sepsis',
        'chronic disease', 'surgery'
    ]
    pattern = '|'.join(keywords)

    # Load ICD-10 codes that match the pattern
    icd10_diag = pd.read_csv('data_raw/icd10cm_codes_2024.csv')
    icd10_diag = icd10_diag.loc[icd10_diag['label'].str.contains(pattern, case=False), 'icd10_code']
    icd10_diag = icd10_diag.tolist()

    sample = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv',
                         nrows=0)
    existing_cols = sample.columns.to_list()

    icd10_diag = [col for col in icd10_diag if col in existing_cols]

    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv', usecols=icd10_diag)
    df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + 'diag.csv', index=False)


def concatenate(obs_win, pred_win):
    scor_sys_df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '.csv')
    excluded_strings = ['226375', '224639'] #crystalloids, daily weight (used only in generated features)
    scor_sys_df = scor_sys_df[[col for col in scor_sys_df.columns if not any(substr in col for substr in excluded_strings)]]

    pf_ratio = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_pfratio.csv')
    shock_index = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_shockindex.csv')
    temp_morethan39 = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_temp.csv')
    map_lessthan65 = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_map.csv')
    fspn = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_fspn.csv')
    qsofa = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_qsofa.csv')
    stats = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_stats.csv')
    hosp_to_icu= pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_stayids.csv', usecols=['hosp_to_icu'])
    gcs = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_gcs.csv')
    diag = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + 'diag.csv')
    dop_lessthan5 = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_dop.csv')
    dop_epineph_nor_a = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_dopepinephnor_a.csv')
    dop_epineph_nor_b = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_dopepinephnor_b.csv')
    y = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed.csv', usecols=['label'])
    df = pd.concat([scor_sys_df, pf_ratio, shock_index, temp_morethan39, map_lessthan65, fspn,
                    qsofa, stats, hosp_to_icu, gcs, diag, dop_lessthan5, dop_epineph_nor_a, dop_epineph_nor_b, y], axis=1)
    df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_all.csv', index=False)


def change_temporal(obs_win, pred_win):
    df = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_all.csv')
    temporal_cols = ['pf_ratio_a', 'shock_index', 'shockindex_morethan1', 'qsofa', 'gcs_sum',
                     '220210', '224689', '223762', '220050', '220045', '220739', '223900', '223901', '229841',
                    '225170', '225690', '220052', '229761', '226631', '223830',
                    '226534', '227464', '226540', '220546', '225624', '227444', '220632', '221662', '221906', '221289']

    for col in temporal_cols:
        for w in range(1, obs_win):
            curr_col = col + '_' + str(w)
            prev_col = col + '_' + str(w-1)
            diff_col = col + '_' + str(w) + '_diff' #for hr_1-hr_0 new col is hr_1_diff, difference of hr at 2nd hour - hr at 1st hour
            df[diff_col] = df[curr_col] - df[prev_col]

    label = df.pop('label')
    df['label'] = label
    df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_all_changes.csv', index=False)


def create_datasets_scorsys():
    obs_win = 24
    pred_win = 12
    choose_scorsys_variables(obs_win, pred_win)
    choose_pf_ratio(obs_win, pred_win)
    generate_shock_index_more_than_1(obs_win, pred_win)
    generate_temp_more_than_39(obs_win, pred_win)
    generate_map_less_than_65(obs_win, pred_win)
    generate_fspn(obs_win, pred_win)
    generate_qsofa(obs_win, pred_win)
    generate_dop_epineph_norepineph(obs_win, pred_win)
    generate_stats(obs_win, pred_win)
    choose_diagnoses(obs_win, pred_win)
    concatenate(obs_win, pred_win)
    change_temporal(obs_win, pred_win)
