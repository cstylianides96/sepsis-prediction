import numpy as np
import pandas as pd
from tqdm import tqdm
from outlier_removal import outlier_imputation
import math
import re
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 
from sklearn.feature_selection import mutual_info_classif


cohort = pd.read_csv('PIPELINE2/data_processed/sepsis3_processed.csv')
print(cohort.label.value_counts()) # 0: 80%, 1: 20%

cohort_cases = cohort[(cohort['label']==1) & (cohort['los']>=36) & (cohort['hours_after_adm']>=36)]
cohort_controls = cohort[(cohort['label']==0) & (cohort['los']>=24)]
cohort = pd.concat([cohort_cases, cohort_controls], axis=0).reset_index(drop=True)
print(cohort.label.value_counts()) 

subjectids = cohort['subject_id'].tolist()
hospids = cohort['hadm_id'].tolist()
stayids = cohort['stay_id'].tolist()
labels = cohort['label'].tolist()
cohort['sepsis_onset'] = pd.to_datetime(cohort['sepsis_onset'])
sepsistimes = cohort['sepsis_onset'].tolist()
cohort['intime'] = pd.to_datetime(cohort['intime'])
admtimes = cohort['intime'].tolist()
cohort['outtime'] = pd.to_datetime(cohort['outtime'])
distimes = cohort['outtime'].tolist()
itemids = pd.read_csv('PIPELINE2/data_raw/d_items.csv')




def icd9_to_icd10(icd_code):
    """
    Convert ICD-9 code to ICD-10 code using the mapping file.
    
    Args:
        icd_code: ICD-9 code (string or numeric)
    
    Returns:
        ICD-10 code (string) or None if no mapping found
    """
    # Load the mapping file
    mapping_path = './utils/mappings/ICD9_to_ICD10_mapping.txt'
    
    if not hasattr(icd9_to_icd10, 'mapping_dict'):
        # Load mapping only once and cache it
        mapping_df = pd.read_csv(mapping_path, sep='\t', dtype=str)
        # Create dictionary mapping icd9cm to icd10cm
        icd9_to_icd10.mapping_dict = dict(zip(mapping_df['icd9cm'], mapping_df['icd10cm']))
    
    # Convert input to string and remove any leading/trailing whitespace
    icd_code_str = str(icd_code).strip()
    
    # Look up the ICD-10 code
    icd10_code = icd9_to_icd10.mapping_dict.get(icd_code_str, None)
    
    # Return None if the result is 'NoDx' (no diagnosis mapping available)
    if icd10_code == 'NoDx':
        return None
    
    return icd10_code


def detect_outliers_chart(): # for chart events

    chart = pd.read_csv("data_processed/sepsis_chartevents.csv")
    chart = outlier_imputation(chart, 'itemid', 'valuenum', 98, left_thresh=2, impute=True)
    chart.to_csv("data_processed/sepsis_chartevents.csv", index=False)


def make_equal_intervals(): # for chart events
 
    chart = pd.read_csv("data_processed/sepsis_chartevents.csv")
    chart['charttime'] = pd.to_datetime(chart['charttime'])

    cohort_stays = set(cohort['stay_id'].unique())
    chart_stays = set(chart['stay_id'].unique())
    missing_before_merge = sorted(list(cohort_stays - chart_stays))
    print(f"[CHECK] cohort stays: {len(cohort_stays)} | chart stays: {len(chart_stays)} | missing from chart: {len(missing_before_merge)}")
    if len(missing_before_merge) > 0:
        pd.DataFrame({'stay_id': missing_before_merge}).to_csv('data_processed/debug_missing_stayids_before_merge.csv', index=False)

    # Calculate charttime from admisison in hours
    chart = cohort[['stay_id', 'intime']].merge(chart, on='stay_id', how='inner')
    merged_stays = set(chart['stay_id'].unique())
    print(f"[CHECK] stays after merge: {len(merged_stays)}")
    chart['chart_from_adm'] = chart['charttime'] - chart['intime']
    chart['chart_from_adm'] = chart['chart_from_adm'].dt.total_seconds() / 3600  # in hours
    chart = chart.drop(columns=['charttime'])
    print(chart.head())

    max_sepsis_onset = cohort['hours_after_adm'].max()
    print("Max sepsis onset (hours):", max_sepsis_onset)

    # Resampling in bins of size=interval
    final_chart = pd.DataFrame()
    
    for i in tqdm(range(0, math.ceil(max_sepsis_onset), 1)):

        # within interval: median of values, ignoring missing values
        sub_chart = (chart[(chart['chart_from_adm'] >= i) & (chart['chart_from_adm'] < i+1)]
                     .groupby(['stay_id', 'itemid']).agg({'valuenum': np.nanmedian}).reset_index())
        sub_chart['chart_from_adm'] = i
        print(sub_chart)
        final_chart = pd.concat([final_chart, sub_chart], axis=0)  
    final_chart = final_chart.reset_index(drop=True)
    final_stays = set(final_chart['stay_id'].unique())
    dropped_after_binning = sorted(list(merged_stays - final_stays))
    print(f"[CHECK] stays after binning: {len(final_stays)} | dropped after binning: {len(dropped_after_binning)}")
    if len(dropped_after_binning) > 0:
        pd.DataFrame({'stay_id': dropped_after_binning}).to_csv('data_processed/debug_missing_stayids_after_binning.csv', index=False)
    print(final_chart)

    return final_chart, max_sepsis_onset


def create_temporal_chart(): # for chart events

    # Resample chart events into equal intervals
    chart, max_sepsis_onset = make_equal_intervals()

    # Keep only stays that actually have selected chart events
    chart_stayids = chart['stay_id'].unique()
    print(len(chart_stayids), 'stay_ids with chart events')
    cohort_filtered = cohort[cohort['stay_id'].isin(chart_stayids)]
    dropped_stays = sorted(list(set(cohort['stay_id'].unique()) - set(chart_stayids)))
    print(f"[CHECK] stays dropped before temporal reshape: {len(dropped_stays)}")
    if len(dropped_stays) > 0:
        pd.DataFrame({'stay_id': dropped_stays}).to_csv('data_processed/debug_missing_stayids_before_temporal_reshape.csv', index=False)
    print(cohort_filtered)
    stayids = cohort_filtered['stay_id'].tolist()
    full_chart = []

    # Reshape to have one row per hour after trach per patient, impute across intervals
    for stayid in stayids:
        df = chart[chart['stay_id'] == stayid]
        df = df.pivot_table(index='chart_from_adm', columns='itemid', values='valuenum')
        add_indices = pd.Index(range(0, math.ceil(max_sepsis_onset))).difference(df.index)
        add_df = pd.DataFrame(index=add_indices, columns=df.columns).fillna(np.nan)
        df = pd.concat([df, add_df])
        df = df.sort_index()
  
        df = df.ffill()
        df = df.bfill()
        df = df.fillna(df.median())  # ffill, bfill, fill with median for missing chart values
        full_chart.append(df)

    full_chart = pd.concat(full_chart, ignore_index=True)
    print(full_chart.isnull().sum()) # missing values here means item not recorded at all during patient stay
    
    # Drop features with more than 70% missing values
    # missing_percent = (full_chart.isnull().sum() / len(full_chart)) * 100
    # print(missing_percent)
    # full_chart = full_chart.loc[:, missing_percent < 70]
    # # print(full_chart)
    stay_id_repeated = []
    for stayid in stayids:
        stay_id_repeated.extend([stayid] * int(math.ceil(max_sepsis_onset)))
    full_chart.insert(0, 'stay_id', stay_id_repeated)
    
    full_chart.to_csv('data_processed/chartevents_resampled.csv', index=False)
    # then merge according to stayids here

# resampled = pd.read_csv('data_processed/chartevents_resampled.csv', usecols=['stay_id'])
# print(resampled['stay_id'].nunique(), 'unique stay_ids in resampled chart events')
# print(cohort['hours_after_adm'].describe())
# print(len(resampled), 'rows in resampled chart events') # 16739808


def create_static_diagnoses():

    cohort_filtered = cohort[cohort['stay_id'].isin(pd.read_csv('data_processed/chartevents_resampled.csv')['stay_id'].unique())]
    cohort_filtered = cohort_filtered[['subject_id', 'hadm_id', 'stay_id']]

    # Preprocess diagnoses with ICD-9 to ICD-10 conversion
    diag = pd.read_csv('data_processed/sepsis_diagnoses.csv')
    diag['icd_code_converted'] = diag.apply(
        lambda row: icd9_to_icd10(row['icd_code']) if row['icd_version'] == 9 else row['icd_code'], axis=1)
    diag = diag.drop(columns=['icd_code', 'icd_version', 'seq_num', 'icd_code_normalized', 'is_relevant'])
    diag = diag.rename(columns={'icd_code_converted': 'new_icd10_code'})
    diag = diag.dropna(subset=['new_icd10_code'])
    diag = diag.merge(cohort_filtered, on=['subject_id', 'hadm_id'], how='right')
    diag = diag.drop(columns=['subject_id', 'hadm_id'])

    # One-hot encode ICD-10 codes per stay_id
    diag = diag.pivot_table(index=['stay_id'], columns='new_icd10_code', values='new_icd10_code', aggfunc='size', fill_value=0)
    diag = (diag > 0).astype(int)
    diag = diag.reset_index()
    
    # Include all cohort stay_ids and fill missing diagnoses with 0 (patient with no diagnose)
    all_stays = cohort[['stay_id']].drop_duplicates()
    diag = all_stays.merge(diag, on='stay_id', how='left')
    diag = diag.fillna(0).astype(int)
    diag.to_csv('data_processed/diagnoses_static.csv', index=False)

def create_static_demographics():

    cohort_filtered = cohort[cohort['stay_id'].isin(pd.read_csv('data_processed/chartevents_resampled.csv')['stay_id'].unique())]
    cohort_filtered = cohort_filtered[['subject_id', 'hadm_id', 'stay_id']]
    demo = pd.read_csv('data_processed/sepsis_demographics.csv')
    demo = demo.drop(columns=['race'], errors='ignore')
    demo = demo.merge(cohort_filtered, on=['subject_id', 'stay_id'], how='right')

    # One-hot encode gender
    demo = pd.get_dummies(demo, columns=['gender'], prefix=['gender'])
    # Convert boolean dummies to integers (True -> 1, False -> 0)
    demo = demo.astype({col: int for col in demo.columns if demo[col].dtype == 'bool'})

    # Rename height and weight columns
    demo = demo.rename(columns={'226512': 'weight_on_adm', '226730': 'height'})

    demo.to_csv('data_processed/demographics_static.csv', index=False)


def handle_temporal():  #do not flatten, just select 24 hour windows, keep dyn format
    
    cohort_filtered = cohort[cohort['stay_id'].isin(pd.read_csv('data_processed/chartevents_resampled.csv')['stay_id'].unique())]
    cohort_filtered = cohort_filtered[['subject_id', 'hadm_id', 'stay_id', 'hours_after_adm', 'label']]
    print(cohort_filtered)

    dyn_full = pd.read_csv('data_processed/chartevents_resampled.csv')
    groups = dyn_full.groupby('stay_id', sort=False)

    X_rows = []

  
    for row in cohort_filtered.itertuples(index=False):
        stay_id = row.stay_id
        y_val = row.label
        # print(f"Processing stay_id {stay_id} with label {y_val} and hours_after_adm {hours_after_adm}")

        try:
            dyn_patient = groups.get_group(stay_id)
        except KeyError:
            continue

        # keep cases & controls with at least 24 hours of data
        if y_val == 1:
            hours_after_adm = math.ceil(row.hours_after_adm)
            time_series = hours_after_adm - 12
            if time_series < 24:
                print('skipping CASE with insufficient data:', stay_id)
                continue
            dyn_patient = dyn_patient.iloc[:time_series, :].iloc[-24:, :]

        else:  # controls
            if len(dyn_patient) < 24:
                print('skipping CONTROL with insufficient data:', stay_id)
                continue
            dyn_patient = dyn_patient.iloc[:24, :]

        if dyn_patient.empty:
            continue

        dyn_patient = dyn_patient.copy().reset_index(drop=True)
        dyn_patient['time_step'] = np.arange(len(dyn_patient))
        dyn_patient['label'] = y_val
        X_rows.append(dyn_patient)

    if len(X_rows) == 0:
        dataset = pd.DataFrame(columns=['stay_id', 'time_step', 'label'])
    else:
        dataset = pd.concat(X_rows, ignore_index=True)

    dataset = dataset.reset_index(drop=True)
    print(dataset.columns)
    dataset.to_csv('data_processed/chartevents_resampled_24hrs.csv', index=False)


def generate_gcs_sum():
    chart = pd.read_csv('data_processed/chartevents_resampled_24hrs.csv')
    print(chart[['220739', '223900', '223901']].head())
    gcs_sum = chart['220739'] + chart['223900'] + chart['223901']
    chart['gcs_sum'] = gcs_sum
    chart.to_csv('data_processed/chartevents_resampled_24hrs.csv', index=False)


def generate_ratios(): # need to add to chartevents before creating datasets
    chart = pd.read_csv('data_processed/chartevents_resampled_24hrs.csv')
    # pf_ratio = chart['220224']/chart['229841'] # PO2(Arterial) / FIO2(CH)
    shock_index = chart['220045']/ chart['220050']  # heart rate/ arterial blood pressure systolic
    # chart['pf_ratio'] = pf_ratio
    chart['shock_index'] = shock_index

    # fill missing 220224 and pf ratio values with -1 - not recorded, MNAR  
    # chart['220224'].fillna(-1, inplace=True)
    # chart['229841'].fillna(-1, inplace=True)
    # chart['pf_ratio'].fillna(-1, inplace=True)
    chart.to_csv('data_processed/chartevents_resampled_24hrs.csv', index=False)


def encode_MNAR():
    chart = pd.read_csv('data_processed/chartevents_resampled_24hrs.csv')

    # Encode binary features for missing values (MNAR) - drop existing
    oxygen_pressure_cols = ['220224']
    oxygen_sat_cols = ['220227']
    ph_arterial_cols = ['223830']
    fio2_cols = ['229841']
    peep_cols = ['224700']
    etco2_cols = ['228640']

    chart['oxygen_pressure_recorded'] = (chart[oxygen_pressure_cols].notna()).astype(int).iloc[:, 0]
    chart['oxygen_sat_recorded'] = (chart[oxygen_sat_cols].notna()).astype(int).iloc[:, 0]
    chart['ph_arterial_recorded'] = (chart[ph_arterial_cols].notna()).astype(int).iloc[:, 0]
    chart['fio2_recorded'] = (chart[fio2_cols].notna()).astype(int).iloc[:, 0]
    chart['peep_recorded'] = (chart[peep_cols].notna()).astype(int).iloc[:, 0]
    chart['etco2_recorded'] = (chart[etco2_cols].notna()).astype(int).iloc[:, 0]
    chart = chart.drop(columns=oxygen_pressure_cols + oxygen_sat_cols + ph_arterial_cols + fio2_cols + peep_cols + etco2_cols)
    chart.to_csv('data_processed/chartevents_resampled_24hrs.csv', index=False)


def create_train_val_test_split(): # just fr temporal - chartevents - then merge static in each split

    chart = pd.read_csv('data_processed/chartevents_resampled_24hrs.csv')

    # Get one label per stay and perform stratified split on stays to avoid leakage of windows
    group_df = chart.groupby('stay_id')['label'].first().reset_index()
    stays = group_df['stay_id'].values
    stay_labels = group_df['label'].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(stays.reshape(-1, 1), stay_labels))
    train_stays = stays[train_idx]
    test_stays = stays[test_idx]

    # Create validation split from training stays (30% of training set), stratified by label
    train_group_df = group_df[group_df['stay_id'].isin(train_stays)].reset_index(drop=True)
    train_stays_all = train_group_df['stay_id'].values
    train_labels_all = train_group_df['label'].values

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_sub_idx, val_idx = next(sss_val.split(train_stays_all.reshape(-1, 1), train_labels_all))
    train_stays_final = train_stays_all[train_sub_idx]
    val_stays = train_stays_all[val_idx]

    # Keep stay_id in train/val/test for group-aware cross-validation downstream
    chart_train = chart[chart['stay_id'].isin(train_stays_final)].reset_index(drop=True)
    chart_val = chart[chart['stay_id'].isin(val_stays)].reset_index(drop=True)
    chart_test = chart[chart['stay_id'].isin(test_stays)].reset_index(drop=True)

    print(chart_train['label'].value_counts(normalize=True, dropna=False))
    print(chart_val['label'].value_counts(normalize=True, dropna=False))
    print(chart_test['label'].value_counts(normalize=True, dropna=False))

    chart_train.to_csv(f'data_processed/chart_resampled_24hrs_train.csv', index=False)
    chart_val.to_csv(f'data_processed/chart_resampled_24hrs_val.csv', index=False)
    chart_test.to_csv(f'data_processed/chart_resampled_24hrs_test.csv', index=False)


def merge_static_with_temporal():

    demo = pd.read_csv('data_processed/demographics_static.csv').drop(columns=['subject_id'])
    diag = pd.read_csv('data_processed/diagnoses_static.csv')
    hosp_to_icu = pd.read_csv('data_processed/sepsis3_processed.csv')[['stay_id', 'hosp_to_icu']]
    chart_train = pd.read_csv(f'data_processed/chart_resampled_24hrs_train.csv')
    chart_val = pd.read_csv(f'data_processed/chart_resampled_24hrs_val.csv')
    chart_test = pd.read_csv(f'data_processed/chart_resampled_24hrs_test.csv')

    # Merge static with temporal for train and test
    train_merged = chart_train.merge(demo, on='stay_id', how='left').merge(diag, on='stay_id', how='left').merge(hosp_to_icu, on='stay_id', how='left')
    val_merged = chart_val.merge(demo, on='stay_id', how='left').merge(diag, on='stay_id', how='left').merge(hosp_to_icu, on='stay_id', how='left')
    test_merged = chart_test.merge(demo, on='stay_id', how='left').merge(diag, on='stay_id', how='left').merge(hosp_to_icu, on='stay_id', how='left')
    train_merged = train_merged.reset_index(drop=True)
    val_merged = val_merged.reset_index(drop=True)
    test_merged = test_merged.reset_index(drop=True)
    train_merged['label'] = train_merged.pop('label')
    val_merged['label'] = val_merged.pop('label')
    test_merged['label'] = test_merged.pop('label')
    

    train_merged.to_csv(f'data_processed/train_merged.csv', index=False)
    val_merged.to_csv(f'data_processed/val_merged.csv', index=False)
    test_merged.to_csv(f'data_processed/test_merged.csv', index=False)


def impute():
    
    train_merged = pd.read_csv(f'data_processed/train_merged.csv')
    val_merged = pd.read_csv(f'data_processed/val_merged.csv')
    test_merged = pd.read_csv(f'data_processed/test_merged.csv')

    train = train_merged.drop(columns=['stay_id', 'time_step', 'label'])
    y_train = train_merged['label']
    id_train = train_merged['stay_id']
    time_step_train = train_merged['time_step']
    val = val_merged.drop(columns=['stay_id', 'time_step', 'label'])
    y_val = val_merged['label']
    id_val = val_merged['stay_id']
    time_step_val = val_merged['time_step']
    test = test_merged.drop(columns=['stay_id', 'time_step', 'label'])
    y_test = test_merged['label']
    id_test = test_merged['stay_id']
    time_step_test = test_merged['time_step']

    # Enforce identical feature order/names across splits for sklearn transformers
    train_cols = train.columns.tolist()
    val = val.reindex(columns=train_cols)
    test = test.reindex(columns=train_cols)

    # IterativeImputer / MICE (fit on train, apply to test)
    imputer = IterativeImputer(
    random_state=123,
    max_iter=15,
    sample_posterior=False,
    skip_complete=True
    )
    train = pd.DataFrame(imputer.fit_transform(train), columns=imputer.feature_names_in_)
    val = pd.DataFrame(imputer.transform(val), columns=imputer.feature_names_in_)
    test = pd.DataFrame(imputer.transform(test), columns=imputer.feature_names_in_)

    train = pd.concat([id_train.reset_index(drop=True), time_step_train.reset_index(drop=True), train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val = pd.concat([id_val.reset_index(drop=True), time_step_val.reset_index(drop=True), val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)    
    test = pd.concat([id_test.reset_index(drop=True), time_step_test.reset_index(drop=True), test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    train.to_csv(f'data_processed/train_merged_imputed.csv', index=False)
    val.to_csv(f'data_processed/val_merged_imputed.csv', index=False)
    test.to_csv(f'data_processed/test_merged_imputed.csv', index=False)


def flatten_imputed_datasets_24hrs():
    """
    Read train/val/test merged-imputed datasets and flatten each stay into one row.
    Feature names are formatted as <feature>_<hour>, where hour is 0..23.
    """

    datasets = {
        'train': 'data_processed/train_merged_imputed.csv',
        'val': 'data_processed/val_merged_imputed.csv',
        'test': 'data_processed/test_merged_imputed.csv'
    }

    for split_name, path in datasets.items():
        df = pd.read_csv(path)
        df = df.sort_values(['stay_id', 'time_step']).reset_index(drop=True)

        # Identify temporal features
        feature_cols = [c for c in df.columns if c not in ['stay_id', 'time_step', 'label'] and c[-1].isdigit() and c[:-2].isdigit()]
        print(feature_cols)
        flattened_rows = []

        for stay_id, group in df.groupby('stay_id', sort=False):
            group = group.sort_values('time_step').reset_index(drop=True)

            # Keep only complete 24-hour sequences (0..23)
            if len(group) != 24 or set(group['time_step'].tolist()) != set(range(24)):
                print('NOT 24 HOURS, SKIPPING STAY_ID:', stay_id)
                continue

            row_dict = {
                'stay_id': stay_id,
                'label': int(group['label'].iloc[0])
            }

            for hour in range(24):
                hour_row = group[group['time_step'] == hour].iloc[0]
                for col in feature_cols:
                    row_dict[f'{col}_{hour}'] = hour_row[col]

            flattened_rows.append(row_dict)

        # Merge static features (diagnoses, demographics, hosp_to_icu)
        static_cols = [c for c in df.columns if c not in ['stay_id', 'time_step', 'label'] and not c[:-2].isdigit()]
        print(static_cols)
        for stay_id, group in df.groupby('stay_id', sort=False):
            for row_dict in flattened_rows:
                if row_dict['stay_id'] == stay_id:
                    for col in static_cols:
                        row_dict[col] = group[col].iloc[0]
                    break

        flattened_df = pd.DataFrame(flattened_rows)
        flattened_df['label'] = flattened_df.pop('label')  # Move label to the end
        flattened_df.to_csv(f'data_processed/{split_name}_merged_imputed_flattened.csv', index=False)
        print(f'{split_name} flattened shape: {flattened_df.shape}')

def create_aggregates():
    
    datasets = {
    # 'train': 'PIPELINE2/data_processed/train_merged_imputed_flattened.csv',
    'val': 'PIPELINE2/data_processed/val_merged_imputed_flattened.csv',
    'test': 'PIPELINE2/data_processed/test_merged_imputed_flattened.csv'
    }

    for split_name, path in datasets.items():
        df = pd.read_csv(path)
        df = df.sort_values(['stay_id']).reset_index(drop=True)

        temporal_cols = [
            c for c in df.columns
            if c not in ['stay_id', 'time_step', 'label'] and re.search(r'_(\d+)$', c) and 0 <= int(re.search(r'_(\d+)$', c).group(1)) < 24
        ]
        agg_rows = []

        for stay_id, row in df.groupby('stay_id', sort=False):
            agg_row = {'stay_id': stay_id, 'label': int(row['label'].iloc[0])}
            
            for feature in sorted(set(col.rsplit('_', 1)[0] for col in temporal_cols)):
                feature_values = [row[f'{feature}_{t}'].iloc[0] for t in range(24)]
                agg_row[f'{feature}_mean'] = np.mean(feature_values)
                agg_row[f'{feature}_min'] = np.min(feature_values)
                agg_row[f'{feature}_max'] = np.max(feature_values)
                agg_row[f'{feature}_range'] = np.max(feature_values) - np.min(feature_values)

                # Calculate differences between consecutive timesteps
                diffs = np.diff(feature_values)
                for i, diff in enumerate(diffs):
                    agg_row[f'{feature}_diff_{i}'] = diff

            # Preserve original temporal windows so downstream binary feature generation
            # has access to all timesteps (0..23)
            for col in temporal_cols:
                agg_row[col] = row[col].iloc[0]

            agg_rows.append(agg_row)

        # Merge static features by taking the first value
        static_cols = [c for c in df.columns if c not in ['stay_id', 'time_step', 'label'] and c not in temporal_cols]
        for stay_id, group in df.groupby('stay_id', sort=False):
            for row_dict in agg_rows:
                if row_dict['stay_id'] == stay_id:
                    for col in static_cols:
                        row_dict[col] = group[col].iloc[0]
                    break

        agg_df = pd.DataFrame(agg_rows)
        agg_df.to_csv(f'PIPELINE2/data_processed/{split_name}_merged_imputed_flattened_aggregated.csv', index=False)
        print(f'{split_name} aggregated shape: {agg_df.shape}')


def generate_binary_features():

    datasets = {
    'train': 'PIPELINE2/data_processed/train_merged_imputed_flattened_aggregated.csv',
    'val': 'PIPELINE2/data_processed/val_merged_imputed_flattened_aggregated.csv',
    'test': 'PIPELINE2/data_processed/test_merged_imputed_flattened_aggregated.csv'
    }

    for split_name, path in datasets.items():
        df = pd.read_csv(path)
        df = df.sort_values(['stay_id']).reset_index(drop=True)

        for t in range(24):
            shock_col = f'shock_index_{t}'
            temp_col = f'223762_{t}'
            map_col = f'220052_{t}'
            resp_col = f'220210_{t}'
            sysbp_col = f'220050_{t}'
            gcs_col = f'gcs_sum_{t}'

            # Threshold-based binary features per timestep
            if shock_col in df.columns:
                df[f'shock_index_more_than_1_{t}'] = (df[shock_col] > 1).astype(int)
            if temp_col in df.columns:
                df[f'temp_more_than_39_{t}'] = (df[temp_col] >= 39).astype(int)
            if map_col in df.columns:
                df[f'map_less_than_65_{t}'] = (df[map_col] < 65).astype(int)

            # qSOFA=3 per timestep
            if resp_col in df.columns and sysbp_col in df.columns and gcs_col in df.columns:
                resp_rate_more_than_22 = (df[resp_col] >= 22).astype(int)
                sysbp_less_than_100 = (df[sysbp_col] <= 100).astype(int)
                gcs_sum_less_than_15 = (df[gcs_col] < 15).astype(int)
                qsofa_score = resp_rate_more_than_22 + sysbp_less_than_100 + gcs_sum_less_than_15
                df[f'qsofa_3_{t}'] = (qsofa_score == 3).astype(int)
        df.to_csv(f'PIPELINE2/data_processed/{split_name}_merged_imputed_flattened_aggregated_binary.csv', index=False)


def feature_selection(): 

    train = pd.read_csv(f'PIPELINE2/data_processed/train_merged_imputed_flattened_aggregated_binary.csv')
    val = pd.read_csv(f'PIPELINE2/data_processed/val_merged_imputed_flattened_aggregated_binary.csv')
    test = pd.read_csv(f'PIPELINE2/data_processed/test_merged_imputed_flattened_aggregated_binary.csv')

    x_train_merged = train.drop(columns=['stay_id', 'hadm_id', 'label'])
    print('No. of features before selection:', x_train_merged.shape[1])
    y_train_merged = train['label']
    x_val_merged = val.drop(columns=['stay_id', 'hadm_id', 'label'])
    y_val_merged = val['label']
    x_test_merged = test.drop(columns=['stay_id', 'hadm_id', 'label'])
    y_test_merged = test['label']

    # Enforce train feature names/order for sklearn compatibility
    train_cols = x_train_merged.columns.tolist()
    x_val_merged = x_val_merged.reindex(columns=train_cols, fill_value=0)
    x_test_merged = x_test_merged.reindex(columns=train_cols, fill_value=0)

    # Variance threshold (fit on train, apply to val and test)
    sel = VarianceThreshold(threshold=0.05)
    train_sel = pd.DataFrame(sel.fit_transform(x_train_merged), columns=sel.get_feature_names_out())
    val_sel = pd.DataFrame(sel.transform(x_val_merged), columns=sel.get_feature_names_out())
    test_sel = pd.DataFrame(sel.transform(x_test_merged), columns=sel.get_feature_names_out())
    print(f'No. of features after selection (variance): {train_sel.shape[1]}')

    # Mutual information feature selection (fit on train, apply to val and test)
    mi_scores = mutual_info_classif(train_sel, y_train_merged, random_state=42)
    mi_feature_indices = np.argsort(mi_scores)[-1000:]  # Keep top 1000 features
    mi_feature_names = train_sel.columns[mi_feature_indices]

    train_sel = train_sel[mi_feature_names]
    val_sel = val_sel[mi_feature_names]
    test_sel = test_sel[mi_feature_names]
    print(f'No. of features after selection (mutual information): {train_sel.shape[1]}')


    # Save selected features train and test
    train_sel = pd.concat([train['stay_id'].reset_index(drop=True), train_sel.reset_index(drop=True), y_train_merged.reset_index(drop=True)], axis=1)
    val_sel = pd.concat([val['stay_id'].reset_index(drop=True), val_sel.reset_index(drop=True), y_val_merged.reset_index(drop=True)], axis=1)
    test_sel = pd.concat([test['stay_id'].reset_index(drop=True), test_sel.reset_index(drop=True), y_test_merged.reset_index(drop=True)], axis=1)
    
    train_sel.to_csv(f'PIPELINE2/data_processed/train_selected.csv', index=False)
    val_sel.to_csv(f'PIPELINE2/data_processed/val_selected.csv', index=False)
    test_sel.to_csv(f'PIPELINE2/data_processed/test_selected.csv', index=False)

def check_missing_values():
    df_train = pd.read_csv('PIPELINE2/data_processed/train_selected.csv')
    df_val = pd.read_csv('PIPELINE2/data_processed/val_selected.csv')
    df_test = pd.read_csv('PIPELINE2/data_processed/test_selected.csv')
        
    print(f"Train missing values:\n{df_train.isnull().sum()}")
    print(f"\nValidation missing values:\n{df_val.isnull().sum()}")
    print(f"\nTest missing values:\n{df_test.isnull().sum()}")
    print(f"Total missing in train: {df_train.isnull().sum().sum()}")
    print(f"Total missing in validation: {df_val.isnull().sum().sum()}")
    print(f"Total missing in test: {df_test.isnull().sum().sum()}")

    print(df_train.label.value_counts(normalize=True, dropna=False))
    print(df_val.label.value_counts(normalize=True, dropna=False))
    print(df_test.label.value_counts(normalize=True, dropna=False))



def preprocess():
    detect_outliers_chart(),
    print('OUTLIERS DETECTED')
    create_temporal_chart(),  
    print('TEMPORAL CHART CREATED')
    create_static_diagnoses(),
    print('STATIC DIAGNOSES CREATED')
    create_static_demographics()
    print('STATIC DEMOGRAPHICS CREATED')
    handle_temporal(),
    print('TEMPORAL DATA HANDLED'),
    generate_gcs_sum(),
    print('GCS SUM GENERATED'),
    generate_ratios(),
    print('RATIOS GENERATED'),
    encode_MNAR(),
    print('MNAR ENCODED'),
    create_train_val_test_split(),
    print('TRAIN-VAL-TEST SPLIT CREATED'),
    merge_static_with_temporal(),
    print('STATIC MERGED WITH TEMPORAL'),
    impute(),
    print('IMPUTATION DONE'),
    flatten_imputed_datasets_24hrs(),
    print('DATASETS FLATTENED'),
    create_aggregates(),
    print('AGGREGATES CREATED'),
    generate_binary_features(),
    print('BINARY FEATURES GENERATED'),
    feature_selection(),
    print('FEATURE SELECTION DONE'),
    # check_missing_values()
    # print('MISSING VALUES CHECKED')
 