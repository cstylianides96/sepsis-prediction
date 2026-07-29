# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import numpy as np
from outlier_removal import outlier_imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 
import math
import joblib
from model_evaluation import evaluate
import gc
import re
import os
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler


def detect_outliers_chart(): # for chart events
    lab = pd.read_csv("/data_processed_eicu/lab_filtered.csv")
    nurse = pd.read_csv("/data_processed_eicu/nurse_filtered.csv")

    # Replace generic "Value" with the corresponding label
    mask = nurse['nursingchartcelltypevalname'] == 'Value'
    nurse.loc[mask, 'nursingchartcelltypevalname'] = nurse.loc[mask, 'nursingchartcelltypevallabel']

    # Apply outlier imputation for each temporal numeric dataset and save the processed files
    for df, name, item, value in zip([lab, nurse], 
                                     ['lab', 'nurse'], 
                                     ['labname', 'nursingchartcelltypevalname'],
                                     ['labresult', 'nursingchartvalue']):
        # Remove non-numeric values
        if value and value in df.columns:
            numeric_vals = pd.to_numeric(df[value], errors='coerce')
            df = df[numeric_vals.notna()].copy()
            df[value] = numeric_vals.loc[df.index]

        df = outlier_imputation(df, item, value, 98, left_thresh=2, impute=True)
        print(df.describe())
        print(df.head())
        df.to_csv(f"/data_processed_eicu/{name}_filtered.csv", index=False)


def make_equal_intervals(): # for temporal events
    lab = pd.read_csv("/data_processed_eicu/lab_filtered.csv")
    nurse = pd.read_csv("/data_processed_eicu/nurse_filtered.csv")

    # Calculate max discharge time from icu admission in hours
    cohort = pd.read_csv("/data_processed_eicu/sepsis3_eicu.csv")
    max_discharge_offset = cohort['diagnosisoffset'].max()
    max_discharge_offset = max_discharge_offset / 60  # convert to hours
    print("Max discharge offset (hours):", max_discharge_offset)

    # Resampling in bins of size=interval - lab, nurse, resp
    final_lab = pd.DataFrame()
    final_nurse = pd.DataFrame()

    for df, target, item, value, time_from_adm in zip(
        [lab, nurse],
        ['lab', 'nurse'],
        ['labname', 'nursingchartcelltypevalname'],
        ['labresult', 'nursingchartvalue'],
        ['labresultoffset', 'nursingchartoffset']
    ):

        final_df = pd.DataFrame()
        interval = 1  # 1 hour intervals
        for i in tqdm(range(0, math.ceil(max_discharge_offset), interval)):  

            # within interval: median of values, ignoring missing values
            sub_df = (df[(df[time_from_adm]/60 >= i) & (df[time_from_adm]/60 < i+interval)]
                        .groupby(['patientunitstayid', item]).agg({value: np.nanmedian}).reset_index())
            sub_df['time_from_adm'] = i
            print(sub_df)
            final_df = pd.concat([final_df, sub_df], axis=0)  
        final_df = final_df.reset_index(drop=True)
        print(final_df)

        if target == 'lab':
            final_lab = final_df
        else:
            final_nurse = final_df

    return final_lab, final_nurse, max_discharge_offset


def create_temporal_chart(): 

    # Resample chart events into equal intervals
    lab, nurse, max_discharge_offset = make_equal_intervals()

    cohort = pd.read_csv('/data_processed_eicu/sepsis3_eicu.csv')
    stayids = cohort['patientunitstayid'].tolist()
    # print(stayids)
   

    # Reshape to have one row per hour after trach per patient, impute across intervals
    for base_df, target, item, value in zip(
        [lab, nurse],
        ['lab', 'nurse'],
        ['labname', 'nursingchartcelltypevalname'],
        ['labresult', 'nursingchartvalue']):

        full_df = []
        for stayid in stayids:
            stay_df = base_df[base_df['patientunitstayid'] == stayid]
            stay_df = stay_df.pivot_table(index='time_from_adm', columns=item, values=value)
            print(stay_df)
            add_indices = pd.Index(range(0, math.ceil(max_discharge_offset))).difference(stay_df.index) 
            print(add_indices)
            add_df = pd.DataFrame(index=add_indices, columns=stay_df.columns).fillna(np.nan)
            print(add_df)
            stay_df = pd.concat([stay_df, add_df])
            print(stay_df)
            stay_df = stay_df.sort_index()
            print(stay_df)
            stay_df.insert(0, 'patientunitstayid', stayid)  
    
            stay_df = stay_df.ffill()
            stay_df = stay_df.bfill()
            stay_df = stay_df.fillna(stay_df.median())  # ffill, bfill, fill with median for missing chart values
            print(stay_df)
            full_df.append(stay_df)

        full_df = pd.concat(full_df, ignore_index=True)
        print(full_df)
        print(full_df.isnull().sum()) # missing values here means item not recorded at all for some patients 
        full_df.to_csv('/data_processed_eicu/' + target + '_resampled.csv', index=False)


def rename():

    lab = pd.read_csv('/data_processed_eicu/lab_resampled.csv').rename(columns={'bedside glucose': 220621, 'WBC x 1000': 220546, 
                                                                               'BUN': 225624, 'Hct': 220545, 'LDH': 220632, 
                                                                               'sodium': 220645, 'fibrinogen': 227468, 
                                                                               'magnesium': 220635, '-lymphs': 225641, 
                                                                               'creatinine': 220615, 'alkaline phos.': 225612, 
                                                                               'CRP': 227444, 'phosphate': 225677})
    lab = lab.drop(columns=['glucose'])

    nurse = pd.read_csv('/data_processed_eicu/nurse_resampled.csv').rename(columns={'Non-Invasive BP Diastolic': 220051, 
                                                                                   'Non-Invasive BP Systolic': 220050,
                                                                              'Non-Invasive BP Mean': 220052, 
                                                                              'Respiratory Rate': 220210, 
                                                                              'Heart Rate': 220045, 
                                                                              'Motor Response': 223901, 
                                                                              'Verbal Response': 223900})
    
    lab.to_csv('/data_processed_eicu/lab_resampled_renamed.csv', index=False)
    nurse.to_csv('/data_processed_eicu/nurse_resampled_renamed.csv', index=False)
    
    demo = pd.read_csv('/data_processed_eicu/hosp_time.csv').rename(columns={'hospitaladmitoffset': 'hosp_to_icu'})
    demo.to_csv('/data_processed_eicu/demo_renamed.csv', index=False)


def handle_temporal(cohort_chunk_size=100):  #do not flatten, just select 24 hour windows, keep dyn format
    
    cohort = pd.read_csv('/data_processed_eicu/sepsis3_eicu.csv')
    output_path = '/data_processed_eicu/temporal_resampled_renamed_24hrs.csv'
    wrote_any_rows = False

    if os.path.exists(output_path):
        os.remove(output_path)

    # Pre-build cohort lookup for faster access
    cohort_dict = {row.patientunitstayid: (row.label, row.diagnosisoffset) 
                   for row in cohort.itertuples(index=False)}
    chunk_size = 10000  # Read CSV in 10k row chunks
    
    print('Building patient temporal data...')
    patient_data = {}

    # Read lab data in chunks, filter by cohort patients
    for lab_chunk in pd.read_csv('/data_processed_eicu/lab_resampled_renamed.csv', 
                                  chunksize=chunk_size):
        lab_chunk = lab_chunk[lab_chunk['patientunitstayid'].isin(cohort_dict.keys())]
        for stay_id, group in lab_chunk.groupby('patientunitstayid'):
            if stay_id not in patient_data:
                patient_data[stay_id] = {'lab': [], 'nurse': []}
            patient_data[stay_id]['lab'].append(group)
    
    print('Building nurse temporal data...')
    # Read nurse data in chunks, filter by cohort patients
    for nurse_chunk in pd.read_csv('/data_processed_eicu/nurse_resampled_renamed.csv', 
                                    chunksize=chunk_size):
        nurse_chunk = nurse_chunk[nurse_chunk['patientunitstayid'].isin(cohort_dict.keys())]
        for stay_id, group in nurse_chunk.groupby('patientunitstayid'):
            if stay_id not in patient_data:
                patient_data[stay_id] = {'lab': [], 'nurse': []}
            patient_data[stay_id]['nurse'].append(group)
    
    print('Processing patients...')
    chunk_rows = []
    
    for stay_id, (y_val, diagnosisoffset) in tqdm(cohort_dict.items(), desc='Processing stays'):
        if stay_id not in patient_data:
            continue
        
        # Concatenate lab and nurse for this patient
        lab_data = pd.concat(patient_data[stay_id]['lab'], ignore_index=True) if patient_data[stay_id]['lab'] else pd.DataFrame()
        nurse_data = pd.concat(patient_data[stay_id]['nurse'], ignore_index=True) if patient_data[stay_id]['nurse'] else pd.DataFrame()
        
        if lab_data.empty or nurse_data.empty:
            continue
        
        # Add time_step
        lab_data['time_step'] = lab_data.groupby('patientunitstayid').cumcount() + 1
        nurse_data['time_step'] = nurse_data.groupby('patientunitstayid').cumcount() + 1
        
        dyn_patient = pd.merge(lab_data, nurse_data, on=['patientunitstayid', 'time_step'], how='inner')
        
        # keep cases & controls with at least 24 hours of data
        if y_val == 1:
            hours_after_adm = math.ceil(diagnosisoffset / 60)  # convert to hours
            time_series = hours_after_adm - 12
            if time_series < 24:
                continue
            dyn_patient = dyn_patient.iloc[:time_series, :].iloc[-24:, :]
        else:  # controls
            if len(dyn_patient) < 24:
                continue
            dyn_patient = dyn_patient.iloc[:24, :]

        if dyn_patient.empty:
            continue

        dyn_patient = dyn_patient.copy().reset_index(drop=True)
        dyn_patient['time_step'] = np.arange(len(dyn_patient))
        dyn_patient['label'] = y_val
        chunk_rows.append(dyn_patient)
        
        # Write in chunks to avoid accumulating too many rows
        if len(chunk_rows) >= cohort_chunk_size:
            chunk_df = pd.concat(chunk_rows, ignore_index=True)
            chunk_df.to_csv(output_path, mode='a', header=not wrote_any_rows, index=False)
            wrote_any_rows = True
            chunk_rows = []
            gc.collect()

    # Write remaining rows
    if len(chunk_rows) > 0:
        chunk_df = pd.concat(chunk_rows, ignore_index=True)
        chunk_df.to_csv(output_path, mode='a', header=not wrote_any_rows, index=False)
        wrote_any_rows = True

    if not wrote_any_rows:
        pd.DataFrame().to_csv(output_path, index=False)

    print('Saved temporal dataset to:', output_path)
    del patient_data
    gc.collect()


def impute():

    temporal = pd.read_csv('/data_processed_eicu/temporal_resampled_renamed_24hrs.csv')
    temporal = temporal.sort_values(['patientunitstayid', 'time_step']).reset_index(drop=True)
    hosp_time = pd.read_csv('/data_processed_eicu/demo_renamed.csv')
    df = pd.merge(temporal, hosp_time, on='patientunitstayid', how='left')  
    y = df['label']
    id = df['patientunitstayid']
    time_step = df['time_step']
    X = df.drop(columns=['patientunitstayid', 'time_step', 'label'])
    
    mimic_train_df = pd.read_csv('/data_processed/train_merged.csv', usecols=X.columns.tolist())
    # IMPORTANT: keep a deterministic feature order for sklearn name checks.
    # pandas `usecols=[...]` preserves file order, not necessarily list order.
    # Reindex both frames to the exact same ordered columns before fit/transform.
    mimic_train_df = mimic_train_df.reindex(columns=X.columns.tolist())

    # IterativeImputer / MICE 
    imputer = IterativeImputer(
    random_state=123,
    max_iter=15,
    sample_posterior=False,
    skip_complete=True)

    imputer.fit(mimic_train_df)
    fit_cols = imputer.feature_names_in_.tolist()
    X = X.reindex(columns=fit_cols)

    mimic_train_df = pd.DataFrame(imputer.transform(mimic_train_df), columns=fit_cols)
    eicu_df = pd.DataFrame(imputer.transform(X), columns=fit_cols)
    eicu_df = eicu_df.reset_index(drop=True)
    print(eicu_df.isnull().sum())
    
    eicu_df = pd.concat([id.reset_index(drop=True), time_step.reset_index(drop=True), eicu_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    eicu_df.to_csv('/data_processed_eicu/df_imputed.csv', index=False)


def flatten_imputed_dataset_24hrs():

    df = pd.read_csv('/data_processed_eicu/df_imputed.csv')
    df = df.sort_values(['patientunitstayid', 'time_step']).reset_index(drop=True)

    # Identify temporal features
    feature_cols = [c for c in df.columns if c not in ['patientunitstayid', 'time_step', 'hosp_to_icu', 'label']]
    print(feature_cols)
    flattened_rows = []

    for stay_id, group in df.groupby('patientunitstayid', sort=False):
        group = group.sort_values('time_step').reset_index(drop=True)

        # Keep only complete 24-hour sequences (0..23)
        if len(group) != 24:
            print('NOT 24 HOURS, SKIPPING STAY_ID:', stay_id)
            continue

        row_dict = {
            'patientunitstayid': stay_id,
            'label': int(group['label'].iloc[0])
        }

        for hour in range(24):
            hour_row = group[group['time_step'] == hour].iloc[0]
            for col in feature_cols:
                row_dict[f'{col}_{hour}'] = hour_row[col]

        flattened_rows.append(row_dict)
    flattened_df = pd.DataFrame(flattened_rows)

    # Merge with static feature hosp_to_icu (one value per patient)
    hosp_time = df[['patientunitstayid', 'hosp_to_icu']].drop_duplicates(subset=['patientunitstayid'])
    flattened_df = pd.merge(flattened_df, hosp_time, on='patientunitstayid', how='left')
    flattened_df['label'] = flattened_df.pop('label')  # Move label to the end
    flattened_df.to_csv(f'/data_processed_eicu/df_imputed_flattened.csv', index=False)
    print(f'flattened shape: {flattened_df.shape}')


def create_aggregates(): 
    df = pd.read_csv('/data_processed_eicu/df_imputed_flattened.csv')  

    temporal_cols = [c for c in df.columns if c not in ['patientunitstayid', 'label', 'hosp_to_icu']]
    agg_rows = []

    for stay_id, row in df.groupby('patientunitstayid', sort=False):
        agg_row = {'patientunitstayid': stay_id, 'hosp_to_icu': int(row['hosp_to_icu'].iloc[0]), 'label': int(row['label'].iloc[0])}
        
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

    agg_df = pd.DataFrame(agg_rows)
    agg_df['label'] = agg_df.pop('label')  # Move label to the end
    agg_df.to_csv(f'/data_processed_eicu/df_imputed_flattened_agg.csv', index=False)
    print(f'flattened shape: {agg_df.shape}')

def train_test_split():
    df = pd.read_csv('/data_processed_eicu/df_imputed_flattened_agg.csv')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    stays = df['patientunitstayid'].values
    stay_labels = df['label'].values
    train_idx, test_idx = next(sss.split(stays.reshape(-1, 1), stay_labels))
    train_stays = stays[train_idx]
    test_stays = stays[test_idx]
    train_df = df[df['patientunitstayid'].isin(train_stays)].reset_index(drop=True)
    test_df = df[df['patientunitstayid'].isin(test_stays)].reset_index(drop=True)
    train_df.to_csv('/data_processed_eicu/train_df.csv', index=False)
    test_df.to_csv('/data_processed_eicu/test_df.csv', index=False)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    stays = train_df['patientunitstayid'].values
    stay_labels = train_df['label'].values
    train_idx, val_idx = next(sss.split(stays.reshape(-1, 1), stay_labels))
    train_stays = stays[train_idx]
    val_stays = stays[val_idx]
    train_split_df = train_df[train_df['patientunitstayid'].isin(train_stays)].reset_index(drop=True)
    val_df = train_df[train_df['patientunitstayid'].isin(val_stays)].reset_index(drop=True)
    train_split_df.to_csv('/data_processed_eicu/train_df.csv', index=False)
    val_df.to_csv('/data_processed_eicu/val_df.csv', index=False)

def normalize():
    train_df = pd.read_csv('/data_processed_eicu/train_df.csv')
    val_df = pd.read_csv('/data_processed_eicu/val_df.csv')
    test_df = pd.read_csv('/data_processed_eicu/test_df.csv')
    
    scaler = MinMaxScaler()
    df_train_norm = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    df_val_norm = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)
    df_test_norm = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    print(df_train_norm.describe())
    df_train_norm.to_csv('/data_processed_eicu/train_df_norm.csv', index=False)
    df_val_norm.to_csv('/data_processed_eicu/val_df_norm.csv', index=False)
    df_test_norm.to_csv('/data_processed_eicu/test_df_norm.csv', index=False)


def preprocess_eICU():
    detect_outliers_chart()
    create_temporal_chart()
    rename()
    handle_temporal()
    impute()
    flatten_imputed_dataset_24hrs()
    create_aggregates()
    train_test_split()
    normalize()
