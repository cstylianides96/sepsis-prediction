
import importlib
import utils.icu_preprocess_util
importlib.reload(utils.icu_preprocess_util)
from utils.icu_preprocess_util import *# module of preprocessing functions

import utils.outlier_removal
importlib.reload(utils.outlier_removal)
from utils.outlier_removal import *

import utils.uom_conversion
importlib.reload(utils.uom_conversion)
from utils.uom_conversion import *


def extract_features():
    diag = pd.read_csv('data_raw/diagnoses_icd.csv')
    diag.to_csv('data_raw/diagnoses_icd.csv.gz', index=False, compression='gzip')

    diag = preproc_icd_module("data_raw/diagnoses_icd.csv.gz", 'data_processed/sepsis3_processed.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
    diag[['subject_id', 'hadm_id', 'stay_id', 'icd_code', 'root_icd10_convert', 'root']].to_csv("data_processed/preproc_diag_icu_sepsis.csv.gz", compression='gzip', index=False)

    out = preproc_out("data_raw/outputevents.csv.gz", 'data_processed/sepsis3_processed.csv.gz', 'charttime', dtypes=None, usecols=None)
    out[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'intime', 'event_time_from_admit']].to_csv("data_processed/preproc_out_icu_sepsis.csv.gz", compression='gzip', index=False)

    chart=preproc_chart("data_raw/chartevents.csv.gz", 'data_processed/sepsis3_processed.csv.gz', 'charttime', dtypes=None, usecols=['stay_id', 'charttime', 'itemid', 'valuenum', 'valueuom'])
    print(chart.loc[chart['itemid'].isin([220739, 223900, 223901])])
    chart = drop_wrong_uom(chart, 0.95)
    print(chart.loc[chart['itemid'].isin([220739, 223900, 223901])])
    chart[['stay_id', 'itemid', 'event_time_from_admit', 'valuenum']].to_csv("data_processed/preproc_chart_icu_sepsis.csv.gz", compression='gzip', index=False)

    proc = preproc_proc("data_raw/procedureevents.csv.gz", 'data_processed/sepsis3_processed.csv.gz', 'starttime', dtypes=None, usecols=['stay_id', 'starttime', 'itemid'])
    proc[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'intime', 'event_time_from_admit']].to_csv("data_processed/preproc_proc_icu_sepsis.csv.gz", compression='gzip', index=False)

    med = preproc_meds("data_raw/inputevents.csv.gz", 'data_processed/sepsis3_processed.csv.gz')
    med[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime','endtime', 'start_hours_from_admit', 'stop_hours_from_admit','rate', 'amount', 'orderid']].to_csv('data_processed/preproc_med_icu_sepsis.csv.gz', compression='gzip', index=False)

    print('FEATURES EXTRACTED')


def preprocess_diag_chart():

    # processing diagnosis data_processed
    diag = pd.read_csv("data_processed/preproc_diag_icu_sepsis.csv.gz", compression='gzip', header=0)
    diag['new_icd_code']=diag['root_icd10_convert']

    diag = diag[['subject_id', 'hadm_id', 'stay_id', 'new_icd_code']].dropna()
    print("Total number of rows",diag.shape[0])
    diag.to_csv("data_processed/preproc_diag_icu_sepsis.csv.gz", compression='gzip', index=False)

    # processing chart events data_processed
    chart = pd.read_csv("data_processed/preproc_chart_icu_sepsis.csv.gz", compression='gzip', header=0)
    chart = outlier_imputation(chart, 'itemid', 'valuenum', 98, left_thresh=2, impute=True)
    print("Total number of rows", chart.shape[0])
    chart.to_csv("data_processed/preproc_chart_icu_sepsis.csv.gz", compression='gzip', index=False)

    print('DIAGNOSIS AND CHART DATA PREPROCESSED')


def generate_summary():  # might remove

    #diagnoses
    diag = pd.read_csv("data_processed/preproc_diag_icu_sepsis.csv.gz", compression='gzip', header=0)
    freq = diag.groupby(['stay_id','new_icd_code']).size().reset_index(name="mean_frequency")
    freq = freq.groupby(['new_icd_code'])['mean_frequency'].mean().reset_index()
    total = diag.groupby('new_icd_code').size().reset_index(name="total_count")
    summary = pd.merge(freq, total, on='new_icd_code', how='right')
    summary = summary.fillna(0)
    summary.to_csv('data_processed/diag_summary_sepsis.csv', index=False)
    summary['new_icd_code'].to_csv('data_processed/diag_features_sepsis.csv', index=False)

    # medication
    med = pd.read_csv("data_processed/preproc_med_icu_sepsis.csv.gz", compression='gzip', header=0)
    freq = med.groupby(['stay_id', 'itemid']).size().reset_index(name="mean_frequency")
    freq = freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()

    missing = med[med['amount'] == 0].groupby('itemid').size().reset_index(name="missing_count")
    total = med.groupby('itemid').size().reset_index(name="total_count")
    summary = pd.merge(missing, total, on='itemid', how='right')
    summary = pd.merge(freq, summary, on='itemid', how='right')
    summary['missing_perc'] = 100*(summary['missing_count']/summary['total_count'])
    summary = summary.fillna(0)

    summary.to_csv('data_processed/med_summary_sepsis.csv', index=False)
    summary['itemid'].to_csv('data_processed/med_features_sepsis.csv', index=False)

    # procedures
    proc = pd.read_csv("data_processed/preproc_proc_icu_sepsis.csv.gz", compression='gzip', header=0)
    freq = proc.groupby(['stay_id', 'itemid']).size().reset_index(name="mean_frequency")
    freq = freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
    total = proc.groupby('itemid').size().reset_index(name="total_count")
    summary = pd.merge(freq, total, on='itemid', how='right')
    summary = summary.fillna(0)
    summary.to_csv('data_processed/proc_summary_sepsis.csv', index=False)
    summary['itemid'].to_csv('data_processed/proc_features_sepsis.csv', index=False)

    # output
    out = pd.read_csv("data_processed/preproc_out_icu_sepsis.csv.gz", compression='gzip', header=0)
    freq = out.groupby(['stay_id', 'itemid']).size().reset_index(name="mean_frequency")
    freq = freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
    total = out.groupby('itemid').size().reset_index(name="total_count")
    summary = pd.merge(freq, total, on='itemid', how='right')
    summary = summary.fillna(0)
    summary.to_csv('data_processed/out_summary_sepsis.csv', index=False)
    summary['itemid'].to_csv('data_processed/out_features_sepsis.csv', index=False)
        
    # chart
    chart = pd.read_csv("data_processed/preproc_chart_icu_sepsis.csv.gz", compression='gzip', header=0)
    freq = chart.groupby(['stay_id', 'itemid']).size().reset_index(name="mean_frequency")
    freq = freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()

    missing = chart[chart['valuenum'] == 0].groupby('itemid').size().reset_index(name="missing_count")
    total = chart.groupby('itemid').size().reset_index(name="total_count")
    summary = pd.merge(missing, total, on='itemid', how='right')
    summary = pd.merge(freq, summary, on='itemid', how='right')
    summary['missing_perc'] = 100*(summary['missing_count']/summary['total_count'])

#         final.groupby('itemid')['missing_count'].sum().reset_index()
#         final.groupby('itemid')['total_count'].sum().reset_index()
#         final.groupby('itemid')['missing%'].mean().reset_index()
    summary = summary.fillna(0)

    summary.to_csv('data_processed/chart_summary_sepsis.csv', index=False)
    summary['itemid'].to_csv('data_processed/chart_features_sepsis.csv', index=False)

    print('SUMMARY AND UNIQUE FEATURE FILES GENERATED')


def preproc_raw_feat():
    extract_features()
    preprocess_diag_chart()
    generate_summary()
