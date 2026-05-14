# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import math
import random


cohort = pd.read_csv("/data_processed_eicu/sepsis3_eicu.csv")
print(cohort.label.value_counts(normalize=True))
cases_ids = cohort[cohort['label'] == 1]['patientunitstayid'].tolist()
controls_ids = cohort[cohort['label'] == 0]['patientunitstayid'].tolist()
patients = pd.read_csv("/data_raw_eicu_v2.0/patient.csv.gz", compression='gzip')
patientunitstayid = patients[patients['patientunitstayid'].isin(cohort['patientunitstayid'])]['patientunitstayid'].tolist()
print(len(patientunitstayid))
discharge_df = (
    patients[patients['patientunitstayid'].isin(cohort['patientunitstayid'])]
    [['patientunitstayid', 'unitdischargeoffset']]
    .drop_duplicates('patientunitstayid'))
print(len(discharge_df))


def extract_lab():
    lab = pd.read_csv("/data_raw_eicu_v2.0/lab.csv.gz", compression='gzip')
    # Ensure values after icu adm, before discharge (vectorized)
    lab_filtered = lab.merge(discharge_df, on='patientunitstayid', how='inner')
    lab_filtered = lab_filtered[
        (lab_filtered['labresultoffset'] >= 0) &
        (lab_filtered['labresultoffset'] < lab_filtered['unitdischargeoffset'])
    ]



    lab_filtered = lab_filtered[lab_filtered['labname'].isin(['glucose', 'bedside glucose','WBC x 1000', 'BUN', 
                                                              'Hct', 'LDH', 'sodium', 'fibrinogen', 'magnesium', '-lymphs',
                                                              'creatinine',  'alkaline phos.', 'CRP', 'phosphate'])]
    print(lab_filtered)
    print(len(lab_filtered.patientunitstayid.unique()))
    lab_filtered.to_csv("/data_processed_eicu/lab_filtered.csv", index=False)
    # lab = lab.groupby(['patientunitstayid', 'labname'], as_index=False)


def extract_nurseCharting():
    nurse = pd.read_csv("/data_raw_eicu_v2.0/nurseCharting.csv.gz", compression='gzip')
    # Ensure values after icu adm, before discharge (vectorized)
    nurse_filtered = nurse.merge(discharge_df, on='patientunitstayid', how='inner')
    nurse_filtered = nurse_filtered[
        (nurse_filtered['nursingchartoffset'] >= 0) &
        (nurse_filtered['nursingchartoffset'] < nurse_filtered['unitdischargeoffset'])
    ]

    nurse1 = nurse_filtered[nurse_filtered['nursingchartcelltypevalname'].isin(['Non-Invasive BP Diastolic','Non-Invasive BP Systolic',
                                                                                'Non-Invasive BP Mean', 'Respiratory Rate', 'Heart Rate'])]
    print(nurse1)
    nurse2 = nurse_filtered[
        (nurse_filtered['nursingchartcelltypevalname'] == 'Value')
        & (nurse_filtered['nursingchartcelltypevallabel'].isin(['Motor Response', 'Verbal Response']))
    ]
    print(nurse2)
    nurse = pd.concat([nurse1, nurse2], axis=0)
    print(nurse)
    print(len(nurse.patientunitstayid.unique()))
    print(nurse.nursingchartcelltypevalname.value_counts()) 
    nurse.to_csv("/data_processed_eicu/nurse_filtered.csv", index=False)


def extract_hosp_time():
    patient = pd.read_csv("/data_raw_eicu_v2.0/patient.csv.gz", compression='gzip')
    patient = patient[patient['patientunitstayid'].isin(cohort['patientunitstayid'])]
    hosp_time = patient[['patientunitstayid', 'hospitaladmitoffset']]
    hosp_time['hospitaladmitoffset'] = -hosp_time['hospitaladmitoffset']
    hosp_time = hosp_time[hosp_time['hospitaladmitoffset'] >= 0]
    hosp_time['hospitaladmitoffset'] = hosp_time['hospitaladmitoffset'] / 60  # convert to hours
    hosp_time.to_csv("/data_processed_eicu/hosp_time.csv", index=False)
    print(patient)


def extract_data_eICU():
    extract_lab()
    extract_nurseCharting()
    extract_hosp_time()
