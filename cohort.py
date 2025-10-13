import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import math
import random


def create_cohort():
    # Get subject_id, stay_id, hadm_id, intime, outtime of ICU patients
    cohort = pd.read_csv('data_raw/icustays.csv')[['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime']]

    # Work on patients with Sepsis-3 obtained form MIMIC_derived
    sepsis3 = pd.read_csv('data_processed/sepsis3.csv')
    sepsis3 = pd.merge(cohort, sepsis3, on=['subject_id', 'stay_id'])

    # Between suspected_infection_time and sofa_time, keep the earliest as sepsis_onset (SEPSIS-3 DEFINITION)
    sepsis3['sepsis_onset'] = pd.Series()
    for i in range(len(sepsis3)):

        sepsis3['suspected_infection_time'][i] = datetime.strptime(sepsis3['suspected_infection_time'][i], '%Y-%m-%d %H:%M:%S')
        sepsis3['sofa_time'][i] = datetime.strptime(sepsis3['sofa_time'][i], '%Y-%m-%d %H:%M:%S')

        if ((sepsis3['suspected_infection_time'][i] - sepsis3['sofa_time'][i] <= timedelta(hours=24)) or
                (sepsis3['sofa_time'][i] - sepsis3['suspected_infection_time'][i] <= timedelta(hours=12))):

            if sepsis3['sofa_time'][i] < sepsis3['suspected_infection_time'][i]:
                sepsis3['sepsis_onset'][i] = sepsis3['sofa_time'][i]
            else:
                sepsis3['sepsis_onset'][i] = sepsis3['suspected_infection_time'][i]
    sepsis3 = sepsis3.dropna().reset_index()  # 32970

    # Convert timestamps to datetime objects
    sepsis3['sofa_time'] = pd.to_datetime(sepsis3['sofa_time'])
    sepsis3['suspected_infection_time'] = pd.to_datetime(sepsis3['suspected_infection_time'])
    sepsis3['intime'] = pd.to_datetime(sepsis3['intime'])
    sepsis3['sepsis_onset'] = pd.to_datetime(sepsis3['sepsis_onset'])

    # Find time of sepsis after ICU admission and sort patients according to that
    # sepsis_onset - intime = time_after_adm
    sepsis3['time_after_adm'] = pd.Series()
    for i in range(len(sepsis3)):
        sepsis3['time_after_adm'][i] = sepsis3['sepsis_onset'][i] - sepsis3['intime'][i]
    sepsis3 = sepsis3.sort_values(by='time_after_adm')
    sepsis3 = sepsis3.reset_index(drop=True)

    # Convert time after ICU admission to hours, keep positive hours
    # sepsis3['hours_after_adm']=pd.to_timedelta(sepsis3['time_after_adm'],unit='h')
    sepsis3['hours_after_adm'] = pd.Series()
    for i in range(len(sepsis3)):
        sepsis3['hours_after_adm'][i] = math.ceil(sepsis3['time_after_adm'][i].total_seconds()/3600)  # 32970
    sepsis3 = sepsis3[sepsis3['hours_after_adm']>0]  # 14623
    # print(sepsis3[['sepsis_onset', 'intime', 'hours_after_adm']])

    # Add label column
    sepsis3['label'] = 1

    # Include just adults
    age = pd.read_csv("data_raw/patients.csv")[['subject_id', 'anchor_year', 'anchor_age', 'anchor_year_group', 'dod', 'gender']]
    age['yob'] = age['anchor_year'] - age['anchor_age']
    sepsis3 = pd.merge(sepsis3, age[['subject_id', 'anchor_year', 'anchor_age', 'yob', 'gender']], on='subject_id', how='inner') #excluded 'dod'
    sepsis3['age'] = sepsis3['intime'].dt.year - sepsis3['yob']  # 14623
    sepsis3 = sepsis3[sepsis3['age'] >= 18]  # 14623

    # Add demographics
    eth = pd.read_csv("data_raw/admissions.csv")[['hadm_id', 'insurance', 'race']]
    sepsis3 = pd.merge(sepsis3, eth, how='inner', on='hadm_id')
    sepsis3 = sepsis3[['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'age', 'gender', 'race', 'insurance', 'hours_after_adm', 'label']]  # 14623

    # Add CONTROLS (use patients that didnt experience sepsis throughout ICU stay)
    cohort = pd.merge(age, cohort, on=['subject_id'], how='inner')  # 'how': refers to rows of column of 'on', all features of both dfs
    cohort['intime'] = pd.to_datetime(cohort['intime'])
    cohort['outtime'] = pd.to_datetime(cohort['outtime'])
    cohort['age'] = cohort['intime'].dt.year - cohort['yob']
    cohort = cohort.loc[cohort['age'] >= 18]  # 73181

    cohort = pd.merge(cohort, eth, how='inner', on='hadm_id')
    cohort = cohort[['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'age', 'gender', 'race', 'insurance']]  # 73181

    sepsis3['intime'] = pd.to_datetime(sepsis3['intime'])
    sepsis3['outtime'] = pd.to_datetime(sepsis3['outtime'])
    cohort = pd.merge(cohort, sepsis3, how='outer', on=['subject_id','hadm_id', 'intime', 'outtime', 'stay_id', 'age', 'gender', 'race', 'insurance'])  # 73181

    # print(cohort['hours_after_adm'].value_counts())
    # print(cohort.info())

    # control: random hours after adm for sepsis onset
    # for i in range(len(cohort)):
    #     if np.isnan(cohort['hours_after_adm'][i]):
    #         cohort['hours_after_adm'][i] = random.randint(1, 7)
    # cohort['hours_after_adm'] = cohort['hours_after_adm'].astype('int')
    # print(cohort['hours_after_adm'].value_counts())
    # print(cohort.info())

    cohort['label'] = cohort['label'].fillna('0')
    cohort['label'] = cohort['label'].astype('int')

    # LOS
    cohort['los'] = cohort['outtime']-cohort['intime']
    for i in range(len(cohort)):
        cohort['los'][i] = cohort['los'][i].total_seconds()/3600
    cohort = cohort[cohort['los']>0] #73181

    # hosp admission to ICU admission time
    # hosp_to_icu = pd.Series()
    hosp_adm_time = pd.read_csv("data_raw/admissions.csv")[['admittime', 'hadm_id']]
    cohort = cohort.merge(hosp_adm_time, how='inner', on='hadm_id')
    cohort['admittime'] = pd.to_datetime(cohort['admittime'])
    cohort['intime'] = pd.to_datetime(cohort['intime'])
    cohort['hosp_to_icu'] = cohort['intime'] - cohort['admittime']  # icu admission time - hospital admission time
    for i in range(len(cohort)):
        cohort['hosp_to_icu'][i] = cohort['hosp_to_icu'][i].total_seconds() / 3600

    # print(cohort.info())
    cohort.to_csv('data_processed/sepsis3_processed.csv', index=False)
    cohort.to_csv('data_processed/sepsis3_processed.csv.gz', compression='gzip', index=False)
    print('COHORT DATA CREATED')
