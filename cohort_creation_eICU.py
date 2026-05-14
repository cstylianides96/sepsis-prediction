import pandas as pd
import numpy as np

def create_cohort_eICU():
    # Seleect patietns with sepsis, infection, SIRS and dysfunction (sepsis-3 criteria)
    diag = pd.read_csv("data_raw_eicu_v2.0/diagnosis.csv.gz", compression='gzip')
    print(diag.head())
    pattern = r'(?i)(?=.*\b(sepsis|infection|SIRS)\b)(?=.*\bdysfunction\b)'
    mask = diag['diagnosisstring'].astype(str).str.contains(pattern, regex=True, na=False)
    diag_filtered = diag.loc[mask].copy()
    print(diag_filtered) #24220

    # Remove diagnoses that occur before or at the time of ICU admission (diagnosisoffset <= 0)
    diag_filtered = diag_filtered[diag_filtered['diagnosisoffset'] > 0]
    print(diag_filtered) #24138

    # Keep only the first sepsis-3 diagnosis for each patient
    diag_filtered = diag_filtered.sort_values(['patientunitstayid', 'diagnosisoffset'])
    cases = diag_filtered.groupby('patientunitstayid', as_index=False).first()
    print(cases) # 3822 unique patients

    cases.to_csv("data_processed_eicu/sepsis3_eicu.csv", index=False)

    # sepsis diagnoses at least 36 hours (24+12 hours) after ICU admission
    cases = cases.loc[cases['diagnosisoffset']>=2160].reset_index(drop=True) # minutes
    cases['label']=1
    cases = cases[['patientunitstayid', 'diagnosisoffset', 'label']]
    print(cases) #360

    # Select controls without sepsis, infection, or SIRS 
    patients = pd.read_csv("data_raw_eicu_v2.0/patient.csv.gz", compression='gzip')
    controls = patients[~patients['patientunitstayid'].isin(cases['patientunitstayid'])].reset_index(drop=True)
    controls = controls[controls['unitdischargeoffset']>=1440] # at least 24 hours in ICU
    print(controls['patientunitstayid'].nunique()) # 132541

    diag = pd.read_csv("data_raw_eicu_v2.0/diagnosis.csv.gz", compression='gzip')
    diag_controls = diag [diag['patientunitstayid'].isin(controls['patientunitstayid'])]
    pattern = r'(?i)(?=.*\b(sepsis|infection|SIRS|septic)\b)'
    mask = diag_controls['diagnosisstring'].astype(str).str.contains(pattern, regex=True, na=False)
    diag_controls = diag_controls.loc[~mask].copy()
    diag_controls = diag_controls.groupby('patientunitstayid', as_index=False).first()
    controls = diag_controls['patientunitstayid'].reset_index(drop=True).to_frame()
    controls['label']=0
    print(controls)

    cohort = pd.merge(cases, controls, on=['patientunitstayid', 'label'], how='outer')
    print(cohort)

    # Exclude patients under 18 years old
    age_num = pd.to_numeric(patients["age"], errors="coerce")
    age_is_string = patients["age"].apply(lambda x: isinstance(x, str))
    cohort = cohort.merge(patients[['patientunitstayid', 'age']], on='patientunitstayid', how='left')
    cohort = cohort[(age_num > 18) | age_is_string]
    print(cohort['patientunitstayid'].nunique())

    print(cohort.label.value_counts(normalize=True))
    cohort.to_csv("data_processed_eicu/sepsis3_eicu.csv", index=False)

    # #select controls such that cases to controls 2:100
    # num_cases = cases.shape[0]
    # print(num_cases) #360
    # num_controls = int(num_cases * 100 / 2)
    # print(num_controls) #18000
    # controls = controls.sample(n=num_controls, random_state=42)
    # cohort = pd.concat([cases, controls], ignore_index=True)
    # cohort.to_csv("data_processed_eicu/sepsis3_eicu.csv", index=False)