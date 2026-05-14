# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd

#chart, diagnoses, demo

cohort = pd.read_csv('data_processed/sepsis3_processed.csv')  #PATHS!
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
itemids = pd.read_csv('data_raw/d_items.csv')


def extract_chartevents():
    # Chartevents of sepsis patients
    chartevents = pd.read_csv('data_raw/chartevents.csv.gz', compression='gzip', usecols=['stay_id', 'charttime', 'itemid', 'valuenum'])
    print('READ')
    chartevents = chartevents[chartevents['stay_id'].isin(stayids)]

    # Chart items
    #clinicians, scoring systems, (literature)
    vital_items = ['Arterial O2 pressure', 'Arterial Blood Pressure mean', 'GCS - Motor Response', 
                   'GCS - Verbal Response', 'GCS - Eye Opening', 'Temperature Celsius', 'Respiratory Rate', 
                   'Heart Rate', 'Arterial Blood Pressure systolic', 'Arterial Blood Pressure diastolic','Arterial O2 Saturation', 
                   'Total PEEP Level', 'EtCO2', 'FiO2 (CH)', 'PH (Arterial)']
    lab_items = ['LDH', 'BUN', 'Hematocrit (serum)', 'Platelets', 'Sodium (serum)', 'Potassium (serum)', 
                 'Calcium non-ionized', 'Phosphorous', 'Magnesium','C Reactive Protein (CRP)', 'WBC', 
                 'Differential-Neuts','Differential-Lymphs', 'Differential-Monos', 'Differential-Basos', 
                 'Fibrinogen', 'INR','Albumin', 'Alkaline Phosphate', 'Total Bilirubin', 'CK (CPK)', 
                 'Creatinine (serum)', 'Glucose (serum)']

    chart_items = vital_items+lab_items
    chart_items = itemids[itemids['label'].isin(chart_items)]['itemid'].tolist()

    chartevents = chartevents[chartevents['itemid'].isin(chart_items)]
    chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])


    # Ensure charttime window is clinically valid.
    # Cases: use data before sepsis onset.
    # Controls: use data until discharge (no sepsis onset boundary).
    sepsis_chart = []
    for stayid, y, sepsistime, admtime, distime in zip(stayids, labels, sepsistimes, admtimes, distimes):
        if y == 1 and pd.notna(sepsistime):
            chart_end = min(sepsistime, distime)
        else:
            chart_end = distime

        chart = chartevents[
            (chartevents['stay_id'] == stayid) &
            (chartevents['charttime'] < chart_end) &
            (chartevents['charttime'] > admtime) &
            (chartevents['charttime'] < distime)]
        sepsis_chart.append(chart)
    sepsis_chart = pd.concat(sepsis_chart, ignore_index=True)
    print(sepsis_chart)
    print(f"Number of unique stay_ids: {sepsis_chart['stay_id'].nunique()}")

    coverage = cohort[['stay_id', 'label']].drop_duplicates()
    coverage = coverage[coverage['stay_id'].isin(sepsis_chart['stay_id'].unique())]
    print('Chart coverage by label:', coverage['label'].value_counts(dropna=False).to_dict())

    sepsis_chart.to_csv('data_processed/sepsis_chartevents.csv', index=False)


def extract_diagnoses():
    diagnoses = pd.read_csv('data_raw/diagnoses_icd.csv')
    diagnoses = diagnoses[diagnoses['hadm_id'].isin(hospids)]

    # ICD codes are already stored without dots; normalize to uppercase strings
    diagnoses['icd_code_normalized'] = diagnoses['icd_code'].astype(str).str.upper()

    def code_in_range(code: str, ranges):
        """Check if ICD-9 code (numeric, no dots) falls within any specified ranges."""
        try:
            if str(code).isdigit():
                code_num = int(code)
                for start, end in ranges:
                    if start <= code_num <= end:
                        return True
        except Exception:
            pass
        return False

    def matches_icd10_prefix(code: str, prefixes):
        """Check if ICD-10 code starts with any of the specified prefixes."""
        if pd.isna(code):
            return False
        code_str = str(code).upper()
        return any(code_str.startswith(prefix) for prefix in prefixes)

    # ICD-9 Diagnosis Ranges (codes are stored without dots in diagnoses_icd.csv)
    # Diagnoses: [Respiratory disease, Hypertension, COPD, Heart Disease,
    #             Diabetes mellitus type II, Malignancy, Stroke, Immunosuppression]
    icd9_ranges = [
        (51881, 51884), #Respiratory failure 
        (490, 496), (515, 516), (517, 519), #Lung disease (chronic)
        (584, 585), #Renal failure
        (1, 139), #Infection
        (401, 405), (490, 496), #Chronic disease (common)
        (0, 86) #Surgery (procedure ranges)
    ]

    # ICD-9 exact codes (string-based, no dots)
    icd9_exact_codes = {
        '7991',  # Respiratory failure
        '39891', '428',  # Heart failure
        '5712', '5715', '5716',  # Cirrhosis
        '3995', '5498',  # Dialysis procedures
        '586',  # Renal failure
        '38', '41', '486',  # Infection
        '570', '5722', '78550', '78552',  # Organ dysfunction
        '458',  # Hypotension
        '99591', '99592',  # Sepsis
        '250', '414', '585'  # Chronic disease (common)
    }

    # ICD-9 V-codes (string-based, no dots)
    icd9_v_codes = {'V4511', 'V56'}

    # ICD-10 Diagnosis Prefixes (prefix matching captures all subcategories)
    icd10_prefixes = [
      'J96', 'R092', #Respiratory failure
      'I50', #Heart failure
      'K74', 'K703', 'K704', 'K717', #Cirrhosis
      'J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J84', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67', 'J68', 'J69', 'J70', 'J80', 'J81', #Lung disease (chronic)
      'Z992', 'Z49', '5A1D', '3E1M39Z', #Dialysis
      'N17', 'N18', 'N19', #Renal failure
      'A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 
      'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41', 
      'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 'A60', 'A61', 'A62', 
      'A63', 'A64', 'A65', 'A66', 'A67', 'A68', 'A69', 'A70', 'A71', 'A72', 'A73', 'A74', 'A75', 'A76', 'A77', 'A78', 'A79', 'A80', 'A81', 'A82', 'A83', 
      'A84', 'A85', 'A86', 'A87', 'A88', 'A89', 'A90', 'A91', 'A92', 'A93', 'A94', 'A95', 'A96', 'A97', 'A98', 'A99', #Infection
      'B00', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
      'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40', 'B41', 
      'B42', 'B43', 'B44', 'B45', 'B46', 'B47', 'B48', 'B49', 'B50', 'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57', 'B58', 'B59', 'B60', 'B61', 'B62', 
      'B63', 'B64', 'B65', 'B66', 'B67', 'B68', 'B69', 'B70', 'B71', 'B72', 'B73', 'B74', 'B75', 'B76', 'B77', 'B78', 'B79', 'B80', 'B81', 'B82', 'B83', 
      'B84', 'B85', 'B86', 'B87', 'B88', 'B89', 'B90', 'B91', 'B92', 'B93', 'B94', 'B95', 'B96', 'B97', 'B98', 'B99', #Infection
      'A40', 'A41', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', #Infection
      'K72', 'J96', 'N17', 'I50', 'R57', 'R652', #Organ dysfunction
      'I95', #Hypotension
      'A40', 'A41', 'R652', 'R572', #Sepsis
      'E10', 'E11', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I25', 'I50', 'N18', 'J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', #Chronic disease (common)
      'Z48', 'Z98' #Surgery
    ]

    # Filter diagnoses based on ICD-9 ranges/V-codes and ICD-10 prefixes
    def is_relevant_diagnosis(code: str) -> bool:
        if matches_icd10_prefix(code, icd10_prefixes):
            return True
        code_str = str(code).upper()
        if code_str in icd9_exact_codes:
            return True
        if code_str.startswith('V') and code_str in icd9_v_codes:
            return True
        if code_in_range(code, icd9_ranges):
            return True
        return False

    diagnoses['is_relevant'] = diagnoses['icd_code_normalized'].apply(is_relevant_diagnosis)
    filtered_diags = diagnoses[diagnoses['is_relevant']]
    print(filtered_diags.icd_version.value_counts())
    filtered_diags.to_csv('data_processed/sepsis_diagnoses.csv', index=False)


def extract_demographics():

    # Age calculation
    age = pd.read_csv("data_raw/patients.csv")[['subject_id', 'anchor_year', 'anchor_age', 'anchor_year_group', 'gender']]
    age['yob'] = age['anchor_year'] - age['anchor_age']
    age = age.merge(cohort[['subject_id', 'intime']], how='right', on='subject_id')
    age ['intime'] = pd.to_datetime(age['intime'])
    age['age'] = age['intime'].dt.year - age['yob']
    age = age[age['age'] >= 18].reset_index(drop=True)

    # Race
    eth = pd.read_csv("data_raw/admissions.csv")[['subject_id', 'race']]
    eth = eth.drop_duplicates(subset=['subject_id']) 
    eth = age.merge(eth, how='left', on='subject_id')

    # Extract height and weight from chartevents
    chartevents = pd.read_csv('data_raw/chartevents.csv.gz', compression='gzip', usecols=['subject_id', 'stay_id','itemid', 'valuenum'])
    chartevents = chartevents[chartevents['stay_id'].isin(stayids)]
    height_weight = ['Height (cm)', 'Admission Weight (Kg)'] 
    height_weight = itemids[itemids['label'].isin(height_weight)]['itemid'].tolist()
    chartevents = chartevents[chartevents['itemid'].isin(height_weight)]
    chartevents = chartevents.pivot_table(index=['subject_id', 'stay_id'], columns='itemid', values='valuenum', aggfunc='first')
    chartevents = chartevents.reset_index()

    # Build demo from cohort to avoid many-to-many duplication
    demo = cohort[['subject_id', 'stay_id']].merge(
        eth[['subject_id', 'age', 'gender', 'race']], how='left', on='subject_id')
    demo = demo.merge(chartevents, how='left', on=['subject_id', 'stay_id'])
    demo = demo[['subject_id', 'stay_id', 'age', 'gender', 'race', height_weight[0], height_weight[1]]]
    print(f"cohort stay_ids: {cohort['stay_id'].nunique()}, demo rows: {len(demo)}, demo unique stay_ids: {demo['stay_id'].nunique()}")
    print(demo.isnull().sum())
    demo = demo.drop_duplicates(subset=['stay_id'])
    demo.to_csv('data_processed/sepsis_demographics.csv', index=False)


def extract_data():
    extract_chartevents()
    extract_diagnoses()
    extract_demographics()