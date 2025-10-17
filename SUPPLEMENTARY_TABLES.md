
### TABLE A: MIMIC-IV TABLE AND MODULE DESCRIPTIONS.

| Tables           | Description of Table                                                                                     | Module        | Description of Module                                                                                                                                                        |
|------------------|-----------------------------------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| patients         | Stores demographic information about each patient, like Gender, date of birth, and date of death.        | HOSP          | Data acquired from the hospital wide electronic health record: admission information, laboratory measurements, microbiology, medication administration.                      |
| admission        | Contains information about each hospital admission, like Timestamps of admission and discharge.          | HOSP          |                                                                                                                                                                               |
| d_icd_diagnoses  | Reference table for ICD diagnosis codes (ICD-9 or ICD-10) with their description.                         | HOSP          |                                                                                                                                                                               |
| diagnoses_icd    | Lists all diagnoses assigned to each hospital admission                                                   | HOSP          |                                                                                                                                                                               |
| d_items          | Dictionary of item IDs used for event data.                                                               | ICU           | Information collected from the clinical information system used within the ICU.                                                                                              |
| icustays         | Contains information about each ICU stay, like ICU admission and discharge timestamps, length of stay.    | ICU           |                                                                                                                                                                               |
| chartevents      | Clinical observations and measurements charted during ICU stays, like vitals, lab results and scores, with timestamps. | ICU           |                                                                                                                                                                               |
| inputevents      | Records all fluid and medication inputs during ICU stays, with timestamps, quantity and rate of input.    | ICU           |                                                                                                                                                                               |
| outputevents     | Records measurable fluid outputs (e.g., urine, drains) during ICU stays, with timestamps.                 | ICU           |                                                                                                                                                                               |
| procedureevents  | Contains information about procedures and complex interventions performed in the ICU, like mechanical ventilation, dialysis, with parameters related to procedures, like ventilator settings. | ICU           |                                                                                                                                                                               |
| sepsis3          | Identifies ICU stays where patients met the Sepsis-3 criteria. Definition Used: Based on Sepsis-3 definition: • Suspected infection (antibiotics + blood cultures). • Acute organ dysfunction (SOFA score increase ≥2). | MIMIC-Derived | Useful views/summaries of MIMIC-IV: demographics, organ failure scores, severity of illness scores, durations of treatment, easier to analyze views.                          |



### TABLE B: ENGINEERED FEATURES AS SUGGESTED BY CLINICAL EXPERTS, SCORING SYSTEMS OR LITERATURE.

| Engineered Features                                                                                                  | Type       | Temporal* / Static |
|----------------------------------------------------------------------------------------------------------------------|------------|--------------------|
| **Clinical Experts’ Suggestions**                                                                                    |            |                    |
| Shock index (Heart rate/Systolic Arterial Blood Pressure) > 1                                                       | Binary     | Temporal           |
| Mean Arterial Pressure < 65 for 1 hour or more                                                                       | Binary     | Static             |
| Aged <=64: Spontaneous breathing rate >25 for more than 1 hour (yes/no), Aged > 64: >27 for more than 1 hour        | Binary     | Static             |
| Respiratory rate >= 22 AND Systolic BP <=100 AND GCS < 15 [qSOFA]                                                   | Binary     | Temporal           |
| Positive or Negative change of all temporal variables at every hour                                                 | Continuous | Temporal           |
| **Key Values in Scoring Systems**                                                                                    |            |                    |
| GCS - lowest value in obs win [SAPS-II]                                                                             | Continuous | Static             |
| Systolic Blood Pressure - min value in obs win [SAPS-II]                                                            | Continuous | Static             |
| Temperature - max value in obs win [SAPS-II]                                                                        | Continuous | Static             |
| Temperature >= 39C [SAPS-II]                                                                                        | Binary     | Temporal           |
| PaO₂/FiO₂ [SAPS-II, SOFA]                                                                                           | Continuous | Temporal           |
| PaO₂/FiO₂ - min value in obs win [SAPS-II]                                                                          | Continuous | Static             |
| Sodium - min value in obs win [SAPS-II]                                                                             | Continuous | Static             |
| Potassium – min & max value in obs win [SAPS-II]                                                                    | Continuous | Static             |
| Bicarbonate - min value in obs win [SAPS-II]                                                                        | Continuous | Static             |
| Bilirubin - max value in obs win [SAPS-II]                                                                          | Continuous | Static             |
| WBC - min value in obs win [SAPS-II]                                                                                | Continuous | Static             |
| GCS Sum [SOFA, QSOFA, SAPS II, APACHE II]                                                                           | Ordinal    | Temporal           |
| **Literature**                                                                                                       |            |                    |
| Hospital admission to ICU admission time (hours)                                                                    | Continuous | Static             |
| Shock index (Heart rate/Systolic Arterial Blood Pressure)                                                           | Continuous | Temporal           |
| Statistics of all temporal variables in obs window (min, max, mean, range)                                          | Continuous | Static             |
*Hourly bins for 24 hours.

### TABLE C: VARIABLE LABELS AND THEIR ORDINAL ENCODINGS ACCORDING TO MEDICAL RANGES
| Feature                                     | Medical Range         | Label                          | Encoding |
|---------------------------------------------|------------------------|--------------------------------|----------|
| **Fspn High (min, max, mean, or any timestep)** |                        |                                |          |
|                                             | [0, 8]                 | Very Low                       | 0        |
|                                             | (8, 11]                | Low                            | 1        |
|                                             | (11, 20]               | Normal                         | 2        |
|                                             | (20, 24]               | High                           | 3        |
|                                             | Above 24               | Very High                      | 4        |
| **PH (Arterial) (min, max, mean, or any timestep)** |                    |                                |          |
|                                             | [0, 7.37]              | Low                            | 0        |
|                                             | (7.37, 7.44]           | Normal                         | 1        |
|                                             | Above 7.44             | High                           | 2        |
| **Tandem Heart Flow (min, max, mean, or any timestep)** |                 |                                |          |
|                                             | [0, 2.99]              | Low                            | 0        |
|                                             | (2.99, 4]              | Normal                         | 1        |
|                                             | Above 4                | High                           | 2        |
| **Arterial Blood Pressure systolic (min, max, mean, or any timestep)** |        |                                |          |
|                                             | [0, 90]                | Very Low                       | 0        |
|                                             | (90, 100]              | Low                            | 1        |
|                                             | (100, 110]             | Pre-Normal                     | 2        |
|                                             | (110, 219]             | Normal                         | 3        |
|                                             | Above 219              | High                           | 4        |
| **O2 Flow (min, max, mean, or any timestep)** |                        |                                |          |
|                                             | [0, 1]                 | Very Low                       | 0        |
|                                             | (1, 5]                 | Low                            | 1        |
|                                             | (5, 10]                | Moderate                       | 2        |
|                                             | (10, 15]               | High                           | 3        |
|                                             | Above 15               | Very High                      | 4        |
| **GCS sum (min, max, mean, or any timestep)** |                        |                                |          |
|                                             | [0, 8]                 | Comatose                       | 0        |
|                                             | (8, 12]                | Confused/lethargic             | 1        |
|                                             | (12, 15]               | Alert/minimally confused       | 2        |
| **Arterial Blood Pressure mean (min, max, mean, or any timestep)** |          |                                |          |
|                                             | [0, 69]                | Low                            | 0        |
|                                             | (69, 100]              | Normal                         | 1        |
|                                             | Above 100              | High                           | 2        |
| **Negative Insp. Force (min, max, mean, or any timestep)** |               |                                |          |
|                                             | Up to -30              | Normal                         | 0        |
|                                             | Above -30              | High                           | 1        |
| **Temperature Celsius (min, max, mean, or any timestep)** |               |                                |          |
|                                             | [0, 35]                | Very Low                       | 0        |
|                                             | (35, 36]               | Low                            | 1        |
|                                             | (36, 38]               | Normal                         | 2        |
|                                             | (38, 39]               | High                           | 3        |
|                                             | Above 39               | Very High                      | 4        |
| **SV (Arterial) (min, max, mean, or any timestep)** |                    |                                |          |
|                                             | [0, 59]                | Low                            | 0        |
|                                             | (59, 100]              | Normal                         | 1        |
|                                             | Above 100              | High                           | 2        |
| **Potassium (whole blood) (min, max, mean, or any timestep)** |             |                                |          |
|                                             | [0, 3.4]               | Low                            | 0        |
|                                             | (3.4, 5.2]             | Normal                         | 1        |
|                                             | Above 5.2              | High                           | 2        |
| **Sodium (whole blood) (min, max, mean, or any timestep)** |                |                                |          |
|                                             | [0, 135]               | Low                            | 0        |
|                                             | (135, 145]             | Normal                         | 1        |
|                                             | Above 145              | High                           | 2        |
| **RRApacheIIValue (min, max, mean, or any timestep)** |                   |                                |          |
|                                             | [0, 8]                 | Very Low                       | 0        |
|                                             | (8, 11]                | Low                            | 1        |
|                                             | (11, 20]               | Normal                         | 2        |
|                                             | (20, 24]               | High                           | 3        |
|                                             | Above 24               | Very High                      | 4        |
| **Any inputevent or diagnosis**             |                        | True                           | 1        |
|                                             |                        | False                          | 0        |
GCS: Glasgow Comma Scale, SV_arterial: Stroke Volume (measured/derived from arterial waveform data), Fspn: spontaneous 
breathing frequency, RRApacheIIValue: Respiratory Rate according to APACHE-II input


### TABLE D: LIST OF DEMOGRAPHICS AND 70 SELECTED FEATURES ACCORDING TO GBM BUILT-IN FEATURE IMPORTANCE WITH THEIR DESCRIPTIVE STATISTICS ACROSS CLASSES.  
N = 45,148 — Cases (N = 1,088) vs Controls (N = 44,060)  
Continuous variables: Age, Length of ICU Stay, Hospital to ICU Admission, Sepsis Onset After ICU Admission, Vital Signs section, Lab Values section
Binary variables: Gender, Infusions/Medications section, Diagnosis section
Nominal variables: Ethnicity
Ordinal variables: Clinical Score section
Engineered features: Hospital to ICU Admission, PO2/FIO2, Shock Index, GCS sum

Median [Q1, Q3] values are reported for Continuous and Ordinal variables. Count (%) values are reported for Binary and 
Nominal variables.

| Section                 | Variable                                     | Cases | Controls | Unit |
|-------------------------|----------------------------------------------|-------|----------|------|
| Demographics            | Age                                          | 67 [54, 77] | 67 [55, 78] | years |
| Demographics            | Gender – Male                                | 571 (52%) | 24,349 (55%) | – |
| Demographics            | Gender – Female                              | 517 (48%) | 19,711 (45%) | – |
| Demographics            | Ethnicity – White                            | 648 (60%) | 28,509 (65%) | – |
| Demographics            | Ethnicity – Black/African American           | 95 (1%) | 4,238 (10%) | – |
| Demographics            | Ethnicity – Unknown                          | 153 (14%) | 3,572 (8%) | – |
| Demographics            | Ethnicity – Other                            | 192 (18%) | 7,741 (18%) | – |
| Demographics            | Length of ICU Stay                           | 275.07 [187.26, 430.89] | 52.32 [35.86, 89.63] | hours |
| Demographics            | Hospital to ICU Admission (hosp_to_icu)*     | 2.65 [0.79, 60.73] | 2.40 [1.08, 26.88] | hours |
| Demographics            | Sepsis Onset After ICU Admission             | 70 [50, 113] | – | hours |
| Vital Signs             | PO2/FIO2 (pf_ratio)* range                   | 0 [0, 0.4] | 0.2 [0, 0.8] | mmHg |
| Vital Signs             | Temperature (1st hr)                         | 36.9 [36.4, 37.6] | 36.7 [36, 37.3] | °C |
| Vital Signs             | Temperature (range)                          | 0.5 [0.2, 0.9] | 0.8 [0.4, 1.2] | °C |
| Vital Signs             | Heart Rate (17th–16th hr)                    | 0 [-3.8, 3] | 0 [-3, 3] | beats/min |
| Vital Signs             | Heart Rate (15th–14th hr)                    | 0 [-3, 3] | 0 [-3, 3] | beats/min |
| Vital Signs             | Spontaneous Respiratory Rate (range)         | 0 [0, 0] | 0 [0, 0] | insp/min |
| Vital Signs             | Shock Index* (15th–14th hr)                  | 0 [0, 0] | 0 [0, 0] | – |
| Vital Signs             | Shock Index* (10th–9th hr)                   | 0 [0, 0] | 0 [0, 0] | – |
| Vital Signs             | Shock Index* (19th–18th hr)                  | 0 [0, 0] | 0 [-0.1, 0] | – |
| Vital Signs             | Tandem Heart Flow (11th hr)                  | 3.2 [2.9, 3.6] | 3.1 [2.9, 3.3] | L/min |
| Vital Signs             | Tandem Heart Flow (9th hr)                   | 3.2 [2.9, 3.5] | 3.0 [2.8, 3.2] | L/min |
| Vital Signs             | Fspn-High (3rd hr)                           | 34.3 [20, 36] | 26 [10.3, 33.3] | insp/min |
| Vital Signs             | Arterial BP Mean (12th hr)                   | 80.7 [72.7, 91] | 78 [72, 85] | mmHg |
| Vital Signs             | Arterial BP Mean (19th hr)                   | 82 [74, 92] | 80.3 [73.3, 87] | mmHg |
| Vital Signs             | Arterial BP Mean (11th–10th hr)              | 0 [-2, 2.3] | 0 [-3, 2.7] | mmHg |
| Vital Signs             | Arterial BP Mean (9th–8th hr)                | 0 [-2, 2.7] | 0 [-3.3, 2.5] | mmHg |
| Vital Signs             | Systolic BP (min)                            | 107 [95, 119.7] | 105 [96.3, 113.7] | mmHg |
| Vital Signs             | Systolic BP (17th hr)                        | 125 [110.7, 137] | 120 [110, 129.7] | mmHg |
| Vital Signs             | Systolic BP (20th–19th hr)                   | 0 [-2.3, 3] | 0 [-3, 4] | mmHg |
| Vital Signs             | Systolic BP (24th hr)                        | 124.3 [111, 136] | 121.7 [110.3, 132.3] | mmHg |
| Vital Signs             | Systolic BP (15th–14th hr)                   | 0 [-3, 4] | 0.3 [-3.3, 6] | mmHg |
| Vital Signs             | Systolic BP (23rd–22nd hr)                   | 0 [-4, 3] | 0 [-4, 4] | mmHg |
| Vital Signs             | Systolic BP (12th–11th hr)                   | 0 [-3.3, 3] | 0 [-4.7, 4] | mmHg |
| Vital Signs             | TFCd (NICOM) Max                             | 3.7 [2.7, 4.8] | 3.1 [1.2, 4.2] | % |
| Vital Signs             | O2 Flow (Mean)                               | 5.9 [3, 10] | 4.1 [2.3, 7.6] | L/min |
| Vital Signs             | O2 Flow (24th hr)                            | 5.7 [2.9, 10] | 3.3 [2, 6] | L/min |
| Vital Signs             | Negative Insp. Force (Mean)                  | -42.1 [-49.8, -34.8] | -37.8 [-46.5, -30.5] | cmH2O |
| Vital Signs             | SV (Arterial) Min                            | 47.8 [42.8, 58.5] | 56.6 [47.7, 65.2] | mL/beat |
| Vital Signs             | SV (Arterial) (14th hr)                      | 53.5 [47.6, 63.8] | 61 [52, 69.4] | mL/beat |
| Vital Signs             | Pisp (Hamilton) Range                        | 0 [0, 2.3] | 2 [0, 4.7] | cmH2O |
| Lab Values              | BUN (2nd–1st hr)                             | 0 [0, 0] | 0 [0, 0] | – |
| Lab Values              | BUN (3rd–2nd hr)                             | 0 [0, 0] | 0 [0, 0] | – |
| Lab Values              | BUN (4th–3rd hr)                             | 0 [0, 0] | 0 [0, 0] | – |
| Lab Values              | BUN (5th–4th hr)                             | 0 [0, 0] | 0 [0, 0] | – |
| Lab Values              | Phosphorous (Range)                          | 0.5 [0.2, 1] | 0 [0, 0.6] | – |
| Lab Values              | Calcium Non-Ionized (Range)                  | 0.3 [0.1, 0.5] | 0 [0, 0.3] | – |
| Lab Values              | WBC (3rd–2nd hr)                             | 0 [0, 0] | 0 [0, 0] | – |
| Lab Values              | WBC (2nd–1st hr)                             | 0 [0, 0] | 0 [0, 0] | – |
| Lab Values              | WBC (4th–3rd hr)                             | 0 [0, 0] | 0 [0, 0] | – |
| Lab Values              | pH (Arterial) (1st hr)                       | 7.4 [7.4, 7.5] | 7.4 [7.4, 7.4] | – |
| Lab Values              | pH (Arterial) (11th hr)                      | 7.4 [7.4, 7.5] | 7.4 [7.4, 7.4] | – |
| Lab Values              | pH (Arterial) (23rd hr)                      | 7.4 [7.4, 7.5] | 7.4 [7.4, 7.4] | – |
| Lab Values              | Potassium (whole blood, max)                 | 3.9 [3.7, 4.2] | 4.1 [3.8, 4.4] | – |
| Lab Values              | Sodium (whole blood, 20th hr)                | 136 [132, 139.3] | 136 [133.7, 139] | – |
| Infusions / Medications | NaCl 0.9% (1st hr)                           | 602 (55%) | 8,871 (20%) | ml |
| Infusions / Medications | NaCl 0.9% (2nd hr)                           | 608 (56%) | 14,244 (32%) | ml |
| Infusions / Medications | Dextrose 5% (1st hr)                         | 319 (29%) | 3,985 (9%) | ml |
| Infusions / Medications | Glucerna 1.2 Full                            | 17 (2%) | 4 (0.009%) | ml |
| Infusions / Medications | Glucerna 1.5 Full                            | 12 (1%) | 5 (0.01%) | ml |
| Infusions / Medications | Calcium Gluconate (CRRT)                     | 32 (3%) | 7 (0.02%) | grams |
| Infusions / Medications | Heparin Sodium                               | 102 (9%) | 542 (1%) | units |
| Infusions / Medications | Jevity 1.2 Full                              | 29 (3%) | 6 (0.01%) | ml |
| Infusions / Medications | Jevity 1.5 Full                              | 19 (2%) | 12 (0.03%) | ml |
| Infusions / Medications | Promote Full                                 | 17 (2%) | 2 (0.005%) | ml |
| Infusions / Medications | Dexmedetomidine (Precedex)                   | 36 (3%) | 56 (0.1%) | mcg |
| Infusions / Medications | Propofol                                     | 259 (24%) | 3,205 (7%) | mg |
| Infusions / Medications | Promote with Fiber Full                      | 17 (2%) | 6 (0.01%) | ml |
| Infusions / Medications | Replete with Fiber Full (1st hr)             | 27 (2%) | 2 (0.005%) | ml |
| Infusions / Medications | Replete with Fiber Full (11th hr)            | 34 (3%) | 77 (0.2%) | ml |
| Infusions / Medications | Furosemide 250/50 (1st hr)                   | 34 (3%) | 67 (0.2%) | mg |
| Infusions / Medications | Milrinone (1st hr)                           | 30 (3%) | 34 (0.08%) | mg |
| Infusions / Medications | Osmolite 1.5 Full                            | 14 (1%) | 3 (0.007%) | ml |
| Infusions / Medications | Nepro Full                                   | 17 (2%) | 11 (0.03%) | ml |
| Infusions / Medications | Fibersource HN Full                          | 11 (1%) | 2 (0.005%) | ml |
| Infusions / Medications | Furosemide (Lasix) (1st hr)                  | 18 (2%) | 43 (0.1%) | mg |
| Clinical Scores         | GCS sum* (24th hr)                           | 12 [9, 15] | 15 [13.6, 15] | – |
| Clinical Scores         | GCS – Eye Opening (2nd–1st hr)               | 0 [0, 0] | 0 [0, 0] | – |
| Clinical Scores         | RRApacheIIValue (6th hr)                     | 27.7 [26, 30] | 30 [26.3, 30] | insp/min |
| Diagnosis               | Urinary Tract Infection (site not specified) | 351 (32%) | 6,642 (15%) | – |

*Engineered Features.
GCS: Glasgow Comma Scale, pinsp_hamilton: inspiratory pressure from Hamilton ventilator, 
BUN: Blood Urea Nitrogen, TFCd_NICOM: Thoracic Fluid Content from the NICOM (non invasive cardiac monitor), 
SV_arterial: Stroke Volume (measured/derived from arterial waveform data), fspn: spontaneous breathing 
frequency, RRApacheIIValue: Respiratory Rate according to APACHE-II input.

### TABLE E: POSITIVE PREDICTION RULE LIST (CONDITIONS INCLUDE ORDINAL ENCODINGS – SEE TABLE C)

| Rule # | Rule | Coverage | Accuracy |
|--------|------|----------|----------|
| 0 | SV_arterial_min <= 0.5 & arterial_blood_pressure_mean_8_diff <= 28.0 & fspn_high_2 > 3.5 & nacl09_0 > 0.5 & pinsp_hamilton_range <= 0.8 | 0.10 | 0.86 |
| 1 | dextrose5_0 > 0.5 & pf_ratio_a_range <= 0.6 & phosphorous_range > 0.1 & temperature_celsius_0 > 1.5 | 0.08 | 0.83 |
| 2 | calcium_non_ionized_range > 0.1 & gcs_sum_23 <= 1.5 & propofol_0 > 0.5 | 0.12 | 0.94 |
| 3 | TFCd_NICOM_max > 3.2 & calcium_non_ionized_range <= 1.8 & calcium_non_ionized_range > 0.1 & hosp_to_icu > 38.6 & pf_ratio_a_range <= 0.7 | 0.80 | 0.78 |
| 4 | BUN_1_diff > -0.5 & GCS_eyeOpening1_diff <= 0.8 & nacl09_0 > 0.5 & pf_ratio_a_range <= 2.2 & urinary_tract_infection_not_specified > 0.5 | 0.09 | 0.87 |
| 5 | PH_arterial_10 > 1.5 & calcium_non_ionized_range <= 0.8 & phosphorous_range > 0.1 & pinsp_hamilton_range <= 0.5 | 0.09 | 0.85 |
| 6 | arterial_blood_pressure_systolic_11_diff <= 0.04 & calcium_non_ionized_range > 0.1 & gcs_sum_23 > 1.5 & pf_ratio_a_range <= 0.003 & phosphorous_range <= 1.8 & pinsp_hamilton_range <= 1.5 | 0.08 | 0.73 |
| 7 | PH_arterial_0 > 0.5 & SV_arterial_13 <= 0.5 & gcs_sum_23 <= 1.5 & phosphorous_range > 0.1 & propofol_0 <= 0.5 & temperature_celsius_0 > 1.5 | 0.07 | 0.83 |
| 8 | dextrose5_0 > 0.5 & gcs_sum_23 > 1.5 & pf_ratio_a_range <= 0.003 | 0.05 | 0.82 |
| 9 | TFCd_NICOM_max > 1.4 & arterial_blood_pressure_mean_18 <= 1.5 & dextrose5_0 > 0.5 & nacl09_0 > 0.5 & pf_ratio_a_range <= 2.3 | 0.09 | 0.87 |
| 10 | TFCd_NICOM_max > 3.2 & nacl09_0 > 0.5 & pf_ratio_a_range <= 0.7 & phosphorous_range > 0.1 & temperature_celsius_range <= 0.6 & urinary_tract_infection_not_specified <= 0.5 | 0.06 | 0.84 |
| 11 | arterial_blood_pressure_mean_18 > 1.5 & arterial_blood_pressure_mean_8_diff <= 11.2 & calcium_non_ionized_range <= 1.3 & phosphorous_range > 0.1 | 0.05 | 0.95 |
| 12 | PH_arterial_10 > 1.5 & TFCd_NICOM_max > 3.2 & arterial_blood_pressure_systolic_14_diff <= 1.4 & pf_ratio_a_range <= 0.7 | 0.09 | 0.74 |
| 13 | BUN_1_diff <= 0.5 & PH_arterial_10 > 1.5 & calcium_non_ionized_range <= 0.8 & fspn_high_2 > 3.5 & phosphorous_range > 0.1 | 0.09 | 0.84 |
| 14 | TFCd_NICOM_max > 3.2 & arterial_blood_pressure_mean_10_diff <= 1.2 & arterial_blood_pressure_mean_10_diff > -0.2 & arterial_blood_pressure_systolic_min > 2.5 & pf_ratio_a_range <= 0.7 & temperature_celsius_range <= 0.6 | 0.05 | 0.85 |
| 15 | O2_flow_23 > 1.5 & PH_arterial_10 > 0.5 & propofol_0 > 0.5 | 0.10 | 0.86 |
| 16 | arterial_blood_pressure_mean_8_diff <= 28.0 & arterial_blood_pressure_systolic_14_diff <= 1.8 & arterial_blood_pressure_systolic_14_diff > -0.1 & calcium_non_ionized_range > 0.1 & hosp_to_icu > 20.2 | 0.05 | 0.86 |
| 17 | arterial_blood_pressure_systolic_min <= 2.5 & gcs_sum_23 <= 1.5 & pf_ratio_a_range <= 0.7117 & urinary_tract_infection_not_specified > 0.5 | 0.04 | 0.79 |
| 18 | TFCd_NICOM_max > 2.6833 & hosp_to_icu <= 0.4185 & nacl09_0 > 0.5 | 0.04 | 0.94 |
| 19 | O2_flow_23 <= 1.5 & SV_arterial_13 <= 0.5 & arterial_blood_pressure_systolic_14_diff <= 3.25 & pf_ratio_a_range <= 0.0583 & urinary_tract_infection_not_specified > 0.5 | 0.03 | 0.80 |
| 20 | TFCd_NICOM_max > -0.1 & hosp_to_icu > 7.9 & nacl09_0 > 0.5 & phosphorous_range > 0.1 & temperature_celsius_0 > 1.5 | 0.09 | 0.82 |
| 21 | arterial_blood_pressure_systolic_19_diff > -0.4 & dextrose5_0 <= 0.5 & nacl09_0 > 0.5 & phosphorous_range > 0.1 & pinsp_hamilton_range <= 0.8 | 0.09 | 0.92 |
| 22 | dextrose5_0 > 0.5 & hosp_to_icu > 36.6854 & tandem_heart_flow_10 > 0.5 | 0.04 | 0.83 |
| 23 | WBC_3_diff > -0.4 & arterial_blood_pressure_mean_18 > 1.5 & gcs_sum_23 <= 1.5 | 0.05 | 0.90 |
| 24 | SV_arterial_13 <= 0.5 & gcs_sum_23 > 1.5 & pf_ratio_a_range <= 0.003 & phosphorous_range > 0.1 & pinsp_hamilton_range <= 1.5 & temperature_celsius_0 > 1.5 | 0.07 | 0.93 |
| 25 | TFCd_NICOM_max <= 3.2 & arterial_blood_pressure_mean_10_diff <= 1.167 & arterial_blood_pressure_mean_10_diff > -0.167 & arterial_blood_pressure_systolic_min > 2.5 & fspn_high_2 > 2.5 & gcs_sum_23 <= 1.5 | 0.02 | 0.86 |
| 26 | BUN_1_diff > 0.5 | 0.01 | 1.00 |
| 27 | BUN_1_diff <= -0.5 | 0.01 | 1.00 |
| 28 | WBC_2_diff <= -0.85 & temperature_celsius_range > 0.67 | 0.00 | 0.00 |
| 29 | phosphorous_range > 0.0667 & spontRR_range > 2.83 | 0.03 | 0.85 |
| 30 | PH_arterial_10 > 1.5 & nacl09_0 > 0.5 & phosphorous_range > 0.1 | 0.08 | 0.81 |
| 31 | arterial_blood_pressure_systolic_19_diff > -4.667 & dextrose5_0 > 0.5 & heparinSodium_0 > 0.5 & nacl09_0 <= 0.5 | 0.01 | 0.50 |
| 32 | hosp_to_icu > 20.1446 & phosphorous_range > 0.0667 & temperature_celsius_0 <= 1.5 & temperature_celsius_range <= 0.6708 | 0.02 | 0.71 |
| 33 | TFCd_NICOM_max <= 3.2 & gcs_sum_23 <= 1.5 & hosp_to_icu <= 1.167 & pinsp_hamilton_range > 0.5 & urinary_tract_infection_not_specified > 0.5 | 0.00 | n/a |
| 34 | O2_flow_23 > 1.5 & TFCd_NICOM_max > 4.983 & arterial_blood_pressure_systolic_11_diff <= 3.167 & phosphorous_range > 0.0667 & temperature_celsius_0 > 1.5 | 0.04 | 0.88 |
| 35 | promoteFiber_0 > 0.5 | 0.00 | 1.00 |
| 36 | arterial_blood_pressure_systolic_min > 2.5 & nacl09_0 > 0.5 & shock_index_18_diff > 0.1274 | 0.00 | n/a |
| 37 | BUN_4_diff > 0.5 & hosp_to_icu > 36.6854 & jevity15_0 <= 0.5 | 0.00 | 1.00 |
| 38 | O2_flow_mean <= 0.5 & heart_rate_14_diff <= 0.292 & hosp_to_icu > 36.6854 | 0.00 | 1.00 |
| 39 | BUN_4_diff > 0.5 & TFCd_NICOM_max <= 3.2 & temperature_celsius_range > 0.6708 | 0.00 | 1.00 |
| 40 | arterial_blood_pressure_mean_18 > 1.5 & pf_ratio_a_range <= 0.7 & pinsp_hamilton_range <= 1.5 | 0.05 | 0.95 |
| 41 | TFCd_NICOM_max > 3.2 & arterial_blood_pressure_systolic_14_diff > 11.833 & phosphorous_range <= 0.0667 & urinary_tract_infection_not_specified > 0.5 | 0.00 | 0.00 |
| 42 | arterial_blood_pressure_mean_10_diff <= 1.2 & arterial_blood_pressure_mean_10_diff > -0.2 & arterial_blood_pressure_systolic_min > 2.5 & nacl09_0 > 0.5 & phosphorous_range > 0.1 | 0.05 | 0.95 |
| 43 | O2_flow_23 > 1.5 & heart_rate_14_diff <= -5.7 & nacl09_0 <= 0.5 & phosphorous_range > 0.0667 & pinsp_hamilton_range <= 0.5 & propofol_0 <= 0.5 | 0.01 | 0.50 |

Feature names ending in a number refer to the mean (if vital sign, lab value, clinical score) or binary 
(if medication/infusion) value in the indicated hour (hour range: 0-23 for 1st to 24th hour, resp.). Features ending in
min/max/mean/range refer to the summary statistic of the feature across the last 24hrs. Features ending in diff refer to 
the difference between the indicated hour and the next. COV: Coverage, ACC: Accuracy, gcs: Glasgow Comma Scale, 
pinsp_hamilton: inspiratory pressure from Hamilton ventilator, BUN: Blood Urea Nitrogen, TFCd_NICOM: Thoracic Fluid 
Content from the NICOM (non invasive cardiac monitor), SV_arterial: Stroke Volume (measured/derived from arterial 
waveform data), fspn: spontaneous breathing frequency, RRApacheIIValue: Respiratory Rate according to APACHE-II input.

### TABLE F: NEGATIVE PREDICTION RULE LIST (CONDITIONS INCLUDE ORDINAL ENCODINGS – SEE TABLE C)
| Rule # | Rule                                                                                                                                                                                                                                                                                            | Coverage | Accuracy |
|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|----------|
| 0 | gcs_sum_23 > 1.5 & pf_ratio_a_range > 0.003 & phosphorous_range <= 0.1                                                                                                                                                                                                                          | 0.17     | 0.95     |
| 1 | O2_flow_23 <= 2.5 & dextrose5_0 <= 0.5 & hosp_to_icu <= 41.3 & nacl09_0 <= 0.5 & temperature_celsius_0 <= 1.5                                                                                                                                                                                   | 0.1      | 0.7      |
| 2 | TFCd_NICOM_max <= 5.0 & dextrose5_0 <= 0.5 & phosphorous_range <= 0.1 & sodium_wholeblood_19 > 0.5 & temperature_celsius_0 > 1.5 & urinary_tract_infection_not_specified <= 0.5                                                                                                                 | 0.11     | 0.86     |
| 3 | dextrose5_0 <= 0.5 & gcs_sum_23 > 1.5 & nacl09_0 <= 0.5 & pf_ratio_a_range > 0.003 & phosphorous_range > 0.1 & pinsp_hamilton_range > 0.5 & urinary_tract_infection_not_specified <= 0.5                                                                                                        | 0.04     | 0.68     |
| 4 | arterial_blood_pressure_mean_18 <= 1.5 & nacl09_0 <= 0.5 & nacl09_1 > 0.5 & propofol_0 <= 0.5                                                                                                                                                                                                   | 0.06     | 0.96     |
| 5 | BUN_1_diff > -0.5 & arterial_blood_pressure_mean_10_diff <= -0.2 & calcium_non_ionized_range <= 0.1 & dextrose5_0 <= 0.5 & nacl09_0 <= 0.5 & spontRR_range <= 2.8                                                                                                                               | 0.08     | 0.81     |
| 6 | PH_arterial_10 <= 1.5 & dextrose5_0 <= 0.5 & gcs_sum_23 > 1.5 & hosp_to_icu <= 35.4 & nacl09_0 <= 0.5 & pf_ratio_a_range > 0.003 & phosphorous_range > 0.1 & pinsp_hamilton_range <= 0.8 & urinary_tract_infection_not_specified <= 0.5                                                         | 0.01     | 0.6      |
| 7 | TFCd_NICOM_max <= 2.6 & arterial_blood_pressure_mean_8_diff <= 28.0 & arterial_blood_pressure_systolic_14_diff <= -0.1 & dextrose5_0 <= 0.5 & pinsp_hamilton_range > 0.5 & urinary_tract_infection_not_specified <= 0.5                                                                         | 0.04     | 0.83     |
| 8 | PH_arterial_10 <= 1.5 & calcium_non_ionized_range > 0.6 & dextrose5_0 <= 0.5 & hosp_to_icu <= 20.2 & nacl09_0 <= 0.5 & phosphorous_range > 0.1 & pinsp_hamilton_range > 0.5                                                                                                                     | 0.02     | 0.7      |
| 9 | BUN_1_diff > -0.5 & GCS_eyeOpening1_diff <= 0.8 & PH_arterial_10 <= 1.5 & calcium_gluconateCRRT_0 <= 0.5 & pf_ratio_a_range > 2.2                                                                                                                                                               | 0.03     | 0.64     |
| 10 | dextrose5_0 <= 0.5 & pf_ratio_a_range <= 0.7 & phosphorous_range <= 0.1 & pinsp_hamilton_range > 1.5 & urinary_tract_infection_not_specified <= 0.5                                                                                                                                             | 0.11     | 0.89     |
| 11 | O2_flow_23 <= 1.5 & PH_arterial_10 <= 1.5 & arterial_blood_pressure_mean_10_diff <= 0.2 & arterial_blood_pressure_mean_8_diff <= 28.0 & fspn_high_2 <= 3.5 & hosp_to_icu <= 20.2 & pf_ratio_a_range > 0.003 & phosphorous_range > 0.1 & urinary_tract_infection_not_specified <= 0.5            | 0.01     | 0.67     |
| 12 | PH_arterial_10 <= 1.5 & TFCd_NICOM_max <= 3.2 & hosp_to_icu <= 36.7 & hosp_to_icu > 0.4 & temperature_celsius_0 <= 1.5 & urinary_tract_infection_not_specified <= 0.5                                                                                                                           | 0.04     | 0.75     |
| 13 | PH_arterial_10 <= 1.5 & arterial_blood_pressure_mean_10_diff <= -0.7 & dextrose5_0 <= 0.5 & nacl09_0 <= 0.5 & pf_ratio_a_range <= 0.003 & phosphorous_range > 0.1 & pinsp_hamilton_range > 0.5 & urinary_tract_infection_not_specified <= 0.5                                                   | 0.02     | 1.0      |
| 14 | WBC_2_diff > -0.2 & arterial_blood_pressure_mean_10_diff > -0.2 & arterial_blood_pressure_mean_18 <= 1.5 & calcium_non_ionized_range <= 0.1 & dextrose5_0 <= 0.5 & nacl09_0 <= 0.5 & pf_ratio_a_range <= 0.003 & potassium_whole_blood_max > 0.5 & urinary_tract_infection_not_specified <= 0.5 | 0.05     | 0.83     |
| 15 | dextrose5_0 <= 0.5 & heart_rate_14_diff > -16.5 & pf_ratio_a_range > 0.7 & pinsp_hamilton_range > 1.2 & propofol_0 <= 0.5                                                                                                                                                                       | 0.07     | 0.83     |
| 16 | O2_flow_23 <= 1.5 & TFCd_NICOM_max <= 3.2 & gcs_sum_23 > 1.5 & heparinSodium_0 <= 0.5 & pf_ratio_a_range > 0.1 & pinsp_hamilton_range <= 0.5                                                                                                                                                    | 0.03     | 0.85     |
| 17 | PH_arterial_10 <= 1.5 & SV_arterial_min > 0.5 & arterial_blood_pressure_mean_8_diff <= 28.0 & arterial_blood_pressure_systolic_14_diff > 0.2 & heart_rate_14_diff > 0.4 & hosp_to_icu <= 35.4 & hosp_to_icu > 1.8                                                                               | 0.03     | 0.75     |
| 18 | O2_flow_23 > 1.5 & WBC_2_diff > -0.6 & heart_rate_14_diff > -5.7 & nacl09_0 <= 0.5 & nacl09_1 <= 0.5 & pf_ratio_a_range > 0.7 & propofol_0 <= 0.5                                                                                                                                               | 0.03     | 0.64     |
| 19 | arterial_blood_pressure_systolic_min <= 2.5 & calcium_non_ionized_range > 0.1 & fspn_high_2 <= 3.5 & gcs_sum_23 > 1.5 & hosp_to_icu <= 41.9 & nacl09_0 <= 0.5 & pf_ratio_a_range <= 0.8 & temperature_celsius_range <= 0.7                                                                      | 0.02     | 0.25     |
| 20 | PH_arterial_10 > 1.5 & calcium_non_ionized_range > 0.8 & heart_rate_14_diff > 0.4 & hosp_to_icu > 0.9 & jevity12_0 <= 0.5 & phosphorous_range > 0.1 & urinary_tract_infection_not_specified <= 0.5                                                                                              | 0.0      | 1.0      |
| 21 | PH_arterial_10 <= 1.5 & TFCd_NICOM_max <= 3.2 & arterial_blood_pressure_mean_10_diff <= -0.2 & dextrose5_0 <= 0.5 & heart_rate_14_diff > 0.4 & hosp_to_icu > 0.9 & jevity12_0 <= 0.5 & phosphorous_range <= 2.5 & urinary_tract_infection_not_specified <= 0.5                                  | 0.02     | 0.78     |
| 22 | PH_arterial_10 <= 1.5 & arterial_blood_pressure_mean_18 > 1.5 & calcium_non_ionized_range > 0.6 & dextrose5_0 <= 0.5 & gcs_sum_23 > 1.5 & hosp_to_icu > 1.8 & nacl09_0 <= 0.5 & phosphorous_range > 0.1 & pinsp_hamilton_range > 0.5 & urinary_tract_infection_not_specified <= 0.5             | 0.01     | 0.67     |
| 23 | gcs_sum_23 > 1.5 & heart_rate_14_diff > -5.7 & hosp_to_icu <= 35.4 & hosp_to_icu > 1.8 & nacl09_0 <= 0.5 & phosphorous_range > 0.1 & pinsp_hamilton_range > 0.5 & urinary_tract_infection_not_specified <= 0.5                                                                                  | 0.01     | 0.56     |
| 24 | dextrose5_0 <= 0.5 & pf_ratio_a_range <= 0.7 & phosphorous_range > 0.1 & pinsp_hamilton_range > 1.5 & urinary_tract_infection_not_specified <= 0.5                                                                                                                                              | 0.01     | 0.72     |
| 25 | PH_arterial_10 <= 1.5 & calcium_non_ionized_range > 0.1 & hosp_to_icu <= 35.4 & hosp_to_icu > 1.8 & nacl09_0 <= 0.5 & phosphorous_range > 0.1                                                                                                                                                   | 0.01     | 0.65     |
| 26     | heart_rate_14_diff > -16.5 & hosp_to_icu <= 41.3 & pf_ratio_a_range > 0.7 & propofol_0 <= 0.5 & temperature_celsius_0 <= 1.5                                                                                                         | 0.06 | 0.72 |
| 27     | O2_flow_23 <= 1.5 & arterial_blood_pressure_mean_18 <= 1.5 & dextrose5_0 <= 0.5 & heartRate_16_diff <= -0.6 & nacl09_0 <= 0.5 & phosphorous_range > 0.1 & pinsp_hamilton_range > 0.5 & urinary_tract_infection_not_specified > 0.5             | 0.01 | 0.80 |
| 28     | arterial_blood_pressure_systolic_11_diff > 5.1 & calcium_non_ionized_range > 0.1 & gcs_sum_23 > 1.5 & hosp_to_icu <= 38.6 & pf_ratio_a_range > 0.003 & promoteFiber_0 <= 0.5 & replete_fiber_10 <= 0.5 & temperature_celsius_0 > 1.5               | 0.00 | 1.00 |
| 29     | TFCd_NICOM_max <= 2.6 & arterial_blood_pressure_mean_8_diff <= 28.0 & arterial_blood_pressure_systolic_14_diff <= -0.1 & calcium_non_ionized_range <= 0.1                                                                          | 0.04 | 0.94 |
| 30     | TFCd_NICOM_max <= 5.0 & calcium_non_ionized_range <= 0.1 & dextrose5_0 <= 0.5 & nacl09_0 <= 0.5 & sodium_wholeblood_19 > 0.5 & temperature_celsius_0 > 1.5                                                                           | 0.09 | 0.92 |
| 31     | O2_flow_23 <= 2.5 & arterial_blood_pressure_mean_18 <= 1.5 & dextrose5_0 <= 0.5 & gcs_sum_23 > 1.5 & nacl09_0 <= 0.5 & pf_ratio_a_range <= 0.003 & propofol_0 > 0.5                                                               | 0.00 | 1.00 |
| 32     | SV_arterial_min > 0.5 & calcium_non_ionized_range > 0.1 & heart_rate_14_diff > 0.4 & hosp_to_icu <= 41.9 & nacl09_0 <= 0.5 & pinsp_hamilton_range > 0.5 & urinary_tract_infection_not_specified <= 0.5                                 | 0.02 | 0.86 |
| 33     | O2_flow_23 <= 1.5 & PH_arterial_10 <= 1.5 & arterial_blood_pressure_systolic_14_diff > 3.25 & dextrose5_0 <= 0.5 & pf_ratio_a_range <= 0.003 & temperature_celsius_range > 0.7 & urinary_tract_infection_not_specified <= 0.5            | 0.02 | 0.70 |
| 34     | PH_arterial_10 <= 1.5 & heartRate_16_diff <= -0.6 & pinsp_hamilton_range > 6.5 & urinary_tract_infection_not_specified > 0.5                                                                                                       | 0.01 | 1.00 |
| 35     | TFCd_NICOM_max <= 3.2 & arterial_blood_pressure_systolic_16 <= 2.5 & dextrose5_0 <= 0.5 & gcs_sum_23 <= 1.5 & pinsp_hamilton_range > 1.2                                                                                           | 0.01 | 0.75 |
| 36     | hosp_to_icu <= 20.1 & nacl09_0 <= 0.5 & phosphorous_range > 0.1 & pinsp_hamilton_range > 0.5 & temperature_celsius_0 <= 1.5                                                                                                         | 0.04 | 0.61 |
| 37     | O2_flow_23 > 1.5 & calcium_non_ionized_range <= 0.1 & gcs_sum_23 > 1.5 & phosphorous_range <= 0.3 & phosphorous_range > 0.1                                                                                                        | 0.01 | 0.50 |
| 38     | PH_arterial_10 <= 1.5 & TFCd_NICOM_max <= 4.8 & hosp_to_icu <= 35.4 & hosp_to_icu > 1.8 & phosphorous_range <= 0.1 & urinary_tract_infection_not_specified <= 0.5                                                                   | 0.06 | 0.92 |
| 39     | arterial_blood_pressure_mean_10_diff <= -0.2 & dextrose5_0 <= 0.5 & hosp_to_icu > 36.7 & pf_ratio_a_range <= 0.003 & pinsp_hamilton_range > 1.2 & tandem_heart_flow_10 <= 0.5 & urinary_tract_infection_not_specified <= 0.5               | 0.00 | n/a  |
| 40     | SV_arterial_13 > 0.5 & TFCd_NICOM_max <= 2.7 & arterial_blood_pressure_systolic_19_diff <= -0.4 & nacl09_0 > 0.5 & phosphorous_range > 0.1 & temperature_celsius_0 > 1.5 & urinary_tract_infection_not_specified <= 0.5                 | 0.01 | 0.67 |
| 41     | PH_arterial_10 <= 1.5 & TFCd_NICOM_max <= 3.2 & arterial_blood_pressure_mean_10_diff <= 0.2 & arterial_blood_pressure_mean_8_diff <= 28.0 & arterial_blood_pressure_systolic_14_diff <= -0.1 & fspn_high_2 <= 3.5 & urinary_tract_infection_not_specified <= 0.5 | 0.01 | 0.40 |
| 42     | BUN_4_diff <= 0.5 & PH_arterial_0 <= 0.5 & calcium_non_ionized_range <= 0.6 & calcium_non_ionized_range > 0.1 & gcs_sum_23 <= 1.5 & hosp_to_icu > 0.3 & jevity15_0 <= 0.5 & propofol_0 <= 0.5 & temperature_celsius_0 > 2.5                  | 0.00 | 0.00 |
| 43     | WBC_2_diff <= -0.2 & heartRate_16_diff <= -16.0                                                                                                                                                                                      | 0.00 | n/a  |
| 44     | PH_arterial_0 <= 0.5 & PH_arterial_10 > 1.5 & fspn_high_2 <= 3.5                                                                                                                                                                     | 0.00 | 1.00 |
| 45     | PH_arterial_10 > 1.5 & fspn_high_2 > 3.5 & phosphorous_range > 2.5                                                                                                                                                                   | 0.00 | 0.00 |
| 46     | TFCd_NICOM_max <= 3.2 & calcium_non_ionized_range > 1.8 & hosp_to_icu > 38.6                                                                                                                                                          | 0.00 | n/a  |
| 47     | PH_arterial_10 > 1.5 & TFCd_NICOM_max > 3.2 & pinsp_hamilton_range <= 1.2 & temperature_celsius_range > 2.0                                                                                                                          | 0.00 | n/a  |
| 48     | arterial_blood_pressure_systolic_14_diff > 1.8 & calcium_non_ionized_range > 0.1 & hosp_to_icu > 20.2 & nacl09_0 <= 0.5 & pf_ratio_a_range > 0.7 & phosphorous_range > 0.1                                                        | 0.00 | 1.00 |
| 49     | O2_flow_23 > 1.5 & PH_arterial_0 <= 0.5 & calcium_non_ionized_range > 0.6 & gcs_sum_23 <= 1.5 & heart_rate_14_diff <= -5.7 & hosp_to_icu <= 20.2 & propofol_0 <= 0.5                                                              | 0.00 | 0.00 |
| 50     | TFCd_NICOM_max > 3.2 & dextrose5_0 <= 0.5 & nacl09_0 <= 0.5 & pf_ratio_a_range <= 0.7 & pf_ratio_a_range > 0.003 & pinsp_hamilton_range > 0.5 & urinary_tract_infection_not_specified <= 0.5                                        | 0.04 | 0.76 |

Feature names ending in a number refer to the mean (if vital sign, lab value, clinical score) or binary 
(if medication/infusion) value in the indicated hour (hour range: 0-23 for 1st to 24th hour, resp.). Features ending in 
min/max/mean/range refer to the summary statistic of the feature across the last 24hrs. Features ending in diff refer 
to the difference between the indicated hour and the next. COV: Coverage, ACC: Accuracy, gcs: Glasgow Comma Scale, 
pinsp_hamilton: inspiratory pressure from Hamilton ventilator, BUN: Blood Urea Nitrogen, TFCd_NICOM: Thoracic Fluid 
Content from the NICOM (non invasive cardiac monitor), SV_arterial: Stroke Volume (measured/derived from arterial 
waveform data), fspn: spontaneous breathing frequency, RRApacheIIValue: Respiratory Rate according to APACHE-II input.
