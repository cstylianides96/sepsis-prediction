
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





### Table B: Data Extracted

| MIMIC-IV Category | Data Extracted |
|---|---|
| `d_items` | Item IDs |
| `icustays` | ICU stay information |
| `admissions` | Hospital admission information |
| Vital Signs (`chartevents`) | Arterial O2 pressure, Arterial Blood Pressure mean, Temperature Celsius, Respiratory Rate, Heart Rate, Arterial Blood Pressure systolic, Arterial Blood Pressure diastolic, Arterial O2 Saturation, Total PEEP Level, EtCO2, FiO2 (CH), PH (Arterial) |
| Lab Values (`chartevents`) | LDH, BUN, Hematocrit (serum), Platelets, Sodium (serum), Potassium (serum), Calcium non-ionized, Phosphorous, Magnesium, C Reactive Protein (CRP), WBC, Differential-Neuts, Differential-Lymphs, Differential-Monos, Differential-Basos, Fibrinogen, INR, Albumin, Alkaline Phosphate, Total Bilirubin, CK (CPK), Creatinine (serum), Glucose (serum) |
| Scoring Values (`chartevents`) | GCS - Motor Response, GCS - Verbal Response, GCS - Eye Opening |
| Diagnoses (`d_icd_diagnoses`, `diagnoses_icd`) | Sepsis-Related ICD9 & ICD10 codes: Respiratory Failure, Heart Failure, Cirrhosis, Dialysis Procedures, Lung Disease, Renal Failure, Infection, Organ Dysfunction, Hypotension, Sepsis, Chronic Disease, Surgeries |
| Demographics (`patients`, `chartevents`) | Age, Gender, Ethnicity, Height, Admission Weight |





### TABLE C: ENGINEERED FEATURES 

| Engineered Features | Type | Temporal* / Static |
|---|---|---|
| Shock index (Heart rate / Systolic Arterial Blood Pressure) > 1 | Binary | Temporal |
| Mean Arterial Pressure < 65 for 1 hour or more | Binary | Temporal |
| Respiratory rate >= 22 AND Systolic BP <= 100 AND GCS < 15 [qSOFA] | Binary | Temporal |
| Positive or Negative change of all temporal variables at every hour | Continuous | Temporal |
| Temperature >= 39°C [SAPS-II] | Binary | Temporal |
| GCS Sum [SOFA, qSOFA, SAPS II, APACHE II] | Ordinal | Temporal |
| Hospital admission to ICU admission time (hours) | Continuous | Static |
| Shock index (Heart rate / Systolic Arterial Blood Pressure) | Continuous | Temporal |
| Statistics of all temporal variables in observation window (min, max, mean, range) | Continuous | Static |
| Occurrence of oxygen pressure | Binary | Temporal |
| Occurrence of oxygen saturation | Binary | Temporal |
| Occurrence of arterial pH | Binary | Temporal |
| Occurrence of FiO2 | Binary | Temporal |
| Occurrence of PEEP | Binary | Temporal |
| Occurrence of EtCO2 | Binary | Temporal |

*Hourly bins for 24 hours.

### TABLE D: VARIABLE LABELS AND THEIR ORDINAL ENCODINGS ACCORDING TO MEDICAL RANGES

| Feature | Medical Range | Label | Encoding |
|---|---|---|---|
| `GCS - Verbal Response` | [0, 1] | None | 0 |
|  | (1, 2] | Incomprehensible | 1 |
|  | (2, 3] | Inappropriate | 2 |
|  | (3, 4] | Confused | 3 |
|  | (4, 5] | Oriented | 4 |
| `GCS - Motor Response` | [0, 1] | None | 0 |
|  | (1, 2] | Abnormal extension to pain | 1 |
|  | (2, 3] | Abnormal flexion to pain | 2 |
|  | (3, 4] | Withdraws from pain | 3 |
|  | (4, 5] | Localizes pain | 4 |
|  | (5, 6] | Obeys commands | 5 |
| `Arterial Blood Pressure systolic` | [0, 90] | Very low | 0 |
|  | (90, 100] | Low | 1 |
|  | (100, 110] | Pre-Normal | 2 |
|  | (110, 219] | Normal | 3 |
|  | Above 219 | High | 4 |
| `Sodium (serum)` | [0, 135] | Low | 0 |
|  | (135, 145] | Normal | 1 |
|  | Above 145 | High | 2 |
| `Arterial Blood Pressure mean` | [0, 69] | Low | 0 |
|  | (69, 100] | Normal | 1 |
|  | Above 100 | High | 2 |
| `Magnesium` | [0, 1.6] | Low | 0 |
|  | (1.6, 2.4] | Normal | 1 |
|  | Above 2.4 | High | 2 |
| `Creatinine (serum)` | [0, 0.6] | Low | 0 |
|  | (0.6, 1.3] | Normal | 1 |
|  | Above 1.3 | High | 2 |
| `Fibrinogen` | [0, 99] | Very Low | 0 |
|  | (99, 199] | Low | 1 |
|  | (199, 400] | Normal | 2 |
|  | (400, 600] | High | 3 |
|  | Above 600 | Very High | 4 |
| `LDH` | [0, 99] | Very Low | 0 |
|  | (99, 139] | Low | 1 |
|  | (139, 280] | Normal | 2 |
|  | (280, 500] | High | 3 |
|  | (500, 1000] | Very High | 4 |
|  | Above 1000 | Critical | 5 |
| `Hematocrit (serum)` | [0, 35] | Low | 0 |
|  | (35, 51] | Normal | 1 |
|  | Above 51 | High | 2 |




### TABLE E: POSITIVE PREDICTION RULE LIST (CONDITIONS INCLUDE ORDINAL ENCODINGS – SEE TABLE D)

| Rule Number | Rule | Eval. Coverage | Eval. Accuracy |
|---|---|---|---|
| 0 | arterial_blood_pressure_diastolic_diff_0 > -0.15532485395669937 & gcs_motor_response_0 > 0.5 & gcs_verbal_response_max <= 3.5 & heart_rate_diff_0 > -3.25 & sodium_serum_mean <= 1.5 & wbc_range > 0.05000000074505806 | 0.13 | 0.77 |
| 1 | arterial_blood_pressure_mean_diff_0 <= -0.6102604568004608 & glucose_serum_range > 0.5 & magnesium_3 > 0.5 & wbc_range > 0.05000000074505806 | 0.14 | 0.80 |
| 2 | alkaline_phosphate_diff_18 <= 0.5181298553943634 & arterial_blood_pressure_diastolic_diff_0 > 0.6847023963928223 | 0.11 | 0.83 |
| 3 | arterial_blood_pressure_mean_diff_10 > -0.3292434811592102 & arterial_blood_pressure_systolic_range <= 11.286303520202637 & glucose_serum_range > 0.5 & magnesium_3 > 0.5 & wbc_range > 0.05000000074505806 | 0.07 | 0.84 |
| 4 | arterial_blood_pressure_diastolic_diff_0 <= -0.38190382719039917 & gcs_verbal_response_max <= 3.5 | 0.08 | 0.89 |
| 5 | glucose_serum_range <= 126.5 & glucose_serum_range > 7.5 & heart_rate_diff_0 > 0.75 & hosp_to_icu > 20.153611183166504 & phosphorous_range > 0.29550568759441376 | 0.05 | 0.81 |
| 6 | bun_diff_0 <= 0.7816289961338043 & c_reactive_protein_crp_diff_12 > -0.13433531671762466 & gcs_verbal_response_max <= 3.5 & glucose_serum_range <= 89.0 & hosp_to_icu > 0.02236111182719469 & respiratory_rate_range <= 12.5 | 0.11 | 0.62 |
| 7 | arterial_blood_pressure_mean_14 > 1.5 & glucose_serum_diff_2 <= 14.0 & hematocrit_serum_diff_2 > -0.2696501985192299 & sodium_serum_mean > 0.5 | 0.04 | 0.84 |
| 8 | arterial_blood_pressure_systolic_range <= 21.305910110473633 & gcs_motor_response_0 > 0.5 & gcs_verbal_response_max <= 3.5 & magnesium_3 > 0.5 | 0.07 | 0.74 |
| 9 | arterial_blood_pressure_systolic_diff_0 > 1.998100221157074 & differential_lymphs_diff_0 > -0.45104674994945526 & differential_lymphs_diff_1 <= 0.2887251079082489 & gcs_verbal_response_max <= 3.5 | 0.10 | 0.81 |
| 10 | arterial_blood_pressure_systolic_diff_0 <= -0.019208671525120735 & glucose_serum_range > 0.5 & phosphorous_range <= 2.850000023841858 & sodium_serum_mean > 1.5 & wbc_range > 0.05000000074505806 | 0.02 | 1.00 |
| 11 | glucose_serum_range <= 96.5 & hosp_to_icu <= 0.7330555617809296 & phosphorous_range > 0.29550568759441376 & respiratory_rate_range > 13.5 & sodium_serum_mean > 0.5 | 0.04 | 0.68 |
| 12 | arterial_blood_pressure_systolic_diff_0 > 0.017756862565875053 & creatinine_serum_mean <= 1.5 & glucose_serum_range > 0.5 & hosp_to_icu > 42.49819564819336 & magnesium_3 > 0.5 | 0.05 | 0.85 |
| 13 | arterial_blood_pressure_mean_diff_10 <= -0.7506462037563324 & gcs_motor_response_0 > 3.5 & gcs_verbal_response_max > 3.5 & glucose_serum_range <= 64.5 & hosp_to_icu > 20.957361221313477 & phosphorous_range > 0.29550568759441376 | 0.01 | 0.50 |
| 14 | differential_lymphs_diff_0 <= -0.00814008922316134 & gcs_verbal_response_23 <= 3.5 & magnesium_3 > 0.5 & wbc_range > 0.07500000111758709 | 0.08 | 0.74 |
| 15 | arterial_blood_pressure_systolic_diff_0 > 2.0778696537017822 & creatinine_serum_mean <= 1.5 & glucose_serum_range > 0.5 & hosp_to_icu <= 1.5561110973358154 & wbc_range > 0.05000000074505806 | 0.07 | 0.88 |
| 16 | sodium_serum_diff_1 <= -0.5233579259365797 | 0.02 | 0.88 |
| 17 | arterial_blood_pressure_diastolic_diff_0 > -0.433398112654686 & differential_lymphs_diff_22 <= -0.5721820592880249 & glucose_serum_range <= 0.5 & sodium_serum_mean > 0.5 | 0.00 | 0.00 |
| 18 | arterial_blood_pressure_systolic_range <= 21.305910110473633 & c_reactive_protein_crp_diff_12 > -0.5088772773742676 & fibrinogen_14 > 2.5 & gcs_verbal_response_23 > 3.5 & magnesium_3 > 0.5 & phosphorous_range > 0.19464480876922607 | 0.03 | 0.75 |
| 19 | glucose_serum_diff_2 > 14.0 | 0.01 | 0.75 |
| 20 | hematocrit_serum_diff_2 <= -0.2696501985192299 & phosphorous_range <= 2.100000023841858 | 0.01 | 1.00 |
| 21 | alkaline_phosphate_diff_18 <= 0.19156409054994583 & arterial_blood_pressure_mean_diff_0 > 1.4841545224189758 & gcs_motor_response_0 > 2.5 & wbc_range > 0.05000000074505806 | 0.08 | 0.79 |
| 22 | arterial_blood_pressure_mean_diff_10 > -1.584976077079773 & gcs_motor_response_0 > 2.5 & ldh_diff_5 > 46.9898681640625 & phosphorous_range > 0.29550568759441376 | 0.01 | 0.60 |
| 23 | c_reactive_protein_crp_diff_12 > -0.13433531671762466 & gcs_motor_response_0 > 2.5 & glucose_serum_range > 7.5 & hosp_to_icu > 20.153611183166504 & phosphorous_range > 0.29550568759441376 & wbc_range <= 2.649999976158142 & wbc_range > 0.949999988079071 | 0.02 | 0.89 |
| 24 | glucose_serum_range <= 7.5 & hosp_to_icu <= 0.7330555617809296 & phosphorous_range > 0.29550568759441376 | 0.02 | 0.71 |
| 25 | arterial_blood_pressure_systolic_diff_0 > 1.998100221157074 & differential_lymphs_diff_0 > -0.45104674994945526 & ldh_min > 3.5 | 0.03 | 0.77 |
| 26 | arterial_blood_pressure_systolic_diff_20 <= 0.3070981949567795 & arterial_blood_pressure_systolic_diff_20 > 0.07076873257756233 & heart_rate_diff_0 <= 0.75 & hosp_to_icu > 42.49819564819336 | 0.00 | 0.00 |
| 27 | arterial_blood_pressure_diastolic_diff_2 > 0.17599450051784515 & glucose_serum_range <= 126.5 & glucose_serum_range > 0.5 & heart_rate_diff_0 > 0.75 | 0.07 | 0.90 |
| 28 | creatinine_serum_mean <= 1.5 & gcs_verbal_response_23 <= 3.5 & heart_rate_diff_0 > 0.75 & phosphorous_range > 0.09500899538397789 | 0.11 | 0.85 |
| 29 | arterial_blood_pressure_diastolic_diff_2 > 0.11173246800899506 & arterial_blood_pressure_systolic_range > 11.286303520202637 & gcs_verbal_response_max <= 3.5 & magnesium_3 > 0.5 & respiratory_rate_range <= 12.5 | 0.05 | 0.71 |
| 30 | arterial_blood_pressure_diastolic_diff_2 > 0.11173246800899506 & arterial_blood_pressure_systolic_range > 11.286303520202637 & c_reactive_protein_crp_diff_12 > -0.13433531671762466 & gcs_motor_response_0 > 2.5 & hosp_to_icu > 5.8044445514678955 & magnesium_3 > 0.5 & wbc_range <= 0.949999988079071 & wbc_range > 0.05000000074505806 | 0.01 | 1.00 |
| 31 | bun_diff_0 > 0.7816289961338043 | 0.03 | 1.00 |
| 32 | heart_rate_diff_0 <= -6.5 & sodium_serum_mean > 1.5 | 0.02 | 0.88 |
| 33 | differential_lymphs_diff_1 <= -0.8960761725902557 & phosphorous_range <= 0.29550568759441376 | 0.00 | 0.50 |
| 34 | arterial_blood_pressure_systolic_diff_20 <= -7.446872234344482 & glucose_serum_range > 0.5 & hosp_to_icu > 42.49819564819336 | 0.03 | 0.91 |
| 35 | arterial_blood_pressure_diastolic_diff_0 > 0.5210173428058624 & arterial_blood_pressure_mean_diff_0 <= -0.6102604568004608 & glucose_serum_range <= 0.5 | 0.00 | 0.00 |
| 36 | gcs_verbal_response_max > 3.5 & ldh_min > 3.5 & phosphorous_range > 3.049999952316284 | 0.00 | n/a |
| 37 | arterial_blood_pressure_systolic_diff_0 <= -0.019208671525120735 & heart_rate_diff_0 > -0.25 & hosp_to_icu <= 0.7330555617809296 | 0.01 | 1.00 |
| 38 | c_reactive_protein_crp_diff_12 > 10.18821907043457 & hosp_to_icu > 42.49819564819336 & magnesium_3 > 0.5 | 0.02 | 0.60 |
| 39 | arterial_blood_pressure_systolic_range <= 10.93578290939331 & heart_rate_diff_0 <= -3.25 & ldh_diff_0 <= -3.455755352973938 & sodium_serum_mean <= 1.5 | 0.00 | n/a |
| 40 | gcs_verbal_response_23 <= 3.5 & heart_rate_diff_0 > 0.75 & ldh_min > 2.5 & sodium_serum_mean > 0.5 | 0.08 | 0.82 |
| 41 | arterial_blood_pressure_systolic_diff_0 > 2.0778696537017822 & arterial_blood_pressure_systolic_diff_20 <= -4.168562650680542 & bun_diff_0 <= 0.7816289961338043 & c_reactive_protein_crp_diff_12 <= -0.13433531671762466 & glucose_serum_range > 0.5 & wbc_range > 0.05000000074505806 | 0.03 | 0.85 |

### TABLE F: NEGATIVE PREDICTION RULE LIST (CONDITIONS INCLUDE ORDINAL ENCODINGS – SEE TABLE D)

| Rule Number | Rule | Eval. Coverage | Eval. Accuracy |
|---|---|---|---|
| 0 | arterial_blood_pressure_diastolic_diff_0 <= 0.556828111410141 & gcs_verbal_response_max > 3.5 & glucose_serum_range <= 0.5 | 0.18 | 0.90 |
| 1 | fibrinogen_14 <= 2.5 & gcs_verbal_response_23 > 3.5 & glucose_serum_diff_0 > -7.711022615432739 & glucose_serum_range > 0.5 & wbc_range <= 0.05000000074505806 | 0.06 | 0.73 |
| 2 | arterial_blood_pressure_systolic_diff_0 <= 5.937050104141235 & bun_diff_1 <= 1.5 & gcs_verbal_response_max > 3.5 & hematocrit_serum_diff_2 > -0.2696501985192299 & hosp_to_icu <= 6.144166707992554 & sodium_serum_mean <= 0.5 | 0.06 | 0.65 |
| 3 | arterial_blood_pressure_diastolic_diff_0 <= 0.6847023963928223 & arterial_blood_pressure_mean_diff_0 > -0.6102604568004608 & arterial_blood_pressure_systolic_range > 11.286303520202637 & differential_lymphs_diff_1 <= 0.1265718713402748 & gcs_motor_response_0 <= 2.5 & respiratory_rate_range > 4.5 | 0.07 | 0.84 |
| 4 | arterial_blood_pressure_systolic_diff_0 <= 5.937050104141235 & arterial_blood_pressure_systolic_diff_20 > -4.658319473266602 & gcs_verbal_response_max > 3.5 & glucose_serum_range > 0.5 & hematocrit_serum_diff_2 > -0.2696501985192299 & magnesium_3 <= 0.5 | 0.03 | 0.83 |
| 5 | arterial_blood_pressure_systolic_range > 10.93578290939331 & gcs_verbal_response_max > 3.5 & heart_rate_diff_0 <= 0.75 & hematocrit_serum_diff_2 > -0.2629629597067833 & phosphorous_range <= 0.29550568759441376 & wbc_range > 1.75 | 0.07 | 0.93 |
| 6 | arterial_blood_pressure_systolic_range > 10.93578290939331 & creatinine_serum_mean > 1.5 & gcs_verbal_response_max > 3.5 & heart_rate_diff_0 <= 0.75 & hosp_to_icu <= 21.078055381774902 & ldh_min <= 3.5 | 0.08 | 0.84 |
| 7 | arterial_blood_pressure_diastolic_diff_0 <= 0.5210173428058624 & arterial_blood_pressure_mean_diff_0 > -0.09405163303017616 & arterial_blood_pressure_systolic_diff_20 > -4.168562650680542 & arterial_blood_pressure_systolic_range > 19.89540672302246 & bun_diff_0 <= 0.7816289961338043 & c_reactive_protein_crp_diff_12 <= -0.13433531671762466 & gcs_verbal_response_max <= 3.5 & heart_rate_diff_0 <= 0.75 & phosphorous_range <= 0.19464480876922607 | 0.01 | 0.40 |
| 8 | arterial_blood_pressure_mean_diff_0 > -0.2466215044260025 & gcs_verbal_response_max > 3.5 & glucose_serum_range > 64.5 & hosp_to_icu > 0.00680555566214025 & phosphorous_range > 0.29550568759441376 & respiratory_rate_range > 13.25 | 0.02 | 1.00 |
| 9 | arterial_blood_pressure_mean_diff_10 <= 4.3310863971710205 & arterial_blood_pressure_mean_diff_10 > 0.12509019672870636 & arterial_blood_pressure_systolic_diff_0 <= 2.2245867252349854 & differential_lymphs_diff_0 > -0.4370129555463791 & glucose_serum_range > 0.5 & wbc_range <= 0.05000000074505806 | 0.03 | 0.92 |
| 10 | arterial_blood_pressure_mean_diff_10 > 0.12304593995213509 & arterial_blood_pressure_systolic_diff_0 <= 1.998100221157074 & differential_lymphs_diff_0 > -0.45104674994945526 & fibrinogen_14 <= 2.5 & gcs_verbal_response_23 > 3.5 & phosphorous_range > 0.19464480876922607 & respiratory_rate_range <= 13.25 & sodium_serum_mean > 0.5 & wbc_range > 0.05000000074505806 | 0.01 | 0.67 |
| 11 | arterial_blood_pressure_diastolic_diff_2 <= 0.15165791660547256 & arterial_blood_pressure_systolic_diff_20 > 0.3070981949567795 & c_reactive_protein_crp_diff_12 <= -0.13433531671762466 & gcs_motor_response_0 > 4.5 & gcs_verbal_response_max > 3.5 & heart_rate_diff_0 <= 0.75 & hosp_to_icu > 0.7330555617809296 & wbc_range > 0.05000000074505806 | 0.03 | 0.67 |
| 12 | arterial_blood_pressure_diastolic_diff_0 > -0.433398112654686 & glucose_serum_range > 99.5 & hosp_to_icu <= 42.49819564819336 & hosp_to_icu > 0.7330555617809296 | 0.03 | 0.67 |
| 13 | gcs_motor_response_0 <= 4.5 & gcs_verbal_response_max > 3.5 | 0.10 | 0.90 |
| 14 | arterial_blood_pressure_diastolic_diff_0 <= -0.19905917346477509 & arterial_blood_pressure_diastolic_diff_0 > -0.433398112654686 & phosphorous_range <= 0.19464480876922607 | 0.06 | 0.79 |
| 15 | alkaline_phosphate_diff_18 > 0.5181298553943634 & arterial_blood_pressure_diastolic_diff_0 > 0.6847023963928223 & arterial_blood_pressure_systolic_diff_20 <= 7.5 & glucose_serum_range <= 0.5 | 0.00 | 0.00 |
| 16 | arterial_blood_pressure_diastolic_diff_2 <= 0.17599450051784515 & arterial_blood_pressure_systolic_diff_0 <= 5.937050104141235 & creatinine_serum_mean > 1.5 & gcs_verbal_response_max > 3.5 & glucose_serum_range <= 126.5 & heart_rate_diff_0 > 0.75 & hematocrit_serum_diff_2 > -0.2696501985192299 & hosp_to_icu <= 21.078055381774902 & ldh_min <= 3.5 | 0.01 | 0.80 |
| 17 | arterial_blood_pressure_systolic_diff_0 <= 3.054964065551758 & arterial_blood_pressure_systolic_range > 21.28965950012207 & differential_lymphs_diff_0 <= 0.3315500319004059 & differential_lymphs_diff_1 > -0.8960761725902557 & gcs_verbal_response_max > 3.5 & glucose_serum_range > 0.5 & hosp_to_icu <= 21.10347270965576 & hosp_to_icu > 1.5561110973358154 & phosphorous_range <= 0.29550568759441376 | 0.01 | 0.67 |
| 18 | arterial_blood_pressure_diastolic_diff_2 <= -0.017171474173665047 & c_reactive_protein_crp_diff_12 <= -2.0300424098968506 & differential_lymphs_diff_22 > 0.026072008535265923 & hosp_to_icu > 42.49819564819336 | 0.01 | 0.33 |
| 19 | arterial_blood_pressure_mean_diff_0 <= 1.4841545224189758 & arterial_blood_pressure_mean_diff_0 > -0.6102604568004608 & hosp_to_icu > 1.9431944489479065 & magnesium_3 <= 0.5 | 0.02 | 0.89 |
| 20 | arterial_blood_pressure_mean_diff_10 <= 4.3310863971710205 & creatinine_serum_mean > 1.5 & glucose_serum_range > 0.5 & hosp_to_icu <= 21.078055381774902 & ldh_min <= 3.5 & wbc_range <= 0.05000000074505806 | 0.03 | 1.00 |
| 21 | arterial_blood_pressure_systolic_diff_0 <= 0.017756862565875053 & arterial_blood_pressure_systolic_range <= 21.305910110473633 & arterial_blood_pressure_systolic_range > 0.6965968608856201 & creatinine_serum_mean <= 1.5 & fibrinogen_14 <= 2.5 & gcs_verbal_response_23 > 3.5 & magnesium_3 > 0.5 | 0.01 | 0.75 |
| 22 | arterial_blood_pressure_diastolic_diff_0 <= 0.6847023963928223 & arterial_blood_pressure_diastolic_diff_0 > -0.433398112654686 & arterial_blood_pressure_systolic_range > 11.286303520202637 & fibrinogen_14 <= 2.5 & gcs_verbal_response_23 > 3.5 & heart_rate_diff_0 <= -3.25 & respiratory_rate_range > 13.25 & sodium_serum_mean <= 1.5 | 0.03 | 0.87 |
| 23 | arterial_blood_pressure_mean_diff_0 > -0.6102604568004608 & arterial_blood_pressure_systolic_range > 16.510486602783203 & differential_lymphs_diff_0 > -0.00814008922316134 & fibrinogen_14 > 2.5 & gcs_verbal_response_23 <= 3.5 & gcs_verbal_response_max > 3.5 & hosp_to_icu <= 279.60137939453125 | 0.01 | 0.75 |
| 24 | arterial_blood_pressure_diastolic_diff_2 <= 0.15165791660547256 & arterial_blood_pressure_mean_diff_10 > 0.12304593995213509 & arterial_blood_pressure_systolic_diff_0 <= 1.998100221157074 & c_reactive_protein_crp_diff_12 <= -0.13433531671762466 & differential_lymphs_diff_0 > -0.45104674994945526 & fibrinogen_14 <= 2.5 & gcs_verbal_response_max > 3.5 & hosp_to_icu > 0.7330555617809296 | 0.04 | 0.89 |
| 25 | differential_lymphs_diff_1 > -0.04483931139111519 & glucose_serum_range <= 0.5 & hematocrit_serum_4 > 0.5 & hosp_to_icu <= 0.7330555617809296 | 0.02 | 0.70 |
| 26 | glucose_serum_diff_2 <= 14.0 & hosp_to_icu <= 6.144166707992554 & magnesium_3 <= 0.5 & respiratory_rate_range > 6.25 & sodium_serum_mean <= 0.5 | 0.01 | 1.00 |
| 27 | gcs_verbal_response_max <= 3.5 & glucose_serum_range <= 7.5 & phosphorous_range > 0.29550568759441376 & respiratory_rate_range > 12.5 & wbc_range > 5.049999952316284 | 0.00 | 1.00 |
| 28 | gcs_motor_response_0 > 4.5 & gcs_verbal_response_max > 3.5 & hosp_to_icu <= 21.10347270965576 & hosp_to_icu > 1.5561110973358154 & wbc_range <= 0.05000000074505806 | 0.06 | 0.71 |
| 29 | arterial_blood_pressure_mean_diff_10 > 0.12509019672870636 & differential_lymphs_diff_0 > -0.4370129555463791 & gcs_verbal_response_max <= 3.5 & phosphorous_range <= 0.19464480876922607 & respiratory_rate_range > 12.5 & wbc_range > 2.25 | 0.00 | 0.00 |
| 30 | arterial_blood_pressure_diastolic_diff_0 > -0.433398112654686 & arterial_blood_pressure_systolic_range > 10.93578290939331 & c_reactive_protein_crp_diff_12 > -0.13433531671762466 & fibrinogen_14 <= 2.5 & gcs_verbal_response_23 > 3.5 & heart_rate_diff_0 <= 0.75 & hosp_to_icu <= 5.8044445514678955 & hosp_to_icu > 0.7330555617809296 & respiratory_rate_range > 13.25 | 0.03 | 0.75 |
| 31 | differential_lymphs_diff_0 <= -0.4370129555463791 & phosphorous_range > 2.350000023841858 | 0.00 | n/a |
| 32 | magnesium_3 <= 0.5 & phosphorous_range <= 0.19464480876922607 | 0.03 | 1.00 |
| 33 | arterial_blood_pressure_diastolic_diff_0 <= 0.5210173428058624 & arterial_blood_pressure_systolic_21 <= 2.5 & ldh_min > 3.5 & phosphorous_range <= 0.19464480876922607 | 0.01 | 1.00 |
| 34 | differential_lymphs_diff_1 <= -0.1845027208328247 & heart_rate_diff_0 > 0.75 & hosp_to_icu <= 0.7330555617809296 & respiratory_rate_range > 13.5 | 0.01 | 0.33 |
| 35 | alkaline_phosphate_diff_18 <= -4.786617755889893 & arterial_blood_pressure_mean_diff_10 <= -1.584976077079773 & gcs_motor_response_0 > 2.5 & hosp_to_icu <= 21.10347270965576 | 0.00 | n/a |
| 36 | glucose_serum_range <= 7.5 & phosphorous_range > 0.29550568759441376 & sodium_serum_mean <= 0.5 & wbc_range > 5.049999952316284 | 0.00 | 1.00 |
| 37 | hosp_to_icu <= 7.390139102935791 & phosphorous_range <= 0.29550568759441376 & phosphorous_range > 0.19464480876922607 & sodium_serum_mean <= 0.5 | 0.00 | 0.00 |
| 38 | fibrinogen_14 > 2.5 & gcs_verbal_response_23 > 3.5 & glucose_serum_range > 79.5 & hosp_to_icu <= 42.49819564819336 & hosp_to_icu > 0.7330555617809296 | 0.01 | 0.33 |
| 39 | fibrinogen_14 <= 2.5 & gcs_verbal_response_23 > 3.5 & heart_rate_diff_0 <= 1.75 & heart_rate_diff_0 > 0.75 & hosp_to_icu <= 0.7330555617809296 | 0.00 | 1.00 |
| 40 | creatinine_serum_mean > 1.5 & gcs_verbal_response_max <= 3.5 & hosp_to_icu > 21.078055381774902 & ldh_diff_0 <= -8.246402978897095 & ldh_min <= 3.5 & wbc_range <= 0.05000000074505806 | 0.00 | n/a |
| 41 | arterial_blood_pressure_diastolic_diff_0 <= 0.5210173428058624 & arterial_blood_pressure_diastolic_diff_2 <= 0.15165791660547256 & arterial_blood_pressure_systolic_range > 5.607791900634766 & c_reactive_protein_crp_diff_12 <= -0.13433531671762466 & glucose_serum_range <= 0.5 & hosp_to_icu > 0.7330555617809296 | 0.05 | 0.85 |
| 42 | arterial_blood_pressure_systolic_range <= 38.798622131347656 & differential_lymphs_diff_0 > -0.00814008922316134 & gcs_verbal_response_23 <= 3.5 & heart_rate_diff_0 > 0.75 & ldh_min <= 2.5 & wbc_range <= 0.05000000074505806 | 0.01 | 0.75 |
| 43 | arterial_blood_pressure_mean_14 <= 1.5 & arterial_blood_pressure_mean_diff_10 > -0.3292434811592102 & arterial_blood_pressure_systolic_range > 19.89540672302246 & glucose_serum_range > 0.5 & phosphorous_range <= 0.19464480876922607 & wbc_range <= 0.05000000074505806 | 0.02 | 0.57 |
| 44 | arterial_blood_pressure_mean_diff_10 <= -0.3292434811592102 & gcs_motor_response_0 > 4.5 & glucose_serum_diff_2 <= 14.0 & glucose_serum_range > 0.5 & phosphorous_range <= 0.19464480876922607 & sodium_serum_mean <= 0.5 | 0.01 | 0.50 |
| 45 | fibrinogen_14 <= 2.5 & gcs_verbal_response_23 <= 3.5 & gcs_verbal_response_max > 3.5 & heart_rate_diff_0 <= -3.25 & ldh_min <= 2.5 & sodium_serum_mean <= 1.5 | 0.00 | 0.50 |
| 46 | arterial_blood_pressure_mean_diff_10 <= 0.09674116969108582 & arterial_blood_pressure_mean_diff_10 > -0.7506462037563324 & arterial_blood_pressure_systolic_range > 0.6965968608856201 & c_reactive_protein_crp_diff_12 > -0.13433531671762466 & creatinine_serum_mean <= 1.5 & fibrinogen_14 <= 2.5 & gcs_verbal_response_23 > 3.5 & gcs_verbal_response_max > 3.5 & glucose_serum_range <= 64.5 & hosp_to_icu <= 5.8044445514678955 & hosp_to_icu > 0.7330555617809296 & ldh_min <= 3.5 & phosphorous_range > 0.29550568759441376 & respiratory_rate_range <= 13.25 | 0.00 | 0.50 |

