import pandas as pd

def compress():
    chartevents = pd.read_csv("data_raw/chartevents.csv")
    chartevents.to_csv("data_raw/chartevents.csv.gz", index=False, compression='gzip')

    diagnosis = pd.read_csv("data_raw_eicu_v2.0/diagnosis.csv")
    diagnosis.to_csv("data_raw_eicu_v2.0/diagnosis.csv.gz", index=False, compression='gzip')

    patient = pd.read_csv("data_raw_eicu_v2.0/patient.csv")
    patient.to_csv("data_raw_eicu_v2.0/patient.csv.gz", index=False, compression='gzip')

    lab = pd.read_csv("data_raw_eicu_v2.0/lab.csv")
    lab.to_csv("data_raw_eicu_v2.0/lab.csv.gz", index=False , compression='gzip')

    nurseCharting = pd.read_csv("data_raw_eicu_v2.0/nurseCharting.csv")
    nurseCharting.to_csv("data_raw_eicu_v2.0/nurseCharting.csv.gz", index=False, compression='gzip')

