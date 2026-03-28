## ADDITIONAL POSTPORCESSING SPECIALTY, DIAGNOSIS, TRIAGE

## import libraries
import pandas as pd


## Load Data from postprocess_specialty_prediction.py, postprocess_diagnosis_prediction.py and postprocess_triage_prediction.py
spec = pd.read_csv("MIMIC-IV-Ext-Specialty-prediction.csv")
diag = pd.read_csv("MIMIC-IV-Ext-Diagnosis-prediction.csv")
triage = pd.read_csv("MIMIC-IV-Ext-Diagnosis-Triage-prediction.csv")


## Convert the specialty rows into lists - data in columns are stored as strings but actually represent lists
spec['specialty_primary_diagnosis'] = spec['specialty_primary_diagnosis'].apply(lambda x: eval(x))


## delete empty specialties and initial vitals
mask1 = spec["specialty_primary_diagnosis"].str.len() == 0
mask2 = spec["initial_vitals"].isna()
mask = pd.Index(mask1 | mask2)
spec = spec[~mask]
diag = diag[~mask]
triage = triage[~mask]

## delete these row due to no possible output from the LLM
spec = spec.drop([795,2176,1208], inplace=False)
diag = diag.drop([795,2176,1208], inplace=False)
triage = triage.drop([795,2176,1208], inplace=False)


## delete where specialty gt is "no answer"
mask = ~spec["specialty_primary_diagnosis"].apply(lambda x: any(item == "no answer" for item in x))
spec = spec[mask]
diag = diag[mask]
triage = triage[mask]


## convert triage/acuity to type int
triage["triage_Claude3.5"] = triage["triage_Claude3.5"].astype(int)
triage["triage_Claude3"] = triage["triage_Claude3"].astype(int)
triage["triage_Haiku"] = triage["triage_Haiku"].astype(int)
triage["triage_Claude3.5_Clinical"] = triage["triage_Claude3.5_Clinical"].astype(int)
triage["triage_Claude3_Clinical"] = triage["triage_Claude3_Clinical"].astype(int)
triage["triage_Haiku_Clinical"] = triage["triage_Haiku_Clinical"].astype(int)


## extract first 2000 values
spec = spec[:2000]
diag = diag[:2000]
triage = triage[:2000]


## save files
spec.to_csv('MIMIC-IV-Ext-Specialty-prediction.csv', index=False)
diag.to_csv('MIMIC-IV-Ext-Diagnosis-prediction.csv', index=False)
triage.to_csv('MIMIC-IV-Ext-Triage-prediction.csv', index=False)