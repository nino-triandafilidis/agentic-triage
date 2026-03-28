## Import Libraries
import gzip
import os
from pathlib import Path

import pandas as pd
import numpy as np
import re
from io import StringIO

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env.local", override=False)

 
# MIMIC-IV-Ext Creation
## Import Datasets from MIMIC-IV, MIMIC-IV-ED, MIMIC-IV-Note
def _first_existing_path(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def _first_existing_file(candidates):
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(f"None of these files exist: {candidates}")


default_mimic_root = "/data/local/llm-evaluation"
mimic_root = os.environ.get("MIMIC_DATA_ROOT", default_mimic_root)

## Load from MIMIC-IV-ED
ed_dir = _first_existing_path([
    os.environ.get("MIMIC_IV_ED_DIR"),
    "/data/local/llm-evaluation/mimic-iv-ed-2.2/ed",
    os.path.join(mimic_root, "mimic-iv-ed-2.2", "ed"),
])

triage = pd.read_csv(_first_existing_file([
    os.path.join(ed_dir, "triage.csv"),
    os.path.join(ed_dir, "triage.csv.gz"),
]), on_bad_lines='skip', low_memory=False)

ed_stays = pd.read_csv(_first_existing_file([
    os.path.join(ed_dir, "edstays.csv"),
    os.path.join(ed_dir, "edstays.csv.gz"),
]))

diagnostics = pd.read_csv(_first_existing_file([
    os.path.join(ed_dir, "diagnosis.csv"),
    os.path.join(ed_dir, "diagnosis.csv.gz"),
]), on_bad_lines='skip')

## Load from MIMIC-IV
patients_path = _first_existing_file([
    os.environ.get("MIMIC_IV_PATIENTS_FILE"),
    "/data/local/llm-evaluation/mimic-iv/mimic-iv-3.0/hosp/patients.csv.gz",
    os.path.join(mimic_root, "mimiciv", "physionet.org", "files", "mimiciv", "3.0", "hosp", "patients.csv.gz"),
    os.path.join(mimic_root, "mimic-iv", "mimic-iv-3.0", "hosp", "patients.csv.gz"),
])
patients = pd.read_csv(patients_path, low_memory=False)


## Load Discharge from MIMIC-IV-Note
discharge_path = _first_existing_file([
    os.environ.get("MIMIC_IV_NOTE_DISCHARGE_FILE"),
    "/data/local/llm-evaluation/mimic-iv-note/discharge.csv",
    os.path.join(mimic_root, "mimic-iv-note", "2.2", "note", "discharge.csv.gz"),
    os.path.join(mimic_root, "mimic-iv-note", "discharge.csv"),
])
try:
    discharge = pd.read_csv(discharge_path, low_memory=False)
except Exception:
    # Fallback path for legacy discharge exports with malformed multiline notes.
    if discharge_path.endswith(".gz"):
        with gzip.open(discharge_path, mode="rt", encoding="utf-8") as f:
            txt = f.read()
    else:
        with open(discharge_path, encoding="utf-8") as f:
            txt = f.read()

    txt = txt.replace('|', ',<vl>')
    txt = txt.replace(',""""\n', ',<br>')
    txt = txt.replace('Followup Instructions:\n___\n""""', 'Followup Instructions:\n___\n</br>|')
    txt = re.sub(r'<br>([^<]*)</br>', lambda x: x.group(0).replace(',', '<comma>'), txt)
    txt = txt.replace('"', '')
    txt = txt.replace('text\n', 'text|')
    discharge = pd.read_csv(StringIO(txt), lineterminator='|', on_bad_lines='skip')

 
## Select MIMIC-IV-Note cases only present in MIMIC-IV-ED dataset
## Add "stay_id" and "text" from edstays dataset
for index, row in discharge.iterrows():
    try:
        hadm_id = float(row['hadm_id'])
        # Find the corresponding 'stay_id' in 'ed_stays' DataFrame that matches the 'hadm_id'
        stay_id = ed_stays[ed_stays['hadm_id'] == hadm_id]['stay_id']

        # If no matching 'stay_id' is found, skip to the next iteration
        if stay_id.empty:
            continue

        discharge.at[index, 'stay_id'] = stay_id.iloc[0]
        
    except Exception as e:
        #print(f"{e} at {index}")
        continue

discharge = discharge[discharge['stay_id'].notnull()]
discharge[['subject_id', 'hadm_id', 'note_seq']] = discharge[['subject_id', 'hadm_id', 'note_seq']].astype(int)

 
## Merge all datasets
df=pd.merge(triage,discharge,on=["subject_id", "stay_id"],how="inner")
df=pd.merge(df,ed_stays,on=["subject_id", "stay_id", "hadm_id"],how="inner")
df = df.drop_duplicates(subset=['subject_id'])
df=pd.merge(df,patients.drop("gender",axis=1),on=["subject_id"],how="inner")
df=pd.merge(df,diagnostics[diagnostics["seq_num"] == 1],on=["subject_id", "stay_id"],how="inner")
df = df.dropna(subset=['icd_code'])

df = df.drop(columns=["note_id", "note_type", "note_seq", "charttime", "storetime", "intime", "outtime", "anchor_year", "anchor_year_group", "dod" ])


 
## Extract Relevant Information from the Clinical Text
#### Extract tests, past medication and HPI (to be continued and refined later on in the code)
def get_tests(text):
    lower_text = text.lower()
    try:
        if "discharge labs" in lower_text.split("pertinent results:")[1].split('brief hospital course:')[0]:
            return lower_text.split("pertinent results:")[1].split('brief hospital course:')[0].split('discharge labs')[0]
        else:
            return lower_text.split("pertinent results:")[1].split('brief hospital course:')[0]
    except:
        return None
    
def get_medication(text):
    lower_text = text.lower()
    try:
        # Extract the text between "medications on admission:" and "discharge medications:"
        return lower_text.split("medications on admission:")[1].split('discharge medications:')[0]
    except:
        # print(lower_text)
        return None

def get_HPI(text):
    # Replace custom placeholders with their intended characters and clean up text markers
    text = text.replace('<comma>', ',').replace('<br>', '').replace('</br>', '')
    
    # Extract the text between "History of Present Illness:" and "Physical Exam:" sections
    text = text.split('History of Present Illness:')[-1].split('Physical Exam:')[0]
    return text

df["tests"] = df['text'].apply(get_tests)
df["past_medication"] = df['text'].apply(get_medication)
df['preprocessed_text'] = df['text'].apply(get_HPI)

 
#### Create Initial Vitals from Temperature, Heartrate, respiration rate, o2 saturation, bloodpressure (dbp, sbp)
def create_vitals(row):
    vitals = []
    
    if not pd.isna(row['temperature']):
        vitals.append(f"Temperature: {row['temperature']}")
    if not pd.isna(row['heartrate']):
        vitals.append(f"Heartrate: {row['heartrate']}")
    if not pd.isna(row['resprate']):
        vitals.append(f"resprate: {row['resprate']}")
    if not pd.isna(row['o2sat']):
        vitals.append(f"o2sat: {row['o2sat']}")
    if not pd.isna(row['sbp']):
        vitals.append(f"sbp: {row['sbp']}")   
    if not pd.isna(row['dbp']):
        vitals.append(f"dbp: {row['dbp']}") 
    
    return ", ".join(vitals)

df['initial_vitals'] = df.apply(create_vitals, axis=1)

 
#### Create Patient Info from Gender, Race and Year
def create_patient_info(row):
    patient_info = []
    
   # Append the gender information with a readable format
    if row["gender"] == "F":
        patient_info.append("Gender: Female")
    elif row["gender"] == "M":
        patient_info.append("Gender: Male")
    else:
        patient_info.append(f"Gender: {row['gender']}")

    patient_info.append(f"Race: {row['race']}")
    patient_info.append(f"Age: {row['anchor_age']}")
    
    return ", ".join(patient_info)

df['patient_info'] = df.apply(create_patient_info, axis=1)
## Drop columns that are no longer needed for the analysis or further processing and rearrange columns
df = df.drop(columns=["gender", "race", "anchor_age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"])

 
#### Cleaning and organizing the DataFrame for clarity
df = df[['stay_id', 'subject_id', 'hadm_id', "text", 'patient_info', 'initial_vitals', 'pain', 'chiefcomplaint', 'preprocessed_text', 'past_medication', 'tests', 'acuity', 'icd_code', 'icd_title', 'icd_version', 'arrival_transport', 'disposition']]

## remove rows that have nans in acuity, because acuity will be predicted and NaNs carry no useful information
df = df.dropna(subset=['acuity'])
df = df.dropna(subset=['tests'])

## convert nans to empty strings
df["pain"] = df['pain'].fillna("")
df["chiefcomplaint"] = df['chiefcomplaint'].fillna("")
df["past_medication"] = df['past_medication'].fillna("")

## convert numpy.float64 to numpy.int64
df['acuity'] = df['acuity'].astype(np.int64)
df['hadm_id'] = df['hadm_id'].astype(np.int64)
df['icd_version'] = df['icd_version'].astype(np.int64)

## rename acuity to triage
df = df.rename(columns={"acuity": "triage"})

## find the rows that have "history of present illness" in the "text" column and keep only these rows
hpi = df['text'].str.contains('history of present illness', case=False, na=False)
hpi_index = hpi[hpi==True].index
df = df.loc[hpi_index]

 
#### Extract HPI
def extract_hpi(text):
    pos_past_med_hist = text.lower().find('past medical history:')
    pos_soc_hist = text.lower().find('social history:')
    pos_fam_hist = text.lower().find('family history:')
    #text = text.replace("\n", " ")
    if pos_past_med_hist != -1:
        return text[:pos_past_med_hist].strip()
    elif pos_soc_hist != -1:
        return text[:pos_soc_hist].strip()
    elif pos_soc_hist != -1:
        return text[:pos_fam_hist].strip()
    else:
        return text

df["HPI"] = df["preprocessed_text"].apply(extract_hpi)

 
#### Extract Diagnosis
def extract_diagnosis(text):
    split_text = text.split("Discharge Diagnosis:" )[-1].split("Discharge Condition:")[0]
    split_text= split_text.replace('<comma>', ', ')
    return("Discharge Diagnosis: " + split_text)

df["diagnosis"] = df["text"].apply(extract_diagnosis)

 
#### Process HPI
## cut length of HPI <2000 and the tests <3000
string_lengths = df['HPI'].str.len()
mask = string_lengths<2000
df = df[mask]

string_lengths = df['HPI'].str.len()
mask = string_lengths>50
df = df[mask]

string_lengths = df['tests'].str.len()
mask = string_lengths<3000
df = df[mask]


## Removing Unwanted Sections Related to ED Course and Initial Vitals
df = df.dropna(subset=['HPI'])
df = df[df['HPI'] != ""]


## HPI preprocess
def extract_only_hpi(text):

    ## remove everything after
    #text = re.sub(re.compile("in the ED.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile(r"in the ED, initial vital.*", re.IGNORECASE | re.DOTALL), "", text)
    text = re.sub(re.compile(r"in the ED initial vital.*", re.IGNORECASE | re.DOTALL), "", text)
    text = re.sub(re.compile(r"\bED Course.*", re.IGNORECASE | re.DOTALL), "", text)
    text = re.sub(re.compile(r"\bIn ED initial VS.*", re.IGNORECASE | re.DOTALL), "", text)
    text = re.sub(re.compile(r"in the ED, initial VS.*", re.IGNORECASE | re.DOTALL), "", text)
    text = re.sub(re.compile(r"\binitial VS.*", re.IGNORECASE | re.DOTALL), "", text)
    text = re.sub(re.compile(r"in the ED.*", re.IGNORECASE | re.DOTALL), "", text)

    return text

tqdm.pandas()
df["HPI"] = df["HPI"].progress_apply(extract_only_hpi)

## Remove the ones that have ED in them
mask = df["HPI"].str.contains(r'\bED', case=False, na=False)
df = df[~mask]

## remove where test is nan to be able to compare between normal user and expert
df = df.dropna(subset=['tests'])

 
#### Process Diagnosis
#### Removing Specific Headers, Unwanted Sections, and Irrelevant Records
## remove the header "discharge diagnosis"
def remove_header(text, header):
    text = re.sub(re.compile(header, re.IGNORECASE), "", text)
    return text


## Remove Header in diagnosis "discharge diagnosis"
df['diagnosis'] = df['diagnosis'].apply(lambda text: remove_header(text, "discharge diagnosis:"))


## Remove all content before and including the "Facility:\n___" marker
def delete_before_string(text):
    text = re.sub(re.compile(r".*Facility:\n___", re.IGNORECASE | re.DOTALL), "", text)
    return text
df['diagnosis'] = df['diagnosis'].apply(delete_before_string)


## Remove all content before and including the "___ Diagnosis:" marker
def delete_before_string(text):
    text = re.sub(re.compile(r".*___ Diagnosis:", re.IGNORECASE | re.DOTALL), "", text)
    return text
df['diagnosis'] = df['diagnosis'].apply(delete_before_string)


## Remove all content after the "PMH" marker (Past Medical History)
def delete_after_string(text):
    text = re.sub(re.compile(r"PMH.*", re.IGNORECASE | re.DOTALL), "", text)
    return text
df['diagnosis'] = df['diagnosis'].apply(delete_after_string)

 
#### Filter Rows with Excessive Information to Preserve Prediction Integrity
# Filter out rows in 'HPI' that contain specific terms like 'ER', 'Emergency room', 'Emergency department', or 'impression'
# These rows likely refer to emergency settings and shouldn't be in the text for further analysis

mask = df["HPI"].str.contains(' ER ', case=False, na=False)
df = df[~mask]
mask = df["HPI"].str.contains('Emergency room', case=False, na=False)
df = df[~mask]
mask = df["HPI"].str.contains('Emergency department', case=False, na=False)
df = df[~mask]
mask = df["HPI"].str.contains('impression', case=False, na=False)
df = df[~mask]

# Filter out rows in 'diagnosis' that contain the terms 'deceased' or 'died'
mask = df["diagnosis"].str.contains('deceased', case=False, na=False)
df = df[~mask]
mask = df["diagnosis"].str.contains('died', case=False, na=False)
df = df[~mask]

# Further filter out rows where 'diagnosis' contains the term 'history of present illness'
# This ensures that diagnosis-related fields don't inadvertently contain HPI-related content
mask_hpi = df["diagnosis"].str.contains('history of present illness', case=False, na=False)
df = df[~mask_hpi]

 
#### Create Primary and Secondary Diagnosis
## Drop rows that include "primary" as primary diagnosis but not surely in the beginning
mask = df["diagnosis"].str.contains('primary', case=False, na=False)
ind = df[mask].index.tolist()
mask2 = df['diagnosis'].str.contains(r'^\s*\nprimary', flags=re.IGNORECASE, regex=True)
ind2 = df[mask2].index.tolist()
ind_drop = set(ind) - set(ind2)
df = df[~df.index.isin(ind_drop)]

## Drop rows that include "secondary" as secondary diagnosis but not surely in the beginning
mask = df["diagnosis"].str.contains('secondary', case=False, na=False)
ind = df[mask].index.tolist()
mask2 = df['diagnosis'].str.contains('\nsecondary', flags=re.IGNORECASE, regex=True)
ind2 = df[mask2].index.tolist()
ind_drop = set(ind) - set(ind2)
df = df[~df.index.isin(ind_drop)]

## Segregate Discharge Diagnosis into Primary and Secondary Categories with Post-Processing and Filtering 
df["primary_diagnosis"] = None
df["secondary_diagnosis"] = None

## divide discharge diagnosis into primary and secondary diangosis if possible
for i in df.index:
    index = df["diagnosis"][i].lower().find('secondary')
    if index != -1:
        df.loc[i, "primary_diagnosis"] = df["diagnosis"][i][:index]
        df.loc[i, "secondary_diagnosis"] = df["diagnosis"][i][index:]
    else:
        df.loc[i, "primary_diagnosis"] = df["diagnosis"][i]
        df.loc[i, "secondary_diagnosis"] = ""


# Remove any text after "___ Condition:" 
def delete_after_string(text):
    text = re.sub(re.compile(r"___ Condition:.*", re.IGNORECASE | re.DOTALL), "", text)
    return text
df['primary_diagnosis'] = df['primary_diagnosis'].apply(delete_after_string)



## Filter rows in the DataFrame where 'primary_diagnosis' has fewer than 16 single newlines (less than 16 diagnoses)
def count_single_newlines(text):
    single_newlines = re.findall(r'(?<!\n)\n(?!\n)', text)
    return len(single_newlines)

# Apply the function to the entire column and get a list of counts
newline_counts = df['primary_diagnosis'].apply(count_single_newlines).tolist()

mask = [value < 16 for value in newline_counts]
df = df[mask]
df = df.drop(columns=['preprocessed_text'], inplace=False)

 
#### Convert Primary and Secondary Diagnosis into a list of diagnoses for each patient
## replace colon without \n to colon with \n
def colon_replacement(text):

    # remove everything after
    text = re.sub(r":\s*(?!\n)", ':\n', text)

    return text

df['primary_diagnosis'] = df['primary_diagnosis'].apply(colon_replacement)
df['secondary_diagnosis'] = df['secondary_diagnosis'].apply(colon_replacement)


## make diagnosis into a list for each row
liste = df['primary_diagnosis'].apply(lambda x: [s for s in x.split('\n') if s.strip()] if pd.notna(x) else x)
liste = liste.apply(lambda lst: [item for item in lst if "primary diagnoses" not in item.lower()])
liste = liste.apply(lambda lst: [item for item in lst if "primary diagnosis" not in item.lower()])
liste = liste.apply(lambda lst: [item for item in lst if "primary" not in item.lower()]) 
liste = liste.apply(lambda lst: [item for item in lst if "====" not in item.lower()])
liste = liste.apply(lambda lst: [item for item in lst if "" != item.lower()])

def remove_number_prefix(item):
    return re.sub(r'^[1-8]\)\s*', '', item)
liste = liste.apply(lambda lst: [remove_number_prefix(item) for item in lst])

df["primary_diagnosis"] = liste


df['secondary_diagnosis'] = df['secondary_diagnosis'].fillna("")
liste = df['secondary_diagnosis'].apply(lambda x: [s for s in x.split('\n') if s.strip()])

liste = liste.apply(lambda lst: [item for item in lst if "secondary diagnoses" not in item.lower()])
liste = liste.apply(lambda lst: [item for item in lst if "secondary diagnosis" not in item.lower()])
liste = liste.apply(lambda lst: [item for item in lst if "secondary" not in item.lower()]) 
liste = liste.apply(lambda lst: [item for item in lst if "====" not in item.lower()])
liste = liste.apply(lambda lst: [item for item in lst if "" != item.lower()])

def remove_number_prefix(item):
    return re.sub(r'^[1-8]\)\s*', '', item)
liste = liste.apply(lambda lst: [remove_number_prefix(item) for item in lst])

df["secondary_diagnosis"] = liste


#### Extract the first 2200 (goal is to predict 2000, 200 are in case rows need to be remove - see postprocessing and additional postprocessing)
df_small = df[:2200]
df_eval = df[2200:].reset_index(drop=True)

## save files — paths are relative to this script's directory regardless of cwd
_here = os.path.dirname(os.path.abspath(__file__))
df_small.to_csv(os.path.join(_here, 'MIMIC-IV-Ext-Triage-Specialty-Diagnosis-Decision-Support.csv'), index=False)
df_eval.to_csv(os.path.join(_here, 'MIMIC-IV-Ext-Dev.csv'), index=False)

## Cleaning and organizing the DataFrame for clarity

df_vital_signs =  df[['stay_id', 'subject_id', 'hadm_id', 'initial_vitals']]
df_patient_demographics = df[['stay_id', 'patient_info']]
df_diagnosis = df[['stay_id', 'HPI', 'patient_info', 'initial_vitals', 'diagnosis', 'primary_diagnosis', 'secondary_diagnosis']]
df_triage = df[['stay_id', 'HPI', 'patient_info', 'initial_vitals', 'triage']]
df_initial_assessment_info = df[['stay_id', 'triage', 'pain', 'chiefcomplaint', 'arrival_transport', 'disposition', 'icd_code', 'icd_title', 'icd_version']]
df_clinical_data = df[['stay_id', 'text', 'HPI', 'tests', 'past_medication', 'diagnosis', 'primary_diagnosis', 'secondary_diagnosis']]
