## LLM TO CREATE GROUND TRUTH FOR SPRCIALTY AND MERGE IT INTO THE DATASET

## import libraries
import pandas as pd
from tqdm import tqdm
import os
import boto3
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock

## import function
from functions.LLM_predictions import get_ground_truth_specialty


## Load Data from mimic_iv_preprocessing.py
df = pd.read_csv('MIMIC-IV-Ext-Triage-Specialty-Diagnosis-Decision-Support.csv')

## Convert the diagnosis rows into lists - data in columns are stored as strings but actually represent lists
df['primary_diagnosis'] = df['primary_diagnosis'].apply(lambda x: eval(x))

## flatten all diagnoses
diagnoses = [diagnosis for sublist in df['primary_diagnosis'] for diagnosis in sublist]
unique_diagnosis = set(diagnoses)
unique_diagnosis = pd.DataFrame(unique_diagnosis, columns=['primary_diagnosis'])


## Define the prompt template
prompt = """You are an experienced healthcare professional with expertise in medical and clinical domains. Determine the medical specialty most appropriate for the patient to consult based on the diagnosis. Please analyze the given diagnosis and predict the medical specialty that would typically manage the condition associated with it. If the condition might be treated by multiple specialties, prioritize the one most likely to manage the majority of cases. Respond with the specialty name only. Give the specialty in a <specialty> tag. If you can't find a specialty return 'no answer' in a <specialty> tag.
Diagnosis: {diagnosis}."""


## set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"]="Enter your AWS Access Key ID"
os.environ["AWS_SECRET_ACCESS_KEY"]="Enter your AWS Secret Access Key"

prompt_chain = PromptTemplate(template=prompt,input_variables=["diagnosis"])
client = boto3.client(service_name="bedrock-runtime", region_name=str("us-east-1"))


## Claude Sonnet 3.5
llm_claude35 = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", model_kwargs={"temperature": 0}, client=client)
chain_claude35 = prompt_chain | llm_claude35


## Run LLM (CLaude Sonnet 3.5) to retrieve ground truth specialties 
tqdm.pandas()
unique_diagnosis["specialty_primary_diagnosis"] = unique_diagnosis.progress_apply(lambda row: get_ground_truth_specialty(row, chain_claude35), axis=1)

unique_diagnosis.to_csv('df_specialty_groundtruth.csv', index=False)


## merge ground truth
## Parsing
def parse_response(specialty):
    #parse whats in between <specialty>  and </specialty> tag
    specialty = specialty.split('<specialty>')[-1].split('</specialty>')[0]
    return specialty

unique_diagnosis["specialty_primary_diagnosis"] = unique_diagnosis["specialty_primary_diagnosis"].apply(parse_response)


## Create a dictionary for fast lookup of specialties
diagnosis_to_specialty = pd.Series(unique_diagnosis.specialty_primary_diagnosis.values, index=unique_diagnosis.primary_diagnosis).to_dict()


## Function to map diagnosis list to a list of specialties
def get_specialties(diagnosis_list_column, specialty_look_up_dict):

    specialty_primary_diagnosis = diagnosis_list_column.apply(lambda diagnosis_list: [specialty_look_up_dict.get(diagnosis, 'Unknown Specialty') for diagnosis in diagnosis_list])
    
    return specialty_primary_diagnosis

## assign each diagnosis in the list of diagnoses of each row a specialty as a ground truth
df['specialty_primary_diagnosis'] = get_specialties(df["primary_diagnosis"], diagnosis_to_specialty)

## save files (triage, diag, spec)
df.to_csv('df_mimic_iv_ext_triage_diag_spec.csv', index=False)


## Create Dataset for each modality (Specialty and Diagnosis, Triage)
df_diag_spec = df.copy()
df_triage = df.copy()


## Cleaning the Dataframe for clarity
df_diag_spec  = df_diag_spec.drop(columns=["subject_id", "hadm_id", "pain", "chiefcomplaint", "tests", "triage", "icd_code", "icd_title", "icd_version"], inplace=False)
df_triage  = df_triage.drop(columns=["subject_id", "hadm_id", "pain", "chiefcomplaint", "tests", "icd_code", "icd_title", "icd_version", "diagnosis", "primary_diagnosis", "secondary_diagnosis", "specialty_primary_diagnosis"], inplace=False)


## save files
df_diag_spec.to_csv('MIMIC-IV-Ext-Diagnosis-Specialty.csv', index=False)
df_triage.to_csv('MIMIC-IV-Ext-Triage.csv', index=False)


