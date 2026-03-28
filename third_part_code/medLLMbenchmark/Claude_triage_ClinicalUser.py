## PREDICT TRIAGE/ACUITY CLINICAL USER CASE

## import libraries
import pandas as pd
from tqdm import tqdm
import os
import boto3
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock


## Import functions
from functions.LLM_predictions import get_prediction_ClinicalUser


## Load Data from create_ground_truth_specialty.py
df = pd.read_csv("MIMIC-IV-Ext-Triage.csv")


## Define the prompt template
prompt = """You are a nurse with emergency and triage experience. Using the patient's history of present illness, his information and initial vitals, determine the triage level based on the Emergency Severity Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 (lowest acuity): 1: Assign if the patient requires immediate lifesaving intervention. 2: Assign if the patient is in a high-risk situation (e.g., confused, lethargic, disoriented, or experiencing severe pain/distress)  3: Assign if the patient requires two or more diagnostic or therapeutic interventions and their vital signs are within acceptable limits for non-urgent care. 4: Assign if the patient requires one diagnostic or therapeutic intervention (e.g., lab test, imaging, or EKG). 5: Assign if the patient does not require any diagnostic or therapeutic interventions beyond a physical exam (e.g., no labs, imaging, or wound care).
History of present illness: {HPI}, patient info: {patient_info} and initial vitals: {initial_vitals}. Respond with the level in an <acuity> tag."""


## set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"]="Enter your AWS Access Key ID"
os.environ["AWS_SECRET_ACCESS_KEY"]="Enter your AWS Secret Access Key"

prompt_chain = PromptTemplate(template=prompt,input_variables=["hpi", "patient_info", "initial_vitals"])
client = boto3.client(service_name="bedrock-runtime", region_name=str("us-east-1"))


## Claude Sonnet 3.5
llm_claude35 = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", model_kwargs={"temperature": 0},client=client)
chain_claude35 = prompt_chain | llm_claude35


## Claude Sonnet 3
llm_claude3 = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0",  model_kwargs={"temperature": 0},client=client)
chain_claude3 = prompt_chain | llm_claude3


## Claude 3 Haiku
llm_haiku = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0",  model_kwargs={"temperature": 0},client=client)
chain_haiku = prompt_chain | llm_haiku


tqdm.pandas()
df['triage_Claude3.5_Clinical'] = df.progress_apply(lambda row: get_prediction_ClinicalUser(row, chain_claude35), axis=1)
df.to_csv('MIMIC-IV-Ext-Triage.csv', index=False)

df['triage_Claude3_Clinical'] = df.progress_apply(lambda row: get_prediction_ClinicalUser(row, chain_claude3), axis=1)
df.to_csv('MIMIC-IV-Ext-Triage.csv', index=False)

df['triage_Haiku_Clinical'] = df.progress_apply(lambda row: get_prediction_ClinicalUser(row, chain_haiku), axis=1)
df.to_csv('MIMIC-IV-Ext-Triage.csv', index=False)
