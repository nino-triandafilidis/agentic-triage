## PREDICT SPECIATLY AND DIAGNOSIS CLINICAL USER CASE

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
df = pd.read_csv("MIMIC-IV-Ext-Diagnosis-Specialty.csv")


## Define the prompt template
prompt = """You are an experienced healthcare professional with expertise in determining the medical specialty and diagnosis based on a patient's history of present illness, personal information and initial vitals. Review the data and identify the three most likely, distinct specialties to manage the condition, followed by the three most likely diagnoses. List specialties first, in order of likelihood, then diagnoses.
Respond with the specialties in <specialty> tags and the diagnoses in <diagnosis> tags.
History of present illness: {hpi}, personal information: {patient_info} and initial vitals: {initial_vitals}."""


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
df['diag_spec_Claude3.5_Clinical'] = df.progress_apply(lambda row: get_prediction_ClinicalUser(row, chain_claude35), axis=1)
df.to_csv('MIMIC-IV-Ext-Diagnosis-Specialty.csv', index=False)

df['diag_spec_Claude3_Clinical'] = df.progress_apply(lambda row: get_prediction_ClinicalUser(row, chain_claude3), axis=1)
df.to_csv('MIMIC-IV-Ext-Diagnosis-Specialty.csv', index=False)

df['diag_spec_Haiku_Clinical'] = df.progress_apply(lambda row: get_prediction_ClinicalUser(row, chain_haiku), axis=1)
df.to_csv('MIMIC-IV-Ext-Diagnosis-Specialty.csv', index=False)