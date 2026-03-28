## LLM TO EVLUATE DIAGNOSIS PREDICTION

## import libraries
import pandas as pd
from tqdm import tqdm
import time
import boto3
import os
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate


## Import Functions
from functions.LLM_predictions import get_evaluation_diagnosis


## Load Data from additional_prostprocessing.py
df = pd.read_csv("MIMIC-IV-Ext-Diagnosis-prediction.csv")


## Convert the diagnosis rows into lists - data in columns are stored as strings but actually represent lists
df['diagnosis_Claude3.5'] = df['diagnosis_Claude3.5'].apply(lambda x: eval(x))
df['diagnosis_Claude3'] = df['diagnosis_Claude3'].apply(lambda x: eval(x))
df['diagnosis_Haiku'] = df['diagnosis_Haiku'].apply(lambda x: eval(x))
df['diagnosis_Claude3.5_Clincal'] = df['diagnosis_Claude3.5_Clincal'].apply(lambda x: eval(x))
df['diagnosis_Claude3_Clinical'] = df['diagnosis_Claude3_Clincal'].apply(lambda x: eval(x))
df['diagnosis_Haiku_Clinical'] = df['diagnosis_Haiku_Clinical'].apply(lambda x: eval(x))


## Define the prompt template
prompt = """You are an experienced healthcare professional with expertise in medical and clinical domains. I will provide a list of real diagnoses for a patient and 3 predicted diagnoses. For each predicted diagnosis, determine if it has the same meaning as one of the real diagnoses or if the prediction falls under a broader category of one of the real diagnoses (e.g., a specific condition falling under a general diagnosis category). If it matches, return 'True'; otherwise, return 'False'. Return only 'True' or 'False' for each predicted diagnosis within <evaluation> tags and nothing else.
Real Diagnoses: {real_diag}, predicted diagnosis 1: {diag1}, predicted diagnosis 2: {diag2}, and predicted diagnosis 3: {diag3}."""


## set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"]="Enter your AWS Access Key ID"
os.environ["AWS_SECRET_ACCESS_KEY"]="Enter your AWS Secret Access Key"

prompt_chain = PromptTemplate(template=prompt,input_variables=["real_diag", "diag1", "diag2", "diag3"])
client = boto3.client(service_name="bedrock-runtime", region_name=str("us-east-1"))


## Claude Sonnet 3.5
llm_claude35 = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", model_kwargs={"temperature": 0}, client=client)
chain_claude35 = prompt_chain | llm_claude35


tqdm.pandas()
keys = ["diagnosis_Claude3.5", "diagnosis_Claude3", 'diagnosis_Haiku', 'diagnosis_Claude3.5_Clinical', 'diagnosis_Claude3_Clinical','diagnosis_Haiku_Clinical']

for key in keys:
    df["eval_"+key] = df.progress_apply(lambda row: get_evaluation_diagnosis(row, key, chain_claude35), axis=1)
    df.to_csv('MIMIC-IV-Ext-Diagnosis-evaluation.csv', index=False)