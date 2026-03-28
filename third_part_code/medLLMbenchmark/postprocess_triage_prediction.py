## POSTPROCESS TRIAGE PREDICTION

## import libraries
import pandas as pd
import re

## Load Data from Claude_triage_ClinicalUser.py and Claude_triage_GeneralUser.py
df = pd.read_csv("MIMIC-IV-Ext-Triage.csv")


## Function to extract the Prediction from the <acuity> tag
def parse_triage(triage):
    #parse whats in between <acuity>  and </acuity> tag
    try:
        triage = triage.split('<acuity>')[-1].split('</acuity>')[0]
    except Exception as e:
        print(triage, f"{e}")
    return triage

df["triage_Claude3.5"] = df["triage_Claude3.5"].apply(parse_triage)
df["triage_Claude3"] = df["triage_Claude3"].apply(parse_triage)
df["triage_Haiku"] = df["triage_Haiku"].apply(parse_triage)
df["triage_Claude3.5_Clinical"] = df["triage_Claude3.5_Clinical"].apply(parse_triage)
df["triage_Claude3_Clinical"] = df["triage_Claude3_Clinical"].apply(parse_triage)
df["triage_Haiku_Clinical"] = df["triage_Haiku_Clinical"].apply(parse_triage)


## function to remove "esi level" string before the triage prediction
def extract_esi_level(text):
    # Regular expression to find "ESI Level" followed by a number
    if pd.isna(text):
        return(text)
    else:
        match = re.search(r'esi level\s*(\d+)', text.lower())
        
        if match:
            return match.group(1)  # Extract the number part
        else:
            return text
        
df["triage_Claude3.5"] = df["triage_Claude3.5"].apply(extract_esi_level)
df["triage_Claude3"] = df["triage_Claude3"].apply(extract_esi_level)
df["triage_Haiku"] = df["triage_Haiku"].apply(extract_esi_level)
df["triage_Claude3.5_Clinical"] = df["triage_Claude3.5_Clinical"].apply(extract_esi_level)
df["triage_Claude3_Clinical"] = df["triage_Claude3_Clinical"].apply(extract_esi_level)
df["triage_Haiku_Clinical"] = df["triage_Haiku_Clinical"].apply(extract_esi_level)


## convert the prediction to "int"
def convert_to_int(text):
    try:
        return int(text)  # Try to convert the string to an integer
    except ValueError:
        return text
    
df["triage_Claude3.5"] = df["triage_Claude3.5"].apply(convert_to_int)
df["triage_Claude3"] = df["triage_Claude3"].apply(convert_to_int)
df["triage_Haiku"] = df["triage_Haiku"].apply(convert_to_int)
df["triage_Claude3.5_Clinical"] = df["triage_Claude3.5_Clinical"].apply(convert_to_int)
df["triage_Claude3_Clinical"] = df["triage_Claude3_Clinical"].apply(convert_to_int)
df["triage_Haiku_Clinical"] = df["triage_Haiku_Clinical"].apply(convert_to_int)


## save file
df.to_csv('MIMIC-IV-Ext-Triage-prediction.csv', index=False)