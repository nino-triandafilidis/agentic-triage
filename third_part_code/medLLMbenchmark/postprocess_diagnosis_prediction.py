## POSTPROCESS DIAGNOSIS PREDICTION

## import libraries
import pandas as pd
import re

## Load Data from Claude_diag_spec_ClinicalUser.py and Claude_diag_spec_GeneralUser.py
df = pd.read_csv("MIMIC-IV-Ext-Diagnosis-Specialty.csv")

## Function to extract the Prediction from the <diagnosis> tag
def parse_diagnosis(diagnosis):
    #parse whats in between <diagnosis>  and </diagnosis> tag
    try:
        diag1 = diagnosis.split('<diagnosis>')[1].split('</diagnosis>')[0]
        diag2 = diagnosis.split('<diagnosis>')[2].split('</diagnosis>')[0]
        diag3 = diagnosis.split('<diagnosis>')[3].split('</diagnosis>')[0]
        diagnosis = [diag1, diag2, diag3]
    except Exception as e1:
        e1
        try: 
            diagnosis = diagnosis.split('<diagnosis>')[-1].split('</diagnosis>')[0]
        except Exception as e2:
            print(f"{e2}")
    return diagnosis

df["diagnosis_Claude3.5"] = df["diag_spec_Claude3.5"].apply(parse_diagnosis)
df["diagnosis_Claude3"] = df["diag_spec_Claude3"].apply(parse_diagnosis)
df["diagnosis_Haiku"] = df["diag_spec_Haiku"].apply(parse_diagnosis)
df["diagnosis_Claude3.5_Clinical"] = df["diag_spec_Claude3.5_Clinical"].apply(parse_diagnosis)
df["diagnosis_Claude3_Clinical"] = df["diag_spec_Claude3_Clinical"].apply(parse_diagnosis)
df["diagnosis_Haiku_Clinical"] = df["diag_spec_Haiku_Clinical"].apply(parse_diagnosis)


## function to remove leading newline 
def remove_leading_newline(text):
    if isinstance(text, str):  # Check if the input is a string
        # Remove leading '\n' (newline) and leading '\\n' (literal backslash followed by 'n')
        if text.startswith("\\n"):  # Handle literal "\\n"
            return text[2:]  # Remove the first two characters (i.e., "\\n")
        else:
            return text.lstrip('\n')  # Remove actual newline characters
    else:
        return text  # Return the input unchanged if it's not a string
    
df["diagnosis_Claude3.5"] = df["diagnosis_Claude3.5"].apply(remove_leading_newline)
df["diagnosis_Claude3"] = df["diagnosis_Claude3"].apply(remove_leading_newline)
df["diagnosis_Haiku"] = df["diagnosis_Haiku"].apply(remove_leading_newline)
df["diagnosis_Claude3.5_Clinical"] = df["diagnosis_Claude3.5_Clinical"].apply(remove_leading_newline)
df["diagnosis_Claude3_Clinical"] = df["diagnosis_Claude3_Clinical"].apply(remove_leading_newline)
df["diagnosis_Haiku_Clinical"] = df["diagnosis_Haiku_Clinical"].apply(remove_leading_newline)


## function to create a list of the predicted diagnoses 
def create_list(text):
    if type(text) == str:
        try:
            text1 = text.split('\n')[0]
            text2 = text.split('\n')[1]
            text3 = text.split('\n')[2]
            text = [text1, text2, text3]
        except Exception as e1:
            try:
                text1 = text.split('\\n')[0]
                text2 = text.split('\\n')[1]
                text3 = text.split('\\n')[2]
                text = [text1, text2, text3]
            except Exception as e2:
                print(text, f"{e2}")
        return text
    else:
        return(text)
    
df["diagnosis_Claude3.5"] = df["diagnosis_Claude3.5"].apply(create_list)
df["diagnosis_Claude3"] = df["diagnosis_Claude3"].apply(create_list)
df["diagnosis_Haiku"] = df["diagnosis_Haiku"].apply(create_list)
df["diagnosis_Claude3.5_Clinical"] = df["diagnosis_Claude3.5_Clinical"].apply(create_list)
df["diagnosis_Claude3_Clinical"] = df["diagnosis_Claude3_Clinical"].apply(create_list)
df["diagnosis_Haiku_Clinical"] = df["diagnosis_Haiku_Clinical"].apply(create_list)


## function to remove the numeration in some of the predictions
def remove_numeration(entry):
    # Use regular expression to remove leading numeration only for '1.', '2.', or '3.'
    return re.sub(r'^[1-3]\.\s*', '', entry)

# Apply the function to each list in the column
df["diagnosis_Claude3.5"] = df["diagnosis_Claude3.5"].apply(lambda lst: [remove_numeration(entry) for entry in lst] if isinstance(lst, list) else lst)
df["diagnosis_Claude3"] = df["diagnosis_Claude3"].apply(lambda lst: [remove_numeration(entry) for entry in lst] if isinstance(lst, list) else lst)
df["diagnosis_Haiku"] = df["diagnosis_Haiku"].apply(lambda lst: [remove_numeration(entry) for entry in lst] if isinstance(lst, list) else lst)
df["diagnosis_Claude3.5_Clinical"] = df["diagnosis_Claude3.5_Clinical"].apply(lambda lst: [remove_numeration(entry) for entry in lst] if isinstance(lst, list) else lst)
df["diagnosis_Claude3_Clinical"] = df["diagnosis_Claude3_Clinical"].apply(lambda lst: [remove_numeration(entry) for entry in lst] if isinstance(lst, list) else lst)
df["diagnosis_Haiku_Clinical"] = df["diagnosis_Haiku_Clinical"].apply(lambda lst: [remove_numeration(entry) for entry in lst] if isinstance(lst, list) else lst)


## Cleaning the Dataframe for clarity
df = df.drop(columns=["diag_spec_Claude3.5", "diag_spec_Claude3", "diag_spec_Haiku", "diag_spec_Claude3.5_Clinical", "diag_spec_Claude3_Clinical", "diag_spec_Haiku_Clinical"])


## save file
df.to_csv('MIMIC-IV-Ext-Diagnosis-prediction.csv', index=False)