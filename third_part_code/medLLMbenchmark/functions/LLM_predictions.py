import time

def get_ground_truth_specialty(row, chain, max_retries=5, initial_wait=1):
    diagnosis = row["primary_diagnosis"]
    attempt = 0
    while attempt < max_retries:
        try:
            # Invoke the chain with the diagnosis and icd_code

            specialty = chain.invoke({"diagnosis": diagnosis}).content
            return specialty  # Return on successful invocation

        except Exception as e:
            # Check if the error is a ThrottlingException or similar
            if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                # Exponential backoff
                wait_time = initial_wait * (2 ** attempt)
                print(f"Throttling detected. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1
            else:
                # Handle other types of exceptions
                return f"Error: {str(e)}"
    # If all retries fail, return an error
    return "Error: Max retries exceeded"


## Function to perform prediction with an LLM for the General User Case with retry logic
def get_prediction_GeneralUser(row, chain, max_retries=5, initial_wait=1):
    hpi = row['HPI']
    patient_info = row["patient_info"]
    attempt = 0
    while attempt < max_retries:
        try:
            # Invoke the chain with the HPI and patient_info
            response = chain.invoke({"HPI": hpi, "patient_info": patient_info}).content
            return response  # Return on successful invocation

        except Exception as e:
            # Check if the error is a ThrottlingException or similar
            if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                # Exponential backoff
                wait_time = initial_wait * (2 ** attempt)
                print(f"Throttling detected. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1
            else:
                # Handle other types of exceptions
                return f"Error: {str(e)}"
    # If all retries fail, return an error
    return "Error: Max retries exceeded"


## Function to perform prediction with an LLM for the Clinical User Case with retry logic
def get_prediction_ClinicalUser(row, chain, max_retries=5, initial_wait=1):
    hpi = row['HPI']
    patient_info = row["patient_info"]
    initial_vitals = row["initial_vitals"]
    attempt = 0
    while attempt < max_retries:
        try:
            # Invoke the chain with the HPI, patient_info and initial_vitals
            response = chain.invoke({"hpi": hpi, "patient_info": patient_info, "initial_vitals": initial_vitals}).content
            return response  # Return on successful invocation
        
        except Exception as e:
            # Check if the error is a ThrottlingException or similar
            if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                # Exponential backoff
                wait_time = initial_wait * (2 ** attempt)
                print(f"Throttling detected. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1
            else:
                # Handle other types of exceptions
                return f"Error: {str(e)}"
    # If all retries fail, return an error
    return "Error: Max retries exceeded"


## Function to evaluate the diagnosis predictions with retry logic
def get_evaluation_diagnosis(row, key, chain, max_retries=5, initial_wait=1):
    diagnosis = row["primary_diagnosis"]
    diag1 = row[key][0]
    diag2 = row[key][1]
    diag3 = row[key][2]

    attempt = 0
    while attempt < max_retries:
        try:
            # Invoke the chain with the diagnosis and icd_code
            evaluation= chain.invoke({"real_diag": diagnosis, "diag1": diag1, "diag2": diag2, "diag3": diag3}).content
            #print(evaluation)
            return evaluation  # Return on successful invocation

        except Exception as e:
            # Check if the error is a ThrottlingException or similar
            if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                # Exponential backoff
                wait_time = initial_wait * (2 ** attempt)
                print(f"Throttling detected. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1
            else:
                # Handle other types of exceptions
                return f"Error: {str(e)}"
    # If all retries fail, return an error
    return "Error: Max retries exceeded"