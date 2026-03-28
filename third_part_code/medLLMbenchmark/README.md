# LLM Benchmark for Clinical-Decision Support
This repository contains the code to create the curated dataset from the original MIMIC-IV-ED, MIMIC-IV and MIMIC-IV-Note datasets.

## Generate the dataset from the MIMIC-IV datasets
Download the following datasets:\
MIMIC-IV-ED: https://physionet.org/content/mimic-iv-ed/2.2/ \
MIMIC-IV: https://physionet.org/content/mimiciv/3.0/ \
MIMIC-IV-Note: https://physionet.org/content/mimic-iv-note/2.2/

Run the jupyter notebooks in the following order:\
```MIMIC-IV-Ext-Creation.ipynb```\
```create_ground_truth_specialty.ipynb```\
```Claude_triage_diagnosis_specialty.ipynb```\
```postprocessing.ipynb```

Run the .py files in the following order:\
```MIMIC-IV-Ext-Creation.py```\
```create_ground_truth_specialty.py```\
```CLaude_diag_spec_GeneralUser.py```\
```CLaude_diag_spec_ClinicalUser.py```\
```CLaude_triage_GeneralUser.py```\
```CLaude_triage_ClinicalUser.py```\
```postprocess_specialty_prediction.py```\
```postprocess_diagnosis_prediction.py```\
```postprocess_triage_prediction.py```\
```additional_postprocess.py```



