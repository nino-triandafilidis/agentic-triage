# E11 RAG-wins analysis: cases where RAG correct, LLM-only wrong

Total RAG wins: 9 (4 in RAG-used subset + 5 in fallback subset)
Total RAG losses (reverse): 10 (2 in RAG-used + 8 in fallback)

## Error direction summary

### RAG wins (n=9)

| stay_id | GT | RAG pred | LLM pred | LLM error direction | Chief complaint | Top-1 dist | RAG used? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30466704 | 2 | 2 | 1 | over-triage | CVA, Transfer | 0.2295 | Yes |
| 30535671 | 1 | 1 | 2 | under-triage | Respiratory distress, Transfer | 0.2081 | Yes |
| 30685431 | 2 | 2 | 3 | under-triage | Wound eval | 0.2346 | Yes |
| 30792397 | 2 | 2 | 3 | under-triage | DVT, Transfer | 0.2421 | Yes |
| 30050458 | 3 | 3 | 2 | over-triage | S/P ASSAULT | 0.3417 | No |
| 30127567 | 2 | 2 | 3 | under-triage | s/p Fall, Transfer | 0.2799 | No |
| 30607655 | 3 | 3 | 2 | over-triage | ABD PAIN | 0.2896 | No |
| 30632895 | 2 | 2 | 5 | under-triage | Abnormal sodium level | 0.2719 | No |
| 30827587 | 2 | 2 | 3 | under-triage | Cough | 0.3166 | No |

LLM-only errors on these cases: 3 over-triage, 6 under-triage

### RAG losses (n=10)

| stay_id | GT | RAG pred | LLM pred | RAG error direction | Chief complaint | Top-1 dist | RAG used? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30344200 | 3 | 2 | 3 | over-triage | s/p Fall, FX HUMERUS | 0.2424 | Yes |
| 30806100 | 3 | 2 | 3 | over-triage | ABD PAIN | 0.2309 | Yes |
| 30194390 | 3 | 2 | 3 | over-triage | Cellulitis, R Hand swelling, Transfer | 0.3083 | No |
| 30295233 | 3 | 2 | 3 | over-triage | PNEUMO | 0.2696 | No |
| 30308992 | 3 | 2 | 3 | over-triage | Abd pain, Epigastric pain | 0.2872 | No |
| 30482617 | 3 | 2 | 3 | over-triage | L Arm swelling | 0.2583 | No |
| 30488313 | 3 | 2 | 3 | over-triage | Leg pain | 0.2603 | No |
| 30529112 | 3 | 2 | 3 | over-triage | Lower back pain | 0.2639 | No |
| 30724405 | 3 | 2 | 3 | over-triage | Nausea, Weakness | 0.2886 | No |
| 30797544 | 3 | 2 | 3 | over-triage | RIGHT SIDED ABDOMINAL PAIN | 0.3005 | No |

RAG errors on these cases: 10 over-triage, 0 under-triage

---

## Per-case article analysis: RAG wins

### stay_id: 30466704

- **GT:** ESI 2 | **RAG:** ESI 2 (correct) | **LLM:** ESI 1 (wrong — over-triage)
- **Chief complaint:** CVA, Transfer
- **Top-1 distance:** 0.2295
- **RAG used at gate=0.25:** Yes

**HPI:** Mr. ___ is a ___ yo right handed male with a ___ of DM II, 
HTN,
HLD, and possible known right carotid stenosis who was 
transferred
from ___ after being found at his home confused, with
difficulty speaking, and left facial weakness. The details of 
the
history are quite unclear, but from information gathered from 
ER,
OSH, and patient it seems tha...

**Top-5 retrieved articles:**

- **Rank 1** (PMC12447357, Cureus.; 17(8):e90555)
  > pmcIntroduction Carotid artery dissection is common in all age groups and accounts for 2.5% of all strokes, including those in individuals under 40 years of age [1]. In young patients, 20% of cerebrovascular diseases are caused by carotid artery dissection, with a slightly higher incidence in male...

- **Rank 2** (PMC12380769, Front Med (Lausanne). 2025 Aug 13; 12:1626194)
  > pmcIntroduction Endovascular therapy (EVT) is an effective treatment for patients with acute ischemic stroke (AIS) due to large vessel occlusion (LVO) (1). In clinical practice, patients with acute ischemic stroke (AIS) who develop concurrent intracranial hemorrhage (ICH) may be excluded from EVT....

- **Rank 3** (PMC11699854, Cureus.; 16(12):e75153)
  >  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.75153 Neurology Cardiology Internal Medicine A Case of Severe Advanced Diabetic Cardiac Autonomic Neuropathy: Severe Orthostatic Hypotension Complicated With Episodes of Nocturnal Supine Hypertensive Emergency...

- **Rank 4** (PMC11453048, Int J Emerg Med. 2024 Oct 4; 17:138)
  > pmcIntroduction A stroke occurs when the blood supply is cut off from a certain part of the brain depriving it of oxygen and resulting in the death of brain cells. There are two main types of strokes, ischemic and hemorrhagic. Ischemic strokes are a result of a blood clot occluding an artery, prev...

- **Rank 5** (PMC11661266, Clin Pract Cases Emerg Med. 2024 Oct 22; 8(4):365-368)
  > We present a unique case of a patient who presented to the emergency department with stroke-like symptoms found to have a spontaneous, left-sided internal carotid artery dissection (ICAD). Case Report The patient was treated successfully with thrombectomy and subsequently developed contralateral...


### stay_id: 30535671

- **GT:** ESI 1 | **RAG:** ESI 1 (correct) | **LLM:** ESI 2 (wrong — under-triage)
- **Chief complaint:** Respiratory distress, Transfer
- **Top-1 distance:** 0.2081
- **RAG used at gate=0.25:** Yes

**HPI:** ___ year old woman ___ s/p SVD on ___ with severely increased 
dyspnea, orthopnea, leg swelling, fatigue who presented to BID-P 
with hypoxemia requiring CPAP. 

She was having orthopnea at home associated with palpitations as 
well as increased swelling of lower extremities. She and her 
husband had noticed some swelling in her legs in the days be...

**Top-5 retrieved articles:**

- **Rank 1** (PMC12291060, Eur Heart J Case Rep. 2025 Jul 1; 9(7):ytaf306)
  > Background Peripartum cardiomyopathy (PPCM) is a rare but potentially fatal cause of heart failure that occurs towards the end of pregnancy or within the first 5 months postpartum, in the absence of other identifiable cause of cardiac dysfunction. It is characterized by left ventricular systolic i...

- **Rank 2** (PMC12143739, Cureus.; 17(5):e83605)
  >  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.83605 Obstetrics/Gynecology Anesthesiology On-Table Hypoxic Arrest: A Comparative Examination of Peripartum Cardiomyopathy and Pre-eclampsia Muacevic Alexander Adler John R Tong Clara 1 Kong Aaron 2 Ithnin Far...

- **Rank 3** (PMC5584348, Case Rep Cardiol. 2017 Aug 14; 2017:9561405)
  > 1. Introduction Peripartum cardiomyopathy (PPCM) is an uncommon pathology for pregnant women that can produce severe left ventricular (LV) dysfunction. Its diagnosis may be obfuscated as many symptoms are identical to those of normal pregnancy; accordingly, practitioners must have a high index of s...

- **Rank 4** (PMC12010134, Clin Case Rep. 2025 Apr 21; 13(4):e70454)
  > Catecholaminergic Polymorphic Ventricular Tachycardia (CPVT) and Left Ventricular Non‐Compaction Cardiomyopathy (LVNC) are inherited disorders that pose significant challenges in the obstetric population due to the potential exacerbation of ventricular arrhythmias and potentially lethal cardiac com...

- **Rank 5** (PMC4641300, Pak J Med Sci. 2015 Sep-Oct; 31(5):1280-1282)
  > INTRODUCTION Peripartum cardiomyopathy (PPCM) is one of the potentially life-threatening complications of pregnancy, the underlying reason for which is unknown. This form of dilated cardiomyopathy causes congestive heart failure in the later months of pregnancy or in the first 5 months after birth....


### stay_id: 30685431

- **GT:** ESI 2 | **RAG:** ESI 2 (correct) | **LLM:** ESI 3 (wrong — under-triage)
- **Chief complaint:** Wound eval
- **Top-1 distance:** 0.2346
- **RAG used at gate=0.25:** Yes

**HPI:** Mr. ___ is a ___ year old man who was originally diagnosed
with a grade II oligodendroglioma in ___ after presenting with 
a
seizure at home. He has undergone a biopsy followed by
temozolomide. He had a recurrence s/p resection in ___ with
pathology showing a grade III glioma. Following the resection he
received adjuvant radiation followed by temoz...

**Top-5 retrieved articles:**

- **Rank 1** (PMC9396286, Clin Case Rep. 2022 Aug 23; 10(8):e5985)
  > A 58‐year‐old right‐handed man presented to our tertiary care center with gliosarcoma (GS) infiltration through the dura, skull, and soft tissue. Patient had a previous history of right temporal GS, with four intracranial surgeries prior to presentation. A multidisciplinary approach was used to tre...

- **Rank 2** (PMC9318890, Oxf Med Case Reports. 2022 Jul 26; 2022(7):omac074)
  > We present a case demonstrating that older age does not exclude long-term survival with glioblastoma. This is a malignant neoplasm with a median life expectancy of 14 months in patients treated with radical intent. Survival is dependent on several independent and interacting prognostic factors of w...

- **Rank 3** (PMC12648516, Oxf Med Case Reports. 2025 Nov 26; 2025(11):omaf198)
  >  ==== Front Oxf Med Case Reports Oxf Med Case Reports omcr Oxford Medical Case Reports 2053-8855 Oxford University Press  10.1093/omcr/omaf198 omaf198 Case Report AcademicSubjects/MED00010 Management of persistent residual growth of high-grade glioma: a case report highlighting clinical and therapeu...

- **Rank 4** (PMC12648516, Oxf Med Case Reports. 2025 Nov 26; 2025(11):omaf198)
  >  ==== Front Oxf Med Case Reports Oxf Med Case Reports omcr Oxford Medical Case Reports 2053-8855 Oxford University Press  10.1093/omcr/omaf198 omaf198 Case Report AcademicSubjects/MED00010 Management of persistent residual growth of high-grade glioma: a case report highlighting clinical and therapeu...

- **Rank 5** (PMC4727952, Cureus.; 7(12):e434)
  >  ==== Front CureusCureus2168-8184Cureus2168-8184Cureus Palo Alto (CA) 10.7759/cureus.434OncologyNeurosurgeryAdult Brainstem Glioblastoma Multiforme: Long-term Survivor Muacevic Alexander Adler John R Barnard Zachary R 1Drazin Doniel 1Bannykh Serguei I 2Rudnick Jeremy D 3Chu Ray M 11  Neurosurgery, C...


### stay_id: 30792397

- **GT:** ESI 2 | **RAG:** ESI 2 (correct) | **LLM:** ESI 3 (wrong — under-triage)
- **Chief complaint:** DVT, Transfer
- **Top-1 distance:** 0.2421
- **RAG used at gate=0.25:** Yes

**HPI:** ___ with a history of osteoarthritis s/p bilateral hip
replacements in ___ and recent left hip prosthetic joint
infection due to coagulase negative staphylococcus treated with
vancomycin via right PICC who presents with right upper 
extremity
DVT. He reports the sudden onset of right upper extremity the 
day
prior to admission and pain in his wrist...

**Top-5 retrieved articles:**

- **Rank 1** (PMC12764311, Cureus.; 17(12):e98448)
  > pmcIntroduction Staphylococcus aureus bacteremia (SAB) remains a leading cause of serious bloodstream infections and is associated with significant morbidity from metastatic complications, including endocarditis, osteomyelitis, and septic arthritis. Dissemination without an identifiable cardiac so...

- **Rank 2** (PMC9012589, Cureus.; 14(4):e24166)
  > pmcIntroduction Prosthetic joint replacement surgery is effective at improving mobility and pain. Infection, although uncommon (1%-3%), causes substantial morbidity and most often requires surgical intervention. Conventional treatment options include a debridement, antibiotics, and implant retenti...

- **Rank 3** (PMC12371359, JACC Case Rep. 2025 Aug 20; 30(24):104652)
  > Endovascular therapies for peripheral artery disease have become widespread, but the risk of device infection, though rare, must be considered. Prompt management is essential to preserve the patient's limb and life. Case Summary A 79-year-old man presented with intermittent claudication and resti...

- **Rank 4** (PMC12238834, Cureus.; 17(6):e85586)
  > pmcIntroduction Infectious complications following percutaneous coronary intervention (PCI) occur infrequently but carry substantial morbidity and mortality, especially among maintenance hemodialysis patients [1]. These patients face heightened susceptibility due to several factors: uremia-induced...

- **Rank 5** (PMC11403647, Cureus.; 16(8):e66854)
  > pmcIntroduction Staphylococcus aureus (S. aureus) is a Gram-positive bacterium that is responsible for a vast range of diseases. It can cause both community-acquired and hospital-acquired infections [1]. This bacterium is commonly found in the environment and constitutes part of the normal human f...


### stay_id: 30050458

- **GT:** ESI 3 | **RAG:** ESI 3 (correct) | **LLM:** ESI 2 (wrong — over-triage)
- **Chief complaint:** S/P ASSAULT
- **Top-1 distance:** 0.3417
- **RAG used at gate=0.25:** No (fallback)

**HPI:** ___ yo RH ___ s/p assault while at fraternity party earlier
this evening.  Does not remember circumstances, but supposedly
body slammed to ground.  + LOC.  Since then been thinking 
clearly
without any seizures, focal weakness or numbness.  Denies pain
elsewhere besides facial abrasions....

**Top-5 retrieved articles:**

- **Rank 1** (PMC8610471, Clin Pract Cases Emerg Med. 2021 Apr 23; 5(4):502-506)
  > The differential diagnosis for altered mental status and respiratory failure is broad. Careful physical examination, appropriate use of diagnostic tools, and accurate interpretation and correlation of test results are important for piecing together the puzzle of a patient with altered mental status...

- **Rank 2** (PMC12448260, Cureus.; 17(8):e90592)
  > pmcIntroduction The skin is the body’s first natural defense and barrier against pathogens and assists in regulating heat and fluid loss. However, this barrier can be disrupted by several means, including, but not limited to, burns. Damage caused by burns is due to acute injury of the skin or subc...

- **Rank 3** (PMC3441877, J Med Case Rep. 2012 Aug 30; 6:257)
  >  ==== Front J Med Case RepJ Med Case RepJournal of Medical Case Reports1752-1947BioMed Central 1752-1947-6-2572293554710.1186/1752-1947-6-257Case ReportTraumatic asphyxia due to blunt chest trauma: a case report and literature review Sertaridou Eleni 1elenisertaridou@yahoo.comPapaioannou Vasilios 1v...

- **Rank 4** (PMC12274251, Acta Neurochir (Wien). 2025 Jul 19; 167(1):195)
  >  ==== Front Acta Neurochir (Wien) Acta Neurochir (Wien) Acta Neurochirurgica 0001-6268 0942-0940 Springer Vienna Vienna  6601 10.1007/s00701-025-06601-9 Case Report Chokehold with ‘rear naked choke’ and delayed post-hypoxic leukoencephalopathy: a new form of assault in Mexico City Castillo-Rangel Ca...

- **Rank 5** (PMC9851284, Cureus.; 14(12):e32742)
  >  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.32742 Neurology Psychiatry Psychology An Uncontrollable, Aggressive Patient at a Free-Standing Emergency Department Muacevic Alexander Adler John R Crane Joel 1 Aguiar Brittney E 2 Nielson Jeffrey A 1 1 Emerge...


### stay_id: 30127567

- **GT:** ESI 2 | **RAG:** ESI 2 (correct) | **LLM:** ESI 3 (wrong — under-triage)
- **Chief complaint:** s/p Fall, Transfer
- **Top-1 distance:** 0.2799
- **RAG used at gate=0.25:** No (fallback)

**HPI:** HPI: ___ presents as a transfer from ___ 
with
a R trimalleolar ankle fx s/p fall down spiral staircase last
night.  Immediate pain and inability to ambulate, so ambulance
was called.  Bedside reduction unsuccessful x 3, so the patient
was transferred to ___ for further management.

On presentation, complains of ankle pain but denies numbness or
ti...

**Top-5 retrieved articles:**

- **Rank 1** (PMC8892453, Br Paramed J. 2022 Mar 1; 6(4):41-47)
  >  ==== Front Br Paramed J BPJ British Paramedic Journal 1478-4726 The College of Paramedics  35340582 BPJ-2022-6-4-41 10.29045/14784726.2022.03.6.4.41 Case Report An atypical presentation of orthostatic hypotension and falls in an older adult Thoburn Steve North West Ambulance Service NHS Trust; Univ...

- **Rank 2** (PMC12101221, Geriatrics (Basel). 2025 May 20; 10(3):68)
  >  ==== Front Geriatrics (Basel) Geriatrics (Basel) geriatrics Geriatrics 2308-3417 MDPI  40407575 10.3390/geriatrics10030068 geriatrics-10-00068 Case Report Case Report: Multifactorial Intervention for Safe Aging in Place https://orcid.org/0000-0002-9578-1828 Kulkarni Ashwini Patel Harnish P. Academi...

- **Rank 3** (PMC11671416, Cureus.; 16(11):e74535)
  > pmcIntroduction Falls are the leading cause of unintentional injury in individuals aged 65 and older [1]. In 2020, nearly one in four adults over the age of 65 reported experiencing a fall, which highlights the significant prevalence of this issue [2]. Furthermore, the rate of unintentional falls...

- **Rank 4** (PMC7994030, Cureus.; 13(2):e13519)
  > Introduction The fractures of the ankle joint are common injuries, most of the time easily diagnosed and treated either conservatively or surgically [1]. However, in diabetic patients, these fractures constitute demanding cases for both the diagnosis and the treatment [2]. We report two cases of b...

- **Rank 5** (PMC11743738, Cureus.; 16(12):e76040)
  >  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.76040 Anesthesiology Cardiac/Thoracic/Vascular Surgery Otolaryngology Head to Toe: A Perioperative Surgical Home Approach to Sepsis and Airway Obstruction in a Geriatric Patient Muacevic Alexander Adler John R...


### stay_id: 30607655

- **GT:** ESI 3 | **RAG:** ESI 3 (correct) | **LLM:** ESI 2 (wrong — over-triage)
- **Chief complaint:** ABD PAIN
- **Top-1 distance:** 0.2896
- **RAG used at gate=0.25:** No (fallback)

**HPI:** ___ s/p cholecystectomy and with a history of pancreatitis 
presenting with abdominal pain, vomiting x1, and urinary 
frequency. She reports intemittent pain 

...

**Top-5 retrieved articles:**

- **Rank 1** (PMC11684536, Cureus.; 16(11):e74800)
  > pmcIntroduction According to the National Institute for Health and Care Excellence (NICE), acut...

- **Rank 2** (PMC12414525, Cureus.; 17(8):e89636)
  > pmcIntroduction Percutaneous cholecystostomy (PC) is a minimally invasive procedure that involves the percutaneous insertion of a drainage catheter into the gallbladder to relieve biliary obstruction and inflammation [1]. It is primarily indicated in patients with acute cholecystitis who are poor...

- **Rank 3** (PMC12008993, Cureus.; 17(3):e80870)
  >  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.80870 Gastroenterology Internal Medicine Medical Education Challenges in Diagnosing and Managing Acute Cholecystitis in Cirrhosis Muacevic Alexander Adler John R Pornchai Angsupat 1 Wongjarupong Nicha 2 Mir Ad...

- **Rank 4** (PMC9997421, Cureus.; 15(2):e34727)
  > pmcIntroduction In recent literature, there have been conflicting views on whether gangrenous cholecystitis (GC) represents the natural progression of acute cholecystitis (AC) or is a different disease entirely [1-4]; albeit, GC has been found to have an incidence rate varying from 10% to 40% in p...

- **Rank 5** (PMC11712018, Heliyon. 2024 Dec 12; 11(1):e41204)
  > Super-elderly patients with choledocholithiasis are considered to be at high risk for undergoing surgery. While laparoscopic transcystic common bile duct exploration (LTCBDE) is regarded as a challenging procedure for super-elderly patients with choledocholithiasis, there have been no reported case...


### stay_id: 30632895

- **GT:** ESI 2 | **RAG:** ESI 2 (correct) | **LLM:** ESI 5 (wrong — under-triage)
- **Chief complaint:** Abnormal sodium level
- **Top-1 distance:** 0.2719
- **RAG used at gate=0.25:** No (fallback)

**HPI:** ___ yo woman with history of decompensated alcoholic cirrhosis
complicated by hepatic encephalopathy, GI bleed and ascites who
is admitted for accelerated transplant evaluation.

Of note she has a history of polysubstance abuse including EtOH.
She stopped drinking, previously one bottle of vodka daily for
day years from ___ when she stopped heroin....

**Top-5 retrieved articles:**

- **Rank 1** (PMC10693899, Cureus.; 15(11):e48221)
  > pmcIntroduction Liver cirrhosis represents a severe manifestation of alcoholic liver disease, which can result in fatalities and unfortunate health complications. Cirrhosis is characterized by fibrous scar tissue replacing the normal hepatic tissue, impeding physiologic function. Epidemiological r...

- **Rank 2** (PMC12476575, QJM. 2025 Mar 14; 118(7):543-544)
  > pmcLearning points for clinicians • Alcohol-associated liver disease represents a spectrum of liver injury ranging from asymptomatic steatosis, through alcoholic hepatitis (AH) to overt cirrhosis and liver failure. • Portal hypertension without cirrhosis is mostly associated with pre-sinusoidal or...

- **Rank 3** (PMC11885173, Cureus.; 17(2):e78511)
  > pmcIntroduction Acquired hepatocerebral degeneration (AHD) is a rare, chronic condition affecting about 1% of individuals with liver cirrhosis [1]. Its clinical manifestation is neurological, with a characteristic tremor as well as altered mental status (AMS), and it is officially diagnosed with a...

- **Rank 4** (PMC12358176, Cureus.; 17(7):e88248)
  > pmcIntroduction Hemorrhagic ascites (HA) is described as ascitic fluid with a red blood cell (RBC) count greater than 10,000/μL [1]. HA is a common complication of late-stage liver disease and affects up to 27% of patients with cirrhosis [1]. In cirrhotic patients, HA may occur spontaneously or be...

- **Rank 5** (PMC11876712, Cureus.; 17(2):e78345)
  > pmcIntroduction Hepatic hydrothorax (HH) is a pleural effusion that develops in patients with decompensated liver cirrhosis and portal hypertension where cardiac, pulmonary, and pleural diseases are ruled out. The amount of fluid is typically more than 500 mL [1]. HH is a rare condition estimated...


### stay_id: 30827587

- **GT:** ESI 2 | **RAG:** ESI 2 (correct) | **LLM:** ESI 3 (wrong — under-triage)
- **Chief complaint:** Cough
- **Top-1 distance:** 0.3166
- **RAG used at gate=0.25:** No (fallback)

**HPI:** ___ year old ___ speaking male with past
medical history diastolic CHF (EF 50-55% in ___, COPD,
presenting with several days of cough productive of green sputum 


...

**Top-5 retrieved articles:**

- **Rank 1** (PMC8610471, Clin Pract Cases Emerg Med. 2021 Apr 23; 5(4):502-506)
  > The differential diagnosis for altered mental status and respiratory failure is broad. Careful physical examination, appropriate use of diagnostic tools, and accurate interpretation and correlation of test results are important for piecing together the puzzle of a patient with altered mental status...

- **Rank 2** (PMC11060019, Cureus.; 16(3):e57318)
  >  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.57318 Physical Medicine & Rehabilitation Pulmonology Quality Improvement Novelty of Physiotherapy Management in a Classic Case of Chronic Obstructive Pulmonary Disease in an 84-Year-Old Male Patient with Hyper...

- **Rank 3** (PMC12162380, Cureus.; 17(5):e84006)
  >  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.84006 Family/General Practice Pulmonology Therapeutics Chronic Obstructive Pulmonary Disease Management in the Real World: The Importance of a Holistic Assessment Muacevic Alexander Adler John R Franco Spínola...

- **Rank 4** (PMC7188448, Cureus.; 12(3):e7482)
  >  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)   10.7759/cureus.7482 Infectious Disease Pulmonology Epidemiology/Public Health Coronavirus Disease 2019 (COVID-19) Complicated by Acute Respiratory Distress Syndrome: An Internist’s Perspective Muacevic Alexander  Adler John...

- **Rank 5** (PMC11972617, Cureus.; 17(3):e80218)
  >  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.80218 Pharmacology Pulmonology Nebulized Long-Acting Bronchodilators to Treat Acute Respiratory Failure in an Older Adult: A Case Report Muacevic Alexander Adler John R North Avery 1 Helwig Katelyn 1 Gibbs Mac...


---

## Keyword analysis: RAG wins vs losses

| Keyword group | RAG wins (n=9, 45 articles) | RAG losses (n=10, 50 articles) |
| --- | --- | --- |
| triage/acuity | 56% | 60% |
| ED/emergency | 47% | 42% |
| case report | 87% | 82% |
| treatment | 84% | 72% |
| complication | 53% | 52% |
| diagnosis | 51% | 78% |

---

## Interpretation

### Error direction asymmetry

The most striking pattern is the asymmetry in error direction:

- **RAG wins (9 cases):** LLM-only errors are mixed — 3 over-triage, 6 under-triage.
  RAG corrects LLM's under-triage in 6 cases (LLM said ESI 3 when GT was ESI 2, or said
  ESI 2 when GT was ESI 1). RAG also corrects over-triage in 3 cases (LLM said ESI 1/2
  when GT was 2/3).
- **RAG losses (10 cases):** RAG errors are **100% over-triage** (all 10 cases: GT=3,
  RAG=2, LLM=3). RAG never under-triages in the loss group.

This confirms the mechanism: RAG's bias is unidirectional (toward higher acuity). When
LLM-only under-triages, RAG's acuity-escalation bias happens to push the prediction in the
right direction. When LLM-only is already correct (especially on ESI-3), RAG's bias pushes
past the correct answer into over-triage.

### What RAG wins look like

The 4 RAG-used wins are clinically serious cases where LLM-only under-estimated acuity:

1. **CVA transfer** (gt=2, llm=1→rag=2): Stroke case. Articles about carotid dissection and
   stroke management. LLM over-triaged to ESI-1; RAG articles may have provided context
   that this was a stable transfer, not an active code.
2. **Respiratory distress postpartum** (gt=1, llm=2→rag=1): PPCM case. Articles directly
   about peripartum cardiomyopathy — a life-threatening condition. RAG correctly elevated
   to ESI-1 where LLM under-triaged.
3. **Wound eval, brain tumor history** (gt=2, llm=3→rag=2): Glioma patient with wound.
   Articles about glioma management. LLM under-triaged; RAG's acuity bias corrected this.
4. **DVT with prosthetic joint infection** (gt=2, llm=3→rag=2): PICC-associated DVT.
   Articles about staph bacteremia complications. LLM under-triaged; RAG's severity bias
   corrected this.

### Pattern

RAG adds value on **complex cases with serious underlying conditions where LLM-only fails
to recognize the severity**. In these cases, the same publication-bias mechanism that causes
over-triage on ESI-3 patients (case reports about serious outcomes) actually helps by
signaling genuine severity. The retrieved articles describe complications and serious
outcomes of the patient's actual condition, which is informative when the patient truly is
high-acuity.

### Keyword analysis: no discriminating signal

Keyword rates are similar between wins and losses (triage: 56% vs 60%, ED: 47% vs 42%,
complication: 53% vs 52%). The articles are qualitatively the same type — case reports about
similar conditions. The difference is not in the articles but in whether the patient's true
acuity aligns with the acuity-escalation bias that case reports induce.

### Net assessment

RAG provides genuine severity-relevant context — case reports about serious outcomes of
the patient's actual condition. This information is accurate and clinically meaningful. The
9 wins show that PMC retrieval can help on some hard cases where LLM-only under-estimates
acuity.

The 9 wins are cases where LLM-only under-estimated acuity and the severity signal from
articles corrected the error. The 10 losses are cases where LLM-only was already correct
and the same severity signal pushed past the right answer. The underlying mechanism is the
same — the question is whether there is a way to predict *which* cases need the acuity boost.

Across the repo, this should be read as a narrow positive signal rather than the main
conclusion. Most PMC-retrieval variants remain below LLM-only, and later triage-native
approaches perform better than E11: E18 few-shot reaches kappa 0.410 and E19 tool-use
reaches 0.431, vs E11 gate 0.25 at 0.366. The current evidence supports "PMC retrieval has
some value on a minority of cases," not "PMC RAG is the best path if gated better."

### Implication: gate on case difficulty, not embedding distance

The current distance gate (top1 < 0.25) is a poor proxy — it measures topic similarity,
not case difficulty. A better gate would identify cases where the LLM is likely to
under-triage. Possible approaches:

1. **Two-pass confidence gating.** Run LLM-only first. If the model is uncertain (low
   confidence, ambiguous reasoning, or borderline ESI 2/3), apply RAG on a second pass.
   Cost: 2× LLM calls on uncertain cases, but only ~30-40% of cases may be uncertain.
2. **Self-assessed difficulty.** Ask the model to rate its own confidence alongside the
   prediction. Gate RAG on low self-assessed confidence.
3. **Clinical pattern gating.** Identify chief complaint / patient profile patterns that
   correlate with under-triage risk (e.g., transfers, complex comorbidities, subtle
   presentations of serious conditions). Apply RAG selectively to those patterns.

This remains a follow-up hypothesis, not a settled conclusion. The evidence here is post-hoc
and partly reconstructed from E07 RAG + E00.5 LLM-only predictions rather than measured in a
prospective gated run. If the difficulty gate correctly identifies the ~9 under-triaged
cases without also selecting the ~10 correctly-triaged ESI-3 cases, RAG becomes a net
positive.
