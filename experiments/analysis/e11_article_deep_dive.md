# E11 Post-Hoc Deep Dive: Retrieved Articles Analysis

Examines the PMC articles retrieved for cases where RAG predictions were
correct vs incorrect, focusing on ESI-3 cases (90.5% over-triage rate).

Data sources:
- E11 post-hoc analysis (gating at threshold=0.25)
- E07 top-10 retrieval diagnostics (PMC IDs per row)
- Retrieval cache (article text + citations)

---

## 1. Aggregate article characteristics by prediction outcome

### RAG correct (all ESI levels, n=27)

- Cases: 27
- Unique articles retrieved (top-5 per case): 114
- Mean snippet length: 5501 chars
- Mean duplicate articles per case (in top-5): 0.37

Top journals:

| Journal | Count |
| --- | --- |
| Front Med (Lausanne) | 4 |
| Medicine (Baltimore) | 3 |
| Clin Case Rep | 3 |
| JACC Case Rep | 3 |
| Clin Pract Cases Emerg Med | 3 |
| Oxf Med Case Reports | 3 |
| Vaccines (Basel) | 2 |
| Int J Emerg Med | 2 |
| Eur Heart J Case Rep | 2 |
| Int J Surg Case Rep | 2 |

### RAG wrong (all ESI levels, n=21)

- Cases: 21
- Unique articles retrieved (top-5 per case): 94
- Mean snippet length: 5533 chars
- Mean duplicate articles per case (in top-5): 0.05

Top journals:

| Journal | Count |
| --- | --- |
| Br Paramed J | 6 |
| Clin Pract Cases Emerg Med | 3 |
| Int J Surg Case Rep | 3 |
| Clin Case Rep | 2 |
| Medicine (Baltimore) | 2 |
| Reports (MDPI) | 2 |
| Surg J (N Y) | 2 |
| ACG Case Rep J | 1 |
| J Surg Case Rep | 1 |
| Pan Afr Med J | 1 |

### ESI-3 RAG correct (n=2)

- Cases: 2
- Unique articles retrieved (top-5 per case): 10
- Mean snippet length: 5600 chars
- Mean duplicate articles per case (in top-5): 0.00

### ESI-3 RAG wrong/over-triaged (n=19)

- Cases: 19
- Unique articles retrieved (top-5 per case): 87
- Mean snippet length: 5546 chars
- Mean duplicate articles per case (in top-5): 0.05

Top journals:

| Journal | Count |
| --- | --- |
| Br Paramed J | 4 |
| Clin Pract Cases Emerg Med | 3 |
| Int J Surg Case Rep | 3 |
| Clin Case Rep | 2 |
| Medicine (Baltimore) | 2 |
| Reports (MDPI) | 2 |
| Surg J (N Y) | 2 |
| ACG Case Rep J | 1 |
| J Surg Case Rep | 1 |
| Pan Afr Med J | 1 |

---

## 2. ESI-3 cases where RAG predicted correctly (n=2)

These are the rare cases where RAG retrieved articles AND the model still
correctly predicted ESI-3 (instead of over-triaging to ESI-2).

### stay_id: 30314156

- **Chief complaint:** Abd pain
- **Top-1 distance:** 0.1946
- **RAG prediction:** ESI 3 (correct)
- **LLM-only prediction:** ESI 3

**HPI excerpt:** ___ y/o male with history of simoid diverticulitis presenting 
with left lower quadrant abdominal pain. Patient reports that he 
had similar symptoms in the past. Five months ago, patient 
presented to ___ with abdominal pain and was 
diagnosed with diverticulitis. He was admitted briefly for 
treatment with IV antibiotics. Patient had a return of symptoms 
2 weeks ago, was admitted to ___, and wa...

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12046870 | Cureus.; 17(4):e81607 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.81607 Gastroenterology Internal Medicine A Rare Case of Terminal Ileal Diverticulitis: Clinical Presentation,... |
| 2 | PMC11998995 | Cureus.; 17(3):e80661 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.80661 Emergency Medicine Gastroenterology Pathology Beyond the Usual Age: A Case Report on Segmental Colitis ... |
| 3 | PMC10181893 | Cureus.; 15(4):e37511 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.37511 Emergency Medicine Internal Medicine General Surgery Atypical Presentation of Diverticulitis in a Young... |
| 4 | PMC7746110 | Spartan Med Res J.; 3(2):6979 |  ==== Front Spartan Med Res J Spartan Med Res J 1364 Spartan Medical Research Journal 2474-7629 MSU College of Osteopathic Medicine Statewide Campus System Website: Spartan Medical Research Journal  3... |
| 5 | PMC12710101 | Cureus.; 17(11):e97058 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.97058 General Surgery Radiology Small Bowel Obstruction Secondary to Sigmoid Diverticulitis With Reactive Ent... |

### stay_id: 30458286

- **Chief complaint:** R HAND SWELLING 
- **Top-1 distance:** 0.2467
- **RAG prediction:** ESI 3 (correct)
- **LLM-only prediction:** ESI 3

**HPI excerpt:** ___ with HTN, CKD, BPH, h/o gout presents with right hand 
swelling and pain. The swelling started 3 days ago and has 
increased. The hand is becoming increasingly painful. Pt has 
been working in his garden, but denies any trauma or abrasions. 
The pain is constant and he points to his ___ digit MCP and PIP. 


He mentions that he had gout about a year ago in his R big toe. 
This morning the pt n...

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12902796 | Cureus.; 18(1):e101502 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.101502 Internal Medicine Rheumatology Chronic Tophaceous Gout Presenting With Severe Polyarticular Erosive Di... |
| 2 | PMC12337738 | Case Reports Plast Surg Hand Surg.; 12(1):2545199 |  ==== Front Case Reports Plast Surg Hand Surg Case Reports Plast Surg Hand Surg Case Reports in Plastic Surgery & Hand Surgery 2332-0885 Taylor & Francis  10.1080/23320885.2025.2545199 2545199 Version... |
| 3 | PMC8021002 | Cureus.; 13(3):e13732 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.13732 Dermatology Rheumatology Unusual Subacute Interphalangeal Tophaceous Gouty Arthritis Muacevic Alexander... |
| 4 | PMC11912807 | Cureus.; 17(2):e79045 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.79045 Internal Medicine Rheumatology Pain Management Severe Tophaceous Polyarticular Gout: A Case Report and ... |
| 5 | PMC12434384 | Cureus.; 17(8):e90199 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.90199 Internal Medicine Rheumatology Preventive Medicine Early-Onset Polyarticular Gout: A Case Report Muacev... |

---

## 3. ESI-3 cases where RAG over-triaged to ESI-2 (sample of 5/19)

Representative sample of the dominant error pattern: true ESI-3 predicted as ESI-2.

### stay_id: 30106140

- **Chief complaint:** SURGICAL SITE EVAL
- **Top-1 distance:** 0.2148
- **RAG prediction:** ESI 2 (over-triaged)
- **LLM-only prediction:** ESI 2

**HPI excerpt:** Ms. ___ is a ___ with Past Surgical History notable for 
total colectomy ___ years ago, with Lysis of Adhesions for small 
bowel obstruction, 
8 months ago, complicated by ventral hernia, repaired with mesh 
(physio mesh and composite type mesh polypropylene w/Monocryl 
coating) 6 months ago at ___. Patient reports that 
has required aspiration and  drain placement x4, with most 
recent drain remo...

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12342112 | Cureus.; 17(7):e87844 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.87844 General Surgery Gastroenterology It’s a Mesh in These Bowels: A Delayed Case of a Decade-Long Mesh Erod... |
| 2 | PMC12085785 | Cureus.; 17(4):e82492 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.82492 Plastic Surgery General Surgery Vacuum-Assisted Closure Therapy Following Mesh Removal Due to Multidrug... |
| 3 | PMC7571861 | J Surg Case Rep. 2020 Oct 20; 2020(10):rjaa409 |  ==== Front J Surg Case Rep J Surg Case Rep jscr Journal of Surgical Case Reports 2042-8812 Oxford University Press   10.1093/jscr/rjaa409 rjaa409 Case Report AcademicSubjects/MED00910 jscrep/040 Dela... |
| 4 | PMC3224069 | Pan Afr Med J. 2011 Oct 10; 10:18 |  ==== Front Pan Afr Med JPAMJThe Pan African Medical Journal1937-8688The African Field Epidemiology Network 22187600PAMJ-10-18Case ReportDon't be scared: insert a mesh! Chichom Mefire Alain 1&Guifo Ma... |
| 5 | PMC4247657 | BMC Surg. 2014 Nov 15; 14:91 |  ==== Front BMC SurgBMC SurgBMC Surgery1471-2482BioMed Central London 2539925053210.1186/1471-2482-14-91Research ArticleAcute inflammatory response in the subcutaneous versus periprosthethic space aft... |

### stay_id: 30523256

- **Chief complaint:** EPIGASTRIC PAIN
- **Top-1 distance:** 0.2205
- **RAG prediction:** ESI 2 (over-triaged)
- **LLM-only prediction:** ESI 2

**HPI excerpt:** Ms. ___ is a very nice ___ yo female with long history of 
asthma, GERD and TBM. Patient underwent LINX procedure on 
___, surgery was uncomplicated and patient was discharged 
the same day.  Ms. ___ presented to clinic on ___ for 
first post op visit. Multiple complains including anorexia, due 
to loss of appetite and fear to eat; excessive burping and PO 
intolerance. She informs that after drin...

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12284699 | Int J Surg Case Rep. 2025 Jul 10; 133:111646 |  ==== Front Int J Surg Case Rep Int J Surg Case Rep International Journal of Surgery Case Reports 2210-2612 Elsevier  S2210-2612(25)00832-6 10.1016/j.ijscr.2025.111646 111646 Case Report More than a m... |
| 2 | PMC12255889 | Cureus.; 17(6):e85893 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.85893 General Surgery Gastroenterology Internal Medicine Dumping Syndrome Post Hiatal Hernia Repair With the ... |
| 3 | PMC4878099 | J Med Case Rep. 2016 May 24; 10:124 |  ==== Front J Med Case RepJ Med Case RepJournal of Medical Case Reports1752-1947BioMed Central London 88710.1186/s13256-016-0887-6Case ReportLINX®, a novel treatment for patients with refractory asthm... |
| 4 | PMC12287130 | Surg Endosc. 2025 Jun 26; 39(8):4956-4964 |  ==== Front Surg Endosc Surg Endosc Surgical Endoscopy 0930-2794 1432-2218 Springer US New York  40571793 11842 10.1007/s00464-025-11842-x Article The failure pattern for the magnetic sphincter augmen... |
| 5 | PMC12275466 | Int J Surg Case Rep. 2025 Jul 9; 133:111642 |  ==== Front Int J Surg Case Rep Int J Surg Case Rep International Journal of Surgery Case Reports 2210-2612 Elsevier  S2210-2612(25)00828-4 10.1016/j.ijscr.2025.111642 111642 Case Report Refractory bi... |

### stay_id: 30806100

- **Chief complaint:** ABD PAIN
- **Top-1 distance:** 0.2309
- **RAG prediction:** ESI 2 (over-triaged)
- **LLM-only prediction:** ESI 3

**HPI excerpt:** ___ with history of pancreatitis s/p cyst gastrostomy and
cholecystectomy on ___ presents with abdominal pain for one
day. Patient was seen in clinic yesterday and reported feeling
well overall with significant improvement since postop. Today, 
he
reports onset of LUQ and epigastric abdominal pain, at the site
of his usual pancreatitis. He denies nausea or vomiting. He has
not eaten since the onse...

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12521821 | Clin Case Rep. 2025 Oct 15; 13(10):e71087 |  ==== Front Clin Case Rep Clin Case Rep 10.1002/(ISSN)2050-0904 CCR3 Clinical Case Reports 2050-0904 John Wiley and Sons Inc. Hoboken  10.1002/ccr3.71087 CCR371087 CCR3-2025-03-0781.R2 Case Report Cas... |
| 2 | PMC11848220 | Cureus.; 17(1):e77947 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.77947 Gastroenterology Internal Medicine Pain Management Chronic Pancreatitis: One Patient, Multiple Etiologi... |
| 3 | PMC10309077 | Cureus.; 15(5):e39704 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.39704 Internal Medicine Gastroenterology General Surgery Gallstone Pancreatitis Post Laparoscopic Cholecystec... |
| 4 | PMC11684536 | Cureus.; 16(11):e74800 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.74800 Emergency Medicine General Surgery Internal Medicine Small Bowel Obstruction as a Complication of Acute... |
| 5 | PMC12206490 | Cureus.; 17(6):e86983 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.86983 Gastroenterology Internal Medicine A Fatal Cascade Following Endoscopic Retrograde Cholangiopancreatico... |

### stay_id: 30344200

- **Chief complaint:** s/p Fall, FX HUMERUS
- **Top-1 distance:** 0.2424
- **RAG prediction:** ESI 2 (over-triaged)
- **LLM-only prediction:** ESI 3

**HPI excerpt:** ___ year old ___ female, with h/o ___, HTN, 
urinary retention, chronic constipation present from ___ 
___ s/p fall.  
Per ___ report, this morning the patient walked to bathroom with 
her walker. On returning from the bathroom, she turned "in a 
funny way" and fell on her left side. She was able to get up 
from the ground herself but reportedly left upper arm pain with 
decresed ROM. She denied a...

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC8892453 | Br Paramed J. 2022 Mar 1; 6(4):41-47 |  ==== Front Br Paramed J BPJ British Paramedic Journal 1478-4726 The College of Paramedics  35340582 BPJ-2022-6-4-41 10.29045/14784726.2022.03.6.4.41 Case Report An atypical presentation of orthostati... |
| 2 | PMC12324059 | Medicine (Baltimore). 2025 Aug 1; 104(31):e41958 |  ==== Front Medicine (Baltimore) Medicine (Baltimore) MD Medicine 0025-7974 1536-5964 Lippincott Williams & Wilkins Hagerstown, MD  40760634 MD-D-24-13214 00134 10.1097/MD.0000000000041958 3 7100 Rese... |
| 3 | PMC8610462 | Clin Pract Cases Emerg Med. 2021 Nov 1; 5(4):369-376 |  ==== Front Clin Pract Cases Emerg Med Clin Pract Cases Emerg Med Clinical Practice and Cases in Emergency Medicine 2474-252X University of California Irvine, Department of Emergency Medicine publishi... |
| 4 | PMC6920307 | Int J Surg Case Rep. 2019 Dec 6; 66:130-135 |  ==== Front Int J Surg Case RepInt J Surg Case RepInternational Journal of Surgery Case Reports2210-2612Elsevier S2210-2612(19)30693-510.1016/j.ijscr.2019.11.058ArticleAxillo-subclavian dissection and... |
| 5 | PMC7783912 | Br Paramed J. 2020 Jun 1; 5(1):15-19 |  ==== Front Br Paramed J BPJ British Paramedic Journal 1478-4726 The College of Paramedics  33456381 BPJ-5-1-15 10.29045/14784726.2020.06.5.1.15 Case Report Unexpected shock in a fallen older adult: a... |

### stay_id: 30552702

- **Chief complaint:** Dysuria
- **Top-1 distance:** 0.2494
- **RAG prediction:** ESI 2 (over-triaged)
- **LLM-only prediction:** ESI 2

**HPI excerpt:** ___ with DM2, hypothyroidism, BPH s/p TURP 2 months ago who 
presents with a referral from PCP for ___ UTI. Patient 
reports some dysuria and notes a cloudy urine for the past few 
weeks.  

Recently hospitalized in early ___ for urinary retention at 
___.  Found to have BPH and underwent TURP at that time.  
Had to straight cath himself for a while after discharge but now 
can urinate on his own....

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12358184 | Cureus.; 17(7):e88251 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.88251 Internal Medicine Endocrinology/Diabetes/Metabolism Subclinical Hyperthyroidism Presenting as Urinary F... |
| 2 | PMC12841809 | Cureus.; 17(12):e100256 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.100256 Emergency Medicine Infectious Disease Multiloculated Prostatic Abscess Complicated by Obstructive Hydr... |
| 3 | PMC12841809 | Cureus.; 17(12):e100256 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.100256 Emergency Medicine Infectious Disease Multiloculated Prostatic Abscess Complicated by Obstructive Hydr... |
| 4 | PMC12158734 | Front Endocrinol (Lausanne). 2025 May 29; 16:1609966 |  ==== Front Front Endocrinol (Lausanne) Front Endocrinol (Lausanne) Front. Endocrinol. Frontiers in Endocrinology 1664-2392 Frontiers Media S.A.  10.3389/fendo.2025.1609966 Endocrinology Original Rese... |
| 5 | PMC12767225 | J Diabetes Res. 2025 Dec 9; 2025:6890754 |  ==== Front J Diabetes Res J Diabetes Res 10.1155/1485 JDR Journal of Diabetes Research 2314-6745 2314-6753 Wiley  10.1155/jdr/6890754 JDR6890754 Research Article Article The Role of Prognostic Nutrit... |

---

## 4. ESI-2 cases where RAG predicted correctly (sample of 5/24)

Contrast group: cases where retrieved articles aligned with the correct
ESI-2 prediction. Are the articles qualitatively different?

### stay_id: 30067117

- **Chief complaint:** Dizziness, Presyncope
- **Top-1 distance:** 0.2464
- **RAG prediction:** ESI 2 (correct)
- **LLM-only prediction:** ESI 2

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12273525 | Cureus.; 17(6):e86281 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.86281 Cardiology Coronary Vasospasm Masquerading as Coronary Artery Disease Muacevic Alexander Adler John R R... |
| 2 | PMC11465705 | Cureus.; 16(9):e69064 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.69064 Neurology Cardiology Substance Use and Addiction Lifestyle Effects on an Unusual Presentation of Syncop... |
| 3 | PMC11699854 | Cureus.; 16(12):e75153 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.75153 Neurology Cardiology Internal Medicine A Case of Severe Advanced Diabetic Cardiac Autonomic Neuropathy:... |
| 4 | PMC12315597 | Cureus.; 17(7):e87159 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.87159 Cardiology Emergency Medicine Syncope as the Initial Presentation of Severe Pulmonary Embolism Without ... |
| 5 | PMC12361826 | Cureus.; 17(7):e88337 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.88337 Geriatrics Internal Medicine Unmasking Postural Hypotension: A Stroke Mimic in an Elderly Patient Muace... |

### stay_id: 30078898

- **Chief complaint:** Chest pain, Elevated troponin, Transfer
- **Top-1 distance:** 0.2267
- **RAG prediction:** ESI 2 (correct)
- **LLM-only prediction:** ESI 2

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12750962 | Cureus.; 17(11):e98139 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.98139 Other Cardiology Emergency Medicine Complete Proximal Right Coronary Artery Occlusion in a Patient With... |
| 2 | PMC12750962 | Cureus.; 17(11):e98139 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.98139 Other Cardiology Emergency Medicine Complete Proximal Right Coronary Artery Occlusion in a Patient With... |
| 3 | PMC12750962 | Cureus.; 17(11):e98139 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.98139 Other Cardiology Emergency Medicine Complete Proximal Right Coronary Artery Occlusion in a Patient With... |
| 4 | PMC12483343 | Cureus.; 17(9):e93586 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.93586 Cardiology Emergency Medicine Internal Medicine An Atypical Presentation of ST-Segment Elevation Myocar... |
| 5 | PMC12267590 | Cureus.; 17(6):e86103 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.86103 Cardiology Emergency Medicine Internal Medicine New Right Bundle Branch Block: A Benign Variant or an O... |

### stay_id: 30160670

- **Chief complaint:** s/p Fall, R Leg injury
- **Top-1 distance:** 0.2376
- **RAG prediction:** ESI 2 (correct)
- **LLM-only prediction:** ESI 2

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC8892453 | Br Paramed J. 2022 Mar 1; 6(4):41-47 |  ==== Front Br Paramed J BPJ British Paramedic Journal 1478-4726 The College of Paramedics  35340582 BPJ-2022-6-4-41 10.29045/14784726.2022.03.6.4.41 Case Report An atypical presentation of orthostati... |
| 2 | PMC11957484 | J Am Coll Emerg Physicians Open. 2025 Mar 13; 6(3):100099 |  ==== Front J Am Coll Emerg Physicians Open J Am Coll Emerg Physicians Open Journal of the American College of Emergency Physicians Open 2688-1152 Elsevier  S2688-1152(25)00057-8 10.1016/j.acepjo.2025... |
| 3 | PMC12259304 | Case Rep Infect Dis. 2025 Jul 2; 2025:4053129 |  ==== Front Case Rep Infect Dis Case Rep Infect Dis CRIID Case Reports in Infectious Diseases 2090-6625 2090-6633 Wiley  10.1155/crdi/4053129 Case Report Perianesthetic Management in a Patient With Al... |
| 4 | PMC12101221 | Geriatrics (Basel). 2025 May 20; 10(3):68 |  ==== Front Geriatrics (Basel) Geriatrics (Basel) geriatrics Geriatrics 2308-3417 MDPI  40407575 10.3390/geriatrics10030068 geriatrics-10-00068 Case Report Case Report: Multifactorial Intervention for... |
| 5 | PMC12324059 | Medicine (Baltimore). 2025 Aug 1; 104(31):e41958 |  ==== Front Medicine (Baltimore) Medicine (Baltimore) MD Medicine 0025-7974 1536-5964 Lippincott Williams & Wilkins Hagerstown, MD  40760634 MD-D-24-13214 00134 10.1097/MD.0000000000041958 3 7100 Rese... |

### stay_id: 30187115

- **Chief complaint:** Influenza, Transfer
- **Top-1 distance:** 0.2352
- **RAG prediction:** ESI 2 (correct)
- **LLM-only prediction:** ESI 2

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12307254 | CEN Case Rep. 2024 Dec 11; 14(4):542-546 |  ==== Front CEN Case Rep CEN Case Rep CEN Case Reports 2192-4449 Springer Nature Singapore Singapore  39658702 951 10.1007/s13730-024-00951-6 Case Report Severe influenza A viral pneumonia in a hemodi... |
| 2 | PMC3109787 | Indian J Nephrol. 2011 Jan-Mar; 21(1):59-61 |  ==== Front Indian J NephrolIJNIndian Journal of Nephrology0971-40651998-3662Medknow Publications India IJN-21-5910.4103/0971-4065.78082Case ReportSuccessful treatment of critically ill chronic kidney... |
| 3 | PMC12846267 | Vaccines (Basel). 2026 Jan 4; 14(1):63 |  ==== Front Vaccines (Basel) Vaccines (Basel) vaccines Vaccines 2076-393X MDPI  10.3390/vaccines14010063 vaccines-14-00063 Article Influenza Vaccine Immunogenicity in Hemodialysis Patients https://orc... |
| 4 | PMC12846267 | Vaccines (Basel). 2026 Jan 4; 14(1):63 |  ==== Front Vaccines (Basel) Vaccines (Basel) vaccines Vaccines 2076-393X MDPI  10.3390/vaccines14010063 vaccines-14-00063 Article Influenza Vaccine Immunogenicity in Hemodialysis Patients https://orc... |
| 5 | PMC6374868 | Case Rep Med. 2019 Jan 30; 2019:1540761 |  ==== Front Case Rep MedCase Rep MedCRIMCase Reports in Medicine1687-96271687-9635Hindawi 10.1155/2019/1540761Case ReportSevere Influenza A(H1N1) Virus Infection Complicated by Myositis, Refractory Rh... |

### stay_id: 30205151

- **Chief complaint:** Melena
- **Top-1 distance:** 0.2438
- **RAG prediction:** ESI 2 (correct)
- **LLM-only prediction:** ESI 2

**Retrieved articles (top-5):**

| Rank | PMC ID | Citation | Snippet (first 200 chars) |
| --- | --- | --- | --- |
| 1 | PMC12439618 | Cureus.; 17(8):e90287 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.90287 Cardiology Cardiac/Thoracic/Vascular Surgery Transplantation Curative Radical Jejunectomy for a Refract... |
| 2 | PMC4828895 | J Cardiothorac Surg. 2016 Apr 12; 11:58 |  ==== Front J Cardiothorac SurgJ Cardiothorac SurgJournal of Cardiothoracic Surgery1749-8090BioMed Central London 44110.1186/s13019-016-0441-zReviewHemorrhoids screening and treatment prior to LVAD: i... |
| 3 | PMC11702465 | Clin Case Rep. 2025 Jan 6; 13(1):e9695 |  ==== Front Clin Case Rep Clin Case Rep 10.1002/(ISSN)2050-0904 CCR3 Clinical Case Reports 2050-0904 John Wiley and Sons Inc. Hoboken  10.1002/ccr3.9695 CCR39695 CCR3-2024-07-1977.R2 Cardiology Case R... |
| 4 | PMC11660727 | Cureus.; 16(11):e74082 |  ==== Front Cureus Cureus 2168-8184 Cureus 2168-8184 Cureus Palo Alto (CA)  10.7759/cureus.74082 Cardiology Preventive Medicine Infectious Disease Complications of Left Ventricular Assist Devices: A C... |
| 5 | PMC11366310 | None |  ==== Front Spartan Med Res J Spartan Med Res J 1364 Spartan Medical Research Journal 2474-7629 MSU College of Osteopathic Medicine Statewide Campus System Website: Spartan Medical Research Journal  1... |

---

## 5. Cross-group comparison: article content themes

Keyword presence in top-5 retrieved articles (% of articles containing keyword group):

| Keyword group | ESI-3 correct (n=10 articles) | ESI-3 wrong (n=95 articles) | ESI-2 correct (n=120 articles) |
| --- | --- | --- | --- |
| triage | 90% | 59% | 50% |
| treatment/management | 90% | 80% | 69% |
| diagnosis | 90% | 64% | 54% |
| epidemiology | 50% | 35% | 18% |
| pathophysiology | 20% | 13% | 12% |
| case report | 90% | 78% | 83% |
| ED/emergency | 60% | 42% | 32% |
| review/meta | 0% | 8% | 5% |

---

## 6. Article overlap and duplicate analysis

- Unique articles in ESI-3 correct group: 10
- Unique articles in ESI-3 wrong group: 87
- Overlap: 0 articles appear in both groups

---

## 7. FAISS duplicate retrieval (same article multiple ranks)

Cases where the same PMC ID appears at multiple ranks in the top-5/10,
indicating the FAISS index contains duplicate embeddings for the same article.

Cases with duplicate articles in top-5: 8 / 48

| stay_id | True ESI | RAG pred | Duplicate PMC IDs (count) |
| --- | --- | --- | --- |
| 30078898 | 2 | 2 | PMC12750962 (×3) |
| 30187115 | 2 | 2 | PMC12846267 (×2) |
| 30549289 | 2 | 2 | PMC12750962 (×2) |
| 30552702 | 3 | 2 | PMC12841809 (×2) |
| 30587111 | 2 | 2 | PMC12750962 (×3) |
| 30685431 | 2 | 2 | PMC12648516 (×2) |
| 30702659 | 2 | 2 | PMC12750962 (×3) |
| 30773795 | 2 | 2 | PMC12796546 (×2) |

