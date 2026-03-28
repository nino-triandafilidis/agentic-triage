# ESI v4 Few-Shot Demonstration Bank (Frozen)

Curated 5-case set for E18 prompt ablation: one case per ESI level, selected
for boundary-teaching value. All text is verbatim from Chapter 9 (Practice
Cases) of the ESI v4 Implementation Handbook.

## Provenance

- **Source:** Gilboy N, Tanabe P, Travers DA, Rosenau AM, Eitel DR.
  *Emergency Severity Index, Version 4: Implementation Handbook.*
  AHRQ Publication No. 05-0046-2. Rockville, MD: Agency for Healthcare
  Research and Quality. May 2005. Pages 63–68.
- **Full case bank file:** `data/corpus/esi_v4_practice_cases.md`
  SHA-256: `3513b82455372c90da12f1194156e4c57eba3d7cc8732ed3c534d9f245c398b7`
- **Frozen date:** 2026-03-09

## Selection Rationale

Each case was chosen to teach a specific ESI boundary decision that LLMs
commonly get wrong:

| ESI | Source    | Boundary lesson |
|-----|-----------|-----------------|
| 1   | Practice #12 | Shock recognition (ESI 1 vs 2): hemodynamic instability + severe pain distinguishes immediate life-saving intervention from high-risk |
| 2   | Practice #4  | Risk-factor recognition (ESI 2 vs 3): DKA risk in a diabetic with vomiting + abnormal vitals = unsafe to wait, even without current hemodynamic collapse |
| 3   | Practice #30 | Resource counting + stable vitals (ESI 3 vs 2): concerning history (prior ectopic) but hemodynamically stable = ESI 3, not 2 |
| 4   | Practice #14 | Single-resource identification (ESI 4 vs 5): one lab test (strep screen) = one resource |
| 5   | Practice #10 | Pain rating ≠ ESI level (ESI 5 vs 2): 10/10 pain alone does not override clinical picture when no resources are needed |

---

## Cases

### ESI 1 — Practice Case #12

**Case:** A 76-year-old male is brought to the ED because of severe abdominal pain. He tells you "it feels like someone is ripping me apart." The pain began about 30 minutes prior to admission and he rates the intensity as 20/10. He has hypertension for which he takes a diuretic. No allergies. The patient is sitting in a wheelchair moaning in pain. His skin is cool and diaphoretic. VS: HR 122, BP 88/68, RR 24, SpO2 94%.

**ESI Level:** 1
**Rationale:** Requires immediate life-saving intervention. The patient is presenting with signs of shock — hypotensive, tachycardic, with decreased peripheral perfusion. He has a history of hypertension and is presenting with signs and symptoms that could be attributed to a dissecting aortic abdominal aneurysm. He needs immediate IV access, aggressive fluid resuscitation, and perhaps blood prior to surgery.

---

### ESI 2 — Practice Case #4

**Case:** A 44-year-old female is retching continuously into a large basin as her son wheels her into the triage area. Her son tells you that his diabetic mother has been vomiting for the past 5 hours and now it is "just this yellow stuff." "She hasn't eaten or taken her insulin," he tells you. No known drug allergies (NKDA). VS: BP 148/70, P 126, RR 24.

**ESI Level:** 2
**Rationale:** High risk. A 44-year-old diabetic with continuous vomiting is at risk for diabetic ketoacidosis. The patient's vital signs are a concern as her heart rate and respiratory rate are both elevated. It is not safe for this patient to wait for an extended period of time in the waiting room.

---

### ESI 3 — Practice Case #30

**Case:** A 27-year-old female wants to be checked by a doctor. She has been experiencing low abdominal pain (6/10) for about 4 days. This morning she began spotting. She denies nausea, vomiting, diarrhea, or urinary symptoms. Her last menstrual period was 7 weeks ago. PMH: previous ectopic pregnancy. VS: T 98° F, HR 66, RR 14, BP 106/68.

**ESI Level:** 3
**Rationale:** Two or more resources. Based on her history, this patient will require two or more resources — lab and an ultrasound. She may in fact be pregnant. Ectopic pregnancy is on the differential diagnosis list, but this patient is currently hemodynamically stable and her pain is generalized across her lower abdomen.

---

### ESI 4 — Practice Case #14

**Case:** "I have a fever and a sore throat. I have finals this week and I am scared this is strep," reports a 19-year-old college student. She is sitting at triage drinking bottled water. No PMH, medications: birth control pills, no allergies to medications. VS: T 100.6° F, HR 88, RR 18, BP 112/76.

**ESI Level:** 4
**Rationale:** One resource. In most EDs, this patient will have a rapid strep screen sent to the lab; one resource. She is able to drink fluids and will be able to swallow pills if indicated.

---

### ESI 5 — Practice Case #10

**Case:** "My dentist can't see me until Monday and my tooth is killing me. Can't you give me something for the pain?" a 38-year-old healthy male asks the triage nurse. He tells you the pain started yesterday and he rates his pain as 10/10. No obvious facial swelling is noted. Allergic to Penicillin. VS: T 99.8° F, HR 78, RR 16, BP 128/74.

**ESI Level:** 5
**Rationale:** No resources should be necessary. He will require a physical exam but, without signs of an abscess or cellulitis, this patient will be referred to a dentist. In the ED he may be given oral medications and prescriptions for antibiotics and/or pain medication. He is not an ESI level 2, even though he rates his pain as 10/10. Based on the triage assessment, he would not be given the last open bed.
