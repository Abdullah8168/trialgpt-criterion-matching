# Error analysis (test, finetuned)

Error counts by category: UNSUPPORTED=28, MISSED_INFO=18, NOT_APPLICABLE=15, EVIDENCE_ONLY=14 (of 206 records)

## E1. UNSUPPORTED — patient sigir-201410, exclusion
- **Criterion:** Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2
- **Gold:** not enough information | evidence (none)
- **Predicted:** not excluded | evidence (none)
- **Raw output:** `{"evidence": [], "label": "not excluded"}`

## E2. UNSUPPORTED — patient sigir-201410, exclusion
- **Criterion:** Patients with previous axillary surgery
- **Gold:** not enough information | evidence (none)
- **Predicted:** not excluded | evidence (none)
- **Raw output:** `{"evidence": [], "label": "not excluded"}`

## E3. MISSED_INFO — patient sigir-201410, inclusion
- **Criterion:** Patients undergoing elective carotid endarterectomy
- **Gold:** not included | evidence (none)
- **Predicted:** not enough information | evidence (none)
- **Raw output:** `{"evidence": [], "label": "not enough information"}`

## E4. MISSED_INFO — patient sigir-201410, inclusion
- **Criterion:** Patients undergoing open abdominal aortic aneurysm repair
- **Gold:** not included | evidence (none)
- **Predicted:** not enough information | evidence [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- **Raw output:** `{"evidence": [0], "label": "not enough information"}`

## E5. NOT_APPLICABLE — patient sigir-201410, exclusion
- **Criterion:** Pregnancy
- **Gold:** not applicable | evidence (none)
- **Predicted:** not excluded | evidence (none)
- **Raw output:** `{"evidence": [], "label": "not excluded"}`

## E6. NOT_APPLICABLE — patient sigir-201422, inclusion
- **Criterion:** If female, using birth control
- **Gold:** not applicable | evidence [3] She has had no sexual partners and her menses are regular.
- **Predicted:** not enough information | evidence [0] A 15-year-old girl presents to the ER with abdominal pain.
- **Raw output:** `{"evidence": [0], "label": "not enough information"}`

## E7. EVIDENCE_ONLY — patient sigir-201411, exclusion
- **Criterion:** Less than 18 years of age
- **Gold:** not excluded | evidence [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
- **Predicted:** not excluded | evidence (none)
- **Raw output:** `{"evidence": [], "label": "not excluded"}`

## E8. EVIDENCE_ONLY — patient sigir-201411, exclusion
- **Criterion:** Decreased level of consciousness
- **Gold:** not excluded | evidence [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.; [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Predicted:** not excluded | evidence (none)
- **Raw output:** `{"evidence": [], "label": "not excluded"}`
