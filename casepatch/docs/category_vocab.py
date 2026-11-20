"""Curated vocabulary for health content and GDPR special-category values.

LICENSE: this list was compiled by hand for CasePatch from common clinical English (generic drug names, common
condition/procedure names, and code-system *structure*). It is dedicated to the public domain under CC0 1.0
(https://creativecommons.org/publicdomain/zero/1.0/) and may be redistributed freely. It contains:
- NO CPT descriptors (CPT is (c) American Medical Association) -- only plain-English procedure names;
- ICD-10-CM / ICD-9-CM code values for listed conditions (ICD-10-CM is published by CDC/NCHS and CMS as a US
  government work; individual codes are facts);
- generic (non-proprietary) drug names only, no brand names.
Grouping (body system / drug class) drives category-preserving substitution: a term is only ever replaced by a
different term from the same group. It is deliberately small (a few hundred entries) and is NOT a clinical
terminology; recall on free-text clinical notes will be limited.

Provenance note for the benchmark: this list was written while assembling case-real-05-health, so that case is not
a blind test of the vocabulary. Terms that appear only in that case's CMS boilerplate were deliberately NOT added.
"""

def _c(term, group, abbr=(), syn=(), icd10=None, icd9=None):
    return {"term": term, "group": group, "abbr": list(abbr), "syn": list(syn), "icd10": icd10, "icd9": icd9}


CONDITIONS = [
    # cardiovascular
    _c("hypertension", "cardio", ["HTN"], ["high blood pressure"], "I10", "401.9"),
    _c("coronary artery disease", "cardio", ["CAD"], ["coronary heart disease"], "I25.10", "414.01"),
    _c("atrial fibrillation", "cardio", ["AFib", "AF"], [], "I48.91", "427.31"),
    _c("congestive heart failure", "cardio", ["CHF"], ["heart failure"], "I50.9", "428.0"),
    _c("hyperlipidemia", "cardio", [], ["high cholesterol"], "E78.5", "272.4"),
    _c("myocardial infarction", "cardio", ["MI"], ["heart attack"], "I21.9", "410.90"),
    _c("stroke", "cardio", ["CVA"], ["cerebrovascular accident"], "I63.9", "434.91"),
    _c("deep vein thrombosis", "cardio", ["DVT"], [], "I82.409", "453.40"),
    _c("peripheral artery disease", "cardio", ["PAD"], ["peripheral vascular disease"], "I73.9", "443.9"),
    # endocrine / metabolic
    _c("diabetes mellitus", "endocrine", ["DM", "T2DM", "T1DM"], ["diabetes", "type 2 diabetes", "type 1 diabetes", "type II diabetes"], "E11.9", "250.00"),
    _c("hypothyroidism", "endocrine", [], [], "E03.9", "244.9"),
    _c("hyperthyroidism", "endocrine", [], ["graves disease", "graves' disease"], "E05.90", "242.90"),
    _c("obesity", "endocrine", [], ["morbid obesity"], "E66.9", "278.00"),
    _c("gout", "endocrine", [], [], "M10.9", "274.9"),
    _c("osteoporosis", "endocrine", [], [], "M81.0", "733.00"),
    # respiratory
    _c("asthma", "respiratory", [], [], "J45.909", "493.90"),
    _c("chronic obstructive pulmonary disease", "respiratory", ["COPD"], ["emphysema"], "J44.9", "496"),
    _c("pneumonia", "respiratory", [], [], "J18.9", "486"),
    _c("obstructive sleep apnea", "respiratory", ["OSA"], ["sleep apnea"], "G47.33", "327.23"),
    _c("bronchitis", "respiratory", [], [], "J40", "490"),
    _c("tuberculosis", "respiratory", ["TB"], [], "A15.9", "011.90"),
    # mental / behavioural
    _c("major depressive disorder", "mental", ["MDD"], ["depression", "clinical depression"], "F32.9", "311"),
    _c("generalized anxiety disorder", "mental", ["GAD"], ["anxiety disorder", "anxiety"], "F41.1", "300.02"),
    _c("bipolar disorder", "mental", [], ["manic depression"], "F31.9", "296.80"),
    _c("schizophrenia", "mental", [], [], "F20.9", "295.90"),
    _c("post-traumatic stress disorder", "mental", ["PTSD"], [], "F43.10", "309.81"),
    _c("attention deficit hyperactivity disorder", "mental", ["ADHD"], [], "F90.9", "314.01"),
    _c("alcohol use disorder", "mental", ["AUD"], ["alcoholism", "alcohol dependence"], "F10.20", "303.90"),
    _c("opioid use disorder", "mental", ["OUD"], ["opioid dependence"], "F11.20", "304.00"),
    _c("substance use disorder", "mental", ["SUD"], ["drug addiction"], "F19.20", "304.90"),
    _c("anorexia nervosa", "mental", [], ["eating disorder"], "F50.00", "307.1"),
    _c("autism spectrum disorder", "mental", ["ASD"], ["autism"], "F84.0", "299.00"),
    _c("dementia", "mental", [], [], "F03.90", "294.20"),
    _c("alzheimer's disease", "neuro", [], ["alzheimers disease", "alzheimer disease"], "G30.9", "331.0"),
    _c("insomnia", "mental", [], [], "G47.00", "780.52"),
    # neurological
    _c("epilepsy", "neuro", [], ["seizure disorder"], "G40.909", "345.90"),
    _c("migraine", "neuro", [], [], "G43.909", "346.90"),
    _c("multiple sclerosis", "neuro", ["MS"], [], "G35", "340"),
    _c("parkinson's disease", "neuro", [], ["parkinsons disease", "parkinson disease"], "G20", "332.0"),
    _c("peripheral neuropathy", "neuro", [], ["neuropathy"], "G62.9", "356.9"),
    # oncology
    _c("breast cancer", "oncology", [], ["malignant neoplasm of breast"], "C50.919", "174.9"),
    _c("lung cancer", "oncology", [], [], "C34.90", "162.9"),
    _c("prostate cancer", "oncology", [], [], "C61", "185"),
    _c("colorectal cancer", "oncology", [], ["colon cancer"], "C18.9", "153.9"),
    _c("leukemia", "oncology", [], [], "C95.90", "208.90"),
    _c("lymphoma", "oncology", [], ["non-hodgkin lymphoma", "hodgkin lymphoma"], "C85.90", "202.80"),
    _c("melanoma", "oncology", [], [], "C43.9", "172.9"),
    # infectious
    _c("HIV infection", "infectious", ["HIV", "AIDS"], ["HIV/AIDS", "HIV positive", "HIV-positive"], "B20", "042"),
    _c("hepatitis C", "infectious", ["HCV"], [], "B18.2", "070.54"),
    _c("hepatitis B", "infectious", ["HBV"], [], "B18.1", "070.32"),
    _c("COVID-19", "infectious", [], ["covid"], "U07.1", None),
    _c("influenza", "infectious", [], ["flu"], "J11.1", "487.1"),
    _c("syphilis", "infectious", [], [], "A53.9", "097.9"),
    _c("chlamydia", "infectious", [], [], "A74.9", "079.98"),
    _c("gonorrhea", "infectious", [], [], "A54.9", "098.0"),
    _c("genital herpes", "infectious", [], ["herpes"], "A60.00", "054.10"),
    # renal / urological
    _c("chronic kidney disease", "renal", ["CKD"], [], "N18.9", "585.9"),
    _c("renal failure", "renal", [], ["kidney failure"], "N19", "586"),
    _c("end-stage renal disease", "renal", ["ESRD"], [], "N18.6", "585.6"),
    _c("urinary incontinence", "renal", [], ["incontinence"], "R32", "788.30"),
    _c("urinary tract infection", "renal", ["UTI"], [], "N39.0", "599.0"),
    _c("kidney stones", "renal", [], ["nephrolithiasis"], "N20.0", "592.0"),
    # digestive
    _c("gastroesophageal reflux disease", "digestive", ["GERD"], ["acid reflux"], "K21.9", "530.81"),
    _c("crohn's disease", "digestive", [], ["crohns disease"], "K50.90", "555.9"),
    _c("ulcerative colitis", "digestive", ["UC"], [], "K51.90", "556.9"),
    _c("irritable bowel syndrome", "digestive", ["IBS"], [], "K58.9", "564.1"),
    _c("cirrhosis", "digestive", [], ["liver cirrhosis"], "K74.60", "571.5"),
    _c("celiac disease", "digestive", [], [], "K90.0", "579.0"),
    # musculoskeletal / autoimmune
    _c("rheumatoid arthritis", "musculoskeletal", ["RA"], [], "M06.9", "714.0"),
    _c("osteoarthritis", "musculoskeletal", ["OA"], [], "M19.90", "715.90"),
    _c("low back pain", "musculoskeletal", ["LBP"], ["chronic back pain"], "M54.50", "724.2"),
    _c("fibromyalgia", "musculoskeletal", [], [], "M79.7", "729.1"),
    _c("systemic lupus erythematosus", "musculoskeletal", ["SLE"], ["lupus"], "M32.9", "710.0"),
    # reproductive
    _c("pregnancy", "reproductive", [], ["pregnant"], "Z33.1", "V22.2"),
    _c("infertility", "reproductive", [], [], "N97.9", "628.9"),
    _c("miscarriage", "reproductive", [], ["spontaneous abortion"], "O03.9", "634.90"),
    _c("endometriosis", "reproductive", [], [], "N80.9", "617.9"),
    _c("erectile dysfunction", "reproductive", ["ED"], [], "N52.9", "607.84"),
    # eye / skin
    _c("glaucoma", "eye_skin", [], [], "H40.9", "365.9"),
    _c("cataract", "eye_skin", [], ["cataracts"], "H26.9", "366.9"),
    _c("macular degeneration", "eye_skin", ["AMD"], [], "H35.30", "362.50"),
    _c("psoriasis", "eye_skin", [], [], "L40.9", "696.1"),
    _c("atopic dermatitis", "eye_skin", [], ["eczema"], "L20.9", "691.8"),
    _c("actinic keratosis", "eye_skin", [], [], "L57.0", "702.0"),
    # blood
    _c("anemia", "blood", [], ["anaemia", "iron deficiency anemia"], "D64.9", "285.9"),
    _c("sickle cell disease", "blood", [], ["sickle cell anemia"], "D57.1", "282.60"),
    _c("hemophilia", "blood", [], [], "D66", "286.0"),
    # genetic
    _c("cystic fibrosis", "genetic", ["CF"], [], "E84.9", "277.00"),
    _c("down syndrome", "genetic", [], ["trisomy 21"], "Q90.9", "758.0"),
    _c("huntington's disease", "genetic", [], ["huntingtons disease"], "G10", "333.4"),
    # sensory / disability
    _c("hearing loss", "sensory", [], ["deafness"], "H91.90", "389.9"),
    _c("blindness", "sensory", [], ["visual impairment"], "H54.7", "369.9"),
]

MEDICATIONS = []
_MED_CLASSES = {
    "antidiabetic": "metformin glipizide glyburide sitagliptin empagliflozin dapagliflozin semaglutide liraglutide dulaglutide pioglitazone insulin_glargine insulin_lispro",
    "statin": "atorvastatin simvastatin rosuvastatin pravastatin lovastatin",
    "ace_arb": "lisinopril enalapril ramipril benazepril losartan valsartan irbesartan olmesartan",
    "beta_blocker": "metoprolol atenolol carvedilol propranolol bisoprolol",
    "ccb": "amlodipine diltiazem nifedipine verapamil",
    "diuretic": "hydrochlorothiazide furosemide chlorthalidone spironolactone",
    "anticoagulant": "warfarin apixaban rivaroxaban dabigatran clopidogrel",
    "antidepressant": "sertraline fluoxetine citalopram escitalopram paroxetine bupropion venlafaxine duloxetine trazodone mirtazapine",
    "antipsychotic_mood": "quetiapine olanzapine risperidone aripiprazole lithium lamotrigine haloperidol clozapine",
    "benzodiazepine_sleep": "alprazolam lorazepam clonazepam diazepam zolpidem",
    "stimulant": "methylphenidate lisdexamfetamine dextroamphetamine",
    "opioid": "oxycodone hydrocodone morphine fentanyl tramadol hydromorphone",
    "addiction_treatment": "buprenorphine methadone naltrexone naloxone",
    "anticonvulsant": "gabapentin pregabalin levetiracetam topiramate carbamazepine phenytoin",
    "thyroid": "levothyroxine methimazole liothyronine",
    "respiratory": "albuterol fluticasone budesonide montelukast tiotropium",
    "gastro": "omeprazole pantoprazole esomeprazole famotidine ondansetron",
    "antibiotic": "amoxicillin azithromycin doxycycline ciprofloxacin cephalexin metronidazole nitrofurantoin penicillin clindamycin",
    "antiretroviral": "tenofovir emtricitabine efavirenz dolutegravir bictegravir raltegravir",
    "antiviral": "oseltamivir acyclovir valacyclovir sofosbuvir",
    "immunology": "adalimumab etanercept infliximab methotrexate hydroxychloroquine prednisone ustekinumab rituximab",
    "oncology": "tamoxifen letrozole anastrozole pembrolizumab nivolumab trastuzumab cisplatin carboplatin paclitaxel",
    "reproductive_hormone": "estradiol progesterone levonorgestrel medroxyprogesterone norethindrone testosterone mifepristone misoprostol",
    "urology": "sildenafil tadalafil finasteride tamsulosin oxybutynin",
    "dementia": "donepezil memantine rivastigmine galantamine",
    "other": "allopurinol alendronate cyclobenzaprine",
}
for _g, _names in _MED_CLASSES.items():
    for _n in _names.split():
        MEDICATIONS.append(_c(_n.replace("_", " "), _g))

PROCEDURES = []
_PROC_GROUPS = {
    "imaging": "MRI|magnetic resonance imaging|CT scan|computed tomography|x-ray|mammogram|mammography|ultrasound|echocardiogram|PET scan|bone density scan",
    "endoscopy": "colonoscopy|upper endoscopy|endoscopy|sigmoidoscopy|bronchoscopy|cystoscopy",
    "cardiac": "cardiac catheterization|angioplasty|coronary artery bypass graft|CABG|pacemaker implantation|stent placement|electrocardiogram|EKG|ECG|cardiac stress test",
    "surgery": "appendectomy|cholecystectomy|hernia repair|hysterectomy|mastectomy|lumpectomy|tonsillectomy|knee replacement|hip replacement|arthroscopy|spinal fusion|cataract surgery|bariatric surgery|gastric bypass",
    "therapy": "chemotherapy|radiation therapy|radiotherapy|dialysis|hemodialysis|physical therapy|psychotherapy|electroconvulsive therapy|blood transfusion|immunotherapy",
    "reproductive": "cesarean section|C-section|abortion|in vitro fertilization|IVF|amniocentesis|vasectomy|tubal ligation",
    "minor": "biopsy|skin biopsy|cryotherapy|lesion excision|excision of lesion|skin lesion removal|vaccination|allergy testing",
}
for _g, _names in _PROC_GROUPS.items():
    for _n in _names.split("|"):
        PROCEDURES.append(_c(_n, _g))

# lab test names (a result = test name followed closely by a value)
LAB_TESTS = ["hemoglobin A1c", "HbA1c", "A1c", "LDL-C", "LDL", "HDL", "total cholesterol", "cholesterol", "triglycerides",
             "fasting glucose", "blood glucose", "glucose", "creatinine", "eGFR", "BUN", "TSH", "free T4", "hemoglobin",
             "hematocrit", "WBC", "platelets", "INR", "PSA", "ALT", "AST", "bilirubin", "albumin", "potassium", "sodium",
             "vitamin D", "CD4 count", "CD4", "viral load", "blood pressure", "BP", "BMI"]

# GDPR Art. 9 / 10 values (used ONLY with a label or a strong phrase; many are ordinary words in other contexts)
SPECIAL_VALUES = {
    "religion": ["Catholic", "Roman Catholic", "Protestant", "Christian", "Muslim", "Islam", "Jewish", "Judaism", "Hindu",
                 "Buddhist", "Sikh", "Mormon", "Latter-day Saint", "Jehovah's Witness", "Orthodox", "Atheist", "Agnostic",
                 "Baptist", "Methodist", "Lutheran", "Evangelical", "Anglican", "Presbyterian"],
    "racial_ethnic_origin": ["Hispanic", "Latino", "Latina", "Black", "African American", "White", "Caucasian", "Asian",
                             "Native American", "American Indian", "Alaska Native", "Pacific Islander", "Native Hawaiian",
                             "Middle Eastern", "Arab", "Roma", "Indigenous", "Asian American", "Not Hispanic or Latino"],
    "political_opinion": ["Democrat", "Republican", "Democratic Party", "Republican Party", "Labour", "Conservative",
                          "Green Party", "Libertarian", "Liberal Democrat", "Socialist", "Independent"],
    "sex_life_orientation": ["gay", "lesbian", "bisexual", "heterosexual", "straight", "homosexual", "queer", "asexual", "pansexual"],
    "trade_union": ["Teamsters", "AFL-CIO", "SEIU", "UAW", "AFSCME", "UNITE HERE", "Unite", "Unison", "GMB", "IG Metall", "Verdi"],
}
