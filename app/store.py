"""Clinical store, curated knowledge base, seeded patients, and Firestore write-through persistence."""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .firebase_client import get_firestore_client

logger = logging.getLogger("sakhi.store")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CLINIC_ID = "1c5b8ed4-0fbf-4131-be3c-f1c4e18ef34d"
DOCTOR_ID = "550e8400-e29b-41d4-a716-446655440001"

clinics: List[Dict[str, str]] = [
    {"id": CLINIC_ID, "name": "Sakhi Fertility & Wellness Clinic", "city": "Hyderabad"},
    {"id": "2d6c9fe5-10cg-4242-cf4d-g2d5f29fg45e", "name": "Medcy IVF & Maternity Center", "city": "Visakhapatnam"},
]

doctors: List[Dict[str, str]] = [
    {"id": DOCTOR_ID, "clinic_id": CLINIC_ID, "name": "Dr. Sample Gynecologist", "department": "Fertility"},
    {"id": "660f9511-f3ac-52e5-b827-557766550002", "clinic_id": CLINIC_ID, "name": "Dr. Sample Physician", "department": "General Medicine"},
]

clinic_patients: Dict[str, Dict[str, Any]] = {}
users_by_id: Dict[str, Dict[str, Any]] = {}
users_by_phone: Dict[str, Dict[str, Any]] = {}
users_by_email: Dict[str, Dict[str, Any]] = {}
user_contexts: Dict[str, Dict[str, Any]] = {}
chat_states: Dict[str, Dict[str, Any]] = {}
conversations: Dict[str, List[Dict[str, Any]]] = {}
user_rewards: Dict[str, int] = {}
user_answers: Dict[str, List[Dict[str, Any]]] = {}
leads: List[Dict[str, Any]] = []
escalations: List[Dict[str, Any]] = []
_processed_message_ids: Dict[str, float] = {}
_cached_care_preview: Optional[Dict[str, Any]] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _firestore_write(collection: str, doc_id: str, payload: Dict[str, Any], merge: bool = True) -> None:
    return  # In-memory only for demo mode, skip all network Firestore calls
    """Persist critical records to Cloud Firestore when initialized on Cloud Run."""
    db = get_firestore_client()
    if db is None:
        return
    try:
        db.collection(collection).document(doc_id).set(payload, merge=merge)
    except Exception:
        logger.warning(
            "Firestore write skipped due to transient backend error.",
            extra={"event": "firestore_write_error"},
        )


def _firestore_read_user(user_id: str) -> Optional[Dict[str, Any]]:
    return None  # In-memory only for demo mode, skip all network Firestore calls
    db = get_firestore_client()
    if db is None:
        return None
    try:
        snap = db.collection("sakhi_users").document(user_id).get()
        if snap.exists:
            data = snap.to_dict() or {}
            return data
    except Exception:
        logger.warning(
            "Firestore read failed; using local cache.",
            extra={"event": "firestore_read_error"},
        )
    return None


def normalize_phone(raw: Optional[str]) -> str:
    if not raw:
        return ""
    trimmed = str(raw).strip()
    if trimmed == "group_all":
        return "group_all"
    if trimmed.startswith("guest_"):
        return trimmed
    digits = re.sub(r"\D", "", trimmed)
    if not digits:
        return trimmed
    if len(digits) == 10:
        return f"+91{digits}"
    return f"+{digits}"


clinical_rules: List[Dict[str, Any]] = [
    {
        "department": "Fertility",
        "treatment_stage": "IUI Cycle",
        "scenario": "Mild cramping after IUI procedure",
        "keywords": ["cramp", "cramping", "pain", "ache", "abdomen", "stomach"],
        "emergency_flag": False,
        "clarifying_question": "Are you experiencing severe pain, heavy bleeding, or a fever above 100.4°F?",
        "escalation_trigger": "Severe pain, heavy bleeding, fever",
        "final_answer": "Mild cramping is very common for 1-2 days after an IUI. You can take Paracetamol for comfort and rest. Avoid strenuous exercise.",
    },
    {
        "department": "Fertility",
        "treatment_stage": "IUI Cycle",
        "scenario": "Light spotting after IUI",
        "keywords": ["spot", "spotting", "bleed", "bleeding", "blood", "discharge"],
        "emergency_flag": False,
        "clarifying_question": "Is the bleeding heavy (soaking a pad in an hour) or accompanied by sharp pelvic pain?",
        "escalation_trigger": "Heavy bleeding, sharp pelvic pain",
        "final_answer": "A small amount of light spotting (pink or brown) is normal after the catheter is inserted during IUI. Wear a pantyliner and avoid intercourse for 24 hours.",
    },
    {
        "department": "Fertility",
        "treatment_stage": "IVF Stimulation Phase",
        "scenario": "Bloating and discomfort during stimulation injections",
        "keywords": ["bloat", "bloating", "discomfort", "swollen", "fullness", "ohss", "abdomen", "stomach"],
        "emergency_flag": False,
        "clarifying_question": "Are you experiencing shortness of breath, severe abdominal pain, nausea, or rapid weight gain (more than 2 lbs in a day)?",
        "escalation_trigger": "Shortness of breath, severe pain, rapid weight gain, OHSS symptoms",
        "final_answer": "Mild to moderate bloating is a normal response as your ovaries enlarge from the stimulation medications. Drink plenty of electrolyte fluids and eat small, frequent meals.",
    },
    {
        "department": "Fertility",
        "treatment_stage": "IVF Stimulation Phase",
        "scenario": "Missed an injection dose",
        "keywords": ["miss", "missed", "forgot", "injection", "dose", "shot", "medication"],
        "emergency_flag": True,
        "clarifying_question": "",
        "escalation_trigger": "Missed dosage requires immediate adjustment",
        "final_answer": "Missing an injection during the stimulation phase can affect your cycle. I am alerting the clinic immediately so a nurse can advise you on how to adjust your dosage timing.",
    },
    {
        "department": "Fertility",
        "treatment_stage": "IVF Stimulation Phase",
        "scenario": "Mild spotting after embryo transfer",
        "keywords": ["spot", "spotting", "bleed", "bleeding"],
        "emergency_flag": False,
        "clarifying_question": "How many days has it been since embryo transfer, and is the bleeding light or heavy?",
        "escalation_trigger": "Heavy bleeding, severe pain, syncope",
        "final_answer": "Light spotting may occur after embryo transfer and does not always indicate a problem. Continue prescribed medications and rest.",
    },
    {
        "department": "Obstetrics",
        "treatment_stage": "First Trimester",
        "scenario": "Morning sickness, nausea, and vomiting",
        "keywords": ["nausea", "vomit", "vomiting", "morning sickness", "queasy", "sick"],
        "emergency_flag": False,
        "clarifying_question": "Are you unable to keep any fluids down for over 12 hours, feeling dizzy, or experiencing dark urine?",
        "escalation_trigger": "Severe dehydration, hyperemesis gravidarum, unable to keep fluids down",
        "final_answer": "Nausea is a common symptom in the first trimester. Try eating dry crackers before getting out of bed, eat small frequent meals, and take ginger or vitamin B6. Stay hydrated with small sips of water.",
    },
    {
        "department": "Obstetrics",
        "treatment_stage": "First Trimester",
        "scenario": "Lower abdominal cramping (mild)",
        "keywords": ["cramp", "cramping", "pain", "abdomen", "stomach", "pelvic"],
        "emergency_flag": False,
        "clarifying_question": "Is the cramping severe, strictly on one side, or accompanied by any vaginal bleeding?",
        "escalation_trigger": "Severe one-sided pain, bleeding (possible ectopic or miscarriage)",
        "final_answer": "Mild, intermittent cramping without bleeding is often caused by your uterus expanding. Rest and stay hydrated. However, always monitor for changes.",
    },
    {
        "department": "Pregnancy",
        "treatment_stage": "Trimester_1",
        "scenario": "Mild nausea and vomiting",
        "keywords": ["nausea", "vomit", "vomiting", "morning sickness"],
        "emergency_flag": False,
        "clarifying_question": "Are you able to keep fluids down?",
        "escalation_trigger": "Unable to tolerate fluids for >24 hours, signs of dehydration, vomiting blood",
        "final_answer": "Mild nausea is common in early pregnancy. Small frequent meals and adequate hydration may help while we assess your symptoms.",
    },
    {
        "department": "Pregnancy",
        "treatment_stage": "Trimester_1",
        "scenario": "Headache during pregnancy",
        "keywords": ["headache", "migraine", "head pain"],
        "emergency_flag": False,
        "clarifying_question": "Is the headache mild or severe, and do you have any blurred vision or swelling?",
        "escalation_trigger": "Severe headache, blurred vision, BP >=140/90, swelling, confusion",
        "final_answer": "Headaches can occur during pregnancy due to hormonal changes, dehydration, or fatigue. Rest, hydrate, and consult your doctor before taking medication.",
    },
    {
        "department": "Pregnancy",
        "treatment_stage": "Trimester_1",
        "scenario": "Vaginal bleeding in pregnancy",
        "keywords": ["bleed", "bleeding", "blood", "spotting"],
        "emergency_flag": True,
        "clarifying_question": "",
        "escalation_trigger": "ALL",
        "final_answer": "",
    },
    {
        "department": "Pregnancy",
        "treatment_stage": "Trimester_1",
        "scenario": "Decreased fetal movement",
        "keywords": ["fetal", "movement", "kicking", "baby not moving"],
        "emergency_flag": True,
        "clarifying_question": "",
        "escalation_trigger": "ALL",
        "final_answer": "",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Fever",
        "keywords": ["fever", "temperature", "chills", "hot", "pyrexia"],
        "emergency_flag": False,
        "clarifying_question": "What is your highest recorded temperature and how long have you had the fever?",
        "escalation_trigger": "Temperature >=39.5°C / 103°F with confusion or persistent fever >72 hrs",
        "final_answer": "Most fevers are caused by infections and are self-limiting. Stay hydrated, rest adequately, and monitor your temperature.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Chest Pain",
        "keywords": ["chest pain", "chest tightness", "heart pain"],
        "emergency_flag": True,
        "clarifying_question": "",
        "escalation_trigger": "ALL",
        "final_answer": "",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Difficulty Breathing",
        "keywords": ["breathing", "shortness of breath", "breathless", "cant breathe", "wheezing"],
        "emergency_flag": True,
        "clarifying_question": "",
        "escalation_trigger": "ALL",
        "final_answer": "",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Headache",
        "keywords": ["headache", "migraine", "head pain"],
        "emergency_flag": False,
        "clarifying_question": "Did the headache start suddenly or gradually, and is it severe?",
        "escalation_trigger": "Sudden worst headache, neurological symptoms, confusion, vision loss",
        "final_answer": "Many headaches are due to stress, dehydration, or viral illness. Rest in a quiet room and stay well hydrated.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Cough",
        "keywords": ["cough", "coughing", "phlegm"],
        "emergency_flag": False,
        "clarifying_question": "Are you coughing up blood or experiencing breathlessness?",
        "escalation_trigger": "Hemoptysis, coughing blood, oxygen saturation <92%, breathlessness",
        "final_answer": "Viral coughs usually improve within 1–3 weeks with fluids and rest.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Sore Throat",
        "keywords": ["sore throat", "throat pain", "throat"],
        "emergency_flag": False,
        "clarifying_question": "Are you having difficulty swallowing saliva or breathing?",
        "escalation_trigger": "Airway compromise, difficulty swallowing saliva, neck swelling",
        "final_answer": "Most sore throats are viral and improve with warm fluids, salt-water gargles, and supportive care.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Vomiting",
        "keywords": ["vomit", "vomiting", "throw up", "nausea"],
        "emergency_flag": False,
        "clarifying_question": "How many times have you vomited today, and can you keep fluids down?",
        "escalation_trigger": "Persistent vomiting with dehydration, vomiting blood",
        "final_answer": "Small frequent sips of oral rehydration fluids can help prevent dehydration.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Diarrhea",
        "keywords": ["diarrhea", "loose motion", "loose stool", "watery stool"],
        "emergency_flag": False,
        "clarifying_question": "Is there blood in the stool or signs of severe dehydration?",
        "escalation_trigger": "Bloody diarrhea or severe dehydration",
        "final_answer": "Oral rehydration (ORS) is important to prevent dehydration. Avoid heavy or spicy foods until it settles.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Abdominal Pain",
        "keywords": ["abdominal pain", "stomach pain", "stomach ache", "belly pain"],
        "emergency_flag": False,
        "clarifying_question": "Where exactly is the pain located and how severe is it?",
        "escalation_trigger": "Severe localized pain, rigid abdomen",
        "final_answer": "Abdominal pain has many causes. Rest, sip water, and monitor whether the pain worsens or localizes.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Dizziness",
        "keywords": ["dizzy", "dizziness", "lightheaded", "vertigo"],
        "emergency_flag": False,
        "clarifying_question": "Did you lose consciousness or feel any weakness on one side?",
        "escalation_trigger": "Syncope, fainting, stroke symptoms",
        "final_answer": "Dizziness may result from dehydration, low blood pressure, or inner ear disorders. Sit or lie down slowly and hydrate.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "High Blood Pressure",
        "keywords": ["blood pressure", "bp", "hypertension"],
        "emergency_flag": False,
        "clarifying_question": "What is your current blood pressure reading, and do you have chest pain or headache?",
        "escalation_trigger": "BP >=180/120 with symptoms",
        "final_answer": "A single high reading does not always indicate an emergency. Rest for 5 minutes and re-check your blood pressure.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Back Pain",
        "keywords": ["back pain", "backache", "spine"],
        "emergency_flag": False,
        "clarifying_question": "Do you have leg weakness, numbness, or loss of bladder control?",
        "escalation_trigger": "Cauda equina symptoms, leg weakness, loss of bladder control",
        "final_answer": "Mechanical back pain is common and often improves with gentle rest, posture care, and activity modification.",
    },
    {
        "department": "General Medicine",
        "treatment_stage": "Initial Assessment",
        "scenario": "Fatigue",
        "keywords": ["fatigue", "tired", "weakness", "exhausted"],
        "emergency_flag": False,
        "clarifying_question": "How long have you been experiencing fatigue, and do you have fever or dizziness?",
        "escalation_trigger": "Severe weakness with altered mental status",
        "final_answer": "Fatigue can stem from poor sleep, dehydration, or nutritional deficiencies. Ensure rest and hydration, and discuss persistent fatigue with your doctor.",
    },
]

medical_knowledge: List[Dict[str, str]] = [
    {
        "id": "mk-ivf-overview",
        "category": "IVF Overview",
        "lens": "fertility",
        "sub_category": "What is IVF",
        "title": "What is IVF (In Vitro Fertilization)?",
        "content": (
            "IVF (In Vitro Fertilization) is an assisted reproductive treatment where a woman's eggs are fertilized with sperm outside the body in a specialized embryology lab, and the resulting healthy embryo is gently transferred into the uterus.\n\n"
            "A typical IVF cycle has 5 main stages:\n"
            "1. Ovarian Stimulation (8–12 days): Hormone injections help your ovaries mature multiple eggs instead of just one.\n"
            "2. Trigger Shot & Egg Retrieval: A final timed injection matures the eggs, followed by a brief 15–20 minute painless procedure under light anesthesia to collect them.\n"
            "3. Fertilization & Embryo Culture: Eggs are fertilized with sperm in the lab (or via ICSI) and monitored for 3–5 days until they reach the blastocyst stage.\n"
            "4. Embryo Transfer: A healthy embryo is placed into your uterus using a thin, soft catheter—usually painless and requiring no anesthesia.\n"
            "5. Two-Week Wait & Beta-hCG Test: About 12–14 days later, a blood test confirms pregnancy.\n\n"
            "IVF is commonly recommended for blocked fallopian tubes, severe male factor infertility, endometriosis, PCOS that hasn't responded to simpler treatments, unexplained infertility, or age-related fertility decline."
        ),
        "language": "en",
    },
    {
        "id": "mk-iui-overview",
        "category": "IUI Overview",
        "lens": "fertility",
        "sub_category": "What is IUI",
        "title": "What is IUI (Intrauterine Insemination)?",
        "content": (
            "IUI (Intrauterine Insemination) is a gentle, minimally invasive fertility treatment where washed and concentrated motile sperm are placed directly inside the uterus right around the time of ovulation.\n\n"
            "It shortens the distance sperm need to travel to reach the egg, increasing the chance of fertilization. IUI is typically quick (5–10 minutes), requires no anesthesia, and is often a first-line option for mild male factor infertility, unexplained infertility, ovulation disorders, or cervical mucus issues."
        ),
        "language": "en",
    },
    {
        "id": "mk-age-fertility",
        "category": "Fertility Planning",
        "lens": "fertility",
        "sub_category": "Age & Fertility",
        "title": "How Age Affects Fertility & Chances of Conceiving",
        "content": (
            "Age is an important factor in fertility because women are born with a finite reserve of eggs, and both egg quantity and egg chromosomal quality naturally decline over time—gradually after age 32 and more noticeably after age 35–37. Male sperm quality can also decline gradually after age 40–45.\n\n"
            "However, age is only one part of the picture. Many women in their mid-to-late 30s and early 40s conceive healthy pregnancies naturally or with treatments like ovulation induction, IUI, or IVF. Checking ovarian reserve markers like AMH (Anti-Müllerian Hormone) and an Antral Follicle Count (AFC) ultrasound gives a much clearer picture of your individual timeline."
        ),
        "language": "en",
    },
    {
        "id": "mk-preg-routines",
        "category": "Pregnancy Care",
        "lens": "nutrition",
        "sub_category": "Healthy Pregnancy Routines",
        "title": "Healthy Daily Routines During Pregnancy",
        "content": (
            "A steady, gentle daily routine supports both you and your baby during pregnancy:\n"
            "• Balanced Indian Meals: Eat small, frequent meals rich in protein (dal, paneer, eggs, curds, sprouts), iron, calcium, leafy greens, and seasonal fruits.\n"
            "• Prenatal Vitamins: Take prescribed folic acid, iron, calcium, and vitamin D consistently.\n"
            "• Hydration & Rest: Aim for 8–10 glasses of water/fluids daily and 7–9 hours of sleep (sleeping on your left side is especially helpful in the 2nd and 3rd trimesters).\n"
            "• Gentle Movement: 20–30 minutes of light walking or prenatal yoga (with your doctor's approval) boosts circulation and reduces stress.\n"
            "• Regular Checkups: Keep up with scheduled scans, blood tests, and blood pressure monitoring."
        ),
        "language": "en",
    },
    {
        "id": "mk-egg-quality",
        "category": "Diet & Nutrition",
        "lens": "nutrition",
        "sub_category": "Egg Quality & Fertility Diet",
        "title": "Natural Foods & Habits to Support Egg Quality",
        "content": (
            "Eggs take about 90 days to mature before ovulation, so consistent nutrition and lifestyle habits make a meaningful difference:\n"
            "• Antioxidant-rich foods: Berries, amla, pomegranate, spinach, broccoli, and colorful vegetables help protect eggs from oxidative stress.\n"
            "• Healthy fats: Walnuts, almonds, flaxseeds, chia seeds, avocado, and cold-pressed oils support hormone production.\n"
            "• Plant & lean proteins: Lentils (dal), chickpeas, paneer, curd, eggs, and fish low in mercury.\n"
            "• Key nutrients: Folate, Vitamin D, Iron, Omega-3s, and CoQ10 (if recommended by your fertility doctor).\n"
            "• Lifestyle: Prioritize 7–8 hours of sleep, limit refined sugar and trans fats, keep caffeine under 200 mg/day, and avoid smoking/alcohol."
        ),
        "language": "en",
    },
    {
        "id": "mk-tww-anxiety",
        "category": "Emotional Support",
        "lens": "emotional",
        "sub_category": "Two-Week Wait & Treatment Anxiety",
        "title": "Coping With the Two-Week Wait & Pre-Scan Anxiety",
        "content": (
            "Waiting for test results or a scan can feel emotionally exhausting—every twinge or symptom makes your mind race. Please know that feeling anxious, hopeful, and overwhelmed all at once is completely normal.\n\n"
            "Gentle strategies that help:\n"
            "• Pause symptom spotting: Progesterone medications often mimic both pregnancy and period symptoms, so early sensations rarely tell the full story.\n"
            "• Grounding breathwork: Try 4-4-6 breathing (inhale 4 seconds, hold 4, exhale 6) whenever worry spikes.\n"
            "• Protect your peace: Set kind boundaries around family questions and take a break from stressful online forums.\n"
            "• Stay gently occupied: Plan light, comforting routines—a walk, warm tea, music, or talking with a trusted partner or friend."
        ),
        "language": "en",
    },
    {
        "id": "mk-iui-1",
        "category": "IUI Cycle",
        "lens": "fertility",
        "sub_category": "Success Rates",
        "title": "IUI Success Rates & Cycle Expectations",
        "content": "The success rate of IUI depends on factors like age and underlying fertility issues. On average, it has a 10-20% success rate per cycle for women under 35. It is normal to require 3-4 cycles to achieve pregnancy.",
        "language": "en",
    },
    {
        "id": "mk-ivf-1",
        "category": "IVF Stimulation Phase",
        "lens": "fertility",
        "sub_category": "Trigger Shot",
        "title": "Understanding the IVF Trigger Shot",
        "content": "The 'trigger shot' (usually hCG or Lupron) is the final injection taken precisely 36 hours before egg retrieval. It matures the eggs. Timing is absolutely critical for this injection.",
        "language": "en",
    },
    {
        "id": "mk-ivf-2",
        "category": "IVF Stimulation Phase",
        "lens": "fertility",
        "sub_category": "Post-Transfer Spotting",
        "title": "Post-Transfer Spotting & Care",
        "content": "Light pink or brown spotting may occur after embryo transfer. Continue prescribed medications, avoid strenuous activity as instructed by the fertility specialist, and report heavy bleeding or severe pain promptly.",
        "language": "en",
    },
    {
        "id": "mk-preg-1",
        "category": "First Trimester",
        "lens": "nutrition",
        "sub_category": "Diet in First Trimester",
        "title": "Diet & Nutrition in the First Trimester",
        "content": "During the first trimester, focus on taking your prenatal vitamins (especially Folic Acid). Avoid raw meats, unpasteurized dairy, and high-mercury fish. Limit caffeine to under 200mg per day.",
        "language": "en",
    },
    {
        "id": "mk-preg-2",
        "category": "Trimester_1",
        "lens": "medical",
        "sub_category": "Nausea & Vomiting",
        "title": "Managing Early Pregnancy Nausea",
        "content": "Nausea and vomiting are common during early pregnancy. Small frequent meals, avoiding triggers, and maintaining hydration are usually recommended. Persistent vomiting, inability to retain fluids, or signs of dehydration require medical assessment.",
        "language": "en",
    },
    {
        "id": "mk-preg-3",
        "category": "First Trimester",
        "lens": "medical",
        "sub_category": "Headache",
        "title": "Headaches During Pregnancy",
        "content": "Mild headaches during pregnancy may be related to hormonal changes, dehydration, or fatigue. Encourage adequate hydration, rest, regular meals, and discuss appropriate pain relief with your clinician. Seek immediate care for severe headache or visual disturbances.",
        "language": "en",
    },
    {
        "id": "mk-pcos-1",
        "category": "PCOS",
        "lens": "medical",
        "sub_category": "Irregular Menstrual Cycles",
        "title": "Managing PCOS & Irregular Cycles",
        "content": "PCOS commonly presents with irregular or absent menstrual cycles due to ovulatory dysfunction. Management focuses on a balanced low-glycemic diet, maintaining a healthy BMI, and individualized care with your clinician.",
        "language": "en",
    },
    {
        "id": "mk-gen-1",
        "category": "Initial Assessment",
        "lens": "medical",
        "sub_category": "Viral Fever",
        "title": "Managing Viral Fever Safely",
        "content": "Fever is a normal immune response. Maintain hydration, rest adequately, monitor temperature, and consider paracetamol for comfort if appropriate. Seek medical review if fever persists beyond 72 hours, exceeds 39.5°C, or is associated with confusion or breathing difficulty.",
        "language": "en",
    },
    {
        "id": "mk-fin-1",
        "category": "Financial Planning",
        "lens": "financial",
        "sub_category": "IVF Cost & Planning",
        "title": "Planning Financially for Fertility Treatment Cycles",
        "content": "Fertility treatments in India vary by protocol, medications, lab diagnostics, and whether multiple cycles or embryo freezing are needed. Ask your clinic for an itemized cycle estimate covering stimulation injections, retrieval, transfer, and cryopreservation.",
        "language": "en",
    },
    {
        "id": "mk-fin-2",
        "category": "Financial Planning",
        "lens": "financial",
        "sub_category": "Insurance & EMI Options",
        "title": "Understanding Insurance Coverage & Clinic EMI Plans",
        "content": "While standard health policies may have waiting periods for maternity and limited ART coverage, many Indian clinics offer transparent milestone billing and zero-cost EMI counseling to reduce upfront family stress.",
        "language": "en",
    },
]


def _seed_initial_data() -> None:
    pre_care_users = [
        {"id": "u-venkatesh", "phone": "+911111111111", "name": "Venkatesh", "preferred_language": "te", "rewards": 25},
        {"id": "u-suresh", "phone": "+912222222222", "name": "Suresh", "preferred_language": "en", "rewards": 15},
        {"id": "u-group", "phone": "group_all", "name": "All Patients Group", "preferred_language": "en", "rewards": 10},
    ]

    for u in pre_care_users:
        user_obj = {
            "user_id": u["id"],
            "id": u["id"],
            "name": u["name"],
            "phone_number": u["phone"],
            "email": f"{u['id']}@sakhi.local",
            "role": "patient",
            "preferred_language": u["preferred_language"],
            "relation": "self",
            "created_at": _utc_now_iso(),
        }
        users_by_id[u["id"]] = user_obj
        users_by_phone[u["phone"]] = user_obj
        user_rewards[u["id"]] = u["rewards"]

    post_care_patients = [
        {"id": "p-ramesh", "mobile": "+913333333333", "name": "Ramesh", "uhid": "TEST-POST-01", "department": "Fertility", "treatment_name": "IVF Stimulation Phase", "rewards": 40},
        {"id": "p-mahesh", "mobile": "+914444444444", "name": "Mahesh", "uhid": "TEST-POST-02", "department": "General Medicine", "treatment_name": "Initial Assessment", "preferred_language": "te", "rewards": 30},
        {"id": "p-genmed", "mobile": "+918888888881", "name": "Rohan Verma", "uhid": "TEST-GEN-02", "department": "General Medicine", "treatment_name": "Initial Assessment", "rewards": 10},
        {"id": "p-ivf", "mobile": "+918888888882", "name": "Ananya Reddy", "uhid": "TEST-IVF-02", "department": "Fertility", "treatment_name": "IVF Stimulation Phase", "rewards": 20},
        {"id": "p-iui", "mobile": "+918888888883", "name": "Priya Sharma", "uhid": "TEST-IUI-02", "department": "Fertility", "treatment_name": "IUI Cycle", "rewards": 15},
        {"id": "p-obs", "mobile": "+918888888884", "name": "Kavya Nair", "uhid": "TEST-OBS-01", "department": "Obstetrics", "treatment_name": "First Trimester", "rewards": 20},
        {"id": "p-preg", "mobile": "+918888888885", "name": "Lakshmi Devi", "uhid": "TEST-PRG-01", "department": "Pregnancy", "treatment_name": "Trimester_1", "rewards": 20},
        {"id": "p-demo-iui", "mobile": "+919988776655", "name": "Meera Krishnan", "uhid": "DEMO-IUI-001", "department": "Fertility", "treatment_name": "IUI Cycle", "rewards": 10},
        {"id": "p-demo-ivf", "mobile": "+919988776644", "name": "Divya Rao", "uhid": "DEMO-IVF-001", "department": "Fertility", "treatment_name": "IVF Stimulation Phase", "rewards": 10},
        {"id": "p-demo-preg", "mobile": "+919988776633", "name": "Sneha Patel", "uhid": "DEMO-PREG-001", "department": "Obstetrics", "treatment_name": "First Trimester", "rewards": 10},
    ]

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for p in post_care_patients:
        patient_record = {
            "id": p["id"],
            "name": p["name"],
            "mobile": p["mobile"],
            "clinic_id": CLINIC_ID,
            "assigned_doctor_id": DOCTOR_ID,
            "uhid": p["uhid"],
            "department": p["department"],
            "status": "Active",
            "timeline": [
                {
                    "event_date": today_str,
                    "event_type": "TREATMENT",
                    "title": p["treatment_name"],
                    "description": "Active",
                }
            ],
        }
        clinic_patients[p["mobile"]] = patient_record

        user_obj = {
            "user_id": p["id"],
            "id": p["id"],
            "name": p["name"],
            "phone_number": p["mobile"],
            "email": f"{p['id']}@sakhi.local",
            "role": "patient",
            "preferred_language": p.get("preferred_language", "en"),
            "relation": "self",
            "created_at": _utc_now_iso(),
        }
        users_by_id[p["id"]] = user_obj
        users_by_phone[p["mobile"]] = user_obj
        user_rewards[p["id"]] = p.get("rewards", 0)


def _load_knowledge_from_json() -> None:
    file_path = DATA_DIR / "emo_hospital_info.json"
    if not file_path.exists():
        return
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        structure = raw.get("document_structure") or []
        counter = 1
        for h1 in structure:
            h1_title = h1.get("title") or "General"
            lower_h1 = h1_title.lower()
            lens = "medical"
            if "emotional" in lower_h1 or "social" in lower_h1:
                lens = "emotional"
            elif "diet" in lower_h1 or "lifestyle" in lower_h1:
                lens = "nutrition"
            elif "treatment" in lower_h1 or "success" in lower_h1:
                lens = "fertility"

            for h2 in h1.get("children") or []:
                sub_category = h2.get("title") or h1_title
                for h3 in h2.get("children") or []:
                    question = h3.get("title") or ""
                    text = " ".join(c.get("text", "") for c in (h3.get("chunks") or []))
                    if question and text:
                        medical_knowledge.append(
                            {
                                "id": f"kb-json-{counter}",
                                "category": h1_title,
                                "lens": lens,
                                "sub_category": sub_category,
                                "title": question,
                                "content": text,
                                "language": "en",
                            }
                        )
                        counter += 1
    except Exception:
        logger.warning("Could not parse emo_hospital_info.json.", extra={"event": "kb_parse_error"})


_seed_initial_data()
_load_knowledge_from_json()


def get_predefined_care_preview() -> Dict[str, Any]:
    global _cached_care_preview
    if _cached_care_preview is not None:
        return _cached_care_preview
    preview_path = DATA_DIR / "sakhi_predefined_care_data.json"
    if preview_path.exists():
        try:
            _cached_care_preview = json.loads(preview_path.read_text(encoding="utf-8"))
            return _cached_care_preview
        except Exception:
            pass
    return {"pre_care": {"topics": []}, "post_care": {"patients": []}}


_guest_sessions: Dict[str, Dict[str, Any]] = {}
MAX_GUEST_SESSIONS = 5000


def get_or_create_guest_session(session_id: str, default_name: Optional[str] = None) -> Dict[str, Any]:
    global _guest_sessions
    if len(_guest_sessions) > MAX_GUEST_SESSIONS:
        oldest_keys = list(_guest_sessions.keys())[:1000]
        for k in oldest_keys:
            _guest_sessions.pop(k, None)
            conversations.pop(k, None)
            user_contexts.pop(k, None)
            chat_states.pop(k, None)

    session = _guest_sessions.get(session_id)
    if not session:
        session = {
            "session_id": session_id,
            "user_id": session_id,
            "id": session_id,
            "name": default_name,
            "phone_number": session_id,
            "email": None,
            "role": "guest",
            "preferred_language": "en",
            "relation": "self",
            "created_at": _utc_now_iso(),
            "last_active": _utc_now_iso(),
        }
        _guest_sessions[session_id] = session
    else:
        session["last_active"] = _utc_now_iso()
        if default_name and not session.get("name"):
            session["name"] = default_name
    return session


def get_user_by_phone(phone: Optional[str]) -> Optional[Dict[str, Any]]:
    if not phone:
        return None
    if phone.startswith("guest_"):
        return _guest_sessions.get(phone) or get_or_create_guest_session(phone)
    norm = normalize_phone(phone)
    return users_by_phone.get(norm) or users_by_phone.get(phone)


def get_user_by_id(user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not user_id:
        return None
    if user_id.startswith("guest_"):
        return _guest_sessions.get(user_id) or get_or_create_guest_session(user_id)
    cached = users_by_id.get(user_id)
    if cached:
        return cached
    persisted = _firestore_read_user(user_id)
    if persisted:
        users_by_id[user_id] = persisted
        if persisted.get("phone_number"):
            users_by_phone[persisted["phone_number"]] = persisted
        if persisted.get("email"):
            users_by_email[str(persisted["email"]).lower()] = persisted
        return persisted
    return None


def save_user_profile(user_row: Dict[str, Any]) -> Dict[str, Any]:
    uid = user_row["user_id"]
    # Guest sessions live in _guest_sessions only — never write to Firestore or users_by_id
    if uid.startswith("guest_"):
        existing = _guest_sessions.get(uid)
        if existing is not None:
            existing.update(user_row)
        else:
            _guest_sessions[uid] = user_row
        return _guest_sessions[uid]
    users_by_id[uid] = user_row
    if user_row.get("phone_number"):
        users_by_phone[user_row["phone_number"]] = user_row
    if user_row.get("email"):
        users_by_email[str(user_row["email"]).lower()] = user_row
    _firestore_write("sakhi_users", uid, user_row, merge=True)
    return user_row


def create_partial_user(phone: str) -> Dict[str, Any]:
    norm = normalize_phone(phone)
    existing = get_user_by_phone(norm)
    if existing:
        return existing
    uid = str(uuid.uuid4())
    user = {
        "user_id": uid,
        "id": uid,
        "name": None,
        "phone_number": norm,
        "email": None,
        "role": "patient",
        "preferred_language": "en",
        "relation": "self",
        "created_at": _utc_now_iso(),
    }
    user_rewards[uid] = 0
    return save_user_profile(user)


def update_user_profile(user_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    user = get_user_by_id(user_id)
    if not user:
        return None
    user.update(patch)
    return save_user_profile(user)


def save_user_answers(user_id: str, answers: List[Dict[str, Any]]) -> int:
    existing = user_answers.setdefault(user_id, [])
    existing.extend(answers)
    _firestore_write(
        "sakhi_user_answers",
        user_id,
        {"user_id": user_id, "answers": existing, "updated_at": _utc_now_iso()},
        merge=True,
    )
    return len(answers)


def get_user_context(user_id: str) -> Dict[str, Any]:
    return user_contexts.get(user_id, {})


def update_user_context(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    current = dict(user_contexts.get(user_id, {}))
    current.update(patch)
    user_contexts[user_id] = current
    return current


def get_chat_state(user_id: str) -> Dict[str, Any]:
    return chat_states.get(user_id, {})


def update_chat_state(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    current = dict(chat_states.get(user_id, {}))
    current.update(patch)
    chat_states[user_id] = current
    return current


def get_patient_by_phone(phone: Optional[str]) -> Optional[Dict[str, Any]]:
    if not phone:
        return None
    if str(phone).startswith("guest_"):
        return None
    norm = normalize_phone(phone)
    return clinic_patients.get(norm) or clinic_patients.get(phone)


def save_message(user_id: str, role: str, content: str, language: str = "en") -> None:
    history = conversations.setdefault(user_id, [])
    entry = {
        "role": role,
        "content": content,
        "language": language,
        "timestamp": _utc_now_iso(),
    }
    history.append(entry)
    if len(history) > 50:
        history.pop(0)
    _firestore_write(
        "sakhi_conversations",
        user_id,
        {"user_id": user_id, "messages": history[-20:], "updated_at": entry["timestamp"]},
        merge=True,
    )


def get_last_messages(user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    return (conversations.get(user_id) or [])[-limit:]


def get_rewards(user_id: str) -> int:
    return user_rewards.get(user_id, 0)


def add_rewards(user_id: str, points: int = 5) -> int:
    total = user_rewards.get(user_id, 0) + points
    user_rewards[user_id] = total
    return total


def record_lead(lead_data: Dict[str, Any], added_by_user_id: str) -> Dict[str, Any]:
    lead = {
        "id": str(uuid.uuid4()),
        **lead_data,
        "status": "New Inquiry",
        "source": "Whatsapp-Sakhi",
        "added_by_user_id": added_by_user_id,
        "created_at": _utc_now_iso(),
    }
    leads.append(lead)
    _firestore_write("sakhi_leads", lead["id"], lead, merge=False)
    return lead


def record_escalation(
    user_id: str,
    phone: str,
    mode: str,
    intent: str,
    reply: str,
) -> Dict[str, Any]:
    esc = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "phone_number": phone,
        "mode": mode,
        "intent": intent,
        "reply": reply,
        "status": "pending_review",
        "created_at": _utc_now_iso(),
    }
    escalations.append(esc)
    _firestore_write("sakhi_escalations", esc["id"], esc, merge=False)
    return esc


def check_and_record_message_id(message_id: Optional[str]) -> bool:
    if not message_id:
        return False
    now = time.time()
    prev = _processed_message_ids.get(message_id)
    if prev is not None and (now - prev) < 300:
        return True
    _processed_message_ids[message_id] = now
    return False
