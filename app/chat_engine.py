"""Core Sakhi multilingual Pre-Care & Post-Care clinical chat engine."""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from . import guardrails, store
from .llm import generate_sakhi_reply

NAME_ONBOARDING_PROMPT = (
    "Welcome to Sakhi — your fertility and post-care companion.\n\n"
    "Before we begin, what should I call you? (Please reply with just your name, e.g., Deepthi)"
)

ONBOARDING_COMPLETE_REPLY = (
    "Thank you! Your profile is all set.\n\n"
    "I am Sakhi, your personal companion for fertility, pregnancy, and recovery. You can ask me about:\n"
    "- Medications and dosages\n"
    "- Diet and nutrition\n"
    "- Activity and recovery milestones\n"
    "- Symptoms and clinical triage\n"
    "- Nearby hospitals (share your 6-digit pincode)\n\n"
    "How can I help you today?"
)

INVALID_ONBOARDING_WORDS = {
    "what", "how", "why", "when", "who", "where", "which", "whose", "whom",
    "explain", "describe", "ivf", "iui", "pregnancy", "pregnant", "fertility", "infertility",
    "symptom", "symptoms", "treatment", "treatments", "doctor", "doctors", "hospital", "hospitals",
    "clinic", "clinics", "cost", "costs", "price", "prices", "fee", "fees", "charge", "charges",
    "medicine", "medicines", "pill", "pills", "drug", "drugs", "pain", "bleeding", "bleed",
    "period", "periods", "cycle", "cycles", "ovulation", "sperm", "egg", "eggs", "embryo",
    "uterus", "fallopian", "ovary", "ovaries", "ovarian", "hormone", "hormones", "thyroid",
    "pcos", "pcod", "miscarriage", "miscarriages", "journey", "success", "rate", "rates",
    "help", "info", "information", "details", "detail", "tell", "side", "effect", "effects",
    "fever", "bath", "walk", "eat", "diet", "mango", "food", "water", "sleep",
}

INVALID_ONBOARDING_STARTERS = {
    "is", "are", "am", "does", "do", "did", "can", "could", "would", "should",
    "has", "have", "had", "will", "shall",
}

SMALLTALK_PATTERNS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "bye", "goodbye", "how are you", "whats up",
    "namaste", "vanakkam", "ok", "okay", "hlo", "hai", "k", "gm", "gn",
}

SYMPTOM_KEYWORDS = [
    "pain", "cramp", "cramping", "spot", "spotting", "bleed", "bleeding", "blood",
    "bloat", "bloating", "nausea", "vomit", "vomiting", "morning sickness",
    "fever", "temperature", "chills", "headache", "migraine", "cough", "sore throat",
    "diarrhea", "loose motion", "dizzy", "dizziness", "chest pain", "breathing",
    "shortness of breath", "breathless", "rash", "swelling", "palpitation", "fatigue",
    "back pain", "missed injection", "missed dose", "forgot injection", "fetal movement",
    "fainting", "seizure", "stroke", "urinary burning", "low blood sugar",
]

EMERGENCY_KEYWORDS = [
    "chest pain", "chest tightness", "heart pain",
    "difficulty breathing", "cant breathe", "can't breathe", "shortness of breath", "breathless",
    "heavy bleeding", "bleeding heavily", "severe bleeding", "soaking a pad",
    "fainting", "fainted", "unconscious", "passed out", "seizure", "stroke",
    "decreased fetal movement", "baby not moving", "severe abdominal pain", "unbearable pain",
    "emergency", "urgent help", "high fever 103", "high fever 104",
]

TREATMENT_TERMS = [
    {"key": "ivf", "label": "IVF", "matches": ["ivf", "stimulation", "egg retrieval", "embryo transfer", "trigger shot", "icsi"]},
    {"key": "iui", "label": "IUI", "matches": ["iui", "insemination"]},
    {"key": "pregnancy", "label": "Pregnancy", "matches": ["pregnancy", "pregnant", "trimester", "prenatal", "antenatal", "fetal", "obstetrics"]},
    {"key": "pcos", "label": "PCOS", "matches": ["pcos", "pcod"]},
    {"key": "genmed", "label": "General Medicine", "matches": ["initial assessment", "general medicine"]},
]

HOSPITAL_SEARCH_REGEX = re.compile(
    r"\b(hospitals?|clinics?|fertility cent(er|re)|near me|nearby|find.*hospital|find.*clinic|pincode|zip\s*code)\b"
    r"|(హాస్పిటల్|ఆసుపత్రి|अस्पताल)",
    re.IGNORECASE,
)

TINGLISH_REGEX = re.compile(
    r"\b(nenu|naaku|naa|meeru|meeku|ela|enti|em|cheppandi|unnaru|undi|undhi|kavali|dhayachesi|namaskaram|bagunnara|daggara)\b",
    re.IGNORECASE,
)

# Maps Telugu & Hindi domain keywords to English search tokens so verified JSON topics match cleanly across languages
MULTILINGUAL_TOPIC_MAP = [
    (r"(ఐవీఎఫ్|आईवीएफ)", "ivf"),
    (r"(ఐయూఐ|आईयूआई)", "iui"),
    (r"(ప్రెగ్నెన్సీ|గర్భధారణ|गर्भावस्था|गर्भवती)", "pregnancy"),
    (r"(వయస్సు|उम्र)", "age fertility"),
    (r"(రోజువారీ|జాగ్రత్తలు|दिनचर्या|स्वस्थ)", "healthy routines pregnancy"),
    (r"(ఆహారం|పోషణ|आहार|पोषण)", "diet nutrition egg quality"),
    (r"(ఆందోళన|ఒత్తిడి|चिंता|तनाव)", "anxiety emotional support"),
    (r"(వికారం|వాంతులు|मतली|उल्टी)", "nausea vomiting"),
    (r"(తలనొప్పి|सिरदर्द)", "headache"),
    (r"(జ్వరం|बुखार)", "fever"),
    (r"(రక్తస్రావం|रक्तस्राव)", "bleeding"),
    (r"(ఛాతీ నొప్పి|सीने में दर्द)", "chest pain"),
    (r"(ఊపిరి|सांस लेने में तकलीफ)", "difficulty breathing"),
]

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have",
    "are", "you", "your", "can", "will", "what", "how", "about",
    "does", "really", "explain", "tell", "please", "want", "know",
    "need", "help", "should", "could", "would",
}


# ---------------------------------------------------------------------------
# Clean, Rephrased Clinical Escalation & Emergency Messages
# ---------------------------------------------------------------------------
def build_doctor_escalation_message(target_lang: str) -> str:
    if target_lang == "Telugu":
        return (
            "*ధృవీకరించబడిన వైద్యునికి పంపబడింది*\n\n"
            "ప్రస్తుతం ఈ విషయంపై నా వద్ద ధృవీకరించబడిన వైద్య సమాచారం అందుబాటులో లేదు.\n\n"
            "మీకు సురక్షితమైన మరియు ఖచ్చితమైన మార్గదర్శకత్వం అందేలా మీ ప్రశ్నను మా *ధృవీకరించబడిన వైద్యునికి (Verified Doctor)* పంపాము. "
            "వైద్యులు సమాధానం ఇచ్చిన వెంటనే మేము మీకు ఇక్కడ అప్‌డేట్ చేస్తాము.\n\n"
            "_ఇది అత్యవసర పరిస్థితి అయితే, దయచేసి వెంటనే మీ కేర్‌టేకర్ స్టాఫ్‌ను సంప్రదించండి._"
        )
    if target_lang == "Hindi":
        return (
            "*सत्यापित डॉक्टर को भेजा गया*\n\n"
            "वर्तमान में मेरे पास इस विषय पर सत्यापित चिकित्सकीय जानकारी उपलब्ध नहीं है।\n\n"
            "आपको सुरक्षित और सही मार्गदर्शन मिल सके, इसके लिए मैंने आपका प्रश्न हमारे *सत्यापित डॉक्टर (Verified Doctor)* को भेज दिया है। "
            "डॉक्टर के उत्तर देते ही हम आपको तुरंत अपडेट करेंगे।\n\n"
            "_यदि यह आपातकालीन स्थिति है, तो कृपया तुरंत अपने केयरटेकर स्टाफ से संपर्क करें।_"
        )
    if target_lang == "Tinglish":
        return (
            "*Verified Doctor ki Forward Chesamu*\n\n"
            "Prastutam ee vishayam pai naa daggara verified medical data ledu.\n\n"
            "Meeku correct mariyu safe guidance andadaniki mee question ni maa *verified doctor* ki pampamu. "
            "Doctor answer ichina ventane memu meeku update chestamu.\n\n"
            "_Idhi urgent or emergency ayithe, dhayachesi ventane mee caretaker staff ni contact cheyandi._"
        )
    return (
        "*Forwarded to a Verified Doctor*\n\n"
        "I don't currently have verified information on this topic in my knowledge base.\n\n"
        "To make sure you receive safe and accurate guidance, I have escalated your question to a *verified doctor* on our care team. "
        "As soon as the doctor responds, we will update you right away.\n\n"
        "_If you are experiencing urgent symptoms, please contact your caretaker staff immediately._"
    )


def build_emergency_caretaker_message(target_lang: str, clinical_detail: str = "") -> str:
    detail_block = f"\n\n{clinical_detail.strip()}" if clinical_detail and clinical_detail.strip() else ""

    if target_lang == "Telugu":
        return (
            "*అత్యవసర హెచ్చరిక — దయచేసి వెంటనే మీ కేర్‌టేకర్ స్టాఫ్‌ను సంప్రదించండి*"
            f"{detail_block}\n\n"
            "మీరు తెలిపిన లక్షణాలకు తక్షణ వైద్య సహాయం అవసరం మరియు వీటిని చాట్ ద్వారా సురక్షితంగా పరీక్షించలేము.\n\n"
            "• దయచేసి వెంటనే మీ *కేర్‌టేకర్ స్టాఫ్ / క్లినిక్ కేర్ టీమ్‌ను* సంప్రదించండి.\n"
            "• లక్షణాలు తీవ్రంగా ఉంటే, ఆలస్యం చేయకుండా సమీపంలోని హాస్పిటల్ ఎమర్జెన్సీ విభాగానికి వెళ్లండి.\n\n"
            "మేము దీనిని మీ కేర్ టీమ్‌కు అత్యవసర అలర్ట్‌గా నమోదు చేసాము."
        )
    if target_lang == "Hindi":
        return (
            "*आपातकालीन सूचना — कृपया तुरंत अपने केयरटेकर स्टाफ से संपर्क करें*"
            f"{detail_block}\n\n"
            "आपने जो लक्षण बताए हैं, उनके लिए तुरंत चिकित्सकीय देखभाल आवश्यक है और इनका मूल्यांकन चैट पर सुरक्षित रूप से नहीं किया जा सकता।\n\n"
            "• कृपया तुरंत अपने *केयरटेकर स्टाफ / क्लिनिक केयर टीम* से संपर्क करें।\n"
            "• यदि लक्षण गंभीर हैं या बढ़ रहे हैं, तो कृपया तुरंत नजदीकी अस्पताल के आपातकालीन विभाग में जाएं।\n\n"
            "हमने इसे आपकी केयर टीम के लिए तत्काल अलर्ट के रूप में दर्ज कर दिया है।"
        )
    if target_lang == "Tinglish":
        return (
            "*Emergency Alert — Please Contact Your Caretaker Staff Immediately*"
            f"{detail_block}\n\n"
            "Meeru cheppina symptoms ki immediate medical care avasaram, veetini chat dwara safely assess cheyalemu.\n\n"
            "• Dhayachesi ventane mee *caretaker staff / clinic care team* ni contact cheyandi.\n"
            "• Symptoms severe ga unte, ventane daggara lo unna hospital emergency department ki vellandi.\n\n"
            "Ee update ni mee care team ki urgent alert ga flag chesamu."
        )
    return (
        "*Emergency Alert — Please Contact Your Caretaker Staff Immediately*"
        f"{detail_block}\n\n"
        "The situation or symptoms you described require immediate medical attention and cannot be safely assessed over chat.\n\n"
        "• Please contact your *caretaker staff / clinic care team* right away.\n"
        "• If symptoms are severe or worsening, please visit the nearest hospital emergency department immediately.\n\n"
        "We have also flagged this as an urgent alert for your care team."
    )


def build_triage_escalation_message(target_lang: str, unresponsive: bool = False) -> str:
    if unresponsive:
        return (
            "*Care Team Follow-Up — Please Contact Your Caretaker Staff*\n\n"
            "Since I couldn't gather enough detail about your symptoms over chat, I want to make sure you stay safe.\n\n"
            "• Please contact your *caretaker staff / clinic care team* directly for prompt assistance.\n"
            "• We have also alerted our care coordinators to follow up with you shortly."
        )
    return build_emergency_caretaker_message(
        target_lang,
        "Thank you for sharing those details. Because your symptoms need prompt clinical attention, we are escalating this right away.",
    )


def detect_language(text: str, user_preferred: str = "en") -> str:
    if not text:
        if user_preferred in ("te", "Telugu"):
            return "Telugu"
        if user_preferred in ("hi", "Hindi"):
            return "Hindi"
        if user_preferred in ("tinglish", "Tinglish"):
            return "Tinglish"
        return "English"

    if re.search(r"[\u0C00-\u0C7F]", text):
        return "Telugu"
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"
    if TINGLISH_REGEX.search(text):
        return "Tinglish"
    if user_preferred in ("te", "Telugu"):
        return "Telugu"
    if user_preferred in ("hi", "Hindi"):
        return "Hindi"
    if user_preferred in ("tinglish", "Tinglish"):
        return "Tinglish"
    return "English"


def is_smalltalk(message: str) -> bool:
    if not message:
        return False
    clean = re.sub(r"[^\w\s]", "", message.lower()).strip()
    if clean in SMALLTALK_PATTERNS:
        return True
    words = [w for w in clean.split() if w]
    if len(words) <= 3:
        for pat in SMALLTALK_PATTERNS:
            if clean == pat or clean.startswith(f"{pat} "):
                return True
    return False


def is_onboarding_input_invalid(message: str) -> bool:
    cleaned = (message or "").strip().lower()
    if not cleaned or "?" in cleaned:
        return True
    words = re.findall(r"[a-z']+", cleaned)
    if not words:
        return True
    if words[0] in INVALID_ONBOARDING_STARTERS:
        return True
    return any(w in INVALID_ONBOARDING_WORDS for w in words)


def extract_name_from_onboarding(message: str) -> Optional[str]:
    if is_onboarding_input_invalid(message):
        return None
    cleaned = (message or "").strip()
    cleaned = re.sub(
        r"^(my name is|i am|i'm|this is|call me|naa peru|na peru)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"[^a-zA-Z\s]", "", cleaned).strip()
    parts = [p for p in cleaned.split() if p]
    if not parts or len(parts) > 2:
        return None
    return " ".join(p[:1].upper() + p[1:].lower() for p in parts)


def extract_treatments_mentioned(message: str) -> List[Dict[str, Any]]:
    lower = (message or "").lower()
    return [t for t in TREATMENT_TERMS if any(m in lower for m in t["matches"])]


def extract_symptoms_mentioned(message: str) -> List[str]:
    lower = (message or "").lower()
    matched = [kw for kw in SYMPTOM_KEYWORDS if kw in lower]
    if re.search(r"\b(miss|missed|forgot|skipped)\b.*\b(injection|dose|shot|medication|medicine)\b", lower):
        if "missed injection" not in matched:
            matched.append("missed injection")
    return matched


def is_direct_emergency(message: str) -> bool:
    lower = (message or "").lower()
    if any(kw in lower for kw in EMERGENCY_KEYWORDS):
        return True
    if re.search(r"(ఛాతీ నొప్పి|सीने में दर्द|ఊపిరి ఆడటం లేదు|सांस लेने में तकलीफ|अत्याधिक रक्तस्राव)", message or ""):
        return True
    return False


def treatment_matches_active(
    mentioned_treatments: List[Dict[str, Any]],
    active_treatment: str,
    department: str,
) -> bool:
    active_lower = f"{active_treatment or ''} {department or ''}".lower()
    for t in mentioned_treatments:
        if any(m in active_lower for m in t["matches"]):
            return True
        if t["key"] == "pregnancy" and (
            "trimester" in active_lower
            or "obstetrics" in active_lower
            or "pregnancy" in active_lower
        ):
            return True
    return False


def _augment_multilingual_query(text: str) -> str:
    extra_tokens: List[str] = []
    for pattern, mapped in MULTILINGUAL_TOPIC_MAP:
        if re.search(pattern, text or "", flags=re.IGNORECASE):
            extra_tokens.append(mapped)
    return f"{text} {' '.join(extra_tokens)}".strip() if extra_tokens else (text or "")


def tokenize(text: str) -> List[str]:
    augmented = _augment_multilingual_query(text)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", augmented.lower())
    return [w for w in cleaned.split() if len(w) > 2 and w not in STOP_WORDS]


def search_medical_knowledge(
    query: str,
    category_filter: Optional[str] = None,
) -> List[Tuple[Dict[str, str], float]]:
    normalized_query = (query or "").lower().strip()
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    is_definition_query = bool(
        re.match(r"^(what\s+is|explain|tell\s+me\s+about|meaning\s+of|overview\s+of)\b", normalized_query)
    )

    scored: List[Tuple[Dict[str, str], float]] = []
    for doc in store.medical_knowledge:
        if category_filter:
            cat_lower = f"{doc['category']} {doc['sub_category']} {doc['lens']}".lower()
            filter_lower = category_filter.lower()
            matches_filter = (
                filter_lower in cat_lower
                or ("ivf" in filter_lower and "ivf" in cat_lower)
                or ("iui" in filter_lower and "iui" in cat_lower)
                or (
                    ("trimester" in filter_lower or "pregnancy" in filter_lower)
                    and ("trimester" in cat_lower or "pregnancy" in cat_lower or "obstetrics" in cat_lower)
                )
                or (
                    "initial assessment" in filter_lower
                    and (doc["category"] == "Initial Assessment" or doc["lens"] == "medical")
                )
            )
            if not matches_filter:
                continue

        title_lower = f"{doc.get('title', '')} {doc.get('sub_category', '')}".lower()
        content_lower = (doc.get("content") or "").lower()
        title_tokens = set(tokenize(title_lower))
        content_tokens = set(tokenize(content_lower))

        title_hits = sum(1 for qt in query_tokens if qt in title_tokens or qt in title_lower)
        content_hits = sum(1 for qt in query_tokens if qt in content_tokens or qt in content_lower)

        if title_hits == 0 and content_hits == 0:
            continue

        score = (title_hits / len(query_tokens)) * 0.65 + (content_hits / len(query_tokens)) * 0.35
        if normalized_query in title_lower:
            score += 0.5

        is_overview_doc = (
            "overview" in doc["category"].lower()
            or doc["sub_category"].lower().startswith("what is")
        )
        if is_definition_query and is_overview_doc and title_hits > 0:
            score += 0.6

        if len(query_tokens) <= 2 and not is_overview_doc and len(title_tokens) > len(query_tokens) + 2:
            score -= 0.12

        scored.append((doc, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def find_best_clinical_rule(
    message: str,
    department: str,
    treatment_stage: str,
) -> Optional[Dict[str, Any]]:
    lower = (message or "").lower()
    stage_lower = (treatment_stage or "").lower()
    dept_lower = (department or "").lower()

    relevant_rules = []
    for r in store.clinical_rules:
        r_stage = (r.get("treatment_stage") or "").lower()
        r_dept = (r.get("department") or "").lower()
        if r_stage == stage_lower or r_dept == dept_lower:
            relevant_rules.append(r)
        elif (
            "trimester" in stage_lower
            or "obstetrics" in dept_lower
            or "pregnancy" in dept_lower
        ) and r_dept in ("obstetrics", "pregnancy"):
            relevant_rules.append(r)

    pool = relevant_rules if relevant_rules else store.clinical_rules
    best: Optional[Dict[str, Any]] = None
    best_score = 0

    for rule in pool:
        score = 0
        for kw in rule.get("keywords") or []:
            if kw in lower:
                score += 2 if " " in kw else 1
        if score > best_score:
            best_score = score
            best = rule

    return best if best_score > 0 else None


def evaluate_triage_hard_stop(
    user_id: str,
    message: str,
    triage_state: Dict[str, Any],
    target_lang: str = "English",
) -> Dict[str, Any]:
    lower = (message or "").lower().strip()
    retry_count = triage_state.get("triage_retry_count", 0)
    clarifying_q = (
        triage_state.get("clarifying_question")
        or "Could you please share more details about how you are feeling?"
    )

    if is_smalltalk(message) and not re.search(r"\b(yes|no|mild|severe|fine|okay)\b", lower):
        if retry_count < 1:
            store.update_chat_state(user_id, {"triage_retry_count": retry_count + 1})
            return {
                "reply": f"To make sure you receive the right medical guidance for your symptoms, could you please answer: {clarifying_q}",
                "mode": "post_care_triage_reask",
                "intent": "symptom_triage_reask",
                "route": "post-care",
            }
        store.update_chat_state(user_id, {"waiting_for_triage_reply": False, "escalation_trigger": None})
        return {
            "reply": build_triage_escalation_message(target_lang, unresponsive=True),
            "mode": "escalation",
            "intent": "unresponsive_triage_escalation",
            "route": "post-care",
        }

    store.update_chat_state(user_id, {"waiting_for_triage_reply": False, "escalation_trigger": None})

    red_flags = re.compile(
        r"\b(yes|severe|heavy|sharp|blood|bleeding|fever|10[1-5]|39\.|40|dizzy|faint|breathless|"
        r"shortness of breath|blurred vision|swollen|swelling|cant keep|unable|chest pain|one side|"
        r"rigid|confusion|unconscious|180|200|worst)\b",
        re.IGNORECASE,
    )
    safe_indicators = re.compile(
        r"\b(no|none|mild|light|little|slight|gradual|98|99|100\.0|okay|manageable|able to keep|keeping fluids)\b",
        re.IGNORECASE,
    )

    is_escalation = bool(red_flags.search(lower)) and not (
        bool(re.match(r"^no\b", lower)) and not bool(re.search(r"\b(unable|cant|cannot)\b", lower))
    )

    if is_escalation or (
        not safe_indicators.search(lower)
        and bool(re.search(r"\b(very bad|terrible|extreme|unbearable|emergency)\b", lower))
    ):
        return {
            "reply": build_triage_escalation_message(target_lang, unresponsive=False),
            "mode": "emergency",
            "intent": "triage_escalation",
            "route": "post-care",
        }

    safe_reply = (
        triage_state.get("provisional_answer")
        or "Thank you for clarifying. Your symptoms appear to be within normal expected bounds. I have logged this update for your doctor."
    )
    return {
        "reply": safe_reply,
        "mode": "post_care_triage_complete",
        "intent": "triage_complete",
        "route": "post-care",
    }


def handle_lead_flow(user_id: str, message: str) -> Dict[str, Any]:
    msg = (message or "").strip()
    state = store.get_chat_state(user_id)
    lead_state = state.get("lead_flow") or {}
    current_step = lead_state.get("step")
    temp_data = dict(lead_state.get("data") or {})

    if msg.lower() == "/newlead":
        store.update_chat_state(user_id, {"lead_flow": {"step": "ask_name", "data": {}}})
        return {
            "reply": "I can help you register a new patient inquiry. Let's start with their name — what should we call them?",
            "mode": "lead_input",
            "intent": "lead_registration",
            "route": "lead_flow",
        }

    if not current_step:
        return {
            "reply": "I lost track of the registration step. Please type /newlead to start again.",
            "mode": "general",
            "intent": "lead_reset",
            "route": "lead_flow",
        }

    next_step = current_step
    reply_text = ""

    if current_step == "ask_name":
        temp_data["name"] = msg
        next_step = "ask_phone"
        reply_text = f"Thanks. Could you share {msg}'s phone number so our clinic team can reach out?"
    elif current_step == "ask_phone":
        temp_data["phone"] = msg
        next_step = "ask_age"
        reply_text = "Got it. May I ask how old they are?"
    elif current_step == "ask_age":
        temp_data["age"] = msg
        next_step = "ask_gender"
        reply_text = "And their gender?"
    elif current_step == "ask_gender":
        temp_data["gender"] = msg
        next_step = "ask_problem"
        reply_text = "Finally, could you briefly share the health concern or treatment inquiry they have?"
    elif current_step == "ask_problem":
        temp_data["problem"] = msg
        store.record_lead(temp_data, user_id)
        store.update_chat_state(user_id, {"lead_flow": None})
        return {
            "reply": "Thank you. I have noted all the details and shared them with our clinic care team. We will take good care of them.",
            "mode": "lead_complete",
            "intent": "lead_complete",
            "route": "lead_flow",
        }

    store.update_chat_state(user_id, {"lead_flow": {"step": next_step, "data": temp_data}})
    return {
        "reply": reply_text,
        "mode": "lead_input",
        "intent": "lead_registration",
        "route": "lead_flow",
    }


async def process_sakhi_chat(
    user_id: Optional[str] = None,
    phone_number: Optional[str] = None,
    message: str = "",
    language: str = "en",
) -> Dict[str, Any]:
    phone = store.normalize_phone(phone_number)
    user = store.get_user_by_id(user_id) if user_id else store.get_user_by_phone(phone)
    patient = store.get_patient_by_phone(phone) if phone else None
    patient_name = patient.get("name") if patient else None

    if not user:
        if not phone:
            return {"statusCode": 400, "error": "user_id or phone_number is required"}
        user = store.create_partial_user(phone)
        if patient and patient_name:
            store.update_user_profile(user["user_id"], {"name": patient_name})
        else:
            store.update_user_context(user["user_id"], {"onboarding_step": "name"})
            return {
                "reply": NAME_ONBOARDING_PROMPT,
                "mode": "onboarding",
                "intent": "ask_name",
                "language": "en",
                "route": "onboarding",
            }

    uid = user.get("user_id") or user.get("id")
    msg = str(message or "").strip()
    user_context = store.get_user_context(uid)

    if not user.get("name") and patient and patient_name:
        store.update_user_profile(uid, {"name": patient_name})

    if not user.get("name") or user_context.get("onboarding_step") == "name":
        if is_onboarding_input_invalid(msg):
            store.update_user_context(uid, {"onboarding_step": "name", "initial_question": msg})
            return {
                "reply": "I'd love to help you with that! First, what should I call you? (Please reply with just your name, e.g., Deepthi)",
                "mode": "onboarding",
                "intent": "ask_name_again",
                "language": "en",
                "route": "onboarding",
            }
        extracted = extract_name_from_onboarding(msg)
        if extracted:
            store.update_user_profile(uid, {"name": extracted})
            store.update_user_context(uid, {"onboarding_step": None})
            welcome = (
                f"Nice to meet you, {extracted}! Your profile is linked with your clinic records. How are you feeling today?"
                if patient and patient.get("id")
                else f"Nice to meet you, {extracted}! I am Sakhi, your companion for fertility, pregnancy, and wellness. How can I help you today?"
            )
            return {
                "reply": welcome,
                "mode": "onboarding_complete",
                "intent": "onboarding_complete",
                "language": "en",
                "route": "onboarding",
            }
        return {
            "reply": "I didn't quite catch that. Could you please share your name with me? (e.g., Deepthi)",
            "mode": "onboarding",
            "intent": "ask_name_again",
            "language": "en",
            "route": "onboarding",
        }

    if msg.lower() == "/rewards":
        total = store.get_rewards(uid)
        return {
            "reply": f"You have earned *{total} reward points*! Keep asking questions and tracking your care to earn more.",
            "mode": "rewards",
            "intent": "show_rewards",
            "route": "rewards",
        }

    target_lang = detect_language(msg, language or user.get("preferred_language") or "en")

    chat_state = store.get_chat_state(uid)
    if chat_state.get("waiting_for_triage_reply"):
        triage_resp = evaluate_triage_hard_stop(uid, msg, chat_state, target_lang)
        if triage_resp["mode"] in ("escalation", "emergency"):
            store.record_escalation(uid, phone, triage_resp["mode"], triage_resp["intent"], triage_resp["reply"])
        return triage_resp

    if msg.lower() == "/newlead" or (chat_state.get("lead_flow") or {}).get("step"):
        return handle_lead_flow(uid, msg)

    if user_context.get("onboarding_step") == "clinic_selection":
        selected = None
        if msg.isdigit():
            idx = int(msg) - 1
            if 0 <= idx < len(store.clinics):
                selected = store.clinics[idx]
        if not selected:
            for c in store.clinics:
                if msg.lower() in c["name"].lower():
                    selected = c
                    break
        if selected:
            store.update_user_context(uid, {"onboarding_step": None, "active_clinic_id": selected["id"]})
            return {
                "reply": f"You're registered at *{selected['name']}*!\n\n{ONBOARDING_COMPLETE_REPLY}",
                "mode": "onboarding_complete",
                "intent": "onboarding_complete",
                "language": "en",
                "route": "onboarding",
            }

    store.save_message(uid, "user", msg, target_lang)

    zip_match = re.search(r"\b\d{6}\b", msg)
    if zip_match:
        store.update_user_context(uid, {"zip_code": zip_match.group(0)})

    awaiting_pincode = bool(user_context.get("awaiting_pincode"))
    if HOSPITAL_SEARCH_REGEX.search(msg) or (awaiting_pincode and zip_match):
        ctx = store.get_user_context(uid)
        zip_code = zip_match.group(0) if zip_match else ctx.get("zip_code")
        if zip_code:
            store.update_user_context(uid, {"awaiting_pincode": False, "zip_code": zip_code})
            map_url = f"https://www.google.com/maps/search/hospitals+near+{zip_code}"
            reply_text = f"Here are the hospitals near you: {map_url}"
            if target_lang == "Telugu":
                reply_text = f"మీకు సమీపంలోని హాస్పిటల్స్ ఇక్కడ ఉన్నాయి: {map_url}"
            elif target_lang == "Tinglish":
                reply_text = f"Meeku daggara lo unna hospitals ivi: {map_url}"
            store.save_message(uid, "assistant", reply_text, target_lang)
            return {
                "reply": reply_text,
                "mode": "general",
                "language": target_lang,
                "route": "hospital_search",
                "intent": "hospital search",
                "zip_code": zip_code,
            }
        store.update_user_context(uid, {"awaiting_pincode": True})
        ask_pin = "Could you please share your 6-digit pincode so I can find fertility and maternity hospitals near you?"
        if target_lang == "Telugu":
            ask_pin = "దయచేసి మీ 6-అంకెల పిన్‌కోడ్‌ను ఇవ్వండి, తద్వారా నేను మీకు సమీపంలోని హాస్పిటల్స్‌ను కనుగొనగలను."
        elif target_lang == "Tinglish":
            ask_pin = "Dhayachesi mee 6-digit pincode ivvandi, appudu nenu meeku daggara lo unna hospitals ni vetukuthanu."
        store.save_message(uid, "assistant", ask_pin, target_lang)
        return {
            "reply": ask_pin,
            "mode": "general",
            "language": target_lang,
            "route": "hospital_search",
            "intent": "ask pincode",
        }

    is_patient = bool(patient and patient.get("id"))
    active_treatment_event = next(
        (
            e
            for e in (patient.get("timeline") or [])
            if e.get("event_type") == "TREATMENT" and str(e.get("description", "")).lower() == "active"
        ),
        None,
    ) if patient else None
    active_treatment = active_treatment_event["title"] if active_treatment_event else "None"
    has_active_treatment = active_treatment != "None"

    mentioned_treatments = extract_treatments_mentioned(msg)
    mentioned_symptoms = extract_symptoms_mentioned(msg)

    redirect = guardrails.get_redirect_for_out_of_scope(msg)
    if redirect:
        store.save_message(uid, "assistant", redirect, target_lang)
        return {
            "reply": redirect,
            "mode": "general",
            "language": target_lang,
            "intent": "out_of_scope",
            "route": "post-care" if is_patient else "pre-care",
        }

    # Immediate emergency guard (works for both Post-Care & Pre-Care users)
    if is_direct_emergency(msg) and not (is_patient and patient):
        emergency_reply = build_emergency_caretaker_message(target_lang)
        store.record_escalation(uid, phone, "emergency", "emergency", emergency_reply)
        store.save_message(uid, "assistant", emergency_reply, target_lang)
        return {
            "reply": emergency_reply,
            "mode": "emergency",
            "intent": "emergency",
            "route": "pre-care",
            "language": target_lang,
        }

    if is_patient and patient:
        first_name = str(user.get("name") or patient.get("name") or "").split()[0]

        if is_smalltalk(msg):
            greeting = (
                f"Hi {first_name}! How are you feeling today with your *{active_treatment}*?"
                if has_active_treatment
                else f"Hello {first_name}! I am Sakhi, your clinic companion. How can I help you with your care today?"
            )
            store.save_message(uid, "assistant", greeting, target_lang)
            return {
                "reply": greeting,
                "mode": "post_care_greeting",
                "route": "post-care",
                "intent": "greeting",
                "language": target_lang,
                "active_treatment": active_treatment,
            }

        if mentioned_symptoms or is_direct_emergency(msg):
            rule = find_best_clinical_rule(msg, patient.get("department", ""), active_treatment)

            # Direct emergency even if no stage-specific rule matched
            if is_direct_emergency(msg) and (not rule or rule.get("emergency_flag")):
                detail = rule.get("final_answer", "") if rule else ""
                emergency_reply = build_emergency_caretaker_message(target_lang, detail)
                store.record_escalation(uid, phone, "emergency", "emergency", emergency_reply)
                store.save_message(uid, "assistant", emergency_reply, target_lang)
                return {
                    "reply": emergency_reply,
                    "mode": "emergency",
                    "intent": "emergency",
                    "route": "post-care",
                    "language": target_lang,
                }

            if not rule:
                esc_reply = build_doctor_escalation_message(target_lang)
                store.record_escalation(uid, phone, "doctor_escalation", "missing_data_escalation", esc_reply)
                store.save_message(uid, "assistant", esc_reply, target_lang)
                return {
                    "reply": esc_reply,
                    "mode": "doctor_escalation",
                    "intent": "missing_data_escalation",
                    "route": "post-care",
                    "language": target_lang,
                }

            if rule.get("emergency_flag"):
                emergency_reply = build_emergency_caretaker_message(
                    target_lang,
                    rule.get("final_answer", ""),
                )
                store.record_escalation(uid, phone, "emergency", "emergency", emergency_reply)
                store.save_message(uid, "assistant", emergency_reply, target_lang)
                return {
                    "reply": emergency_reply,
                    "mode": "emergency",
                    "intent": "emergency",
                    "route": "post-care",
                    "language": target_lang,
                }

            store.update_chat_state(
                uid,
                {
                    "waiting_for_triage_reply": True,
                    "escalation_trigger": rule.get("escalation_trigger"),
                    "provisional_answer": rule.get("final_answer"),
                    "original_symptom": msg,
                    "clarifying_question": rule.get("clarifying_question"),
                    "triage_retry_count": 0,
                },
            )
            triage_question = rule.get("clarifying_question") or rule.get("final_answer")
            store.save_message(uid, "assistant", triage_question, target_lang)
            return {
                "reply": triage_question,
                "mode": "post_care_triage",
                "intent": "symptom_triage",
                "route": "post-care",
                "language": target_lang,
            }

        if mentioned_treatments and has_active_treatment:
            if not treatment_matches_active(mentioned_treatments, active_treatment, patient.get("department", "")):
                reject_reply = (
                    f"I can only assist you with questions related to your current treatment (*{active_treatment}*). "
                    "I cannot answer questions about other treatments."
                )
                store.save_message(uid, "assistant", reject_reply, target_lang)
                return {
                    "reply": reject_reply,
                    "mode": "post_care_rejected",
                    "route": "post-care",
                    "intent": "treatment_mismatch",
                    "language": target_lang,
                    "active_treatment": active_treatment,
                }

        if has_active_treatment:
            matches = search_medical_knowledge(msg, active_treatment)
            if matches and matches[0][1] >= 0.28:
                top_doc = matches[0][0]
                answer = top_doc["content"]
                llm_res = await generate_sakhi_reply(
                    system_instruction=(
                        f"You are Sakhi, an empathetic Post-Care Educational Assistant for a patient currently in {active_treatment}. "
                        f"Respond in {target_lang} clearly and warmly. Ground your response ONLY in this verified clinical context:\n"
                        f"{top_doc['content']}\n"
                        "Do not add unverified medical claims outside this context. Write clear, friendly prose with short bullet points if helpful."
                    ),
                    user_message=msg,
                    history=store.get_last_messages(uid, 6),
                )
                if llm_res:
                    answer = guardrails.clean_output(llm_res[0])
                store.add_rewards(uid, 5)
                store.save_message(uid, "assistant", answer, target_lang)
                return {
                    "reply": answer,
                    "mode": "post_care_education",
                    "intent": "education",
                    "route": "post-care",
                    "language": target_lang,
                }

            # Zero-trust: Post-Care question has no matching JSON data -> escalate to verified doctor
            esc_reply = build_doctor_escalation_message(target_lang)
            store.record_escalation(uid, phone, "doctor_escalation", "missing_data_escalation", esc_reply)
            store.save_message(uid, "assistant", esc_reply, target_lang)
            return {
                "reply": esc_reply,
                "mode": "doctor_escalation",
                "intent": "missing_data_escalation",
                "route": "post-care",
                "language": target_lang,
            }

    if is_smalltalk(msg):
        first_name = str(user.get("name") or "").split()[0] if user.get("name") else ""
        reply = (
            f"Hi {first_name}! I'm Sakhi, your fertility and wellness companion. How can I support you today? "
            "You can ask me about IVF, IUI, pregnancy nutrition, emotional support, nearby hospitals (with your 6-digit pincode), or try /rewards and /newlead."
            if first_name
            else "Hello! I'm Sakhi, your fertility and wellness companion. How can I help you today?"
        )
        if target_lang == "Telugu":
            reply = f"నమస్కారం {first_name}! నేను సఖి, మీ ఆరోగ్య మరియు సంతానోత్పత్తి సహాయకురాలిని. IVF, IUI, గర్భధారణ సంరక్షణ లేదా సమీపంలోని హాస్పిటల్స్ గురించి మీరు నన్ను ఏదైనా అడగవచ్చు."
        elif target_lang == "Hindi":
            reply = f"नमस्ते {first_name}! मैं सखी हूँ, आपकी प्रजनन और मातृत्व साथी। आप मुझसे IVF, IUI, गर्भावस्था देखभाल, पोषण या नजदीकी अस्पतालों के बारे में कुछ भी पूछ सकती हैं।"
        elif target_lang == "Tinglish":
            reply = f"Hi {first_name}! Nenu Sakhi, mee fertility mariyu wellness companion ni. IVF, IUI, pregnancy diet, leda meeku daggara lo unna hospitals gurinchi nannu edaina adagandi."
        store.add_rewards(uid, 2)
        store.save_message(uid, "assistant", reply, target_lang)
        return {
            "reply": reply,
            "mode": "general",
            "language": target_lang,
            "route": "direct_greeting",
            "intent": "greeting",
        }

    supabase_matches: List[Dict[str, Any]] = []
    local_matches = search_medical_knowledge(msg, None)[:3]

    context_sections: List[str] = []
    youtube_link: Optional[str] = None
    infographic_url: Optional[str] = None

    for sb in supabase_matches:
        context_sections.append(f"{sb['title']}: {sb['content']}")
        if not youtube_link and sb.get("youtube_link"):
            youtube_link = sb["youtube_link"]
        if not infographic_url and sb.get("infographic_url"):
            infographic_url = sb["infographic_url"]

    for doc, score in local_matches:
        if score >= 0.30:
            context_sections.append(f"{doc['title']}: {doc['content']}")

    # Strict verified-data gate: if neither Supabase nor our JSON knowledge base contains data for this question,
    # escalate to a verified doctor instead of generating an ungrounded reply.
    if not context_sections:
        esc_reply = build_doctor_escalation_message(target_lang)
        store.record_escalation(uid, phone, "doctor_escalation", "missing_data_escalation", esc_reply)
        store.save_message(uid, "assistant", esc_reply, target_lang)
        return {
            "reply": esc_reply,
            "mode": "doctor_escalation",
            "language": target_lang,
            "route": "doctor_escalation",
            "intent": "missing_data_escalation",
        }

    context_block = (
        "\n\nVerified knowledge base snippets (answer strictly based on these verified snippets; do not invent unverified facts):\n"
        + "\n\n".join(context_sections[:4])
    )

    system_prompt = (
        "You are Sakhi, a warm, compassionate, culturally-aware Indian fertility, pregnancy, and women's health companion built by JanmaSethu. "
        f"Respond in {target_lang}. Address {user.get('name') or 'the user'} gently when natural. "
        "Directly and clearly answer the user's question using the verified knowledge snippets below, structured cleanly for easy reading (short paragraphs or bullet points, 120-220 words unless more detail is asked). "
        "Do not diagnose or prescribe specific medication dosages; gently remind the user to consult their fertility specialist or doctor for personalized clinical decisions."
        f"{context_block}"
    )

    final_reply = ""
    route_used = "local_kb"

    llm_res = await generate_sakhi_reply(
        system_instruction=system_prompt,
        user_message=msg,
        history=store.get_last_messages(uid, 6),
    )
    if llm_res:
        final_reply = guardrails.clean_output(llm_res[0])
        route_used = f"{llm_res[1]}_rag"
    elif supabase_matches:
        top_sb = supabase_matches[0]
        final_reply = f"{top_sb['title']}\n\n{top_sb['content']}\n\nFor personalized medical advice, please consult your treating specialist."
        route_used = "supabase_direct"
    else:
        best_doc = local_matches[0][0]
        final_reply = f"{best_doc['title']}\n\n{best_doc['content']}\n\nFor personalized medical advice or dosage decisions, please consult your treating specialist."
        route_used = "local_kb"

    store.add_rewards(uid, 5)
    store.save_message(uid, "assistant", final_reply, target_lang)

    return {
        "reply": final_reply,
        "mode": "medical",
        "language": target_lang,
        "youtube_link": youtube_link,
        "infographic_url": infographic_url,
        "route": route_used,
        "intent": local_matches[0][0]["sub_category"] if local_matches else "medical_inquiry",
    }
