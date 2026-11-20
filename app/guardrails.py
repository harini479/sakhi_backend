"""Out-of-scope guardrails and output sanitization for Sakhi."""

import random
import re
from typing import Optional

OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"^who is (?!my doctor|the doctor|dr\.|specialist)", re.IGNORECASE),
    re.compile(r"^what is the capital", re.IGNORECASE),
    re.compile(r"^tell me about (?!ivf|iui|pregnancy|fertility|pcos|infertility)", re.IGNORECASE),
    re.compile(r"\b(cricket|cricketer|ipl|bcci|match score|virat|kohli|dhoni|sachin|rohit sharma)\b", re.IGNORECASE),
    re.compile(r"\b(football|soccer|fifa|messi|ronaldo|neymar|premier league)\b", re.IGNORECASE),
    re.compile(r"\b(basketball|nba|tennis|wimbledon|olympics)\b", re.IGNORECASE),
    re.compile(r"\b(sports?|player|team won|world cup)\b", re.IGNORECASE),
    re.compile(r"\b(movie|film|cinema|actor|actress|bollywood|hollywood|netflix|webseries)\b", re.IGNORECASE),
    re.compile(r"\b(song|singer|music|album|concert|spotify)\b", re.IGNORECASE),
    re.compile(r"\b(politics|election|minister|government|party|modi|rahul gandhi)\b", re.IGNORECASE),
    re.compile(r"\b(parliament|congress|bjp|vote|politician)\b", re.IGNORECASE),
    re.compile(r"\b(news|headlines|current affairs|breaking news)\b", re.IGNORECASE),
    re.compile(r"\b(stock|share market|trading|bitcoin|crypto|sensex|nifty)\b", re.IGNORECASE),
    re.compile(r"\b(invest|mutual fund|ipo|business news)\b", re.IGNORECASE),
    re.compile(r"\b(iphone|android|laptop|computer|programming|software|coding)\b", re.IGNORECASE),
    re.compile(r"\b(html|css|javascript|python|java|sql|api|database|server|cloud)\b", re.IGNORECASE),
    re.compile(r"\b(weather forecast|temperature today|rain)\b", re.IGNORECASE),
    re.compile(r"\b(game|gaming|playstation|xbox|pubg|fortnite)\b", re.IGNORECASE),
    re.compile(r"\b(shopping|discount|sale|amazon|flipkart|offer)\b", re.IGNORECASE),
    re.compile(r"\b(vacation|hotel|flight booking|tourist)\b", re.IGNORECASE),
]

REDIRECT_RESPONSES = [
    (
        "I appreciate you sharing that with me! While I'm not equipped to chat about that topic, "
        "I'm here as your companion for anything related to fertility, pregnancy, post-care, and family wellness.\n\n"
        "Is there anything on your mind about your health journey today?"
    ),
    (
        "That's an interesting question, though my expertise is really in fertility, pregnancy, and clinical wellness support.\n\n"
        "If you have treatment questions, nutrition doubts, or emotional concerns weighing on your mind, "
        "I'm right here to listen and help."
    ),
]


def get_redirect_for_out_of_scope(message: str) -> Optional[str]:
    if not message:
        return None
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(message):
            return random.choice(REDIRECT_RESPONSES)
    return None


def clean_output(response: str) -> str:
    if not response:
        return response
    cleaned = re.sub(r"(?:\\r\\n|\\n|\\r)+$", "", response)
    cleaned = re.sub(r"\\r\\n|\\n|\\r", "\n", cleaned)
    cleaned = re.sub(r"^as an ai[,\s]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^as a language model[,\s]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(openai|chatgpt|gpt-4)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned.strip()
