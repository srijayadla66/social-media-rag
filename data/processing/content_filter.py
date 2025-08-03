# content_filter.py
from typing import List, Dict

BLACKLIST = {"badword1", "badword2", "spamlink"}

def filter_profanity(posts: List[Dict]) -> List[Dict]:
    return [p for p in posts if not any(b in p["text"].lower() for b in BLACKLIST)]
