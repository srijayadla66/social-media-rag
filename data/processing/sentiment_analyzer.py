# data/processing/sentiment_analyzer.py

from typing import List, Dict
from textstat import flesch_reading_ease

def analyze_sentiment(posts: List[Dict]) -> List[Dict]:
    for p in posts:
        score = flesch_reading_ease(p["text"])
        p["sentiment"] = "positive" if score >= 60 else "negative"
    return posts
