# app/models/trend_detector.py

from collections import Counter
from datetime import datetime, timedelta
from typing import List, Dict

class TrendDetector:
    def __init__(self, window_hours: int = 24, min_mentions: int = 2):
        self.window_hours = window_hours
        self.min_mentions = min_mentions
    
    def extract_keywords(self, posts: List[Dict]) -> List[str]:
        """Extract hashtags and words longer than 3 chars."""
        keywords = []
        for post in posts:
            text = post.get('text', '').lower()
            # Hashtags
            keywords += [w for w in text.split() if w.startswith('#')]
            # Words >3 chars
            keywords += [w for w in text.split() if w.isalpha() and len(w) > 3]
        return keywords
    
    def _is_recent(self, timestamp) -> bool:
        cutoff = datetime.now() - timedelta(hours=self.window_hours)
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return timestamp > cutoff
    
    def detect_trending_topics(self, posts: List[Dict]) -> List[Dict]:
        """Return top keywords by simple mention count in timeframe."""
        recent = [p for p in posts if self._is_recent(p.get('created_at', datetime.now()))]
        keywords = self.extract_keywords(recent)
        counts = Counter(keywords)
        
        trending = []
        for word, cnt in counts.most_common(10):
            if cnt >= self.min_mentions:
                trending.append({'keyword': word, 'mentions': cnt})
        return trending
