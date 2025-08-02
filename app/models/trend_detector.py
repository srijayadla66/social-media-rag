# app/models/trend_detector.py
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import nltk
from textstat import flesch_reading_ease

class TrendDetector:
    def __init__(self, window_hours: int = 24, min_mentions: int = 10):
        self.window_hours = window_hours
        self.min_mentions = min_mentions
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
    
    def extract_keywords(self, posts: List[Dict]) -> List[str]:
        """Extract relevant keywords and hashtags from posts"""
        keywords = []
        for post in posts:
            text = post['text'].lower()
            # Extract hashtags
            hashtags = [word for word in text.split() if word.startswith('#')]
            keywords.extend(hashtags)
            
            # Extract important words (exclude stopwords)
            words = nltk.word_tokenize(text)
            important_words = [
                word for word in words 
                if word.isalpha() and len(word) > 3 and word not in self.stopwords
            ]
            keywords.extend(important_words)
        
        return keywords
    
    def calculate_trend_score(self, keyword: str, posts: List[Dict]) -> float:
        """Calculate trend score based on engagement metrics"""
        recent_posts = [
            post for post in posts 
            if keyword.lower() in post['text'].lower()
            and self._is_recent(post['created_at'])
        ]
        
        if len(recent_posts) < self.min_mentions:
            return 0.0
        
        # Calculate engagement score
        total_engagement = sum(
            post.get('retweet_count', 0) + 
            post.get('favorite_count', 0) + 
            post.get('upvotes', 0)
            for post in recent_posts
        )
        
        # Velocity factor (how quickly it's spreading)
        time_span = self._calculate_time_span(recent_posts)
        velocity = len(recent_posts) / max(time_span, 1)
        
        # Combine factors
        trend_score = (total_engagement * 0.4) + (len(recent_posts) * 0.4) + (velocity * 0.2)
        
        return trend_score
    
    def detect_trending_topics(self, posts: List[Dict]) -> List[Dict]:
        """Detect trending topics from social media posts"""
        keywords = self.extract_keywords(posts)
        keyword_counts = Counter(keywords)
        
        trending_topics = []
        for keyword, count in keyword_counts.most_common(50):
            trend_score = self.calculate_trend_score(keyword, posts)
            
            if trend_score > 0:
                trending_topics.append({
                    'keyword': keyword,
                    'mentions': count,
                    'trend_score': trend_score,
                    'sentiment': self._analyze_sentiment(keyword, posts),
                    'platforms': self._get_platforms(keyword, posts)
                })
        
        return sorted(trending_topics, key=lambda x: x['trend_score'], reverse=True)[:20]
    
    def _is_recent(self, timestamp) -> bool:
        """Check if timestamp is within the analysis window"""
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        cutoff = datetime.now() - timedelta(hours=self.window_hours)
        return timestamp > cutoff
    
    def _calculate_time_span(self, posts: List[Dict]) -> float:
        """Calculate time span of posts in hours"""
        timestamps = [post['created_at'] for post in posts]
        if len(timestamps) < 2:
            return 1.0
        
        latest = max(timestamps)
        earliest = min(timestamps)
        
        if isinstance(latest, str):
            latest = datetime.fromisoformat(latest.replace('Z', '+00:00'))
        if isinstance(earliest, str):
            earliest = datetime.fromisoformat(earliest.replace('Z', '+00:00'))
        
        return max((latest - earliest).total_seconds() / 3600, 1.0)