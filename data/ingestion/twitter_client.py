import tweepy
import asyncio
from typing import List, Dict
from datetime import datetime, timedelta

class TwitterClient:
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str):
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
    
    async def collect_trending_tweets(self, keywords: List[str], count: int = 100) -> List[Dict]:
        tweets = []
        for keyword in keywords:
            try:
                tweet_results = tweepy.Cursor(
                    self.api.search_tweets,
                    q=keyword,
                    result_type="recent",
                    lang="en"
                ).items(count)
                
                for tweet in tweet_results:
                    tweets.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'author': tweet.author.screen_name,
                        'created_at': tweet.created_at,
                        'retweet_count': tweet.retweet_count,
                        'favorite_count': tweet.favorite_count,
                        'platform': 'twitter',
                        'keyword': keyword
                    })
            except Exception as e:
                print(f"Error collecting tweets for {keyword}: {e}")
        
        return tweets
