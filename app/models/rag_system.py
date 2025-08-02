from sentence_transformers import SentenceTransformer
from typing import List, Dict
import openai
from datetime import datetime

class SocialMediaRAG:
    def __init__(self, openai_api_key: str = None):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.posts = []  # Simple in-memory storage

        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_openai = True
        else:
            self.use_openai = False
    
    def add_posts(self, posts: List[Dict]):
        """Add posts to in-memory store with embeddings"""
        for post in posts:
            doc_text = f"Platform: {post.get('platform', 'unknown')}\n"
            doc_text += f"Author: {post.get('author', 'unknown')}\n"
            doc_text += f"Content: {post['text']}\n"
            doc_text += f"Engagement: {post.get('retweet_count', 0) + post.get('favorite_count', 0)}"

            embedding = self.embedding_model.encode([doc_text])[0]
            self.posts.append({
                'text': doc_text,
                'metadata': post,
                'embedding': embedding
            })
    
    def retrieve_relevant_posts(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve posts based on similarity"""
        query_emb = self.embedding_model.encode([query])[0]
        similarities = []

        for post in self.posts:
            sim = self._cosine_similarity(query_emb, post['embedding'])
            similarities.append((sim, post))

        top_posts = sorted(similarities, key=lambda x: x[0], reverse=True)[:n_results]

        return [
            {
                'document': p['text'],
                'metadata': p['metadata'],
                'similarity_score': s
            }
            for s, p in top_posts
        ]
    
    def generate_contextual_response(self, query: str, context_posts: List[Dict]) -> str:
        context = "\n\n".join([post['document'] for post in context_posts])

        if self.use_openai:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a social media trend analyst."},
                    {"role": "user", "content": f"Query: {query}\n\nRelevant Posts:\n{context}"}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        else:
            return f"""Based on analysis of "{query}":

Summary:
{context[:500]}...

Insights:
- Found {len(context_posts)} posts.
- Platforms: {', '.join(set(p['metadata']['platform'] for p in context_posts))}"""

    def analyze_query(self, query: str) -> Dict:
        relevant_posts = self.retrieve_relevant_posts(query)
        response = self.generate_contextual_response(query, relevant_posts)
        return {
            'query': query,
            'relevant_posts': relevant_posts,
            'generated_response': response,
            'timestamp': datetime.now().isoformat()
        }

    def _cosine_similarity(self, vec1, vec2):
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
