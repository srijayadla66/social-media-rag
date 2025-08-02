# app/models/rag_system.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import openai
from datetime import datetime

class SocialMediaRAG:
    def __init__(self, openai_api_key: str = None):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("social_media_posts")
        
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_openai = True
        else:
            self.use_openai = False
    
    def add_posts(self, posts: List[Dict]):
        """Add social media posts to the vector database"""
        documents = []
        metadatas = []
        ids = []
        
        for i, post in enumerate(posts):
            # Create document text combining post content and metadata
            doc_text = f"Platform: {post.get('platform', 'unknown')}\n"
            doc_text += f"Author: {post.get('author', 'unknown')}\n"
            doc_text += f"Content: {post['text']}\n"
            doc_text += f"Engagement: {post.get('retweet_count', 0) + post.get('favorite_count', 0)}"
            
            documents.append(doc_text)
            metadatas.append({
                'platform': post.get('platform', 'unknown'),
                'author': post.get('author', 'unknown'),
                'created_at': str(post.get('created_at', datetime.now())),
                'engagement_score': post.get('retweet_count', 0) + post.get('favorite_count', 0),
                'post_id': post.get('id', f"post_{i}")
            })
            ids.append(f"post_{post.get('id', i)}_{datetime.now().timestamp()}")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
    
    def retrieve_relevant_posts(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve most relevant posts for a query"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        relevant_posts = []
        for i in range(len(results['documents'][0])):
            relevant_posts.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return relevant_posts
    
    def generate_contextual_response(self, query: str, context_posts: List[Dict]) -> str:
        """Generate a contextual response using retrieved posts"""
        context = "\n\n".join([post['document'] for post in context_posts])
        
        if self.use_openai:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a social media trend analyst. 
                     Analyze the provided social media posts and provide insights about trends, 
                     viral content, cultural context, and social movements. Be objective and 
                     provide evidence-based analysis."""},
                    {"role": "user", "content": f"Query: {query}\n\nRelevant Social Media Posts:\n{context}"}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        else:
            # Fallback: Simple template-based response
            return f"""Based on the analysis of recent social media posts related to "{query}":

Context Summary:
{context[:500]}...

Key Insights:
- Found {len(context_posts)} relevant posts discussing this topic
- Posts span across platforms: {', '.join(set(post['metadata']['platform'] for post in context_posts))}
- Average engagement appears to be moderate to high based on the retrieved content

For more detailed analysis, please configure an OpenAI API key."""

    def analyze_query(self, query: str) -> Dict:
        """Complete RAG pipeline: retrieve and generate response"""
        relevant_posts = self.retrieve_relevant_posts(query)
        response = self.generate_contextual_response(query, relevant_posts)
        
        return {
            'query': query,
            'relevant_posts': relevant_posts,
            'generated_response': response,
            'timestamp': datetime.now().isoformat()
        }