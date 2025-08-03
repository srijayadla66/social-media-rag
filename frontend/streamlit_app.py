# frontend/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.rag_system import SocialMediaRAG
#from app.models.trend_detector import TrendDetector
from data.ingestion.twitter_client import TwitterClient
from data.processing.content_filter import filter_profanity
from data.processing.sentiment_analyzer import analyze_sentiment

st.set_page_config(
    page_title="Social Media RAG & Trend Analysis",
    page_icon="📱",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = SocialMediaRAG()
#if 'trend_detector' not in st.session_state:
    #st.session_state.trend_detector = TrendDetector()
if 'sample_data_loaded' not in st.session_state:
    st.session_state.sample_data_loaded = False

def load_sample_data():
    """Load sample social media data for demonstration"""
    sample_posts = [
        {
            'id': 1,
            'text': 'Just tried the new #ChatGPT update and it\'s incredible! AI is changing everything. #AI #Technology',
            'author': 'tech_enthusiast',
            'created_at': datetime.now() - timedelta(hours=2),
            'retweet_count': 45,
            'favorite_count': 123,
            'platform': 'twitter'
        },
        {
            'id': 2,
            'text': 'Climate change protests happening worldwide today. #ClimateAction #Environment #SaveThePlanet',
            'author': 'eco_warrior',
            'created_at': datetime.now() - timedelta(hours=1),
            'retweet_count': 234,
            'favorite_count': 567,
            'platform': 'twitter'
        },
        # Add more sample posts...
    ]

    # 1) Filter out posts containing blacklisted words
    sample_posts = filter_profanity(sample_posts)

    # 2) Analyze sentiment for each post
    sample_posts = analyze_sentiment(sample_posts)

    # 3) Add cleaned & annotated posts to the RAG system
    st.session_state.rag_system.add_posts(sample_posts)

    return sample_posts


def main():
    st.title("🌐 Social Media RAG with Trend Analysis")
    st.markdown("Analyze trending topics and get contextual insights from social media content")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # API Keys section
    with st.sidebar.expander("API Configuration"):
        openai_key = st.text_input("OpenAI API Key", type="password")
        twitter_keys = {
            'api_key': st.text_input("Twitter API Key", type="password"),
            'api_secret': st.text_input("Twitter API Secret", type="password"),
            'access_token': st.text_input("Twitter Access Token", type="password"),
            'access_token_secret': st.text_input("Twitter Access Token Secret", type="password")
        }
    
    # Load sample data
    if st.sidebar.button("Load Sample Data"):
        sample_posts = load_sample_data()
        st.session_state.sample_data_loaded = True
        st.sidebar.success(f"Loaded {len(sample_posts)} sample posts")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Trending Topics", "🤖 RAG Query", "📊 Analytics", "⚙️ Data Collection"])
    
    with tab1:
        st.header("Trending Topics Dashboard")
        
        if st.session_state.sample_data_loaded:
            # Mock trending topics for demo
            trending_data = [
                {'keyword': '#AI', 'mentions': 156, 'trend_score': 8.5, 'sentiment': 'positive'},
                {'keyword': '#ClimateChange', 'mentions': 234, 'trend_score': 7.8, 'sentiment': 'neutral'},
                {'keyword': '#Technology', 'mentions': 89, 'trend_score': 6.2, 'sentiment': 'positive'},
            ]
            
            # Display trending topics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                df = pd.DataFrame(trending_data)
                fig = px.bar(df, x='keyword', y='trend_score', 
                           title='Trending Topics by Score',
                           color='sentiment',
                           color_discrete_map={'positive': 'green', 'neutral': 'blue', 'negative': 'red'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top Trends")
                for trend in trending_data:
                    with st.container():
                        st.metric(
                            trend['keyword'], 
                            f"{trend['trend_score']:.1f}",
                            f"{trend['mentions']} mentions"
                        )
        else:
            st.info("Please load sample data from the sidebar to see trending topics.")
    
    with tab2:
        st.header("RAG-based Query Interface")
        
        if st.session_state.sample_data_loaded:
            query = st.text_input("Enter your query about social media trends:")
            
            if st.button("Analyze Query") and query:
                with st.spinner("Analyzing social media content..."):
                    result = st.session_state.rag_system.analyze_query(query)
                
                st.subheader("Generated Response")
                st.write(result['generated_response'])
                
                st.subheader("Relevant Posts")
                for i, post in enumerate(result['relevant_posts']):
                    with st.expander(f"Post {i+1} (Similarity: {post['similarity_score']:.2f})"):
                        st.text(post['document'])
                        st.json(post['metadata'])
        else:
            st.info("Please load sample data from the sidebar to use the RAG system.")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        # Platform distribution
        col1, col2 = st.columns(2)
        
        with col1:
            platform_data = {'Twitter': 65, 'Reddit': 25, 'Instagram': 10}
            fig = px.pie(values=list(platform_data.values()), 
                        names=list(platform_data.keys()),
                        title="Content Distribution by Platform")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Engagement timeline (mock data)
            dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
            engagement = [100, 150, 200, 180, 220, 300, 250]
            df_timeline = pd.DataFrame({'Date': dates, 'Engagement': engagement})
            
            fig = px.line(df_timeline, x='Date', y='Engagement', 
                         title='Engagement Timeline')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Data Collection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Collection Settings")
            keywords = st.text_area("Keywords to track (one per line)", 
                                   value="AI\nClimate Change\nTechnology")
            max_posts = st.slider("Max posts per keyword", 10, 500, 100)
        
        with col2:
            st.subheader("Collection Status")
            st.info("Real-time collection requires valid API keys")
            
            if st.button("Start Collection"):
                if all(twitter_keys.values()):
                    st.success("Collection started! (This is a demo)")
                else:
                    st.error("Please provide Twitter API keys")

if __name__ == "__main__":
    main()