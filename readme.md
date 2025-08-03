**Social Media RAG with Trend Analysis**

This repository contains a Retrieval-Augmented Generation (RAG) system for social media data, combined with a demonstration dashboard built using Streamlit. The system ingests posts from multiple platforms (Twitter stubbed in this demo), identifies trending topics, and provides contextual responses using an in-memory RAG implementation.

---

## 📁 Project Structure

```
social-media-rag/
├── app/
│   ├── main.py                # FastAPI backend
│   └── models/
│       ├── rag_system.py      # RAG implementation (in-memory)
│       └── trend_detector.py  # (Removed for demo) stubbed trending
├── data/
│   ├── ingestion/
│   │   └── twitter_client.py  # Twitter API client (async stub)
│   └── processing/
│       ├── content_filter.py  # Simple profanity filter
│       └── sentiment_analyzer.py # Basic sentiment via textstat
├── frontend/
│   └── streamlit_app.py       # Streamlit dashboard UI
├── requirements.txt           # Python dependencies
├── Dockerfile                 # For container deployment
├── .env.example               # Environment variable template
└── README.md                  # This documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- (Optional) Docker for containerized deployment

### 1. Clone the Repository

```bash
git clone https://github.com/srijayadla66/social-media-rag.git
cd social-media-rag
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux
# Edit .env to add API keys for OpenAI, Twitter, etc.
```

### 5. Run the Streamlit App

```bash
cd frontend
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) to view the dashboard.

---

## 📊 Features

- **Sample Data Pipeline:** Loads stubbed Twitter posts via `load_sample_data`.
- **Content Filtering:** Removes posts with blacklisted words (`content_filter.py`).
- **Sentiment Analysis:** Assigns "positive"/"negative" tags based on Flesch Reading Ease (`sentiment_analyzer.py`).
- **RAG System:** In-memory vector storage via SentenceTransformers (`rag_system.py`).
- **Streamlit Dashboard:** Four tabs—Trending Topics (mock), RAG Query, Analytics, Data Collection.

---

## 📦 Deployment

### 1. Docker

```bash
docker build -t social-media-rag .
docker run -p 8501:8501 social-media-rag
```

### 2. Streamlit Community Cloud

Push to GitHub and deploy via [share.streamlit.io](https://share.streamlit.io).

Live Demo: [https://srijayadla66-social-media-rag.streamlit.app](https://srijayadla66-social-media-rag.streamlit.app)

---

## 🛠️ Future Work

- Integrate real-time data ingestion from Reddit and Instagram.
- Replace mock trending with live `TrendDetector` once platform supports NLTK or use pure-Python approach.
- Implement persistent vector DB (e.g., Pinecone) for RAG storage.
- Add user authentication and role-based access.

---

## 📜 License

MIT License

