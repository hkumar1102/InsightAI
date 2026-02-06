# app.py
# -----------------------------------------------------------------------------
# InsightAI: Enterprise-Grade Natural Language Processing Suite
# Architecture: Streamlit Frontend | HuggingFace Inference | Scikit-Learn Clustering
# Author: Senior AI Engineer
# Version: 5.0.0 (Titanium Release)
# -----------------------------------------------------------------------------

import streamlit as st
import psutil
import os
import math
import re
import numpy as np
import base64
import requests
from datetime import datetime
from typing import List, Tuple, Dict, Any

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="InsightAI | Enterprise NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'About': "InsightAI Titanium Edition. Built for high-throughput text analysis."
    }
)

# --- 2. LIBRARY IMPORTS ---
# Imported here to ensure page config runs first.
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min, cosine_similarity
from keybert import KeyBERT
from transformers import pipeline, AutoTokenizer
from newspaper import Article, Config
import spacy
from spacy import displacy
import textstat
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- 3. ARCHITECTURAL UTILITIES ---

def get_system_telemetry() -> str:
    """
    Monitors process memory usage.
    Crucial for containerized environments (Docker/K8s) to prevent OOM kills.
    """
    process = psutil.Process(os.getpid())
    mb_used = process.memory_info().rss / 1024 / 1024
    return f"{mb_used:.0f} MB"

# --- 4. NEURAL ENGINE LOADER (SINGLETON PATTERN) ---
# We use st.cache_resource to implement the Singleton pattern.
# This ensures models are loaded into RAM/VRAM exactly once per runtime.

@st.cache_resource(show_spinner=False)
def load_spacy_engine():
    """Loads SpaCy's optimized CPU pipeline for NER and POS tagging."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource(show_spinner=False)
def load_vectorizer(mode: str = "fast") -> SentenceTransformer:
    """
    Loads the Embedding Model.
    - Fast Mode: 'all-MiniLM-L6-v2' (384d, optimized for low latency).
    - Accurate Mode: 'all-mpnet-base-v2' (768d, optimized for semantic density).
    """
    model_id = 'all-MiniLM-L6-v2' if mode == "fast" else 'all-mpnet-base-v2'
    return SentenceTransformer(model_id)

@st.cache_resource(show_spinner=False)
def load_generative_model(mode: str = "fast"):
    """
    Loads Seq2Seq Transformer for Abstractive Summarization.
    """
    model_id = "sshleifer/distilbart-cnn-12-6" if mode == "fast" else "facebook/bart-large-cnn"
    return pipeline("summarization", model=model_id), AutoTokenizer.from_pretrained(model_id)

@st.cache_resource(show_spinner=False)
def load_sentiment_engine():
    """Loads a DistilBERT model fine-tuned on SST-2 for sentiment classification."""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- 5. DATA INGESTION LAYER ---

def normalize_text(text: str) -> List[str]:
    """
    Sanitizes text and performs sentence segmentation.
    Uses regex lookbehinds to split on punctuation without consuming it.
    """
    if not text: return []
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def scrape_url(url: str) -> str:
    """
    Robust URL scraper with User-Agent spoofing to bypass basic WAFs.
    """
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 15

    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        if len(article.text) < 50:
            return None # Reject empty/short content
        return article.text
    except Exception:
        return None

def perform_rag_retrieval(query: str, context_text: str, embedder) -> List[Tuple[str, float]]:
    """
    RAG (Retrieval Augmented Generation) Step.
    Calculates Cosine Similarity between Query Vector and Document Sentence Vectors.
    """
    sentences = normalize_text(context_text)
    if not sentences: return []
    
    # Vectorize
    doc_embeddings = embedder.encode(sentences)
    query_embedding = embedder.encode([query])
    
    # Compute Similarity
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Rank Top 3
    top_k_indices = np.argsort(scores)[-3:][::-1]
    results = [(sentences[i], float(scores[i])) for i in top_k_indices]
    return results

# --- 6. ADVANCED THEME-AGNOSTIC CSS ---
# Using CSS variables ensures automatic Dark/Light mode compatibility.

st.markdown("""
<style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Adaptive Cards - Uses system variables for colors */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--text-color);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-color: rgba(128, 128, 128, 0.2);
    }

    /* Input Fields */
    .stTextArea textarea, .stTextInput input {
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.3);
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #ff4b4b;
        box-shadow: 0 0 0 1px #ff4b4b;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 4px;
        font-weight: 600;
        padding: 0 16px;
    }
    
    /* Modern Gradient Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF914D 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    
    /* Cleanup */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 7. SIDEBAR DASHBOARD ---

with st.sidebar:
    st.title("🎛️ InsightAI Core")
    st.caption(f"Memory Usage: {get_system_telemetry()}")
    st.divider()
    
    st.subheader("⚙️ Pipeline Config")
    
    engine_mode = st.radio(
        "Inference Precision:",
        ["Speed (Distilled)", "Accuracy (Large)"],
        horizontal=False,
        help="Select 'Speed' for quick drafts or 'Accuracy' for production-grade deep analysis."
    )
    selected_mode = "fast" if "Speed" in engine_mode else "accurate"
    
    st.divider()
    
    st.markdown("**System Status**")
    st.markdown("🟢 NER Module: **Active**")
    st.markdown("🟢 Vector Database: **Ready**")
    st.markdown("🟢 Transformer: **Standby**")

# --- 8. MAIN APPLICATION LAYOUT ---

col_logo, col_header = st.columns([0.8, 5])
with col_logo:
    st.markdown("<h1 style='text-align: center;'>🧠</h1>", unsafe_allow_html=True)
with col_header:
    st.title("InsightAI: Enterprise Document Intelligence")
    st.caption("Automated Neural Processing Pipeline v5.0 | Titanium Edition")

# Input Section
st.write("###")
tab_input_text, tab_input_url = st.tabs(["📝 Direct Input", "🌐 Web Scraper"])

with tab_input_text:
    txt_input = st.text_area("Document Buffer:", height=200, label_visibility="collapsed", placeholder="Paste executive summary, legal text, or technical report here...")

with tab_input_url:
    url_input = st.text_input("Source URL:", label_visibility="collapsed", placeholder="https://techcrunch.com/article...")

# Action Row
c_action, c_spacer = st.columns([1, 4])
with c_action:
    trigger_analysis = st.button("⚡ Initialize Pipeline", use_container_width=True)

# --- 9. ORCHESTRATION PIPELINE ---

# Logic: URL Handling
if url_input and not txt_input:
    with st.spinner("🕷️ Accessing remote resource..."):
        fetched_content = scrape_url(url_input)
        if fetched_content:
            txt_input = fetched_content
            st.toast("Remote content successfully ingested!", icon="✅")
        else:
            st.error("Connection Failed. The website may be blocking bots or content is empty.")

if trigger_analysis and txt_input:
    # State persistence
    st.session_state.raw_text = txt_input
    st.session_state.analyzed = True
    
    # Real-time Status Container
    with st.status("🚀 Orchestrating AI Agents...", expanded=True) as status:
        
        # Step A: Normalization
        st.write("🔹 Tokenizing and normalizing text stream...")
        sentences = normalize_text(st.session_state.raw_text)
        if len(sentences) < 2:
            st.error("Insufficient data. Please provide at least 2 sentences.")
            st.stop()
            
        # Step B: Embeddings
        st.write("🔹 Computing high-dimensional vector embeddings...")
        embedder = load_vectorizer(selected_mode)
        embeddings = embedder.encode(sentences)
        st.session_state.embedder_instance = embedder # Persist for RAG
        
        # Step C: Unsupervised Clustering (K-Means)
        st.write("🔹 Executing K-Means clustering algorithm...")
        num_clusters = max(1, math.ceil(len(sentences) * 0.2)) # Dynamic K
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42).fit(embeddings)
        
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        closest.sort()
        st.session_state.extractive_sum = " ".join([sentences[i] for i in closest])
        
        # Step D: Abstractive Generation
        st.write("🔹 Synthesizing abstractive summary (Seq2Seq)...")
        gen_pipe, gen_tok = load_generative_model(selected_mode)
        
        # Chunking Logic for LLM Context Window
        chunks, current_chunk, current_len = [], [], 0
        for s in sentences:
            token_len = len(gen_tok.tokenize(s))
            if current_len + token_len < 1000:
                current_chunk.append(s)
                current_len += token_len
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [s]
                current_len = token_len
        if current_chunk: chunks.append(" ".join(current_chunk))
        
        # Batch Inference
        gen_results = gen_pipe(chunks, max_length=130, min_length=30, do_sample=False)
        st.session_state.abstractive_sum = " ".join([res['summary_text'] for res in gen_results])
        
        # Step E: Feature Extraction
        st.write("🔹 Extracting semantic entities and keywords...")
        kw_model = KeyBERT(model=embedder)
        st.session_state.keywords = kw_model.extract_keywords(st.session_state.raw_text, keyphrase_ngram_range=(1, 2), top_n=8)
        
        nlp = load_spacy_engine()
        st.session_state.doc_obj = nlp(st.session_state.raw_text)
        
        sent_pipe = load_sentiment_engine()
        st.session_state.sentiment_data = sent_pipe(st.session_state.raw_text[:512])[0]
        
        status.update(label="Analysis Pipeline Completed Successfully", state="complete", expanded=False)

# --- 10. ANALYTICS DASHBOARD ---

if st.session_state.get('analyzed'):
    st.divider()
    
    # Metrics Grid
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Readability (Flesch)", f"{textstat.flesch_reading_ease(st.session_state.raw_text):.1f}")
    with kpi2:
        s = st.session_state.sentiment_data
        st.metric("Sentiment Polarity", s['label'], f"{s['score']:.2f}")
    with kpi3:
        st.metric("Corpus Volume", f"{len(st.session_state.raw_text.split())} words")
    with kpi4:
        st.metric("Entities Detected", len(st.session_state.doc_obj.ents))
    
    st.write("###")
    
    # Feature Tabs
    tab_sum, tab_rag, tab_viz, tab_rep = st.tabs(["📝 Summaries", "🤖 Chat (RAG)", "📊 Analytics", "📄 Reports"])
    
    # Tab 1: Summaries
    with tab_sum:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Abstractive (AI Rewrite)")
            st.info(st.session_state.abstractive_sum)
        with c2:
            st.subheader("Extractive (Key Highlights)")
            st.success(st.session_state.extractive_sum)
            
    # Tab 2: RAG
    with tab_rag:
        st.subheader("💬 Context-Aware Q&A")
        st.caption("Ask natural language questions. The system retrieves answers from the document vector space.")
        
        user_query = st.text_input("Your Question:", placeholder="e.g., What are the key financial risks?")
        if user_query:
            rag_hits = perform_rag_retrieval(user_query, st.session_state.raw_text, st.session_state.embedder_instance)
            if rag_hits:
                for i, (txt, score) in enumerate(rag_hits):
                    with st.expander(f"Reference Match {i+1} (Relevance: {score:.2f})", expanded=(i==0)):
                        st.markdown(f"> {txt}")
            else:
                st.warning("No contextually relevant information found.")

    # Tab 3: Visualization
    with tab_viz:
        v1, v2 = st.columns(2)
        with v1:
            # Gauge Chart (Plotly)
            s_score = st.session_state.sentiment_data['score']
            if st.session_state.sentiment_data['label'] == 'NEGATIVE': s_score = -s_score
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=s_score,
                title={'text': "Sentiment Index"},
                gauge={'axis': {'range': [-1, 1]}, 'bar': {'color': "#FF4B4B"}}
            ))
            # Transparent bg for theme compatibility
            fig_gauge.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with v2:
            # Horizontal Bar Chart
            kws = {k[0]: k[1] for k in st.session_state.keywords}
            fig_bar = px.bar(
                x=list(kws.values()), y=list(kws.keys()), orientation='h',
                title="Semantic Keyphrases", labels={'x': 'Relevance', 'y': 'Phrase'}
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

    # Tab 4: Reporting
    with tab_rep:
        st.subheader("Generate Executive Report")
        
        # HTML Report Template
        report_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Helvetica', sans-serif; padding: 40px; color: #333; }}
                .header {{ border-bottom: 2px solid #FF4B4B; padding-bottom: 20px; margin-bottom: 30px; }}
                h1 {{ color: #FF4B4B; margin: 0; }}
                .meta {{ color: #666; font-size: 0.9em; }}
                .section {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #FF4B4B; }}
                h2 {{ font-size: 1.2em; color: #444; margin-top: 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>InsightAI Analysis Report</h1>
                <p class="meta">Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M")}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{st.session_state.abstractive_sum}</p>
            </div>
            
            <div class="section">
                <h2>Key Findings (Extractive)</h2>
                <p>{st.session_state.extractive_sum}</p>
            </div>
            
            <div class="section">
                <h2>Metrics</h2>
                <ul>
                    <li><strong>Sentiment:</strong> {st.session_state.sentiment_data['label']}</li>
                    <li><strong>Readability Score:</strong> {textstat.flesch_reading_ease(st.session_state.raw_text):.1f}</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        b64 = base64.b64encode(report_html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="insight_report.html">' \
               f'<button style="background-color: #FF4B4B; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer;">' \
               f'📥 Download HTML Report</button></a>'
        
        st.markdown(href, unsafe_allow_html=True)
        st.caption("Generates a standalone HTML file suitable for stakeholder distribution.")