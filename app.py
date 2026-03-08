import html
import hashlib
import io
import json
import logging
import math
import os
import re
import warnings
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import psutil
import requests
import spacy
import streamlit as st
import textstat
import torch
from bs4 import BeautifulSoup
from keybert import KeyBERT
from newspaper import Article, Config
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin_min
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, logging as hf_logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
try:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except Exception:
    pass

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="InsightAI | Advanced Document Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAX_INPUT_CHARS = 45000
SUMMARY_CHUNK_TOKEN_LIMIT = 850
DEFAULT_RAG_RESULTS = 3
SUMMARY_INPUT_LIMIT = 1024
MAX_HISTORY_ITEMS = 8

SAMPLE_TEXT = (
    "InsightAI processed the Q4 operations report for Acme Robotics. "
    "Revenue grew 14 percent year-over-year, driven by enterprise subscriptions and "
    "service renewals. Manufacturing lead time improved from 26 days to 18 days after "
    "the supply-chain redesign. However, the risk office flagged two concerns: rising "
    "component costs in Asia and delayed compliance documentation for one product line. "
    "The executive team plans to offset cost pressure through multi-vendor procurement, "
    "while legal and engineering teams will complete documentation by next month. "
    "Customer sentiment remains positive, with support tickets down 11 percent and "
    "retention above target."
)

STATE_DEFAULTS = {
    "analyzed": False,
    "analysis_result": None,
    "last_error": "",
    "input_text_buffer": "",
    "url_input_buffer": "",
    "selected_mode_label": "Speed (Distilled)",
    "ui_theme_mode": "Auto (System)",
    "analysis_preset": "Balanced",
    "last_applied_preset": "",
    "rag_top_k": DEFAULT_RAG_RESULTS,
    "keyword_top_n": 8,
    "summary_detail": 2,
    "rag_question": "",
    "rag_last_question": "",
    "rag_answer_draft": "",
    "embedder_instance": None,
    "analysis_history": [],
    "upload_widget_nonce": 0,
}


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def init_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    st.session_state.analyzed = False
    st.session_state.analysis_result = None
    st.session_state.last_error = ""
    st.session_state.embedder_instance = None
    st.session_state.rag_question = ""
    st.session_state.rag_last_question = ""
    st.session_state.rag_answer_draft = ""


def load_demo_input() -> None:
    st.session_state.input_text_buffer = SAMPLE_TEXT
    st.session_state.url_input_buffer = ""
    st.session_state.upload_widget_nonce += 1
    reset_analysis_state()


def clear_all_inputs() -> None:
    st.session_state.input_text_buffer = ""
    st.session_state.url_input_buffer = ""
    st.session_state.upload_widget_nonce += 1
    reset_analysis_state()


def get_system_telemetry() -> str:
    process = psutil.Process(os.getpid())
    mb_used = process.memory_info().rss / 1024 / 1024
    return f"{mb_used:.0f} MB"


def resolve_theme_mode(theme_choice: str) -> str:
    if theme_choice == "Light":
        return "light"
    if theme_choice == "Dark":
        return "dark"
    return "auto"


def theme_tokens(mode: str) -> Dict[str, str]:
    if mode == "dark":
        return {
            "text": "#e5e7eb",
            "muted_text": "#a3aab8",
            "surface": "rgba(16, 24, 37, 0.82)",
            "surface_soft": "rgba(22, 32, 50, 0.72)",
            "surface_strong": "rgba(11, 19, 32, 0.88)",
            "border": "rgba(148, 163, 184, 0.30)",
            "shadow": "rgba(0, 0, 0, 0.42)",
            "accent": "#2dd4bf",
            "accent_soft": "#14b8a6",
            "chip_bg": "rgba(45, 212, 191, 0.14)",
            "chip_border": "rgba(45, 212, 191, 0.35)",
            "chip_label": "#9fe9dd",
            "chip_text": "#eafdf8",
            "hero_a": "#09111f",
            "hero_b": "#0f2a49",
            "hero_c": "#0f766e",
            "bg_a": "rgba(45, 212, 191, 0.14)",
            "bg_b": "rgba(251, 146, 60, 0.12)",
        }
    return {
        "text": "#0f172a",
        "muted_text": "#334155",
        "surface": "rgba(255, 255, 255, 0.82)",
        "surface_soft": "rgba(248, 250, 252, 0.78)",
        "surface_strong": "rgba(255, 255, 255, 0.95)",
        "border": "rgba(100, 116, 139, 0.28)",
        "shadow": "rgba(2, 8, 23, 0.16)",
        "accent": "#0f766e",
        "accent_soft": "#14b8a6",
        "chip_bg": "rgba(15, 118, 110, 0.08)",
        "chip_border": "rgba(15, 118, 110, 0.26)",
        "chip_label": "#0f766e",
        "chip_text": "#0f172a",
        "hero_a": "#0b1324",
        "hero_b": "#1a3a61",
        "hero_c": "#0f766e",
        "bg_a": "rgba(20, 184, 166, 0.16)",
        "bg_b": "rgba(251, 146, 60, 0.12)",
    }


def runtime_chart_mode(theme_mode: str) -> str:
    if theme_mode != "auto":
        return theme_mode
    base_theme = str(st.get_option("theme.base") or "").lower()
    return "dark" if base_theme == "dark" else "light"


def theme_scope_css(theme_mode: str) -> str:
    def to_css_vars(tokens: Dict[str, str]) -> str:
        lines = []
        for key, value in tokens.items():
            css_key = key.replace("_", "-")
            lines.append(f"--ui-{css_key}: {value};")
        return " ".join(lines)

    light_vars = to_css_vars(theme_tokens("light"))
    dark_vars = to_css_vars(theme_tokens("dark"))

    if theme_mode == "light":
        return f":root{{{light_vars}}}"
    if theme_mode == "dark":
        return f":root{{{dark_vars}}}"
    return (
        f":root{{{light_vars}}}"
        f"@media (prefers-color-scheme: dark){{:root{{{dark_vars}}}}}"
    )


def theme_mode_label(theme_mode: str) -> str:
    if theme_mode == "auto":
        return "Auto (System)"
    return theme_mode.title()


def normalize_text(text: str) -> List[str]:
    if not text:
        return []
    cleaned = text.replace("\n", " ").replace("\r", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = re.split(r"(?<=[.?!])\s+", cleaned)
    return [s.strip() for s in sentences if s.strip()]


def lexical_diversity(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def decode_text_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def extract_text_from_upload(uploaded_file: Any) -> Optional[str]:
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        return None

    if filename.endswith((".txt", ".md", ".csv", ".log")):
        text = decode_text_bytes(raw_bytes).strip()
        return text or None

    if filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = " ".join(pages).strip()
            return text or None
        except Exception:
            return None

    return None


def quality_score(readability: float, lexical_div: float, sentence_count: int) -> int:
    read_component = max(0.0, min(1.0, readability / 100.0)) * 45
    lexical_component = max(0.0, min(1.0, lexical_div)) * 35
    structure_component = max(0.0, min(1.0, sentence_count / 50.0)) * 20
    return int(round(read_component + lexical_component + structure_component))


def sentence_salience(sentences: Sequence[str], sentence_embeddings: np.ndarray) -> List[Tuple[int, str, float]]:
    if len(sentences) == 0:
        return []
    centroid = np.mean(sentence_embeddings, axis=0, keepdims=True)
    salience_scores = cosine_similarity(sentence_embeddings, centroid).flatten()
    ranking = np.argsort(salience_scores)[::-1]
    return [(int(idx), sentences[int(idx)], float(salience_scores[int(idx)])) for idx in ranking]


def push_history(result: Dict[str, Any]) -> None:
    history_entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "words": result["word_count"],
        "sentiment": result["sentiment"]["label"],
        "score": result["quality_score"],
    }
    history = st.session_state.analysis_history
    history.insert(0, history_entry)
    st.session_state.analysis_history = history[:MAX_HISTORY_ITEMS]


def build_business_readout(result: Dict[str, Any]) -> List[str]:
    sentiment_label = str(result["sentiment"]["label"]).upper()
    sentiment_score = float(result["sentiment"]["score"])
    tone = "Neutral"
    if sentiment_label == "POSITIVE":
        tone = "Growth-positive"
    elif sentiment_label == "NEGATIVE":
        tone = "Risk-heavy"

    top_keywords = ", ".join([k for k, _ in result["keywords"][:4]]) or "No major keyphrases"
    reading_time = max(1, int(round(result["word_count"] / 220)))

    risk_terms = {"risk", "delay", "loss", "decline", "issue", "concern", "compliance", "cost"}
    opportunity_terms = {"growth", "increase", "improve", "gain", "opportunity", "retention", "efficiency"}
    risk_hits = 0
    opportunity_hits = 0
    for _, sentence_text, _ in result.get("salience", []):
        lowered = sentence_text.lower()
        if any(term in lowered for term in risk_terms):
            risk_hits += 1
        if any(term in lowered for term in opportunity_terms):
            opportunity_hits += 1

    return [
        f"Document signal: {tone} ({sentiment_score:.2f} confidence).",
        f"Estimated reading time: {reading_time} minute(s) for full text.",
        f"Primary themes: {top_keywords}.",
        f"Risk markers detected in salient statements: {risk_hits}.",
        f"Opportunity markers detected in salient statements: {opportunity_hits}.",
    ]


def extract_action_items(sentences: Sequence[str]) -> List[str]:
    action_markers = (
        "should",
        "must",
        "need to",
        "next step",
        "plan to",
        "will",
        "recommend",
        "priority",
        "action",
    )
    items: List[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(marker in lowered for marker in action_markers):
            item = sentence.strip()
            if item and item not in items:
                items.append(item)
        if len(items) >= 8:
            break
    return items


def apply_preset_settings(preset_name: str) -> None:
    presets = {
        "Balanced": {
            "selected_mode_label": "Speed (Distilled)",
            "rag_top_k": 3,
            "keyword_top_n": 8,
            "summary_detail": 2,
        },
        "Executive Brief": {
            "selected_mode_label": "Speed (Distilled)",
            "rag_top_k": 2,
            "keyword_top_n": 6,
            "summary_detail": 1,
        },
        "Risk Audit": {
            "selected_mode_label": "Accuracy (Large)",
            "rag_top_k": 5,
            "keyword_top_n": 12,
            "summary_detail": 3,
        },
        "Technical Deep Dive": {
            "selected_mode_label": "Accuracy (Large)",
            "rag_top_k": 4,
            "keyword_top_n": 14,
            "summary_detail": 3,
        },
    }
    selected = presets.get(preset_name, presets["Balanced"])
    for key, value in selected.items():
        st.session_state[key] = value
    st.session_state.last_applied_preset = preset_name


def preset_description(preset_name: str) -> str:
    descriptions = {
        "Balanced": "Balanced output quality and speed for regular long-form analysis.",
        "Executive Brief": "Crisp summaries and focused retrieval for decision-ready briefings.",
        "Risk Audit": "Higher depth and recall to surface risk signals and compliance concerns.",
        "Technical Deep Dive": "Detail-oriented mode for dense technical and engineering content.",
    }
    return descriptions.get(preset_name, descriptions["Balanced"])


def render_narrative_card(title: str, body: str, tone: str = "neutral") -> None:
    safe_title = html.escape(title)
    safe_body = html.escape(body).replace("\n", "<br/>")
    tone_class = "narrative-neutral"
    if tone == "accent":
        tone_class = "narrative-accent"
    if tone == "success":
        tone_class = "narrative-success"
    st.markdown(
        (
            f"<div class='narrative-card {tone_class}'>"
            f"<div class='narrative-title'>{safe_title}</div>"
            f"<div class='narrative-body'>{safe_body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def estimate_run_profile(
    text_chars: int,
    has_url: bool,
    has_file: bool,
    mode_label: str,
) -> Tuple[str, str, str]:
    effective_chars = text_chars
    if effective_chars == 0 and has_url:
        effective_chars = 2200
    if effective_chars == 0 and has_file:
        effective_chars = 3200

    if effective_chars < 1200:
        load_tier = "Light"
        eta = "4-8s"
    elif effective_chars < 6000:
        load_tier = "Standard"
        eta = "8-18s"
    elif effective_chars < 16000:
        load_tier = "Heavy"
        eta = "18-40s"
    else:
        load_tier = "Extended"
        eta = "40s+"

    if mode_label.startswith("Accuracy"):
        eta = "15-25s" if load_tier in {"Light", "Standard"} else "30s+"

    readiness = "Ready"
    if text_chars == 0 and not has_url and not has_file:
        readiness = "Missing Input"

    return readiness, load_tier, eta


def scrape_url(url: str) -> Optional[str]:
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 15

    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = (article.text or "").strip()
        if len(text) < 80:
            raise ValueError("Content too short for article parser")
        return text
    except Exception:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": user_agent},
                timeout=15,
            )
            if resp.status_code >= 400:
                return None
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
                tag.decompose()
            blocks = [node.get_text(" ", strip=True) for node in soup.find_all(["p", "li"])]
            cleaned_blocks = [block for block in blocks if len(block) > 45]
            text = " ".join(cleaned_blocks)
            text = re.sub(r"\s+", " ", text).strip()
            return text if len(text) >= 80 else None
        except Exception:
            return None


def build_summary_chunks(
    sentences: Sequence[str], tokenizer: AutoTokenizer, token_limit: int = SUMMARY_CHUNK_TOKEN_LIMIT
) -> List[str]:
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        token_count = len(tokenizer.tokenize(sentence))

        if token_count >= token_limit:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.append(sentence)
            continue

        if current_chunk and (current_tokens + token_count > token_limit):
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = token_count
        else:
            current_chunk.append(sentence)
            current_tokens += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [chunk for chunk in chunks if chunk.strip()]


def summary_length_bounds(token_count: int) -> Tuple[int, int]:
    max_len = int(min(200, max(70, token_count * 0.45)))
    min_len = int(min(90, max(25, max_len * 0.45)))
    if min_len >= max_len:
        min_len = max(20, max_len - 20)
    return min_len, max_len


def summarize_chunk(
    chunk: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    min_len: int,
    max_len: int,
) -> str:
    max_input_tokens = SUMMARY_INPUT_LIMIT
    if isinstance(tokenizer.model_max_length, int) and tokenizer.model_max_length > 0:
        max_input_tokens = min(max_input_tokens, tokenizer.model_max_length)

    encoded = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            min_length=min_len,
            max_length=max_len,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def build_extractive_summary(sentences: Sequence[str], embeddings: np.ndarray) -> str:
    if len(sentences) <= 3:
        return " ".join(sentences)

    cluster_count = max(1, min(len(sentences), math.ceil(len(sentences) * 0.2)))
    kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
    kmeans.fit(embeddings)

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    selected_indices = sorted(set(int(i) for i in closest))
    return " ".join(sentences[idx] for idx in selected_indices)


def perform_rag_retrieval(
    query: str,
    sentences: Sequence[str],
    sentence_embeddings: np.ndarray,
    embedder: Any,
    top_k: int = DEFAULT_RAG_RESULTS,
) -> List[Tuple[str, float]]:
    if not query.strip() or not sentences:
        return []

    query_embedding = embedder.encode([query])
    scores = cosine_similarity(query_embedding, sentence_embeddings)[0]

    top_k = max(1, min(top_k, len(sentences)))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score >= 0.05:
            results.append((sentences[idx], score))
    return results


def synthesize_rag_answer(query: str, hits: Sequence[Tuple[str, float]], mode: str = "fast") -> str:
    if not hits:
        return ""

    context_text = " ".join([snippet for snippet, _ in hits[:3]])
    prompt = f"Question: {query}\nContext: {context_text}\nAnswer:"
    try:
        model, tokenizer = load_generative_model("fast" if mode == "accurate" else mode)
        token_count = len(tokenizer.tokenize(prompt))
        min_len, max_len = summary_length_bounds(token_count)
        return summarize_chunk(prompt, model, tokenizer, min_len=min_len, max_len=max_len)
    except Exception:
        return hits[0][0]


def build_report_html(result: Dict[str, Any]) -> str:
    keyword_items = "".join(
        f"<li>{html.escape(keyword)} <em>({score:.2f})</em></li>"
        for keyword, score in result["keywords"]
    )

    entity_items = "".join(
        f"<li>{html.escape(label)}: {count}</li>"
        for label, count in result["entity_counts"].items()
    )
    executive_items = "".join(
        f"<li>{html.escape(line)}</li>"
        for line in build_business_readout(result)
    )
    action_items = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in result.get("action_items", [])
    )
    warning_items = "".join(
        f"<li>{html.escape(warn)}</li>"
        for warn in result.get("warnings", [])
    )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>InsightAI Report</title>
    <style>
        body {{
            font-family: Segoe UI, Arial, sans-serif;
            margin: 0;
            padding: 28px;
            background: #f8fafc;
            color: #0f172a;
        }}
        .container {{
            max-width: 980px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 14px;
            box-shadow: 0 12px 32px rgba(2, 6, 23, 0.08);
            padding: 28px;
        }}
        .header {{
            border-bottom: 2px solid #0f766e;
            margin-bottom: 20px;
            padding-bottom: 14px;
        }}
        .meta {{ color: #334155; font-size: 0.95rem; }}
        .section {{
            margin: 16px 0;
            background: #f1f5f9;
            border-left: 5px solid #0f766e;
            padding: 14px;
            border-radius: 8px;
        }}
        h1, h2 {{ margin: 0 0 10px 0; }}
        ul {{ margin: 0; padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>InsightAI Analysis Report</h1>
            <div class="meta">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
        </div>

        <div class="section">
            <h2>Abstractive Summary</h2>
            <p>{html.escape(result['abstractive_summary'])}</p>
        </div>

        <div class="section">
            <h2>Extractive Highlights</h2>
            <p>{html.escape(result['extractive_summary'])}</p>
        </div>

        <div class="section">
            <h2>Executive Readout</h2>
            <ul>{executive_items}</ul>
        </div>

        <div class="section">
            <h2>Metrics</h2>
            <ul>
                <li>Word Count: {result['word_count']}</li>
                <li>Sentence Count: {result['sentence_count']}</li>
                <li>Insight Score: {result['quality_score']}/100</li>
                <li>Readability (Flesch): {result['readability']:.1f}</li>
                <li>Sentiment: {html.escape(result['sentiment']['label'])} ({result['sentiment']['score']:.2f})</li>
                <li>Lexical Diversity: {result['lexical_diversity']:.2f}</li>
                <li>Average Sentence Length: {result['avg_sentence_length']:.1f} words</li>
                <li>Processing Time: {result.get('processing_seconds', 0.0):.1f} seconds</li>
            </ul>
        </div>

        <div class="section">
            <h2>Top Keywords</h2>
            <ul>{keyword_items}</ul>
        </div>

        <div class="section">
            <h2>Named Entities</h2>
            <ul>{entity_items or '<li>No named entities detected.</li>'}</ul>
        </div>

        <div class="section">
            <h2>Action Items</h2>
            <ul>{action_items or '<li>No explicit action items detected.</li>'}</ul>
        </div>

        <div class="section">
            <h2>Pipeline Notes</h2>
            <ul>{warning_items or '<li>No warnings reported.</li>'}</ul>
        </div>
    </div>
</body>
</html>
""".strip()


# -----------------------------------------------------------------------------
# Cached Model Loaders
# -----------------------------------------------------------------------------
class LightweightEmbedder:
    """Deterministic local fallback encoder when transformer embeddings are unavailable."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _encode_sentence(self, sentence: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"\b\w+\b", sentence.lower())
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, byteorder="little", signed=False) % self.dim
            vec[idx] += 1.0

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, sentences: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        del show_progress_bar
        if isinstance(sentences, str):
            sentences = [sentences]
        if not sentences:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self._encode_sentence(sentence) for sentence in sentences])


@st.cache_resource(show_spinner=False)
def load_lightweight_embedder() -> LightweightEmbedder:
    return LightweightEmbedder()


@st.cache_resource(show_spinner=False)
def load_spacy_engine() -> Any:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        try:
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception:
            # Fallback keeps the app usable even when model download is blocked.
            return spacy.blank("en")


@st.cache_resource(show_spinner=False)
def load_vectorizer(mode: str = "fast") -> SentenceTransformer:
    model_id = "all-MiniLM-L6-v2" if mode == "fast" else "all-mpnet-base-v2"
    return SentenceTransformer(model_id)


@st.cache_resource(show_spinner=False)
def load_generative_model(mode: str = "fast") -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    model_id = "sshleifer/distilbart-cnn-12-6" if mode == "fast" else "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_sentiment_engine() -> Any:
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    try:
        return pipeline("sentiment-analysis", model=model_id)
    except Exception:
        return pipeline("text-classification", model=model_id)


# -----------------------------------------------------------------------------
# Analysis Pipeline
# -----------------------------------------------------------------------------
def run_analysis(
    text: str,
    mode: str,
    keyword_top_n: int,
    summary_chunk_token_limit: int,
) -> Tuple[Dict[str, Any], Any]:
    if not text.strip():
        raise ValueError("Input text is empty.")

    sentences = normalize_text(text)
    if len(sentences) < 2:
        raise ValueError("Please provide at least two complete sentences.")

    pipeline_warnings: List[str] = []
    try:
        embedder = load_vectorizer(mode)
    except Exception as exc:
        if mode == "accurate":
            try:
                embedder = load_vectorizer("fast")
                pipeline_warnings.append(
                    f"High-accuracy embedding model unavailable, switched to fast mode: {exc}"
                )
            except Exception as fallback_exc:
                embedder = load_lightweight_embedder()
                pipeline_warnings.append(
                    "Embedding models unavailable, switched to lightweight local embeddings."
                )
                pipeline_warnings.append(f"Model load details: {fallback_exc}")
        else:
            embedder = load_lightweight_embedder()
            pipeline_warnings.append(
                f"Fast embedding model unavailable, switched to lightweight local embeddings: {exc}"
            )

    sentence_embeddings = embedder.encode(sentences, show_progress_bar=False)

    extractive_summary = build_extractive_summary(sentences, sentence_embeddings)

    abstractive_summary = extractive_summary
    try:
        summarizer_model, tokenizer = load_generative_model(mode)
        chunks = build_summary_chunks(sentences, tokenizer, token_limit=summary_chunk_token_limit)
        if not chunks:
            chunks = [" ".join(sentences)]

        abstractive_parts: List[str] = []
        for chunk in chunks:
            token_count = len(tokenizer.tokenize(chunk))
            if token_count < 40:
                abstractive_parts.append(chunk)
                continue

            min_len, max_len = summary_length_bounds(token_count)
            summary_text = summarize_chunk(chunk, summarizer_model, tokenizer, min_len, max_len)
            abstractive_parts.append(summary_text or chunk)

        joined_summary = " ".join(abstractive_parts).strip()
        if joined_summary:
            abstractive_summary = joined_summary
    except Exception as exc:
        if mode == "accurate":
            try:
                summarizer_model, tokenizer = load_generative_model("fast")
                chunks = build_summary_chunks(
                    sentences, tokenizer, token_limit=summary_chunk_token_limit
                ) or [" ".join(sentences)]
                fast_parts: List[str] = []
                for chunk in chunks:
                    token_count = len(tokenizer.tokenize(chunk))
                    if token_count < 40:
                        fast_parts.append(chunk)
                        continue
                    min_len, max_len = summary_length_bounds(token_count)
                    summary_text = summarize_chunk(chunk, summarizer_model, tokenizer, min_len, max_len)
                    fast_parts.append(summary_text or chunk)
                fallback_summary = " ".join(fast_parts).strip()
                if fallback_summary:
                    abstractive_summary = fallback_summary
                pipeline_warnings.append(
                    f"High-accuracy summarizer unavailable, switched to fast mode: {exc}"
                )
            except Exception as fallback_exc:
                pipeline_warnings.append(f"Summarization fallback used: {fallback_exc}")
        else:
            pipeline_warnings.append(f"Summarization fallback used: {exc}")

    keywords: List[Tuple[str, float]] = []
    try:
        kw_model = KeyBERT(model=embedder)
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=keyword_top_n,
        )
    except Exception as exc:
        pipeline_warnings.append(f"Keyword extraction unavailable: {exc}")

    nlp = load_spacy_engine()
    doc = nlp(text)
    entity_counts = dict(Counter(ent.label_ for ent in doc.ents).most_common())

    try:
        sentiment_engine = load_sentiment_engine()
        sentiment = sentiment_engine(text[:512])[0]
    except Exception as exc:
        sentiment = {"label": "UNAVAILABLE", "score": 0.0}
        pipeline_warnings.append(f"Sentiment unavailable: {exc}")

    sentence_lengths = [len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences]
    salience = sentence_salience(sentences, sentence_embeddings)
    readability = textstat.flesch_reading_ease(text)
    lex_div = lexical_diversity(text)

    return {
        "raw_text": text,
        "sentences": sentences,
        "sentence_embeddings": sentence_embeddings,
        "abstractive_summary": abstractive_summary,
        "extractive_summary": extractive_summary,
        "keywords": keywords,
        "entity_counts": entity_counts,
        "entities": [(ent.text, ent.label_) for ent in doc.ents[:40]],
        "sentiment": sentiment,
        "readability": readability,
        "word_count": len(re.findall(r"\b\w+\b", text)),
        "sentence_count": len(sentences),
        "lexical_diversity": lex_div,
        "sentence_lengths": sentence_lengths,
        "salience": salience[:10],
        "quality_score": quality_score(readability, lex_div, len(sentences)),
        "avg_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "action_items": extract_action_items(sentences),
        "processing_seconds": 0.0,
        "warnings": pipeline_warnings,
    }, embedder


# -----------------------------------------------------------------------------
# UI Styling
# -----------------------------------------------------------------------------
init_state()
current_theme_mode = resolve_theme_mode(st.session_state.ui_theme_mode)
chart_mode = runtime_chart_mode(current_theme_mode)
chart_tokens = theme_tokens(chart_mode)
is_dark_ui = chart_mode == "dark"
plotly_template = "plotly_dark" if is_dark_ui else "plotly_white"
plot_neg = "rgba(248, 113, 113, 0.28)" if is_dark_ui else "#fee2e2"
plot_neu = "rgba(251, 191, 36, 0.28)" if is_dark_ui else "#fef3c7"
plot_pos = "rgba(74, 222, 128, 0.24)" if is_dark_ui else "#dcfce7"
scope_css = theme_scope_css(current_theme_mode)

style_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
    __THEME_SCOPE__
    :root {
        color-scheme: light dark;
    }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }
    .stApp {
        font-family: 'Manrope', sans-serif;
        background:
            radial-gradient(circle at 85% 12%, var(--ui-bg-a), transparent 35%),
            radial-gradient(circle at 18% 88%, var(--ui-bg-b), transparent 30%),
            var(--background-color);
        color: var(--ui-text);
    }
    p, li {
        color: var(--ui-text);
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.01em;
    }
    .section-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.06rem;
        font-weight: 700;
        color: var(--ui-text);
        margin: 0.35rem 0 0.65rem 0;
        letter-spacing: 0.01em;
        position: relative;
        padding-bottom: 0.18rem;
    }
    .section-head::after {
        content: '';
        position: absolute;
        left: 0;
        bottom: -2px;
        width: 52px;
        height: 2px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--ui-accent) 0%, color-mix(in srgb, var(--ui-accent-soft) 70%, #ffffff) 100%);
    }
    .section-head small {
        color: var(--ui-muted-text);
        font-family: 'Manrope', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        text-align: right;
    }
    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes heroSweep {
        0% {
            transform: translateX(-130%) rotate(11deg);
            opacity: 0.0;
        }
        30% {
            opacity: 0.18;
        }
        100% {
            transform: translateX(210%) rotate(11deg);
            opacity: 0.0;
        }
    }
    .preset-note {
        border: 1px dashed var(--ui-border);
        border-radius: 12px;
        background: color-mix(in srgb, var(--ui-surface-soft) 86%, transparent);
        padding: 10px 12px;
        margin: 8px 0 2px 0;
    }
    .preset-note b {
        color: var(--ui-text);
        font-size: 0.84rem;
        margin-right: 6px;
    }
    .preset-note span {
        color: var(--ui-muted-text);
        font-size: 0.84rem;
    }
    .status-pills {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 14px;
    }
    .status-pill {
        border: 1px solid var(--ui-chip-border);
        background: linear-gradient(135deg, var(--ui-chip-bg) 0%, color-mix(in srgb, var(--ui-surface) 72%, transparent) 100%);
        border-radius: 12px;
        padding: 10px 12px;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 22px var(--ui-shadow);
        animation: fadeUp 320ms ease both;
        transition: transform 0.24s ease, box-shadow 0.24s ease, border-color 0.24s ease;
    }
    .status-pill:hover {
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--ui-accent) 46%, var(--ui-chip-border));
        box-shadow: 0 12px 26px var(--ui-shadow);
    }
    .status-pill .k {
        display: block;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--ui-chip-label);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }
    .status-pill .v {
        display: block;
        font-size: 0.93rem;
        font-weight: 700;
        color: var(--ui-chip-text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .result-banner {
        border: 1px solid var(--ui-border);
        border-radius: 14px;
        padding: 12px 14px;
        background: linear-gradient(120deg, var(--ui-surface-strong) 0%, var(--ui-surface-soft) 100%);
        margin-bottom: 12px;
        box-shadow: 0 10px 26px var(--ui-shadow);
        animation: fadeUp 260ms ease both;
    }
    .result-banner b {
        color: var(--ui-text);
        margin-right: 6px;
    }
    .result-banner span {
        color: var(--ui-muted-text);
    }
    .narrative-card {
        border: 1px solid var(--ui-border);
        border-radius: 14px;
        background: linear-gradient(155deg, var(--ui-surface-strong) 0%, var(--ui-surface-soft) 100%);
        padding: 12px 14px;
        min-height: 170px;
        box-shadow: 0 10px 24px var(--ui-shadow);
        animation: fadeUp 320ms ease both;
        transition: transform 0.24s ease, box-shadow 0.24s ease, border-color 0.24s ease;
    }
    .narrative-card:hover {
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--ui-accent) 45%, var(--ui-border));
        box-shadow: 0 14px 30px var(--ui-shadow);
    }
    .narrative-accent {
        border-left: 4px solid var(--ui-accent);
    }
    .narrative-success {
        border-left: 4px solid var(--ui-accent-soft);
    }
    .narrative-neutral {
        border-left: 4px solid color-mix(in srgb, var(--ui-accent) 40%, var(--ui-border));
    }
    .narrative-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: var(--ui-text);
        font-size: 1rem;
        margin-bottom: 8px;
        letter-spacing: 0.01em;
    }
    .narrative-body {
        color: var(--ui-muted-text);
        line-height: 1.56;
        font-size: 0.92rem;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 4px 0 14px 0;
    }
    .feature-card {
        border: 1px solid var(--ui-border);
        border-radius: 12px;
        padding: 12px;
        background: linear-gradient(160deg, var(--ui-surface-strong) 0%, var(--ui-surface-soft) 100%);
        box-shadow: 0 8px 20px var(--ui-shadow);
        animation: fadeUp 300ms ease both;
        transition: transform 0.24s ease, box-shadow 0.24s ease, border-color 0.24s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--ui-accent) 45%, var(--ui-border));
        box-shadow: 0 12px 28px var(--ui-shadow);
    }
    .feature-card b {
        display: block;
        color: var(--ui-text);
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    .feature-card span {
        display: block;
        color: var(--ui-muted-text);
        font-size: 0.83rem;
        line-height: 1.35;
    }
    .readiness-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 8px 0 14px 0;
    }
    .readiness-card {
        border: 1px solid var(--ui-border);
        border-radius: 12px;
        padding: 10px 12px;
        background: linear-gradient(155deg, var(--ui-surface-strong) 0%, var(--ui-surface-soft) 100%);
        box-shadow: 0 8px 20px var(--ui-shadow);
        transition: transform 0.24s ease, box-shadow 0.24s ease, border-color 0.24s ease;
    }
    .readiness-card:hover {
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--ui-accent) 45%, var(--ui-border));
        box-shadow: 0 12px 28px var(--ui-shadow);
    }
    .readiness-card .k {
        display: block;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--ui-muted-text);
        font-weight: 700;
        margin-bottom: 3px;
    }
    .readiness-card .v {
        display: block;
        font-size: 0.93rem;
        color: var(--ui-text);
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--ui-border);
        background: linear-gradient(125deg, var(--ui-hero-a) 0%, var(--ui-hero-b) 46%, var(--ui-hero-c) 100%);
        border-radius: 18px;
        padding: 24px;
        color: #eff8ff;
        margin-bottom: 18px;
        box-shadow: 0 18px 46px var(--ui-shadow);
        animation: fadeUp 360ms ease both;
    }
    .hero > * {
        position: relative;
        z-index: 1;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -34%;
        left: -22%;
        width: 52%;
        height: 190%;
        background: linear-gradient(115deg, transparent 0%, rgba(255, 255, 255, 0.24) 48%, transparent 100%);
        transform: translateX(-130%) rotate(11deg);
        animation: heroSweep 8s linear infinite;
        pointer-events: none;
    }
    .hero::after {
        content: '';
        position: absolute;
        width: 240px;
        height: 240px;
        top: -70px;
        right: -80px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.24) 0%, transparent 70%);
    }
    .hero h2 {
        margin: 0 0 8px 0;
        line-height: 1.1;
        font-size: 2.15rem;
        background: linear-gradient(90deg, #f8fafc 0%, #cffafe 45%, #d1fae5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p { margin: 0; color: rgba(248, 250, 252, 0.92); }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .hero-meta span {
        border-radius: 999px;
        padding: 0.18rem 0.56rem;
        border: 1px solid rgba(255, 255, 255, 0.28);
        background: rgba(8, 47, 73, 0.24);
        color: rgba(241, 245, 249, 0.95);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .hero-chip {
        display: inline-block;
        margin-top: 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        padding: 0.28rem 0.6rem;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.32);
    }
    div[data-testid="metric-container"] {
        border: 1px solid var(--ui-border);
        border-radius: 14px;
        background: linear-gradient(145deg, var(--ui-surface-strong) 0%, var(--ui-surface) 100%);
        backdrop-filter: blur(10px);
        padding: 12px;
        box-shadow: 0 8px 24px var(--ui-shadow);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px var(--ui-shadow);
    }
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--ui-muted-text) !important;
        font-size: 0.72rem;
        font-weight: 700;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ui-text) !important;
        font-size: 1.56rem;
        font-weight: 700;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: var(--ui-muted-text) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid var(--ui-border);
        padding-bottom: 4px;
        overflow-x: auto;
        flex-wrap: nowrap;
        scrollbar-width: thin;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 700;
        background: var(--ui-surface-soft);
        color: var(--ui-text);
        white-space: nowrap;
        border: 1px solid var(--ui-border);
        transition: all 0.22s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: color-mix(in srgb, var(--ui-accent) 40%, var(--ui-border));
        transform: translateY(-1px);
    }
    .stTabs [aria-selected="true"] {
        color: var(--ui-accent) !important;
        border-bottom: 2px solid var(--ui-accent) !important;
        background: color-mix(in srgb, var(--ui-chip-bg) 84%, transparent) !important;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--ui-border);
    }
    .stExpander {
        border-radius: 12px !important;
        border: 1px solid var(--ui-border) !important;
        background: var(--ui-surface-soft) !important;
    }
    [data-testid="stWidgetLabel"] p {
        color: var(--ui-text) !important;
        font-weight: 700;
    }
    .stTextArea textarea, .stTextInput input {
        background: var(--ui-surface) !important;
        border-radius: 12px !important;
        border: 1px solid var(--ui-border) !important;
        color: var(--ui-text) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--ui-accent) !important;
        box-shadow: 0 0 0 1px color-mix(in srgb, var(--ui-accent) 50%, transparent) !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: var(--ui-surface) !important;
        border: 1px solid var(--ui-border) !important;
        color: var(--ui-text) !important;
    }
    [data-baseweb="radio"] label {
        color: var(--ui-text) !important;
        opacity: 1 !important;
    }
    [data-baseweb="slider"] [role="slider"] {
        border: 1px solid var(--ui-border);
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: var(--ui-muted-text) !important;
    }
    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--ui-border);
        background: linear-gradient(115deg, var(--ui-accent) 0%, var(--ui-accent-soft) 100%);
        color: #ffffff;
        font-weight: 700;
        transition: all 0.25s ease;
        box-shadow: 0 8px 22px var(--ui-shadow);
        min-height: 2.7rem;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 28px var(--ui-shadow);
    }
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    .stTextArea textarea:focus-visible,
    .stTextInput input:focus-visible {
        outline: 2px solid color-mix(in srgb, var(--ui-accent) 75%, #ffffff);
        outline-offset: 2px;
    }
    [data-testid="stSidebar"] {
        background: var(--secondary-background-color) !important;
        border-right: 1px solid var(--ui-border);
    }
    [data-testid="stSidebar"] * {
        color: var(--ui-text) !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stCaption p {
        color: var(--ui-muted-text) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown ul {
        margin-top: 0.2rem;
    }
    header[data-testid="stHeader"] {
        background: color-mix(in srgb, var(--background-color) 88%, transparent) !important;
        border-bottom: none !important;
    }
    .control-card {
        border: 1px solid var(--ui-border);
        border-radius: 14px;
        padding: 12px 14px;
        background: var(--ui-surface-soft);
        margin-bottom: 14px;
        box-shadow: 0 10px 26px var(--ui-shadow);
        animation: fadeUp 320ms ease both;
        position: relative;
        overflow: hidden;
    }
    .control-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, color-mix(in srgb, var(--ui-accent) 62%, transparent) 50%, transparent 100%);
    }
    .control-card > * {
        position: relative;
        z-index: 1;
    }
    .action-strip {
        border: 1px solid var(--ui-border);
        border-radius: 14px;
        padding: 10px;
        margin: 0 0 14px 0;
        background: color-mix(in srgb, var(--ui-surface-soft) 92%, transparent);
        box-shadow: 0 10px 24px var(--ui-shadow);
    }
    [data-testid="column"] {
        min-width: 0;
    }
    .input-hint {
        font-size: 0.86rem;
        color: var(--ui-muted-text);
        margin: 0 0 8px 0;
    }
    .char-counter {
        color: var(--ui-muted-text);
        font-size: 0.79rem;
        margin-top: 6px;
    }
    .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid var(--ui-border);
        font-weight: 700;
        min-height: 2.7rem;
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px !important;
        border: 1px dashed var(--ui-border) !important;
        background: var(--ui-surface-soft) !important;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--ui-accent) !important;
        transform: translateY(-1px);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--ui-border);
        border-radius: 12px;
    }
    *::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    *::-webkit-scrollbar-thumb {
        background: color-mix(in srgb, var(--ui-accent) 36%, var(--ui-border));
        border-radius: 999px;
    }
    *::-webkit-scrollbar-track {
        background: color-mix(in srgb, var(--ui-surface-soft) 84%, transparent);
    }
    [data-testid="stSidebar"] {
        min-width: min(20rem, 84vw) !important;
    }
    @media (max-width: 1180px) {
        .status-pills {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .readiness-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .hero h2 {
            font-size: 1.95rem !important;
        }
        .feature-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 900px) {
        .block-container {
            padding-top: 0.9rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
            padding-bottom: 1.5rem;
        }
        .status-pills {
            grid-template-columns: 1fr;
            gap: 8px;
        }
        .readiness-grid {
            grid-template-columns: 1fr;
            gap: 8px;
        }
        .feature-grid {
            grid-template-columns: 1fr;
            gap: 8px;
        }
        .narrative-card {
            min-height: auto;
        }
        .hero {
            padding: 18px;
            border-radius: 14px;
        }
        .hero h2 {
            font-size: 1.45rem !important;
        }
        .hero-meta span {
            font-size: 0.69rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px;
            font-size: 0.9rem;
        }
        .control-card {
            padding: 10px 12px;
        }
        [data-baseweb="radio"] > div {
            flex-direction: column !important;
            align-items: flex-start !important;
        }
        [data-testid="stSidebar"] {
            min-width: min(18rem, 92vw) !important;
        }
        [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .stButton > button {
            width: 100% !important;
        }
        .action-strip {
            position: sticky;
            bottom: 0.45rem;
            z-index: 6;
            backdrop-filter: blur(8px);
        }
        [data-testid="stDataFrame"] {
            overflow-x: auto;
        }
    }
    @media (max-width: 640px) {
        .hero h2 {
            font-size: 1.3rem !important;
            line-height: 1.2;
        }
        .hero p {
            font-size: 0.92rem;
        }
        .hero-meta {
            gap: 6px;
        }
        .hero-meta span {
            font-size: 0.66rem;
            padding: 0.15rem 0.48rem;
        }
        .section-head {
            font-size: 0.98rem;
            align-items: flex-start;
            flex-direction: column;
            gap: 2px;
        }
        .section-head small {
            text-align: left;
        }
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background:
                radial-gradient(circle at 80% 10%, rgba(45, 212, 191, 0.14), transparent 35%),
                radial-gradient(circle at 20% 90%, rgba(251, 146, 60, 0.12), transparent 30%),
                var(--background-color);
        }
        .hero {
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.45);
        }
    }
    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
        }
    }
</style>
"""
style_css = style_css.replace("__THEME_SCOPE__", scope_css)
st.markdown(style_css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# App Layout
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("InsightAI Control")
    st.caption("Theme-aware workspace with resilient pipeline fallbacks.")
    st.divider()

    st.metric("Runtime Memory", get_system_telemetry())
    st.caption(
        f"Mode: {st.session_state.selected_mode_label} | "
        f"Theme: {theme_mode_label(resolve_theme_mode(st.session_state.ui_theme_mode))} | "
        f"RAG Top-K: {st.session_state.rag_top_k} | "
        f"Keywords: {st.session_state.keyword_top_n}"
    )

    st.divider()
    if st.button("Reset Analysis", width="stretch"):
        reset_analysis_state()

    st.markdown("**Pipeline Status**")
    st.markdown("- Embeddings: ready")
    st.markdown("- Summarizer: ready")
    st.markdown("- NER engine: ready")

    if st.session_state.analysis_history:
        st.divider()
        st.markdown("**Recent Analyses**")
        history_rows = [
            f"{item['time']} | {item['words']}w | {item['sentiment']} | score {item['score']}"
            for item in st.session_state.analysis_history[:5]
        ]
        for row in history_rows:
            st.caption(row)

if st.session_state.last_applied_preset != st.session_state.analysis_preset:
    apply_preset_settings(st.session_state.analysis_preset)

st.markdown(
    """
<div class="hero">
    <h2>InsightAI Premium Analysis Studio</h2>
    <p>Enterprise-grade summarization, retrieval Q&A, sentiment, named entities, and executive reporting with resilient fallbacks.</p>
    <div class="hero-meta">
        <span>Theme-Adaptive UI</span>
        <span>Mobile-Optimized Workflow</span>
        <span>Fallback-Safe Pipeline</span>
    </div>
    <span class="hero-chip">Production-Safe NLP Pipeline</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="status-pills">
    <div class="status-pill"><span class="k">Theme</span><span class="v">{html.escape(theme_mode_label(current_theme_mode))}</span></div>
    <div class="status-pill"><span class="k">Preset</span><span class="v">{html.escape(st.session_state.analysis_preset)}</span></div>
    <div class="status-pill"><span class="k">Inference</span><span class="v">{html.escape(st.session_state.selected_mode_label)}</span></div>
    <div class="status-pill"><span class="k">Runtime Memory</span><span class="v">{html.escape(get_system_telemetry())}</span></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='section-head'>Control Deck<small>Configure mode, depth, and theme before running analysis</small></div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='control-card'>", unsafe_allow_html=True)
control_row1_col1, control_row1_col2, control_row1_col3, control_row1_col4 = st.columns([2.0, 1.0, 1.1, 1.1])
with control_row1_col1:
    mode_label = st.radio(
        "Inference Profile",
        ["Speed (Distilled)", "Accuracy (Large)"],
        key="selected_mode_label",
        horizontal=True,
        help="Speed is low latency. Accuracy is deeper but heavier.",
    )
with control_row1_col2:
    summary_detail = st.slider(
        "Summary Depth",
        min_value=1,
        max_value=3,
        key="summary_detail",
        help="Higher depth keeps more detail in generated summaries.",
    )
with control_row1_col3:
    st.selectbox(
        "Analysis Preset",
        ["Balanced", "Executive Brief", "Risk Audit", "Technical Deep Dive"],
        key="analysis_preset",
        help="Preset adjusts model profile and analysis depth for specific use-cases.",
    )
with control_row1_col4:
    st.selectbox(
        "Theme Mode",
        ["Auto (System)", "Light", "Dark"],
        key="ui_theme_mode",
        help="Auto follows your app/system theme. You can force light or dark mode here.",
    )

control_row2_col1, control_row2_col2 = st.columns(2)
with control_row2_col1:
    rag_top_k = st.slider("RAG Results", min_value=1, max_value=5, key="rag_top_k")
with control_row2_col2:
    keyword_top_n = st.slider("Keyword Count", min_value=5, max_value=15, key="keyword_top_n")
st.markdown(
    (
        "<div class='preset-note'>"
        f"<b>Active Preset:</b><span>{html.escape(st.session_state.analysis_preset)} - "
        f"{html.escape(preset_description(st.session_state.analysis_preset))}</span>"
        "</div>"
    ),
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

selected_mode = "fast" if mode_label.startswith("Speed") else "accurate"
summary_token_limit = 650 if summary_detail == 1 else 850 if summary_detail == 2 else 1050

st.markdown(
    "<div class='section-head'>Input Workspace<small>Add text, URL, or file content</small></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='input-hint'>Tip: for best output quality, provide 2+ complete sentences and avoid very noisy OCR text.</p>",
    unsafe_allow_html=True,
)
input_tab_text, input_tab_url, input_tab_file = st.tabs(["Text", "URL", "File Upload"])

with input_tab_text:
    st.caption("Paste raw content directly for fastest analysis.")
    st.text_area(
        "Document Text",
        key="input_text_buffer",
        height=220,
        placeholder="Paste long-form text, report sections, legal content, or product docs.",
        label_visibility="collapsed",
    )
    input_chars = len(st.session_state.input_text_buffer.strip())
    st.markdown(
        f"<div class='char-counter'>Characters: {input_chars:,} / {MAX_INPUT_CHARS:,}</div>",
        unsafe_allow_html=True,
    )

with input_tab_url:
    st.caption("Provide an article/report URL to fetch readable body content.")
    st.text_input(
        "Source URL",
        key="url_input_buffer",
        placeholder="https://example.com/article",
        label_visibility="collapsed",
    )

with input_tab_file:
    st.caption("Supported types: .txt, .md, .csv, .log, .pdf")
    uploaded_file = st.file_uploader(
        "Upload text or PDF",
        type=["txt", "md", "csv", "log", "pdf"],
        key=f"uploaded_file_{st.session_state.upload_widget_nonce}",
    )

input_has_url = bool(st.session_state.url_input_buffer.strip())
input_has_file = uploaded_file is not None
readiness_state, load_tier, eta_window = estimate_run_profile(
    text_chars=input_chars,
    has_url=input_has_url,
    has_file=input_has_file,
    mode_label=st.session_state.selected_mode_label,
)
input_sources: List[str] = []
if input_chars > 0:
    input_sources.append("Text")
if input_has_url:
    input_sources.append("URL")
if input_has_file:
    input_sources.append(f"File ({uploaded_file.name})")
source_summary = ", ".join(input_sources) if input_sources else "None"
mode_summary = "Fast Inference" if selected_mode == "fast" else "High-Accuracy Inference"

st.markdown(
    "<div class='section-head'>Run Readiness<small>Validate source and complexity before execution</small></div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"""
<div class="readiness-grid">
    <div class="readiness-card"><span class="k">Input Status</span><span class="v">{html.escape(readiness_state)}</span></div>
    <div class="readiness-card"><span class="k">Active Sources</span><span class="v">{html.escape(source_summary)}</span></div>
    <div class="readiness-card"><span class="k">Workload Tier</span><span class="v">{html.escape(load_tier)}</span></div>
    <div class="readiness-card"><span class="k">Est. Pipeline</span><span class="v">{html.escape(eta_window)} ({html.escape(mode_summary)})</span></div>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.analyzed:
    st.markdown(
        "<div class='section-head'>Workspace Preview<small>Output artifacts generated after each run</small></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="feature-grid">
    <div class="feature-card"><b>Executive Summaries</b><span>Abstractive and extractive summaries with strategic highlights.</span></div>
    <div class="feature-card"><b>Context Q&A</b><span>Retrieval-grounded responses with top evidence snippets and relevance scores.</span></div>
    <div class="feature-card"><b>Insight Analytics</b><span>Sentiment gauge, salience curve, keyword relevance, and entity distribution.</span></div>
    <div class="feature-card"><b>Action Signals</b><span>Recommended action items and risk/opportunity markers for faster decisions.</span></div>
    <div class="feature-card"><b>Export Layer</b><span>Download polished HTML and JSON outputs for stakeholders and integrations.</span></div>
    <div class="feature-card"><b>Resilient Pipeline</b><span>Fallback-safe components keep analysis running when heavy models fail.</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='section-head'>Action Center<small>Run, demo, or clear your current workspace input</small></div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='action-strip'>", unsafe_allow_html=True)
action_col1, action_col2, action_col3 = st.columns([1.2, 1, 1])
with action_col1:
    run_clicked = st.button("Run Analysis", type="primary", width="stretch")
with action_col2:
    st.button(
        "Load Demo Text",
        width="stretch",
        on_click=load_demo_input,
    )
with action_col3:
    st.button(
        "Clear Inputs",
        width="stretch",
        on_click=clear_all_inputs,
    )
st.markdown("</div>", unsafe_allow_html=True)

if run_clicked:
    reset_analysis_state()

    source_text = st.session_state.input_text_buffer.strip()

    if not source_text and uploaded_file is not None:
        source_text = extract_text_from_upload(uploaded_file) or ""

    if not source_text and st.session_state.url_input_buffer.strip():
        with st.spinner("Fetching and parsing web content..."):
            fetched = scrape_url(st.session_state.url_input_buffer.strip())
        if fetched:
            source_text = fetched
            st.toast("Web content loaded.")
        else:
            st.error("Unable to extract readable content from this URL.")

    if not source_text:
        st.error("Provide text input, upload a file, or enter a URL.")
    else:
        truncated = False
        if len(source_text) > MAX_INPUT_CHARS:
            source_text = source_text[:MAX_INPUT_CHARS]
            truncated = True

        pipeline_start = datetime.now()
        with st.status("Running analysis pipeline...", expanded=True) as status:
            try:
                st.write("Step 1/5: Building embeddings")
                st.write("Step 2/5: Generating summaries")
                st.write("Step 3/5: Extracting keywords and entities")
                st.write("Step 4/5: Running sentiment and readability")
                st.write("Step 5/5: Finalizing dashboard artifacts")

                result, embedder_instance = run_analysis(
                    text=source_text,
                    mode=selected_mode,
                    keyword_top_n=keyword_top_n,
                    summary_chunk_token_limit=summary_token_limit,
                )
                elapsed_seconds = max(
                    0.0, (datetime.now() - pipeline_start).total_seconds()
                )
                result["processing_seconds"] = elapsed_seconds

                st.session_state.analysis_result = result
                st.session_state.embedder_instance = embedder_instance
                st.session_state.analyzed = True
                push_history(result)

                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as exc:
                st.session_state.last_error = str(exc)
                st.session_state.analyzed = False
                status.update(label="Analysis failed", state="error", expanded=True)
                st.error(f"Analysis failed: {exc}")

        if truncated and st.session_state.analyzed:
            st.info(
                f"Input exceeded {MAX_INPUT_CHARS} characters and was truncated for performance stability."
            )

if st.session_state.last_error:
    st.info(f"Last error: {st.session_state.last_error}")

if st.session_state.analyzed and st.session_state.analysis_result:
    result = st.session_state.analysis_result

    if result.get("warnings"):
        for warning_msg in result["warnings"]:
            st.info(warning_msg)

    top_keywords = ", ".join([k for k, _ in result.get("keywords", [])[:3]]) or "No dominant keyphrases"
    reading_time = max(1, int(round(result["word_count"] / 220)))
    warning_count = len(result.get("warnings", []))
    st.markdown(
        (
            "<div class='result-banner'>"
            f"<b>Analysis Ready</b><span>Score {result['quality_score']}/100 | "
            f"Sentiment {html.escape(result['sentiment']['label'])} | "
            f"Top themes: {html.escape(top_keywords)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="status-pills">
    <div class="status-pill"><span class="k">Read Time</span><span class="v">{reading_time} min</span></div>
    <div class="status-pill"><span class="k">Warnings</span><span class="v">{warning_count}</span></div>
    <div class="status-pill"><span class="k">Entities</span><span class="v">{len(result.get('entities', []))}</span></div>
    <div class="status-pill"><span class="k">Actions</span><span class="v">{len(result.get('action_items', []))}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-head'>Result Dashboard<small>Explore summaries, retrieval Q&A, analytics, and export-ready reports</small></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Insight Score", f"{result['quality_score']}/100")
    m2.metric("Readability", f"{result['readability']:.1f}")
    m3.metric("Sentiment", result["sentiment"]["label"], f"{result['sentiment']['score']:.2f}")
    m4.metric("Latency", f"{result.get('processing_seconds', 0.0):.1f}s")
    m5, m6, m7 = st.columns(3)
    m5.metric("Words", f"{result['word_count']}")
    m6.metric("Sentences", f"{result['sentence_count']}")
    m7.metric("Lexical Diversity", f"{result['lexical_diversity']:.2f}")

    tab_summary, tab_rag, tab_analytics, tab_entities, tab_report = st.tabs(
        ["Summaries", "Chat (RAG)", "Analytics", "Entities", "Report"]
    )

    with tab_summary:
        c1, c2 = st.columns(2)
        with c1:
            render_narrative_card("Abstractive Summary", result["abstractive_summary"], tone="accent")
        with c2:
            render_narrative_card("Extractive Highlights", result["extractive_summary"], tone="success")

        if result["salience"]:
            st.markdown("**Strategic Sentence Highlights**")
            for rank, (_, sentence_text, salience_score) in enumerate(result["salience"][:5], start=1):
                st.markdown(f"{rank}. ({salience_score:.2f}) {sentence_text}")

        st.markdown("**Executive Readout**")
        for insight_line in build_business_readout(result):
            st.markdown(f"- {insight_line}")

        st.markdown("**Recommended Action Items**")
        if result.get("action_items"):
            for item in result["action_items"][:8]:
                st.markdown(f"- {item}")
        else:
            st.caption("No explicit action-item language was detected.")

    with tab_rag:
        st.subheader("Context-Aware Q&A")
        question = st.text_input("Ask a question about this document", key="rag_question")
        if question:
            if st.session_state.rag_last_question != question:
                st.session_state.rag_answer_draft = ""
                st.session_state.rag_last_question = question

            hits = perform_rag_retrieval(
                query=question,
                sentences=result["sentences"],
                sentence_embeddings=result["sentence_embeddings"],
                embedder=st.session_state.embedder_instance,
                top_k=rag_top_k,
            )
            if hits:
                if st.button("Generate Answer Draft", key="rag_generate_answer"):
                    st.session_state.rag_answer_draft = synthesize_rag_answer(
                        query=question,
                        hits=hits,
                        mode=selected_mode,
                    )

                if st.session_state.rag_answer_draft:
                    st.markdown("**Answer Draft**")
                    st.success(st.session_state.rag_answer_draft)

                for i, (chunk, score) in enumerate(hits, start=1):
                    with st.expander(f"Match {i} | relevance {score:.2f}", expanded=(i == 1)):
                        st.write(chunk)
            else:
                st.info("No relevant context was found for that question.")

    with tab_analytics:
        c1, c2 = st.columns(2)

        with c1:
            sentiment_value = result["sentiment"]["score"]
            if result["sentiment"]["label"].upper() == "NEGATIVE":
                sentiment_value *= -1

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=sentiment_value,
                    title={"text": "Sentiment Index"},
                    gauge={
                        "axis": {"range": [-1, 1]},
                        "bar": {"color": chart_tokens["accent"]},
                        "steps": [
                            {"range": [-1, -0.2], "color": plot_neg},
                            {"range": [-0.2, 0.2], "color": plot_neu},
                            {"range": [0.2, 1], "color": plot_pos},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                height=320,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                template=plotly_template,
            )
            st.plotly_chart(fig_gauge, width="stretch")

        with c2:
            if result["keywords"]:
                kw_df = {
                    "phrase": [k for k, _ in result["keywords"]],
                    "score": [float(v) for _, v in result["keywords"]],
                }
                fig_kw = px.bar(
                    kw_df,
                    x="score",
                    y="phrase",
                    orientation="h",
                    title="Keyword Relevance",
                    color="score",
                    color_continuous_scale="Blues",
                )
                fig_kw.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    paper_bgcolor="rgba(0,0,0,0)",
                    template=plotly_template,
                    margin=dict(t=50, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_kw, width="stretch")
            else:
                st.info("No keywords extracted.")

        length_df = {"sentence_length": result["sentence_lengths"]}
        fig_hist = px.histogram(
            length_df,
            x="sentence_length",
            nbins=20,
            title="Sentence Length Distribution",
            color_discrete_sequence=[chart_tokens["accent_soft"]],
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            template=plotly_template,
            margin=dict(t=45, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_hist, width="stretch")

        if result["salience"]:
            salience_df = {
                "sentence_index": [idx + 1 for idx, _, _ in result["salience"]],
                "salience": [score for _, _, score in result["salience"]],
            }
            fig_salience = px.line(
                salience_df,
                x="sentence_index",
                y="salience",
                markers=True,
                title="Narrative Salience Curve",
                color_discrete_sequence=[chart_tokens["accent"]],
            )
            fig_salience.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                template=plotly_template,
                margin=dict(t=45, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_salience, width="stretch")

    with tab_entities:
        st.subheader("Named Entity Breakdown")
        if result["entity_counts"]:
            entity_df = {
                "label": list(result["entity_counts"].keys()),
                "count": list(result["entity_counts"].values()),
            }
            c1, c2 = st.columns(2)
            with c1:
                fig_entities = px.pie(
                    entity_df,
                    values="count",
                    names="label",
                    title="Entity Type Share",
                    hole=0.35,
                )
                fig_entities.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    template=plotly_template,
                    margin=dict(t=45, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_entities, width="stretch")
            with c2:
                st.dataframe(entity_df, width="stretch")

            st.markdown("**Detected Entities (sample)**")
            st.dataframe(
                {"entity": [e for e, _ in result["entities"]], "label": [l for _, l in result["entities"]]},
                width="stretch",
            )
        else:
            st.info("No named entities detected in this document.")

    with tab_report:
        st.subheader("Executive HTML Report")
        report_html = build_report_html(result)
        report_name = f"insight_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        st.download_button(
            label="Download HTML Report",
            data=report_html,
            file_name=report_name,
            mime="text/html",
            width="stretch",
        )
        report_json = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "quality_score": result["quality_score"],
            "readability": result["readability"],
            "sentiment": result["sentiment"],
            "word_count": result["word_count"],
            "sentence_count": result["sentence_count"],
            "lexical_diversity": result["lexical_diversity"],
            "avg_sentence_length": result["avg_sentence_length"],
            "processing_seconds": result.get("processing_seconds", 0.0),
            "keywords": result["keywords"],
            "entity_counts": result["entity_counts"],
            "action_items": result.get("action_items", []),
            "executive_readout": build_business_readout(result),
            "warnings": result.get("warnings", []),
        }
        json_name = f"insight_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report_json, indent=2),
            file_name=json_name,
            mime="application/json",
            width="stretch",
        )
        st.caption("The report includes summaries, key metrics, keywords, and entity distribution.")

