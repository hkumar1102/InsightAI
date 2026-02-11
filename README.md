# InsightAI – Enterprise Document Intelligence

InsightAI is a production-grade NLP web application built with Streamlit that performs deep document analysis using modern transformer models, semantic embeddings, and unsupervised learning.

It is designed for analyzing long-form text such as reports, articles, legal documents, and technical content.

## 🚀 Features

- Abstractive \& extractive text summarization
- Context-aware question answering (RAG-style)
- Semantic keyword extraction
- Named Entity Recognition (NER)
- Sentiment analysis with confidence scores
- Readability scoring (Flesch)
- Interactive analytics \& visualizations
- Auto-generated executive HTML reports
- Optimized model loading using caching (single load per session)

## 🧠 Tech Stack

**Frontend**

- Streamlit
- Plotly

**NLP & ML**

- HuggingFace Transformers
- Sentence Transformers
- SpaCy
- KeyBERT
- Scikit-Learn

**Utilities**

- Newspaper3k (web scraping)
- TextStat
- NumPy
- PSUtil

## 📁 Project Structure

.
|
├── app.py # Main Streamlit application
|
├── requirements.txt # Python dependencies
|
├── download\_nltk.py # Optional NLP setup script
|
├── .gitignore
|
└── README.md

## Deployed Link

```bash
https://insightai-hk1102.streamlit.app/
```

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/insightai.git
cd insightai
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

**Activate it:**

***Windows***

```bash
venv\\Scripts\\activate
```

***Linux / Mac***

```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download SpaCy model

```bash
python -m spacy download en\_core\_web\_sm
```

**(Optional)**

```bash
python download\_nltk.py
```

### 5️⃣ Run the application

```bash
streamlit run app.py
```

**Open browser:**

```bash
http://localhost:8501
```

## 🧪 How to Use

- Paste text or provide a URL
-Choose speed vs accuracy mode
-Click Initialize Pipeline

### Explore:

-Summaries
-Chat with document
-Analytics dashboard
-Downloadable report

### 📈 Performance Notes

- Models are loaded once using Streamlit caching
- Designed to avoid memory spikes
- Works efficiently on CPU (no GPU required)

## 📄 License

This project is for educational and demonstration purposes.

