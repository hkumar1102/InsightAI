# InsightAI

InsightAI is a Streamlit application for long-form document analysis.
It accepts raw text, URLs, and uploaded files, then produces summaries, retrieval-based Q&A, sentiment, entities, analytics, and exportable reports.

This project is designed to balance three things:
- strong UX
- practical NLP depth
- resilient behavior when heavy models or external endpoints fail

## Why This Project Exists

Many NLP demos look good in ideal conditions but break in real use.
This project was built to solve real workflow needs:
- analyze business or technical text quickly
- keep the app usable on low-resource systems
- fail gracefully when model/network issues happen
- give polished outputs that are easy to present to stakeholders

## Core Features

- Multi-input workspace:
  - direct text
  - URL extraction
  - file upload (`.txt`, `.md`, `.csv`, `.log`, `.pdf`)
- Dual model profile:
  - `Speed (Distilled)` for fast runs
  - `Accuracy (Large)` for deeper generation/embedding quality
- Document intelligence pipeline:
  - abstractive summary
  - extractive summary
  - keyword extraction
  - named entity extraction
  - sentiment scoring
  - readability scoring
  - salience ranking
  - action-item extraction
- Retrieval Q&A (RAG-style):
  - semantic sentence retrieval
  - answer draft generation
- Advanced UI/UX:
  - responsive desktop/mobile design
  - theme-aware (`Auto`, `Light`, `Dark`)
  - keyboard focus states and reduced-motion support
- Report export:
  - HTML report download
  - JSON report download
- Reliability safeguards:
  - model fallback paths
  - local lightweight embedding fallback
  - safe state reset behavior

## Tech Stack

- Frontend: `Streamlit`
- NLP/ML:
  - `transformers`
  - `sentence-transformers`
  - `spaCy`
  - `KeyBERT`
  - `scikit-learn`
  - `torch`
- Data/Utilities:
  - `plotly`
  - `requests`
  - `newspaper3k`
  - `beautifulsoup4`
  - `pypdf`
  - `textstat`
  - `psutil`

## Project Structure

```text
.
|-- app.py                  # Main Streamlit app (UI + pipeline + orchestration)
|-- requirements.txt        # Python package dependencies
|-- packages.txt            # System-level dependencies (useful for cloud buildpacks)
|-- download_nltk.py        # Optional NLTK bootstrap script
|-- README.md               # Project documentation
|-- .gitignore              # Ignore rules
```

## Setup

### 1) Clone Repository

```bash
git clone https://github.com/your-username/insightai.git
cd insightai
```

### 2) Create Virtual Environment

```bash
python -m venv venv
```

### 3) Activate Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 4) Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5) Install spaCy English Model

```bash
python -m spacy download en_core_web_sm
```

### 6) (Optional) Download NLTK Tokenizer Assets

```bash
python download_nltk.py
```

### 7) Run the App

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## How to Use

1. Add input in one of three tabs:
   - `Text`
   - `URL`
   - `File Upload`
2. Set controls in `Control Deck`:
   - inference profile
   - analysis preset
   - summary depth
   - RAG top-k
   - keyword count
   - theme mode
3. Click `Run Analysis`.
4. Explore result tabs:
   - `Summaries`
   - `Chat (RAG)`
   - `Analytics`
   - `Entities`
   - `Report`
5. Export HTML/JSON report when needed.

## Pipeline Walkthrough

At a high level, each run follows this order:

1. Input collection and validation
2. Sentence normalization and embedding generation
3. Extractive summary generation via clustering
4. Abstractive summary generation via seq2seq model
5. Keyword/entity/sentiment/readability extraction
6. Salience scoring and action-item mining
7. KPI/analytics rendering and report generation

## Presets

- `Balanced`: default mixed profile
- `Executive Brief`: concise output and lighter retrieval
- `Risk Audit`: deeper recall for risk/compliance signals
- `Technical Deep Dive`: denser analysis for technical content

## Reliability and Fallback Strategy

The app is designed to remain usable when heavy components fail.

- If `Accuracy` embedding model fails:
  - fallback to `Speed` embedding model
  - if that also fails, fallback to a deterministic local lightweight embedder
- If summarization model fails:
  - fallback to extractive summary or fast summarizer path
- If sentiment/keyword submodule fails:
  - app still completes with warning notes in UI
- Session-state safety:
  - input action buttons use callbacks to avoid mutation errors
  - uploader reset nonce ensures `Clear Inputs` truly clears all input surfaces

## Performance Notes

- First run is usually slower because models download and cache.
- Later runs are faster due to `st.cache_resource`.
- `Speed (Distilled)` is recommended for low-memory systems.
- Very large text is truncated to a safe maximum (`MAX_INPUT_CHARS`) to prevent instability.

## Troubleshooting

- `streamlit` command not found:
  - use `python -m streamlit run app.py`
- URL extraction fails:
  - many sites block bots; paste text directly
- Empty/poor PDF extraction:
  - scanned PDFs may not contain extractable text
- Slow first run:
  - expected due to model warmup/download
- Network-limited machine:
  - app can still run with fallback behaviors, but large model quality may be reduced

## Development Notes

- Main logic lives in `app.py`.
- UI, pipeline, and helper utilities are intentionally colocated for a single-file demo architecture.
- For production scaling, split into modules:
  - `ui/`
  - `nlp/`
  - `retrieval/`
  - `reporting/`
  - `tests/`

## License

Educational and demonstration use.
