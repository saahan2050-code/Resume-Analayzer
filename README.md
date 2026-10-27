# 📄 ATS Resume Analyzer

Upload a resume (PDF, DOCX or TXT) and get:

- an **ATS score out of 100**, with a weighted per-category breakdown
- your **strengths** and **missing keywords**
- **prioritised improvements**, each with a concrete example rewrite
- optional **job-description matching** for better keyword feedback
- a downloadable Markdown report

Built with [Streamlit](https://streamlit.io) and Google's Gemini Flash model.

## How it works

1. The resume text is extracted locally (`pypdf` for PDF, `python-docx` for DOCX).
2. The text (plus the optional job description) is sent to Gemini with a strict JSON schema.
3. Gemini scores six categories. The **overall score is computed in code** from fixed weights,
   so it always matches the breakdown shown:

| Category | Weight |
|---|---|
| Keywords & Relevance | 25% |
| Experience & Impact | 25% |
| ATS Formatting & Parseability | 20% |
| Skills | 15% |
| Structure & Completeness | 10% |
| Language & Clarity | 5% |

> **Note:** This is an AI-based estimate, not the output of any real ATS (Workday, Greenhouse, etc. each behave
> differently). Use it as guidance for improving your resume, not as a guarantee.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at <https://aistudio.google.com/apikey>, then either:

- paste it into the app sidebar, **or**
- copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `GEMINI_API_KEY`, **or**
- set an environment variable: `export GEMINI_API_KEY="..."`

```bash
streamlit run app.py
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets or env var or sidebar | none (required) |
| `GEMINI_MODEL` | Streamlit secrets or env var or sidebar | `gemini-flash-latest` |

`gemini-flash-latest` always points to Google's newest Flash model. If you prefer to pin a version,
set `GEMINI_MODEL` to e.g. `gemini-2.5-flash`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with Streamlit access granted).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add: `GEMINI_API_KEY = "your-key"`
5. Click **Deploy**.

## Limitations

- Scanned / image-only PDFs have no extractable text and are rejected (an ATS can't read them either).
- Files over 5 MB and resumes over 20,000 characters are rejected or truncated.
- The free Gemini tier has rate limits; you may see a "rate limit" message under heavy use.
- Your resume is sent to the Gemini API for analysis. The app itself does not store it.

## Project structure

```
app.py                              # the whole app
requirements.txt                    # dependencies
README.md
.gitignore
.streamlit/secrets.toml.example     # template for your API key
```
