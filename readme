# 📄 AI Resume ATS Checker

A Streamlit app that analyzes a resume with Google Gemini Flash and returns:

- An **estimated ATS score** (0–100) with a 5-category breakdown
- Strengths and issues found in the resume
- **Missing keywords** (matched against an optional job description)
- Prioritized, actionable **improvements** with example rewrites
- A downloadable JSON report

> ⚠️ The score is an AI-based estimate, not the output of a real ATS. Use it as guidance.
> Resume text is sent to Google's Gemini API, so don't upload documents you can't share.

## Features
- Supports PDF, DOCX and TXT (max 5 MB)
- Optional job description for tailored keyword matching
- API key from Streamlit secrets, an environment variable, or the sidebar
- Model name configurable (default `gemini-2.5-flash`)

## Project structure
```
.
├── app.py
├── requirements.txt
└── README.md
```

## Run locally
1. Get a free Gemini API key: https://aistudio.google.com/app/apikey
2. Install and run:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Provide the key either by pasting it in the sidebar, or by creating `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   # GEMINI_MODEL = "gemini-2.5-flash"   # optional
   ```
   **Never commit `secrets.toml` to GitHub.**

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → select your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## How it works
1. Text is extracted from the file (`pypdf` / `python-docx`).
2. A structured prompt is sent to Gemini, requesting JSON output.
3. The JSON is parsed, validated and clamped, then rendered in Streamlit.

## Limitations
- Scanned (image-only) PDFs can't be read; export a text-based PDF instead.
- Scores can vary slightly between runs.
