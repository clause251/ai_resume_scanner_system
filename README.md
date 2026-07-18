# AI Resume Screener

A simple, beginner-friendly Streamlit web app that screens resumes against a job description using TF-IDF and cosine similarity.

## Features
- Paste a job description
- Upload one or more resumes (PDF, DOCX, or TXT)
- Get a match score (0-100%) for each resume
- See matched and missing keywords
- Rank multiple resumes from best to worst match

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Sample Data

The `sample_data/` folder includes two sample resumes and two sample job descriptions for testing:
- `sample_description_1` + `resume_1_ananya.txt` → high match
- `sample_description_2` + `resume_2_rahul.txt` → high match

## How it works

The app uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to vectorize the job description and resume text, then computes **cosine similarity** between them to produce a match score. This is a simple, fully explainable approach that doesn't require any external APIs or heavy transformer models — a great starting point before moving to embedding-based semantic matching.

## Tech Stack
- Streamlit (UI)
- scikit-learn (TF-IDF + cosine similarity)
- PyPDF2 (PDF text extraction)
- python-docx (DOCX text extraction)
