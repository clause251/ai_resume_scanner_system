import re
import io
import streamlit as st
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text_from_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        return ""

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

STOPWORDS = set("""
a an the and or of to in for with on at by is are was were be been being
this that these those it its as from into your you we our their they he she
his her will can may should must have has had do does did not no yes but if
""".split())


def get_keywords(text: str) -> set:
    words = clean_text(text).split()
    return {w for w in words if w not in STOPWORDS and len(w) > 2}

def compute_similarity(job_desc: str, resume_text: str) -> float:
    documents = [clean_text(job_desc), clean_text(resume_text)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)


def get_matched_and_missing_keywords(job_desc: str, resume_text: str):
    job_keywords = get_keywords(job_desc)
    resume_keywords = get_keywords(resume_text)

    matched = sorted(job_keywords & resume_keywords)
    missing = sorted(job_keywords - resume_keywords)
    return matched, missing

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

st.title("AI-Based Resume Screening System")
st.write(
    "Paste a job description, upload one or more resumes, and get a match "
    "score with simple, explainable insights."
)

st.header("Job Description")
job_description = st.text_area(
    "Paste the job description here",
    height=200,
    placeholder="e.g. We are looking for a Python developer with experience in "
                "machine learning, NLP, and REST APIs...",
)

st.header("Upload Resume(s)")
uploaded_files = st.file_uploader(
    "Upload one or more resumes (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

st.header("Results")

if st.button("Screen Resumes", type="primary"):
    if not job_description.strip():
        st.warning("Please paste a job description first.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        results = []

        with st.spinner("Analyzing resumes..."):
            for uploaded_file in uploaded_files:
                resume_text = extract_text(uploaded_file)

                if not resume_text.strip():
                    st.error(f"Could not extract text from **{uploaded_file.name}**. Skipping.")
                    continue

                score = compute_similarity(job_description, resume_text)
                matched, missing = get_matched_and_missing_keywords(job_description, resume_text)

                results.append({
                    "name": uploaded_file.name,
                    "score": score,
                    "matched": matched,
                    "missing": missing,
                })

        if results:
            results.sort(key=lambda r: r["score"], reverse=True)

            st.subheader("Ranking")
            table_data = [
                {"Rank": i + 1, "Resume": r["name"], "Match Score (%)": r["score"]}
                for i, r in enumerate(results)
            ]
            st.table(table_data)

            st.subheader("Detailed Insights")
            for i, r in enumerate(results, start=1):
                with st.expander(f"#{i} — {r['name']} ({r['score']}% match)"):
                    st.progress(min(int(r["score"]), 100))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Matched Keywords**")
                        if r["matched"]:
                            st.write(", ".join(r["matched"][:40]))
                        else:
                            st.write("No overlapping keywords found.")

                    with col2:
                        st.markdown("**Missing Keywords**")
                        if r["missing"]:
                            st.write(", ".join(r["missing"][:40]))
                        else:
                            st.write("Great! No major keywords missing.")