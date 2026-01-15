# ============================
# Resume RAG Screener (DEBUGGABLE VERSION)
# ============================

import os
import re
import fitz
import tempfile
import traceback
import streamlit as st
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import logging
from groq import Groq
import json
from datetime import datetime
import pandas as pd
import time

# ============================
# DEBUG MODE
# ============================
DEBUG_MODE = True   # 🔥 SET FALSE FOR PRODUCTION

# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="Resume RAG Screener", layout="wide", page_icon="📊")

# ============================
# LOGGING SETUP
# ============================
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def log(step: str, request_id: str, data=None):
    msg = f"[{request_id}] {step}"
    if data is not None:
        msg += f" | {data}"
    logger.debug(msg)

def log_error(step: str, request_id: str, e: Exception):
    logger.error(f"[{request_id}] {step}: {str(e)}")
    logger.error(traceback.format_exc())

# ============================
# ENV SETUP
# ============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================
# EMBEDDING MODEL
# ============================
@st.cache_resource
def load_embedder():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded")
    return model

embedder = load_embedder()

# ============================
# GROQ CLIENT
# ============================
@st.cache_resource
def load_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)

groq_client = load_groq_client()

# ============================
# UTILITIES
# ============================
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# ============================
# PARSING
# ============================
def parse_resume(file_bytes, filename, request_id):
    log("PARSE_START", request_id, filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        doc = fitz.open(tmp.name)
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        os.unlink(tmp.name)

    email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phone = re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)

    parsed = {
        "filename": filename,
        "text": re.sub(r"\s+", " ", text.strip()),
        "email": email.group(0) if email else "Not found",
        "phone": phone.group(0) if phone else "Not found",
        "word_count": len(text.split())
    }

    log("PARSE_DONE", request_id, {
        "words": parsed["word_count"],
        "email": parsed["email"]
    })

    return parsed

# ============================
# EMBEDDING
# ============================
def embed_text(text, request_id, label):
    vec = embedder.encode([text], show_progress_bar=False)[0]
    log("EMBED", request_id, {
        "label": label,
        "dim": len(vec),
        "preview": vec[:5].tolist()
    })
    return vec

# ============================
# RANKING
# ============================
def rank_resumes(jd_text, resumes, request_id):
    jd_vec = embed_text(jd_text, request_id, "JD")

    resume_vectors = []
    for r in resumes:
        vec = embed_text(r["text"][:2000], request_id, r["filename"])
        resume_vectors.append(vec)

    similarities = cosine_similarity([jd_vec], resume_vectors)[0]

    for r, s in zip(resumes, similarities):
        log("SIMILARITY_SCORE", request_id, {
            "resume": r["filename"],
            "score": float(s)
        })

    ranked = sorted(
        zip(resumes, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    result = []
    for i, (r, score) in enumerate(ranked, 1):
        result.append({
            **r,
            "similarity": float(score),
            "rank": i
        })

    return result

# ============================
# LLM ANALYSIS
# ============================
def analyze_with_llm(jd_text, ranked_resumes, request_id):
    if not groq_client:
        return "Groq not configured"

    prompt = f"""
Job Description:
{jd_text[:1500]}

Top Candidates:
"""

    for r in ranked_resumes[:5]:
        prompt += f"\n{r['filename']} - {r['similarity']:.2%}\n{r['text'][:500]}\n"

    log("LLM_PROMPT_READY", request_id, {"chars": len(prompt)})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )

    log("LLM_RESPONSE_RECEIVED", request_id)
    return response.choices[0].message.content

# ============================
# UI
# ============================
st.title("📊 Resume RAG Screener (Debuggable)")

uploaded_files = st.file_uploader(
    "Upload resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

jd_text = st.text_area("Job Description", height=200)

if st.button("🚀 Screen Resumes"):
    if not uploaded_files or not jd_text.strip():
        st.warning("Upload resumes and JD")
        st.stop()

    request_id = f"REQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log("REQUEST_START", request_id)

    resumes = []
    start = time.time()

    # 🔥 DEBUG MODE = NO THREADING
    for f in uploaded_files:
        resumes.append(parse_resume(f.read(), f.name, request_id))

    ranked = rank_resumes(jd_text, resumes, request_id)
    llm_result = analyze_with_llm(jd_text, ranked, request_id)

    st.success("Done")
    st.markdown("## 🤖 AI Result")
    st.markdown(llm_result)

    st.markdown("## 📊 Rankings")
    for r in ranked:
        st.write(f"{r['rank']}. {r['filename']} — {r['similarity']:.1%}")

    log("REQUEST_END", request_id, {"time": time.time() - start})
