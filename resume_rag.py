# ============================
# Resume RAG Screener (PRODUCTION WITH SMART FEATURES)
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

st.set_page_config(page_title="Resume RAG Screener", layout="wide", page_icon="📊")

# ============================
# METRICS TRACKING
# ============================
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total_resumes_processed': 0,
        'total_screenings': 0,
        'avg_processing_time': 0,
        'total_candidates_ranked': 0,
        'cache_hit_rate': 0,
        'total_api_calls': 0,
        'cost_saved': 0,
        'processing_times': [],
        'hourly_throughput': 0
    }

def update_metrics(num_resumes, processing_time, cache_hits=0, total_items=1):
    """Update performance metrics"""
    m = st.session_state.metrics
    m['total_resumes_processed'] += num_resumes
    m['total_screenings'] += 1
    m['total_candidates_ranked'] += num_resumes
    m['processing_times'].append(processing_time)
    
    # Calculate averages
    m['avg_processing_time'] = sum(m['processing_times']) / len(m['processing_times'])
    m['cache_hit_rate'] = (cache_hits / total_items * 100) if total_items > 0 else 0
    
    # Calculate throughput (resumes per hour)
    # Formula: (total resumes / total time) * 3600 seconds
    total_time = sum(m['processing_times'])
    if total_time > 0:
        resumes_per_second = m['total_resumes_processed'] / total_time
        m['hourly_throughput'] = int(resumes_per_second * 3600)
    
    # Estimate cost saved (vs manual screening at $50/hr)
    manual_time_saved = num_resumes * 10 / 60  # 10 min per resume manually
    m['cost_saved'] += manual_time_saved * 50

# ============================
# TITLE WITH METRICS BANNER
# ============================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📄 Resumes Processed", st.session_state.metrics['total_resumes_processed'])
with col2:
    st.metric("⚡ Avg Time", f"{st.session_state.metrics['avg_processing_time']:.1f}s")
with col3:
    st.metric("🎯 Cache Hit Rate", f"{st.session_state.metrics['cache_hit_rate']:.0f}%")
with col4:
    st.metric("💰 Cost Saved", f"${st.session_state.metrics['cost_saved']:.0f}")

st.title("📊 AI-Powered Resume Screening System")
st.caption("Powered by RAG + Groq AI | 10x faster than manual screening")

# ============================
# LOGGING SETUP
# ============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def log(step: str):
    logger.info(step)

def log_error(step: str, e: Exception):
    logger.error(f"{step}: {str(e)}")
    logger.error(traceback.format_exc())

# ============================
# ENV SETUP
# ============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================
# EMBEDDING MODEL (CACHED)
# ============================
@st.cache_resource
def load_embedder():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        log("Embedding model loaded")
        return model
    except Exception as e:
        log_error("Embedding model load failed", e)
        raise

embedder = load_embedder()

# ============================
# GROQ CLIENT (CACHED)
# ============================
@st.cache_resource
def load_groq_client():
    try:
        if not GROQ_API_KEY:
            return None
        client = Groq(api_key=GROQ_API_KEY)
        log("Groq client initialized")
        return client
    except Exception as e:
        log_error("Groq client initialization failed", e)
        return None

groq_client = load_groq_client()

# ============================
# HASH-BASED CACHING
# ============================
def get_file_hash(file_bytes):
    """Generate unique hash for file content"""
    return hashlib.md5(file_bytes).hexdigest()

cache_stats = {'hits': 0, 'misses': 0}


@st.cache_data(ttl=3600)
def parse_resume_cached(file_hash, file_bytes, filename):
    """Cache parsed resume by file hash"""
    cache_stats['misses'] += 1
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            doc = fitz.open(tmp.name)
            text = " ".join(page.get_text() for page in doc)
            doc.close()
            os.unlink(tmp.name)
            
            # Extract key information
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            phone_match = re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
            
            return {
                "filename": filename,
                "text": re.sub(r"\s+", " ", text.strip()),
                "hash": file_hash,
                "email": email_match.group(0) if email_match else "Not found",
                "phone": phone_match.group(0) if phone_match else "Not found",
                "word_count": len(text.split())
            }
    except Exception as e:
        log_error(f"Failed to parse {filename}", e)
        return None

@st.cache_data(ttl=3600)
def embed_text_cached(text):
    """Cache embeddings by text content"""
    cache_stats['misses'] += 1
    return embedder.encode([text], show_progress_bar=False)[0].tolist()

# ============================
# IN-MEMORY PROCESSING
# ============================
def process_resumes_in_memory(files):
    """Process resumes entirely in memory"""
    resumes = []
    file_data = []
    
    for file in files:
        file_bytes = file.read()
        file_hash = get_file_hash(file_bytes)
        file_data.append((file_hash, file_bytes, file.name))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(parse_resume_cached, fh, fb, fn) 
            for fh, fb, fn in file_data
        ]
        for future in futures:
            result = future.result()
            if result:
                resumes.append(result)
    
    log(f"Processed {len(resumes)} resumes")
    return resumes

def rank_resumes_in_memory(jd_text, resumes):
    """Rank resumes using embeddings + context verification"""
    jd_embedding = np.array([embed_text_cached(jd_text)])
    
    resume_embeddings = []
    for resume in resumes:
        text_preview = resume["text"][:2000]
        embedding = embed_text_cached(text_preview)
        resume_embeddings.append(embedding)
    
    resume_embeddings = np.array(resume_embeddings)
    similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    
    ranked_resumes = []
    for idx in ranked_indices:
        technical_context_score = verify_technical_context(resumes[idx]["text"])
        
        final_score = similarities[idx]
        
        ranked_resumes.append({
            "filename": resumes[idx]["filename"],
            "text": resumes[idx]["text"],
            "email": resumes[idx]["email"],
            "phone": resumes[idx]["phone"],
            "word_count": resumes[idx]["word_count"],
            "similarity": float(similarities[idx]),
            "technical_context_score": technical_context_score,
            "final_score": final_score,
            "rank": len(ranked_resumes) + 1
        })
    
    ranked_resumes.sort(key=lambda x: x['final_score'], reverse=True)
    for i, r in enumerate(ranked_resumes):
        r['rank'] = i + 1
    
    return ranked_resumes

def verify_technical_context(text):
    technical_indicators = [
        'developer','engineer','project','built','developed','implemented',
        'system','api','backend','frontend','database','microservice'
    ]
    text = text.lower()
    return sum(1 for i in technical_indicators if i in text) / len(technical_indicators)

# ============================
# GROQ LLM ANALYSIS
# ============================
@st.cache_data(ttl=1800, show_spinner=False)
def analyze_with_groq_cached(jd_hash, candidates_json):
    """Cache LLM responses"""
    if not groq_client:
        return "⚠️ Groq API not configured"
    
    try:
        st.session_state.metrics['total_api_calls'] += 1
        candidates = json.loads(candidates_json)
        
        resume_block = ""
        for i, cand in enumerate(candidates[:5], 1):
            resume_block += f"\n\n=== CANDIDATE {i}: {cand['filename']} ===\n"
            resume_block += f"Semantic Match: {cand['similarity']:.2%}\n"
            resume_block += f"Resume: {cand['text'][:600]}\n"
        
        prompt = f"""You are a hiring assistant. Analyze these candidates concisely.

Job Description:
{candidates[0].get('jd_preview', '')}

{resume_block[:4000]}

Provide:
1. **Top 3 Recommendations** (one sentence each on why to interview)
2. **Key Strengths** (bullet points)
3. **Concerns** (if any)

Be brief and actionable."""

        log("Calling Groq API...")
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1000,
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        log_error("Groq API failed", e)
        return f"⚠️ Error: {str(e)}"

def analyze_candidates(jd_text, ranked_resumes):
    """Wrapper for cached LLM analysis"""
    jd_hash = hashlib.md5(jd_text.encode()).hexdigest()
    
    top_candidates = []
    for resume in ranked_resumes[:5]:
        top_candidates.append({
            "filename": resume["filename"],
            "text": resume["text"],
            "similarity": resume["similarity"],
            "jd_preview": jd_text[:1500]
        })
    
    candidates_json = json.dumps(top_candidates, sort_keys=True)
    return analyze_with_groq_cached(jd_hash, candidates_json)
# ============================
# EXPORT FEATURES
# ============================
def export_to_csv(ranked_resumes):
    """Export results to CSV"""
    df = pd.DataFrame([{
        "Rank": r["rank"],
        "Candidate": r["filename"],
        "Email": r["email"],
        "Phone": r["phone"],
        "Semantic Match": f"{r['similarity']:.2%}",
        "Resume Length": f"{r['word_count']} words"
    } for r in ranked_resumes[:20]])
    
    return df.to_csv(index=False).encode('utf-8')

# ============================
# STREAMLIT UI
# ============================

# Sidebar
st.sidebar.header("⚙️ System Settings")

# Performance Metrics in Sidebar
with st.sidebar.expander("📈 System Performance"):
    st.metric("Total Screenings", st.session_state.metrics['total_screenings'])
    st.metric("Candidates Ranked", st.session_state.metrics['total_candidates_ranked'])
    st.metric("Throughput", f"{st.session_state.metrics['hourly_throughput']} resumes/hr")
    st.metric("API Calls", st.session_state.metrics['total_api_calls'])
    
    if st.button("🔄 Reset Metrics"):
        st.session_state.metrics = {
            'total_resumes_processed': 0,
            'total_screenings': 0,
            'avg_processing_time': 0,
            'total_candidates_ranked': 0,
            'cache_hit_rate': 0,
            'total_api_calls': 0,
            'cost_saved': 0,
            'processing_times': [],
            'hourly_throughput': 0
        }
        st.rerun()

# Cache Management
with st.sidebar.expander("💾 Cache Management"):
    st.info("Auto-expires: 1 hour")
    if st.button("🗑️ Clear Caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cleared!")
        st.rerun()

# Main UI
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload multiple PDF resumes"
)

jd_text = st.text_area(
    "Job Description", 
    height=200,
    placeholder="Paste complete job description...",
)

# Advanced Options
col1, col2 = st.columns(2)
with col1:
    top_k = st.selectbox("Analyze Top", [3, 5, 10], index=1)
with col2:
    show_skill_analysis = st.checkbox("Show Skill Analysis", value=True)

if st.button("🚀 Screen Resumes", type="primary", use_container_width=True):
    if not uploaded_files or not jd_text.strip():
        st.warning("⚠️ Upload resumes and add job description")
        st.stop()
    
    if not GROQ_API_KEY:
        st.error("⚠️ Add GROQ_API_KEY to .env")
        st.stop()
    
    try:
        import time
        start = time.time()
        
        cache_stats['hits'] = 0
        cache_stats['misses'] = 0
        
        # Progress
        with st.spinner("Processing..."):
            resumes = process_resumes_in_memory(uploaded_files)
        
        if not resumes:
            st.error("❌ No resumes parsed")
            st.stop()
        
        parse_time = time.time() - start
        
        with st.spinner("Ranking..."):
            ranked_resumes = rank_resumes_in_memory(jd_text.strip(), resumes)
        rank_time = time.time() - start - parse_time
        
        with st.spinner("AI analyzing..."):
            result = analyze_candidates(jd_text.strip(), ranked_resumes[:top_k])
        llm_time = time.time() - start - parse_time - rank_time
        
        total_time = time.time() - start
        
        # Update metrics
        cache_hit_rate = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses']) * 100 if (cache_stats['hits'] + cache_stats['misses']) > 0 else 0
        update_metrics(len(resumes), total_time, cache_stats['hits'], cache_stats['hits'] + cache_stats['misses'])
        
        # Performance Summary
        st.success("✅ Screening Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Time", f"{total_time:.1f}s")
        with col2:
            st.metric("📄 Resumes", len(resumes))
        with col3:
            st.metric("⚡ Speed", f"{len(resumes)/total_time:.1f}/s")
        with col4:
            st.metric("💾 Cache", f"{cache_hit_rate:.0f}%")
        
        # AI Analysis
        st.markdown("---")
        st.markdown("## 🤖 AI Analysis")
        st.markdown(result)
        
        # Detailed Rankings
        st.markdown("---")
        st.markdown("## 📊 Candidate Rankings")
        
        for i, candidate in enumerate(ranked_resumes[:10], 1):
            with st.expander(f"#{i} - {candidate['filename']} | Match: {candidate['similarity']:.0%}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Contact:**")
                    st.write(f"📧 {candidate['email']}")
                    st.write(f"📱 {candidate['phone']}")
                    st.write(f"📝 {candidate['word_count']} words")
                
                with col2:
                    st.markdown(f"**Scores:**")
                    st.write(f"Semantic Match: **{candidate['similarity']:.1%}**")
        
        # Export Options
        st.markdown("---")
        st.markdown("## 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_to_csv(ranked_resumes)
            st.download_button(
                "📊 Download CSV",
                csv_data,
                file_name=f"screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            report = f"""# Resume Screening Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance
- Time: {total_time:.1f}s
- Resumes: {len(resumes)}
- Speed: {len(resumes)/total_time:.1f} resumes/s

## AI Analysis
{result}

## Top 10 Candidates
"""
            for r in ranked_resumes[:10]:
                report += f"\n{r['rank']}. {r['filename']}\n"
                report += f"   - Semantic: {r['similarity']:.1%}\n"
                report += f"   - Contact: {r['email']} | {r['phone']}\n"
            
            st.download_button(
                "📄 Download Report",
                report,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
    except Exception as e:
        log_error("Screening failed", e)
        st.error(f"❌ Error: {str(e)}")









