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
# REQUEST ID TRACKING
# ============================
if 'request_id' not in st.session_state:
    st.session_state.request_id = None

def generate_request_id():
    """Generate unique request ID for tracing"""
    import uuid
    request_id = str(uuid.uuid4())[:8]
    st.session_state.request_id = request_id
    return request_id

def get_request_id():
    """Get current request ID"""
    try:
        return st.session_state.request_id or "no-req-id"
    except (AttributeError, KeyError):
        return "no-req-id"

# ============================
# METRICS TRACKING
# ============================
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total_resumes_processed': 0,
        'total_screenings': 0,
        'avg_processing_time': 0,
        'total_candidates_ranked': 0,
        'total_api_calls': 0,
        'processing_times': [],
        'hourly_throughput': 0
    }

def update_metrics(num_resumes, processing_time, total_items=1):
    """Update performance metrics"""
    req_id = get_request_id()
    log(f"[{req_id}] [ENTER] update_metrics | num_resumes={num_resumes}, processing_time={processing_time:.2f}s")
    m = st.session_state.metrics
    m['total_resumes_processed'] += num_resumes
    m['total_screenings'] += 1
    m['total_candidates_ranked'] += num_resumes
    m['processing_times'].append(processing_time)
    
    # Calculate averages
    m['avg_processing_time'] = sum(m['processing_times']) / len(m['processing_times'])
    
    # Calculate throughput (resumes per hour)
    # Formula: (total resumes / total time) * 3600 seconds
    total_time = sum(m['processing_times'])
    if total_time > 0:
        resumes_per_second = m['total_resumes_processed'] / total_time
        m['hourly_throughput'] = int(resumes_per_second * 3600)
    
    log(f"[{req_id}] [EXIT] update_metrics | avg_time={m['avg_processing_time']:.2f}s, throughput={m['hourly_throughput']}/hr")


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
    req_id = get_request_id()
    logger.error(f"[{req_id}] {step}: {str(e)}")
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
    req_id = get_request_id()
    log(f"[{req_id}] [ENTER] load_embedder")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        log(f"[{req_id}] [EXIT] load_embedder | Embedding model loaded successfully")
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
    req_id = get_request_id()
    log(f"[{req_id}] [ENTER] load_groq_client")
    try:
        if not GROQ_API_KEY:
            log(f"[{req_id}] [EXIT] load_groq_client | No API key found")
            return None
        client = Groq(api_key=GROQ_API_KEY)
        log(f"[{req_id}] [EXIT] load_groq_client | Groq client initialized successfully")
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
    req_id = get_request_id()
    hash_val = hashlib.md5(file_bytes).hexdigest()
    log(f"[{req_id}] [get_file_hash] Generated hash: {hash_val[:8]}...")
    return hash_val


@st.cache_data(ttl=3600)
def parse_resume_cached(file_hash, file_bytes, filename, req_id="no-req-id"):
    """Cache parsed resume by file hash"""
    log(f"[{req_id}] [ENTER] parse_resume_cached | filename={filename}, hash={file_hash[:8]}...")
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
            
            result = {
                "filename": filename,
                "text": re.sub(r"\s+", " ", text.strip()),
                "hash": file_hash,
                "email": email_match.group(0) if email_match else "Not found",
                "phone": phone_match.group(0) if phone_match else "Not found",
                "word_count": len(text.split())
            }
            log(f"[{req_id}] [EXIT] parse_resume_cached | filename={filename}, word_count={result['word_count']}, email={result['email']}, phone={result['phone']}")
            return result
    except Exception as e:
        log_error(f"Failed to parse {filename}", e)
        return None

@st.cache_data(ttl=3600)
def embed_text_cached(text, req_id="no-req-id"):
    """Cache embeddings by text content"""
    text_preview = text[:100] if len(text) > 100 else text
    log(f"[{req_id}] [ENTER] embed_text_cached | text_length={len(text)}, preview='{text_preview}...'")
    embedding = embedder.encode([text], show_progress_bar=False)[0].tolist()
    log(f"[{req_id}] [EXIT] embed_text_cached | embedding_dim={len(embedding)}")
    return embedding

# ============================
# IN-MEMORY PROCESSING
# ============================
def process_resumes_in_memory(files):
    """Process resumes entirely in memory"""
    req_id = get_request_id()
    log(f"[{req_id}] [ENTER] process_resumes_in_memory | num_files={len(files)}")
    resumes = []
    file_data = []
    seen_hashes = set()
    
    for file in files:
        file_bytes = file.read()
        file_hash = get_file_hash(file_bytes)
        
        if file_hash in seen_hashes:
            log(f"[{req_id}] Skipping duplicate: {file.name}")
            continue
        
        seen_hashes.add(file_hash)
        file_data.append((file_hash, file_bytes, file.name))
    
    log(f"[{req_id}] Processing {len(file_data)} unique files (skipped {len(files) - len(file_data)} duplicates)")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(parse_resume_cached, fh, fb, fn, req_id) 
            for fh, fb, fn in file_data
        ]
        for future in futures:
            result = future.result()
            if result:
                resumes.append(result)
    
    log(f"[{req_id}] [EXIT] process_resumes_in_memory | processed={len(resumes)} resumes")
    return resumes

def rank_resumes_in_memory(jd_text, resumes):
    """Rank resumes using embeddings + context verification"""
    req_id = get_request_id()
    log(f"[{req_id}] [ENTER] rank_resumes_in_memory | jd_length={len(jd_text)}, num_resumes={len(resumes)}")
    jd_embedding = np.array([embed_text_cached(jd_text, req_id)])
    
    resume_embeddings = []
    for resume in resumes:
        text_preview = resume["text"][:2000]
        embedding = embed_text_cached(text_preview, req_id)
        resume_embeddings.append(embedding)
    
    resume_embeddings = np.array(resume_embeddings)
    similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    
    log(f"[{req_id}] Calculated similarities | min={similarities.min():.3f}, max={similarities.max():.3f}, mean={similarities.mean():.3f}")
    
    ranked_resumes = []
    for idx in ranked_indices:
        technical_context_score = verify_technical_context(resumes[idx]["text"])
        
        final_score = (0.8 * similarities[idx]) + (0.2 * technical_context_score)
        
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
        log(f"[{req_id}] Ranked: {resumes[idx]['filename']} | similarity={similarities[idx]:.3f}, tech_score={technical_context_score:.3f}, final={final_score:.3f}")
    
    ranked_resumes.sort(key=lambda x: x['final_score'], reverse=True)
    for i, r in enumerate(ranked_resumes):
        r['rank'] = i + 1
    
    log(f"[{req_id}] [EXIT] rank_resumes_in_memory | top_candidate={ranked_resumes[0]['filename']}, top_score={ranked_resumes[0]['final_score']:.3f}")
    return ranked_resumes

def verify_technical_context(text):
    req_id = get_request_id()
    log(f"[{req_id}] [ENTER] verify_technical_context | text_length={len(text)}")
    technical_indicators = [
        'developer','engineer','project','built','developed','implemented',
        'system','api','backend','frontend','database','microservice'
    ]
    text = text.lower()
    score = sum(1 for i in technical_indicators if i in text) / len(technical_indicators)
    log(f"[{req_id}] [EXIT] verify_technical_context | score={score:.3f}")
    return score

# ============================
# GROQ LLM ANALYSIS
# ============================
@st.cache_data(ttl=1800, show_spinner=False)
def analyze_with_groq_cached(jd_hash, candidates_json):
    """Cache LLM responses"""
    req_id = get_request_id()
    log(f"[{req_id}] [ENTER] analyze_with_groq_cached | jd_hash={jd_hash[:8]}...")
    if not groq_client:
        log(f"[{req_id}] [EXIT] analyze_with_groq_cached | Groq API not configured")
        return "⚠️ Groq API not configured"
    
    try:
        st.session_state.metrics['total_api_calls'] += 1
        candidates = json.loads(candidates_json)
        log(f"[{req_id}] Analyzing {len(candidates)} candidates with Groq")
        
        resume_block = ""
        for i, cand in enumerate(candidates, 1):
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

        log(f"[{req_id}] Calling Groq API...")
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1000,
        )
        
        response = chat_completion.choices[0].message.content
        log(f"[{req_id}] [EXIT] analyze_with_groq_cached | response_length={len(response)}")
        return response
        
    except Exception as e:
        log_error("Groq API failed", e)
        return f"❌ Error: {str(e)}"

def analyze_candidates(jd_text, ranked_resumes, top_k):
    """Wrapper for cached LLM analysis"""
    req_id = get_request_id()
    log(f"[{req_id}] [ENTER] analyze_candidates | top_k={top_k}, total_resumes={len(ranked_resumes)}")
    jd_hash = hashlib.md5(jd_text.encode()).hexdigest()
    
    top_candidates = []
    for resume in ranked_resumes[:top_k]:
        top_candidates.append({
            "filename": resume["filename"],
            "text": resume["text"],
            "similarity": resume["similarity"],
            "jd_preview": jd_text[:1500]
        })
        log(f"[{req_id}] Adding to analysis: {resume['filename']} (similarity={resume['similarity']:.3f})")
    
    candidates_json = json.dumps(top_candidates, sort_keys=True)
    result = analyze_with_groq_cached(jd_hash, candidates_json)
    log(f"[{req_id}] [EXIT] analyze_candidates")
    return result


# ============================
# EXPORT FEATURES
# ============================
def export_to_csv(ranked_resumes):
    """Export results to CSV"""
    req_id = get_request_id()
    log(f"[{req_id}] [ENTER] export_to_csv | num_resumes={len(ranked_resumes)}")
    df = pd.DataFrame([{
        "Rank": r["rank"],
        "Candidate": r["filename"],
        "Email": r["email"],
        "Phone": r["phone"],
        "Semantic Match": f"{r['similarity']:.2%}",
        "Resume Length": f"{r['word_count']} words"
    } for r in ranked_resumes[:20]])
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    log(f"[{req_id}] [EXIT] export_to_csv | csv_size={len(csv_data)} bytes")
    return csv_data


# ============================
# CUSTOM CSS FOR LOADING OVERLAY
# ============================
st.markdown("""
<style>
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        backdrop-filter: blur(5px);
    }
    .loading-content {
        background: white;
        padding: 3rem 4rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        text-align: center;
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #ff4b4b;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 1.5rem;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .loading-text {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .loading-subtext {
        font-size: 0.95rem;
        color: #666;
    }
    .loading-progress {
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #888;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# STREAMLIT UI
# ============================

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
    # Generate unique request ID for this screening session
    req_id = generate_request_id()
    log("=" * 80)
    log(f"[{req_id}] SCREENING PROCESS STARTED")
    log("=" * 80)
    
    if not uploaded_files or not jd_text.strip():
        log(f"[{req_id}] Validation failed: Missing resumes or JD")
        st.warning("⚠️ Upload resumes and add job description")
        st.stop()
    
    if not GROQ_API_KEY:
        log(f"[{req_id}] Validation failed: Missing GROQ_API_KEY")
        st.error("❌ Add GROQ_API_KEY to .env")
        st.stop()
    
    # Create placeholder for loading overlay
    loading_placeholder = st.empty()
    
    try:
        import time
        start = time.time()
        
        # Show loading overlay - Parsing phase
        loading_placeholder.markdown(f"""
        <div class="loading-overlay">
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-text">🚀 Screening in Progress</div>
                <div class="loading-subtext">Processing {len(uploaded_files)} resumes...</div>
                <div class="loading-progress">This may take a few moments</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Process resumes
        resumes = process_resumes_in_memory(uploaded_files)
        
        if not resumes:
            loading_placeholder.empty()
            log(f"[{req_id}] No resumes parsed successfully")
            st.error("❌ No resumes parsed")
            st.stop()
        
        parse_time = time.time() - start
        log(f"[{req_id}] Parse phase completed in {parse_time:.2f}s")
        
        # Update loading text - Ranking phase
        loading_placeholder.markdown(f"""
        <div class="loading-overlay">
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-text">📊 Ranking Candidates</div>
                <div class="loading-subtext">Analyzing {len(resumes)} resumes with AI...</div>
                <div class="loading-progress">Calculating semantic matches</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        ranked_resumes = rank_resumes_in_memory(jd_text.strip(), resumes)
        rank_time = time.time() - start - parse_time
        log(f"[{req_id}] Ranking phase completed in {rank_time:.2f}s")
        
        # Update loading text - AI analysis phase
        loading_placeholder.markdown(f"""
        <div class="loading-overlay">
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-text">🤖 AI Analysis</div>
                <div class="loading-subtext">Generating insights for top {top_k} candidates...</div>
                <div class="loading-progress">Using Groq AI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        result = analyze_candidates(jd_text.strip(), ranked_resumes, top_k)
        llm_time = time.time() - start - parse_time - rank_time
        log(f"[{req_id}] LLM analysis completed in {llm_time:.2f}s")
        
        # Clear loading overlay
        loading_placeholder.empty()
        
        total_time = time.time() - start
        log(f"[{req_id}] TOTAL SCREENING TIME: {total_time:.2f}s")
        
        # Update metrics
        update_metrics(len(resumes), total_time)
        
        # Performance Summary
        st.success("✅ Screening Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Time", f"{total_time:.1f}s")
        with col2:
            st.metric("📄 Resumes", len(resumes))
        with col3:
            st.metric("⚡ Speed", f"{len(resumes)/total_time:.1f}resumes/s")
        with col4:
            st.metric("🎯 Top Match", f"{ranked_resumes[0]['similarity']:.0%}")
        
        # System Metrics
        st.markdown("---")
        st.markdown("## 📈 System Performance")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Total Screenings", st.session_state.metrics['total_screenings'])
        with col2:
            st.metric("Total Resumes", st.session_state.metrics['total_resumes_processed'])
        with col3:
            st.metric("Candidates Ranked", st.session_state.metrics['total_candidates_ranked'])
        with col4:
            st.metric("Throughput", f"{st.session_state.metrics['hourly_throughput']}/hr")
        with col5:
            st.metric("API Calls", st.session_state.metrics['total_api_calls'])
        with col6:
            avg_time = st.session_state.metrics['avg_processing_time']
            st.metric("Avg Time", f"{avg_time:.1f}s" if avg_time > 0 else "0s")

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
        
        log("=" * 80)
        log(f"[{req_id}] SCREENING PROCESS COMPLETED SUCCESSFULLY")
        log("=" * 80)
        
    except Exception as e:
        # Clear loading overlay on error
        loading_placeholder.empty()
        log_error("Screening failed", e)
        st.error(f"❌ Error: {str(e)}")