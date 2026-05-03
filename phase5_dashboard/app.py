# ============================================================
# PHASE 5: Streamlit Dashboard — UAE Anti-Scam Intelligence
# ============================================================
# WHAT YOU WILL LEARN:
#   1. How to build a Streamlit web application
#   2. How to show real-time agent progress in a UI
#   3. How to deploy a Python app free on Streamlit Cloud
#   4. How to connect your GitHub repo to a live web app
#   5. Bilingual UI design (English + Arabic)
#
# PREREQUISITES:
#   - Your Phase 4 code (all agents + memory system)
#   - Groq API key in Streamlit secrets
#   - VirusTotal API key in Streamlit secrets
#   - GitHub repo: SMKProj/agentic-antiscam-uae
#
# DEPLOYMENT (free):
#   1. Push this file to GitHub
#   2. Go to share.streamlit.io
#   3. Connect your GitHub repo
#   4. Set secrets (API keys)
#   5. Deploy — get a public URL
#
# FILE STRUCTURE NEEDED IN YOUR REPO:
#   agentic-antiscam-uae/
#   ├── phase5_dashboard/
#   │   ├── app.py              ← this file
#   │   └── requirements.txt    ← dependencies
#   └── README.md
# ============================================================

import streamlit as st
import os
import json
import time
import re
import requests
import whois
import dns.resolver
import chromadb
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from groq import Groq
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from typing import TypedDict


# ── PAGE CONFIG ───────────────────────────────────────────────
# Must be first Streamlit command
st.set_page_config(
    page_title="UAE Anti-Scam Intelligence | كاشف الاحتيال",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 2rem;
    }
    .arabic-text {
        font-family: 'Arial', sans-serif;
        direction: rtl;
        text-align: right;
        color: #666;
        font-size: 0.9rem;
    }
    .agent-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        border-left: 3px solid #e0e0e0;
        font-size: 0.9rem;
    }
    .agent-running {
        border-left-color: #2196F3;
        background: #E3F2FD;
    }
    .agent-complete {
        border-left-color: #4CAF50;
        background: #E8F5E9;
    }
    .agent-waiting {
        border-left-color: #e0e0e0;
        background: #f8f9fa;
        color: #999;
    }
    .verdict-scam {
        background: #FFEBEE;
        border: 2px solid #EF5350;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-safe {
        background: #E8F5E9;
        border: 2px solid #66BB6A;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .evidence-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .memory-card {
        background: #EDE7F6;
        border-left: 3px solid #7C4DFF;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .score-bar-container {
        background: #e0e0e0;
        border-radius: 10px;
        height: 24px;
        width: 100%;
        margin: 0.5rem 0;
    }
    .red-flag {
        background: #FFEBEE;
        color: #C62828;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        display: inline-block;
        margin: 3px;
    }
    .safe-flag {
        background: #E8F5E9;
        color: #2E7D32;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        display: inline-block;
        margin: 3px;
    }
    .stat-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ── LOAD API KEYS ─────────────────────────────────────────────
# Streamlit secrets work like Colab secrets
# Set these in Streamlit Cloud dashboard under Settings → Secrets
# Format in secrets.toml:
#   GROQ_API_KEY = "gsk_..."
#   VIRUSTOTAL_API_KEY = "..."

try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
    VT_KEY   = st.secrets["VIRUSTOTAL_API_KEY"]
except:
    # Fallback to environment variables for local testing
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
    VT_KEY   = os.environ.get("VIRUSTOTAL_API_KEY", "")

if not GROQ_KEY:
    st.error("GROQ_API_KEY not found. Add it to Streamlit secrets.")
    st.stop()

client = Groq(api_key=GROQ_KEY)
MODEL  = "llama-3.3-70b-versatile"


# ── CACHED RESOURCES ──────────────────────────────────────────
# @st.cache_resource loads once and reuses across all users
# Critical for heavy objects like the embedding model
# Without caching, model reloads on every page interaction

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_chroma():
    client_db = chromadb.PersistentClient(path="/tmp/chroma_db")
    collection = client_db.get_or_create_collection(
        name="scam_cases",
        metadata={"hnsw:space": "cosine"}
    )
    # Seed with known cases if empty
    if collection.count() == 0:
        _seed_memory(collection, load_embedding_model())
    return collection

def _seed_memory(collection, emb_model):
    """Seeds ChromaDB with known scam and safe cases."""
    seed_cases = [
        {
            "id": "seed_001",
            "msg": "hralert@wadialsagroup.com ADNOC contractor interview "
                   "salary AED 18000 bring Emirates ID pay AED 300 processing fee",
            "is_scam": True, "confidence": 80, "type": "job_scam"
        },
        {
            "id": "seed_002",
            "msg": "Emirates NBD account suspended verify immediately "
                   "http://emiratesnbd-verify.tk failure 2 hours permanent closure",
            "is_scam": True, "confidence": 95, "type": "phishing"
        },
        {
            "id": "seed_003",
            "msg": "Congratulations won AED 500000 Dubai Government Lucky Draw "
                   "send Emirates ID pay AED 250 processing fee expires 24 hours",
            "is_scam": True, "confidence": 98, "type": "lottery_fraud"
        },
        {
            "id": "seed_004",
            "msg": "ZaviyarHayat Group WAS Group FastInsu hiring book slot "
                   "first come first served marketing@zaviyarhayatgroup.com "
                   "no experience required",
            "is_scam": True, "confidence": 80, "type": "job_scam"
        },
        {
            "id": "seed_005",
            "msg": "Noon order UAE shipped expected delivery tomorrow track "
                   "https://noon.com customer service",
            "is_scam": False, "confidence": 95, "type": "not_a_scam"
        },
        {
            "id": "seed_006",
            "msg": "FastInsu WAS Group walk-in interviews book slot "
                   "first come first served multiple positions no experience",
            "is_scam": True, "confidence": 75, "type": "job_scam"
        },
        {
            "id": "seed_007",
            "msg": "Dear candidate open roles might interest you visit website "
                   "current openings apply directly linkedin consulting firm",
            "is_scam": False, "confidence": 88, "type": "not_a_scam"
        },
    ]
    for case in seed_cases:
        vec = emb_model.encode(case["msg"]).tolist()
        collection.upsert(
            ids=[case["id"]],
            embeddings=[vec],
            documents=[case["msg"]],
            metadatas=[{
                "is_scam"   : str(case["is_scam"]),
                "confidence": case["confidence"],
                "scam_type" : case["type"],
                "timestamp" : datetime.now(timezone.utc).isoformat(),
            }]
        )


# ── STATE DEFINITION ──────────────────────────────────────────
class AgentState(TypedDict):
    original_message  : str
    orchestrator_plan : str
    entities_found    : dict
    nlp_findings      : dict
    technical_findings: dict
    cultural_findings : dict
    risk_score        : dict
    final_verdict     : dict
    agents_completed  : list
    error_log         : list
    similar_cases     : list
    memory_context    : str
    memory_boost      : dict


# ── ALL TOOLS ─────────────────────────────────────────────────
def extract_entities_python(message):
    emails  = re.findall(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', message)
    urls    = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', message)
    from urllib.parse import urlparse
    email_domains = list(set([e.split("@")[1] for e in emails if "@" in e]))
    url_domains   = []
    for url in urls:
        try:
            d = urlparse(url).netloc
            if d: url_domains.append(d)
        except: pass
    return {
        "emails" : emails,
        "urls"   : urls,
        "domains": list(set(email_domains + url_domains))
    }

def whois_lookup(domain):
    try:
        w        = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list): creation = creation[0]
        if creation:
            if creation.tzinfo is None:
                creation = creation.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - creation).days
        else: age_days = None
        if age_days is None:       age_risk = "unknown"
        elif age_days < 30:        age_risk = "VERY HIGH"
        elif age_days < 180:       age_risk = "HIGH"
        elif age_days < 365:       age_risk = "MEDIUM"
        else:                      age_risk = "LOW"
        return {"domain": domain, "age_days": age_days,
                "age_risk": age_risk,
                "registrar": str(w.registrar) if w.registrar else "unknown",
                "country": str(w.country) if w.country else "unknown",
                "status": "success"}
    except Exception as e:
        return {"domain": domain, "status": "failed",
                "error": str(e), "age_risk": "unknown"}

def virustotal_scan(url):
    headers = {"x-apikey": VT_KEY}
    try:
        r = requests.post("https://www.virustotal.com/api/v3/urls",
                          headers=headers, data={"url": url}, timeout=15)
        if r.status_code != 200:
            return {"url": url, "status": "failed"}
        aid = r.json()["data"]["id"]
        time.sleep(5)
        r2  = requests.get(
            f"https://www.virustotal.com/api/v3/analyses/{aid}",
            headers=headers, timeout=15)
        stats = r2.json()["data"]["attributes"]["stats"]
        mal   = stats.get("malicious", 0)
        total = sum(stats.values())
        if mal >= 5: vt_risk = "VERY HIGH"
        elif mal >= 2: vt_risk = "HIGH"
        elif mal >= 1: vt_risk = "MEDIUM"
        else:          vt_risk = "LOW"
        return {"url": url, "malicious": mal,
                "total_engines": total, "vt_risk": vt_risk,
                "status": "success"}
    except Exception as e:
        return {"url": url, "status": "failed", "error": str(e)}

def scrape_website(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r    = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()
        text = " ".join(soup.get_text().split())[:2000].lower()
        flags = {
            "login_form"       : bool(soup.find("form")),
            "urgency_language" : any(w in text for w in [
                "urgent","expires","act now","immediately"]),
            "prize_language"   : any(w in text for w in [
                "congratulations","winner","prize"]),
            "financial_request": any(w in text for w in [
                "bank account","wire transfer","processing fee"]),
        }
        found = [k for k,v in flags.items() if v]
        return {"url": url, "flags_found": found,
                "flags_count": len(found), "status": "success"}
    except requests.exceptions.ConnectionError:
        return {"url": url, "status": "unreachable",
                "note": "Website unreachable — strong scam signal"}
    except Exception as e:
        return {"url": url, "status": "failed", "error": str(e)}

def analyze_email_domain(email):
    if "@" not in email:
        return {"email": email, "status": "invalid"}
    prefix, domain = email.split("@", 1)
    free_providers = ["gmail.com","yahoo.com","hotmail.com","outlook.com"]
    sus_prefixes   = ["hralert","hr-alert","noreply-hr","jobs-alert",
                      "marketing","alert-team","recruitment-alert"]
    is_free    = domain.lower() in free_providers
    sus_prefix = any(p in prefix.lower() for p in sus_prefixes)
    try:
        dns.resolver.resolve(domain, "MX")
        has_mx = True
    except: has_mx = False
    signals = []
    if is_free:    signals.append("free provider")
    if sus_prefix: signals.append(f"suspicious prefix: {prefix}")
    if not has_mx: signals.append("no MX records")
    risk = "HIGH" if len(signals)>=2 else "MEDIUM" if signals else "LOW"
    return {"email": email, "domain": domain, "prefix": prefix,
            "is_free_provider": is_free, "suspicious_prefix": sus_prefix,
            "has_mx_records": has_mx, "risk_signals": signals,
            "email_risk": risk, "status": "success"}

def check_company_existence(company_name, domain):
    signals = {"domain_checked": domain, "company_name": company_name,
               "existence_signals": [], "absence_signals": [],
               "existence_risk": "LOW"}
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r    = requests.get(f"https://{domain}", headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text().lower()
        if any(w in text for w in ["about us","our team","contact"]):
            signals["existence_signals"].append("has business content")
        else:
            signals["absence_signals"].append("lacks business content")
        if any(w in text for w in ["careers","jobs","vacancies"]):
            signals["existence_signals"].append("has careers section")
        else:
            signals["absence_signals"].append("no careers section")
        name_words = [w for w in company_name.lower().split() if len(w)>4]
        if any(w in text for w in name_words):
            signals["existence_signals"].append("name found on site")
        else:
            signals["absence_signals"].append("name not on site")
    except requests.exceptions.ConnectionError:
        signals["absence_signals"].append("website unreachable")
    except Exception as e:
        signals["absence_signals"].append(f"check failed")
    website_unreachable = any("unreachable" in s
                               for s in signals["absence_signals"])
    absence_count = len(signals["absence_signals"])
    if website_unreachable and absence_count >= 2:
        signals["existence_risk"] = "HIGH"
    elif website_unreachable or absence_count >= 3:
        signals["existence_risk"] = "HIGH"
    elif absence_count >= 2:
        signals["existence_risk"] = "MEDIUM"
    return signals


# ── LLM HELPER ────────────────────────────────────────────────
def call_llm(system_prompt, user_message, temperature=0.1):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_message}],
        temperature=temperature, max_tokens=1500)
    return response.choices[0].message.content.strip()

def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try: return json.loads(raw.strip())
    except: return {"error": "parse failed", "raw": raw}


# ── MEMORY FUNCTIONS ──────────────────────────────────────────
def retrieve_similar(message, collection, emb_model,
                     n_results=3, threshold=0.65):
    if collection.count() == 0:
        return []
    query_vec = emb_model.encode(message).tolist()
    results   = collection.query(
        query_embeddings=[query_vec],
        n_results=min(n_results, collection.count()),
        include=["documents","metadatas","distances"]
    )
    similar = []
    for i in range(len(results["ids"][0])):
        distance   = results["distances"][0][i]
        similarity = round(1 - (distance / 2), 3)
        if similarity >= threshold:
            similar.append({
                "case_id"   : results["ids"][0][i],
                "similarity": similarity,
                "message"   : results["documents"][0][i],
                "metadata"  : results["metadatas"][0][i],
            })
    return similar

def calculate_memory_boost(similar_cases, current_score):
    # Guard: only boost if base score already suggests suspicion
    if current_score < 45:
        return {"boosted_score": current_score, "boost_applied": 0,
                "boost_reason": ["Base score too low for memory boost"]}
    if not similar_cases:
        return {"boosted_score": current_score, "boost_applied": 0,
                "boost_reason": ["No similar past cases"]}
    boost   = 0
    reasons = []
    confirmed_scams = [c for c in similar_cases
                       if c["metadata"].get("is_scam") == "True"]
    confirmed_safe  = [c for c in similar_cases
                       if c["metadata"].get("is_scam") == "False"]
    for case in confirmed_scams:
        sim = case["similarity"]
        if sim >= 0.90:
            boost += 10
            reasons.append(f"Very similar past scam ({sim:.0%})")
        elif sim >= 0.80:
            boost += 5
            reasons.append(f"Similar past scam ({sim:.0%})")
    if len(confirmed_scams) >= 2:
        boost += 5
        reasons.append(f"{len(confirmed_scams)} similar scam pattern")
    for case in confirmed_safe:
        if case["similarity"] >= 0.85:
            boost -= 5
            reasons.append(f"Similar safe case ({case['similarity']:.0%})")
    boost = max(-10, min(20, boost))
    return {
        "boosted_score"    : min(100, max(0, current_score + boost)),
        "boost_applied"    : boost,
        "boost_reason"     : reasons,
        "similar_scams"    : len(confirmed_scams),
        "similar_safe"     : len(confirmed_safe),
    }

def store_case(case_id, message, verdict,
               agents_findings, collection, emb_model):
    vec = emb_model.encode(message).tolist()
    collection.upsert(
        ids=[case_id],
        embeddings=[vec],
        documents=[message[:500]],
        metadatas=[{
            "case_id"   : case_id,
            "is_scam"   : str(verdict.get("is_scam", False)),
            "confidence": int(verdict.get("confidence", 0)),
            "scam_type" : str(verdict.get("scam_type", "unknown")),
            "timestamp" : datetime.now(timezone.utc).isoformat(),
        }]
    )


# ── STREAMLIT UI ──────────────────────────────────────────────
def main():
    emb_model  = load_embedding_model()
    collection = load_chroma()

    # ── HEADER ────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1 style="font-size:2rem; font-weight:600; margin:0;">
            🛡️ UAE Anti-Scam Intelligence System
        </h1>
        <p class="arabic-text">نظام الكشف عن الاحتيال الإماراتي</p>
        <p style="color:#888; font-size:0.9rem; margin-top:0.5rem;">
            6 AI agents · Real-time analysis · Memory-augmented detection
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📊 System Status")

        # Memory stats
        total_cases = collection.count()
        all_meta    = collection.get(include=["metadatas"])["metadatas"]
        scam_count  = sum(1 for m in all_meta
                          if m.get("is_scam") == "True")
        safe_count  = sum(1 for m in all_meta
                          if m.get("is_scam") == "False")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div style="font-size:1.6rem; font-weight:600;
                            color:#7C4DFF;">{total_cases}</div>
                <div style="font-size:0.75rem; color:#888;">
                    Cases in memory</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div style="font-size:1.6rem; font-weight:600;
                            color:#E53935;">{scam_count}</div>
                <div style="font-size:0.75rem; color:#888;">
                    Scams stored</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🤖 Agent Pipeline")
        agents = [
            ("0", "Memory Retrieval", "نظام الذاكرة"),
            ("1", "Orchestrator",     "المنسق"),
            ("2", "NLP Specialist",   "تحليل النص"),
            ("3", "Technical",        "التحقق التقني"),
            ("4", "Cultural Context", "السياق الثقافي"),
            ("5", "Risk Scorer",      "تقييم المخاطر"),
            ("6", "Final Verdict",    "الحكم النهائي"),
        ]
        for num, name, arabic in agents:
            st.markdown(f"""
            <div class="agent-card agent-waiting">
                <span style="font-weight:500;">Agent {num}:</span>
                {name}
                <span style="color:#aaa; font-size:0.8rem;">
                    — {arabic}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        Built by **Sundas Mohsin Khan**
        UAE Data Scientist

        [GitHub](https://github.com/SMKProj/agentic-antiscam-uae)

        *Portfolio project — 5 phases, 6 agents,
        real UAE scam detection*
        """)

    # ── MAIN CONTENT ──────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍 Investigate | تحقق",
        "🧠 Memory | الذاكرة",
        "📖 How It Works | كيف يعمل"
    ])

    # ── TAB 1: INVESTIGATE ────────────────────────────────────
    with tab1:
        st.markdown("#### Paste a suspicious message below")
        st.markdown(
            '<p class="arabic-text">'
            'الصق الرسالة المشبوهة أدناه للتحليل</p>',
            unsafe_allow_html=True
        )

        # Sample messages for quick testing
        with st.expander("📋 Load a sample message for testing"):
            sample = st.selectbox("Choose a sample:", [
                "Select a sample...",
                "UAE job scam (ADNOC impersonation)",
                "Bank phishing (Emirates NBD)",
                "Lottery fraud (Dubai Government)",
                "Legitimate message (Noon delivery)",
            ])
            sample_messages = {
                "UAE job scam (ADNOC impersonation)":
                    "From: hralert@wadialsagroup.com\n"
                    "Subject: Urgent Interview — ADNOC Contractor\n\n"
                    "Dear Candidate, You have been selected for an interview "
                    "at ADNOC contractor Wadi Al Salam Group. "
                    "Salary AED 18,000/month tax free. "
                    "Interview Monday 10am. Bring Emirates ID and pay "
                    "AED 300 processing fee upon arrival.\nHR Department",

                "Bank phishing (Emirates NBD)":
                    "Dear Emirates NBD Customer,\n"
                    "Your account requires immediate verification.\n"
                    "Click: http://emiratesnbd-secure-login.tk/verify\n"
                    "Failure to verify within 2 hours will result "
                    "in account suspension.\nEmirates NBD Security Team",

                "Lottery fraud (Dubai Government)":
                    "Congratulations! You have won AED 500,000 in the "
                    "Dubai Government Lucky Draw. To claim your prize, "
                    "send your Emirates ID and pay AED 250 processing fee. "
                    "Offer expires in 24 hours!",

                "Legitimate message (Noon delivery)":
                    "Hi, your Noon order #UAE-2847361 has been shipped.\n"
                    "Expected delivery: tomorrow between 2-6pm.\n"
                    "Track here: https://noon.com/uae/track/2847361\n"
                    "Noon Customer Service",
            }
            if sample != "Select a sample...":
                st.session_state["loaded_sample"] = \
                    sample_messages.get(sample, "")

        # Message input
        default_msg = st.session_state.get("loaded_sample", "")
        message     = st.text_area(
            "Message text | نص الرسالة",
            value=default_msg,
            height=180,
            placeholder="Paste any suspicious SMS, email, or WhatsApp "
                         "message here...\nالصق أي رسالة مشبوهة هنا..."
        )

        col_btn1, col_btn2, col_btn3 = st.columns([2,1,1])
        with col_btn1:
            investigate_btn = st.button(
                "🔍 Investigate | تحقق",
                type="primary",
                use_container_width=True
            )
        with col_btn2:
            store_result = st.checkbox("Store result in memory", value=True)
        with col_btn3:
            st.markdown(
                '<p class="arabic-text" style="padding-top:0.5rem;">'
                'حفظ في الذاكرة</p>',
                unsafe_allow_html=True
            )

        if investigate_btn and message.strip():
            st.markdown("---")
            st.markdown("### 🔄 Live Agent Activity | نشاط الوكلاء")

            # Agent progress placeholders
            agent_placeholders = {}
            agent_names = [
                ("memory",      "Agent 0: Memory Retrieval",  "نظام الذاكرة"),
                ("orchestrator","Agent 1: Orchestrator",       "المنسق"),
                ("nlp",         "Agent 2: NLP Specialist",     "تحليل النص"),
                ("technical",   "Agent 3: Technical",          "التحقق التقني"),
                ("cultural",    "Agent 4: Cultural Context",   "السياق الثقافي"),
                ("risk",        "Agent 5: Risk Scorer",        "تقييم المخاطر"),
                ("verdict",     "Agent 6: Final Verdict",      "الحكم النهائي"),
            ]

            for key, name, arabic in agent_names:
                agent_placeholders[key] = st.empty()
                agent_placeholders[key].markdown(
                    f'<div class="agent-card agent-waiting">'
                    f'⏳ {name} — {arabic}</div>',
                    unsafe_allow_html=True
                )

            result_placeholder = st.empty()
            state = {}

            # ── RUN AGENTS ONE BY ONE ─────────────────────────

            # AGENT 0: Memory retrieval
            agent_placeholders["memory"].markdown(
                '<div class="agent-card agent-running">'
                '🔄 Agent 0: Memory Retrieval — searching past cases...</div>',
                unsafe_allow_html=True
            )
            similar_cases  = retrieve_similar(message, collection, emb_model)
            memory_context = ""
            if similar_cases:
                lines = [f"MEMORY: {len(similar_cases)} similar past cases:\n"]
                for i, case in enumerate(similar_cases, 1):
                    meta = case["metadata"]
                    lines.append(
                        f"Case #{i} (similarity {case['similarity']:.0%}): "
                        f"{'SCAM' if meta['is_scam']=='True' else 'SAFE'} "
                        f"({meta['confidence']}% confidence)"
                    )
                memory_context = "\n".join(lines)
            state["similar_cases"]  = similar_cases
            state["memory_context"] = memory_context
            agent_placeholders["memory"].markdown(
                f'<div class="agent-card agent-complete">'
                f'✅ Agent 0: Memory Retrieval — '
                f'{len(similar_cases)} similar cases found</div>',
                unsafe_allow_html=True
            )

            # AGENT 1: Orchestrator
            agent_placeholders["orchestrator"].markdown(
                '<div class="agent-card agent-running">'
                '🔄 Agent 1: Orchestrator — extracting entities...</div>',
                unsafe_allow_html=True
            )
            python_entities = extract_entities_python(message)
            memory_section  = ""
            if memory_context and "No similar" not in memory_context:
                memory_section = f"\nMEMORY CONTEXT:\n{memory_context}\n"
            orc_prompt = f"""
            You are the lead fraud investigator.{memory_section}
            IDENTITY CHECK:
            1. Multiple company names in one sender → SUSPICIOUS
            2. Email domain not matching claimed company → MISMATCH
            3. Book a slot / first come first served → mass spam
            Return ONLY JSON:
            {{"claimed_identity":"...","requested_action":"...",
              "identity_flags":[],"identity_risk":"HIGH/MEDIUM/LOW",
              "memory_influenced":true/false,"priority":"HIGH/MEDIUM/LOW"}}
            """
            orc_result = parse_json(call_llm(orc_prompt,
                                             f"Analyze:\n\n{message}"))
            state["entities_found"] = {
                **python_entities,
                "claimed_identity": orc_result.get("claimed_identity",""),
                "requested_action": orc_result.get("requested_action",""),
                "identity_flags"  : orc_result.get("identity_flags",[]),
                "identity_risk"   : orc_result.get("identity_risk","LOW"),
            }
            agent_placeholders["orchestrator"].markdown(
                f'<div class="agent-card agent-complete">'
                f'✅ Agent 1: Orchestrator — '
                f'{len(python_entities["emails"])} emails, '
                f'{len(python_entities["urls"])} URLs found</div>',
                unsafe_allow_html=True
            )

            # AGENT 2: NLP
            agent_placeholders["nlp"].markdown(
                '<div class="agent-card agent-running">'
                '🔄 Agent 2: NLP Specialist — analyzing language patterns...</div>',
                unsafe_allow_html=True
            )
            nlp_prompt = """
            Analyze ONLY text. Look for: urgency, threats, authority claims,
            reward claims, info harvesting, generic greetings, grammar issues,
            mass recruitment language (book a slot, first come first served).
            Return ONLY JSON:
            {"urgency_found":bool,"threat_found":bool,"authority_claim":bool,
             "reward_claim":bool,"info_harvesting":bool,"generic_greeting":bool,
             "grammar_issues":bool,"mass_recruitment":bool,
             "key_phrases":[],"sentiment":"...","manipulation_score":0,
             "nlp_risk":"HIGH/MEDIUM/LOW","nlp_summary":"..."}
            """
            nlp_result = parse_json(call_llm(nlp_prompt,
                                             f"Analyze:\n\n{message}"))
            # Rule-based override
            signals = sum([
                bool(nlp_result.get("urgency_found")),
                bool(nlp_result.get("threat_found")),
                bool(nlp_result.get("authority_claim")),
                bool(nlp_result.get("reward_claim")),
                bool(nlp_result.get("info_harvesting")),
                bool(nlp_result.get("mass_recruitment")),
                bool(nlp_result.get("generic_greeting")),
            ])
            if signals >= 4:   nlp_result["nlp_risk"] = "HIGH"
            elif signals >= 2: nlp_result["nlp_risk"] = "MEDIUM"
            else:              nlp_result["nlp_risk"] = "LOW"
            state["nlp_findings"] = nlp_result
            agent_placeholders["nlp"].markdown(
                f'<div class="agent-card agent-complete">'
                f'✅ Agent 2: NLP Specialist — '
                f'{signals} signals, risk: {nlp_result["nlp_risk"]}</div>',
                unsafe_allow_html=True
            )

            # AGENT 3: Technical
            agent_placeholders["technical"].markdown(
                '<div class="agent-card agent-running">'
                '🔄 Agent 3: Technical — running WHOIS, VirusTotal, '
                'scraper... (may take 1-2 min)</div>',
                unsafe_allow_html=True
            )
            entities = state["entities_found"]
            evidence = {"whois_results":[],"virustotal":[],
                        "email_analysis":[],"website_scrapes":[],
                        "company_existence":{},"technical_risk":"LOW"}

            all_domains = list(set(
                entities.get("domains",[]) +
                [e.split("@")[1] for e in entities.get("emails",[])
                 if "@" in e]
            ))
            for domain in all_domains:
                evidence["whois_results"].append(whois_lookup(domain))
                time.sleep(1)
            for email in entities.get("emails",[]):
                evidence["email_analysis"].append(
                    analyze_email_domain(email))
            claimed = entities.get("claimed_identity","")
            if claimed and all_domains:
                evidence["company_existence"] = check_company_existence(
                    claimed, all_domains[0])
            for url in entities.get("urls",[])[:2]:
                evidence["virustotal"].append(virustotal_scan(url))
                time.sleep(15)
            for url in entities.get("urls",[])[:1]:
                evidence["website_scrapes"].append(scrape_website(url))

            # Risk calculation
            high_signals = 0
            for w in evidence["whois_results"]:
                if w.get("age_risk") in ["VERY HIGH","HIGH"]:
                    high_signals += 1
            for e in evidence["email_analysis"]:
                if e.get("email_risk") == "HIGH": high_signals += 1
            for v in evidence["virustotal"]:
                if v.get("vt_risk") in ["VERY HIGH","HIGH"]:
                    high_signals += 2
            for s in evidence["website_scrapes"]:
                if s.get("flags_count",0) >= 2: high_signals += 1
            company_risk = evidence["company_existence"].get(
                "existence_risk","LOW")
            if company_risk == "HIGH":   high_signals += 2
            elif company_risk == "MEDIUM": high_signals += 1
            if entities.get("identity_risk") == "HIGH": high_signals += 1
            nlp_risk = state["nlp_findings"].get("nlp_risk","LOW")
            if (not entities.get("urls") and
                    nlp_risk in ["HIGH","VERY HIGH"]):
                high_signals += 2
            if high_signals >= 4:   evidence["technical_risk"] = "VERY HIGH"
            elif high_signals >= 3: evidence["technical_risk"] = "HIGH"
            elif high_signals >= 2: evidence["technical_risk"] = "MEDIUM"
            elif high_signals >= 1: evidence["technical_risk"] = "LOW-MEDIUM"
            state["technical_findings"] = evidence
            agent_placeholders["technical"].markdown(
                f'<div class="agent-card agent-complete">'
                f'✅ Agent 3: Technical — '
                f'{high_signals} high-risk signals, '
                f'{evidence["technical_risk"]}</div>',
                unsafe_allow_html=True
            )

            # AGENT 4: Cultural
            agent_placeholders["cultural"].markdown(
                '<div class="agent-card agent-running">'
                '🔄 Agent 4: Cultural Context — checking UAE patterns...</div>',
                unsafe_allow_html=True
            )
            cultural_prompt = """
            UAE fraud specialist. Detect:
            - Fake ADNOC/Emirates/Etisalat/bank impersonation
            - Unrealistic salaries (AED 15,000+ entry level)
            - Upfront fees before interviews
            - Unsolicited recruitment (3+ flags = HIGH):
              unsolicited contact, multiple company names,
              book a slot / first come first served,
              no job requirements, no CV reference
            Return ONLY JSON:
            {"uae_entity_impersonated":"name or null",
             "known_uae_scam_pattern":bool,
             "unrealistic_offer":bool,
             "unsolicited_recruitment":bool,
             "multiple_company_names":bool,
             "mass_recruitment_language":bool,
             "cultural_red_flags":[],
             "cultural_risk":"HIGH/MEDIUM/LOW",
             "cultural_summary":"..."}
            """
            cultural_result = parse_json(
                call_llm(cultural_prompt, f"Analyze:\n\n{message}"))
            state["cultural_findings"] = cultural_result
            agent_placeholders["cultural"].markdown(
                f'<div class="agent-card agent-complete">'
                f'✅ Agent 4: Cultural Context — '
                f'risk: {cultural_result.get("cultural_risk","LOW")}, '
                f'{len(cultural_result.get("cultural_red_flags",[]))} '
                f'UAE flags</div>',
                unsafe_allow_html=True
            )

            # AGENT 5: Risk Scorer
            agent_placeholders["risk"].markdown(
                '<div class="agent-card agent-running">'
                '🔄 Agent 5: Risk Scorer — calculating weighted score...</div>',
                unsafe_allow_html=True
            )
            def risk_to_score(r):
                return {"VERY HIGH":95,"HIGH":75,"LOW-MEDIUM":55,
                        "MEDIUM":50,"LOW":15,"unknown":30}.get(r, 30)

            tech_risk     = evidence["technical_risk"]
            cultural_risk = cultural_result.get("cultural_risk","LOW")
            nlp_risk      = state["nlp_findings"].get("nlp_risk","LOW")
            has_tech      = bool(evidence.get("whois_results") or
                                 evidence.get("email_analysis"))
            if has_tech:
                weights = {"t":0.50,"c":0.30,"n":0.20}
            else:
                weights = {"t":0.20,"c":0.50,"n":0.30}

            ts = risk_to_score(tech_risk)
            cs = risk_to_score(cultural_risk)
            ns = risk_to_score(nlp_risk)
            base_score  = int(ts*weights["t"] + cs*weights["c"] +
                              ns*weights["n"])
            mem_boost   = calculate_memory_boost(
                similar_cases, base_score)
            final_score = mem_boost["boosted_score"]
            confidence  = ("VERY HIGH" if final_score>=85 else
                           "HIGH" if final_score>=65 else
                           "MEDIUM" if final_score>=45 else "LOW")
            state["risk_score"] = {
                "base_score"      : base_score,
                "final_score"     : final_score,
                "confidence"      : confidence,
                "is_scam"         : final_score >= 55,
                "breakdown"       : {"tech":ts,"cultural":cs,"nlp":ns},
                "individual_risks": {"tech":tech_risk,
                                     "cultural":cultural_risk,
                                     "nlp":nlp_risk},
            }
            state["memory_boost"] = mem_boost
            agent_placeholders["risk"].markdown(
                f'<div class="agent-card agent-complete">'
                f'✅ Agent 5: Risk Scorer — '
                f'base: {base_score}, boost: '
                f'{mem_boost["boost_applied"]:+d}, '
                f'final: {final_score}/100</div>',
                unsafe_allow_html=True
            )

            # AGENT 6: Verdict
            agent_placeholders["verdict"].markdown(
                '<div class="agent-card agent-running">'
                '🔄 Agent 6: Final Verdict — synthesizing report...</div>',
                unsafe_allow_html=True
            )
            verdict_prompt = """
            Lead fraud investigator. Synthesize all findings.
            Return ONLY JSON:
            {"is_scam":bool,"confidence":0-100,
             "scam_type":"phishing/lottery_fraud/advance_fee/romance_scam/
                          investment_fraud/impersonation/job_scam/
                          tech_support_scam/not_a_scam/unknown",
             "verdict_summary":"2-3 sentences plain English",
             "key_evidence":[],"red_flags":[],"safe_indicators":[],
             "recommended_action":"clear action",
             "memory_note":"how past cases influenced verdict or null"}
            """
            all_findings = {
                "message"  : message[:300],
                "entities" : state["entities_found"],
                "nlp"      : state["nlp_findings"],
                "technical": state["technical_findings"],
                "cultural" : state["cultural_findings"],
                "score"    : state["risk_score"],
            }
            verdict_result = parse_json(
                call_llm(verdict_prompt,
                         json.dumps(all_findings, default=str)))
            verdict_result["confidence"] = final_score
            verdict_result["is_scam"]    = final_score >= 55
            state["final_verdict"]       = verdict_result
            agent_placeholders["verdict"].markdown(
                '<div class="agent-card agent-complete">'
                '✅ Agent 6: Final Verdict — complete</div>',
                unsafe_allow_html=True
            )

            # Store in memory
            if store_result:
                case_id = f"case_{int(time.time())}"
                store_case(case_id, message, verdict_result,
                           {}, collection, emb_model)

            # ── DISPLAY VERDICT ────────────────────────────────
            st.markdown("---")
            st.markdown("### 📋 Investigation Report | تقرير التحقيق")

            is_scam = verdict_result.get("is_scam", False)

            if is_scam:
                st.markdown(f"""
                <div class="verdict-scam">
                    <h2 style="color:#C62828; margin:0;">
                        🚨 SCAM DETECTED | تم اكتشاف احتيال</h2>
                    <p style="font-size:1.1rem; margin:0.5rem 0;">
                        Risk Score: {final_score}/100</p>
                    <p style="color:#C62828; margin:0;">
                        {verdict_result.get("scam_type","").replace("_"," ").title()}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-safe">
                    <h2 style="color:#2E7D32; margin:0;">
                        ✅ LIKELY SAFE | يبدو آمناً</h2>
                    <p style="font-size:1.1rem; margin:0.5rem 0;">
                        Risk Score: {final_score}/100</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Score breakdown
            col_score, col_memory = st.columns(2)
            with col_score:
                st.markdown("##### Score Breakdown | تفاصيل النتيجة")
                breakdown_data = {
                    "Technical (50%)": ts,
                    "Cultural (30%)": cs,
                    "NLP (20%)": ns,
                }
                for label, score in breakdown_data.items():
                    color = ("#E53935" if score>=75 else
                             "#FB8C00" if score>=50 else "#43A047")
                    pct   = int(score)
                    st.markdown(f"""
                    <div style="margin:0.5rem 0;">
                        <div style="display:flex; justify-content:space-between;
                                    font-size:0.85rem; margin-bottom:3px;">
                            <span>{label}</span>
                            <span style="font-weight:500;">{pct}</span>
                        </div>
                        <div class="score-bar-container">
                            <div style="background:{color};
                                        border-radius:10px; height:24px;
                                        width:{pct}%;
                                        transition:width 0.5s;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="margin-top:0.5rem; padding:0.5rem 0.75rem;
                            background:#f5f5f5; border-radius:8px;
                            font-size:0.85rem;">
                    Base: {base_score} |
                    Memory boost: {mem_boost['boost_applied']:+d} |
                    Final: {final_score}
                </div>""", unsafe_allow_html=True)

            with col_memory:
                st.markdown("##### Memory Matches | تطابق الذاكرة")
                if similar_cases:
                    for case in similar_cases:
                        meta    = case["metadata"]
                        is_scam_case = meta.get("is_scam") == "True"
                        color   = "#FFEBEE" if is_scam_case else "#E8F5E9"
                        label   = "SCAM" if is_scam_case else "SAFE"
                        st.markdown(f"""
                        <div class="memory-card" style="background:{color};">
                            <strong>{case['case_id']}</strong><br>
                            Similarity: {case['similarity']:.0%} |
                            {label} | Score: {meta.get('confidence')}%
                        </div>""", unsafe_allow_html=True)
                    if mem_boost.get("boost_reason"):
                        for r in mem_boost["boost_reason"]:
                            if "too low" not in r and "No similar" not in r:
                                st.markdown(
                                    f"📈 *{r}*",
                                    unsafe_allow_html=False)
                else:
                    st.info("No similar past cases found in memory")

            # Evidence tabs
            st.markdown("##### Evidence Details | تفاصيل الأدلة")
            ev_tab1, ev_tab2, ev_tab3, ev_tab4 = st.tabs([
                "🔬 Technical", "💬 NLP", "🇦🇪 Cultural", "📝 Summary"
            ])

            with ev_tab1:
                for w in evidence.get("whois_results",[]):
                    risk_color = ("#E53935" if w.get("age_risk")
                                  in ["VERY HIGH","HIGH"]
                                  else "#43A047")
                    st.markdown(f"""
                    <div class="evidence-card">
                        <strong>WHOIS: {w.get('domain')}</strong><br>
                        Age: {w.get('age_days')} days |
                        <span style="color:{risk_color};">
                            {w.get('age_risk')}</span> |
                        Registrar: {w.get('registrar','unknown')} |
                        Country: {w.get('country','unknown')}
                    </div>""", unsafe_allow_html=True)
                for e in evidence.get("email_analysis",[]):
                    st.markdown(f"""
                    <div class="evidence-card">
                        <strong>Email: {e.get('email')}</strong><br>
                        Risk: {e.get('email_risk')} |
                        Signals: {', '.join(e.get('risk_signals',[])) or 'none'}
                    </div>""", unsafe_allow_html=True)
                comp = evidence.get("company_existence",{})
                if comp:
                    st.markdown(f"""
                    <div class="evidence-card">
                        <strong>Company existence: {comp.get('domain_checked')}</strong><br>
                        Risk: {comp.get('existence_risk')} |
                        Absence signals: {', '.join(comp.get('absence_signals',[]))}
                    </div>""", unsafe_allow_html=True)

            with ev_tab2:
                nlp = state.get("nlp_findings",{})
                flags_found = []
                if nlp.get("urgency_found"):
                    flags_found.append("Urgency language")
                if nlp.get("threat_found"):
                    flags_found.append("Threat language")
                if nlp.get("authority_claim"):
                    flags_found.append("Authority impersonation")
                if nlp.get("reward_claim"):
                    flags_found.append("Reward claim")
                if nlp.get("info_harvesting"):
                    flags_found.append("Info harvesting")
                if nlp.get("mass_recruitment"):
                    flags_found.append("Mass recruitment language")
                if nlp.get("generic_greeting"):
                    flags_found.append("Generic greeting")
                st.markdown(
                    f"**NLP Risk:** {nlp.get('nlp_risk')} | "
                    f"**Signals:** {len(flags_found)}")
                st.markdown(
                    f"**Key phrases:** "
                    f"{', '.join(nlp.get('key_phrases',[]))}")
                if flags_found:
                    html_flags = "".join([
                        f'<span class="red-flag">{f}</span>'
                        for f in flags_found
                    ])
                    st.markdown(html_flags, unsafe_allow_html=True)

            with ev_tab3:
                cultural = state.get("cultural_findings",{})
                st.markdown(
                    f"**Cultural Risk:** "
                    f"{cultural.get('cultural_risk','LOW')}")
                if cultural.get("uae_entity_impersonated"):
                    st.warning(
                        f"UAE Entity Impersonated: "
                        f"{cultural.get('uae_entity_impersonated')}")
                flags = cultural.get("cultural_red_flags",[])
                if flags:
                    html_flags = "".join([
                        f'<span class="red-flag">{f}</span>'
                        for f in flags
                    ])
                    st.markdown(html_flags, unsafe_allow_html=True)

            with ev_tab4:
                st.markdown(
                    f"**Summary:** "
                    f"{verdict_result.get('verdict_summary','')}")
                st.markdown("---")

                col_r, col_s = st.columns(2)
                with col_r:
                    st.markdown("**🚩 Red Flags:**")
                    for f in verdict_result.get("red_flags",[]):
                        st.markdown(
                            f'<span class="red-flag">{f}</span>',
                            unsafe_allow_html=True)
                with col_s:
                    st.markdown("**✅ Safe Indicators:**")
                    for f in verdict_result.get("safe_indicators",[]):
                        st.markdown(
                            f'<span class="safe-flag">{f}</span>',
                            unsafe_allow_html=True)

                st.markdown("---")
                st.info(
                    f"💡 **Recommended Action:** "
                    f"{verdict_result.get('recommended_action','')}")
                if verdict_result.get("memory_note"):
                    st.markdown(
                        f"🧠 **Memory Note:** "
                        f"{verdict_result.get('memory_note')}")

        elif investigate_btn and not message.strip():
            st.warning(
                "Please paste a message to investigate. "
                "الرجاء لصق رسالة للتحليل")

    # ── TAB 2: MEMORY ─────────────────────────────────────────
    with tab2:
        st.markdown("#### Memory Database | قاعدة بيانات الذاكرة")
        st.markdown(
            '<p class="arabic-text">'
            'جميع الحالات المحفوظة في قاعدة البيانات</p>',
            unsafe_allow_html=True
        )

        all_data  = collection.get(include=["metadatas","documents"])
        metadatas = all_data["metadatas"]
        documents = all_data["documents"]
        total     = len(metadatas)

        if total == 0:
            st.info("No cases stored yet. Run an investigation first.")
        else:
            # Stats row
            scams   = sum(1 for m in metadatas
                          if m.get("is_scam") == "True")
            safe    = total - scams
            avg_conf= (sum(int(m.get("confidence",0))
                           for m in metadatas) / total
                       if total > 0 else 0)

            c1, c2, c3, c4 = st.columns(4)
            for col, label, value, color in [
                (c1, "Total Cases", total,       "#7C4DFF"),
                (c2, "Scams",       scams,        "#E53935"),
                (c3, "Safe",        safe,         "#43A047"),
                (c4, "Avg Score",   f"{avg_conf:.0f}%", "#1976D2"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div style="font-size:1.6rem; font-weight:600;
                                    color:{color};">{value}</div>
                        <div style="font-size:0.75rem;
                                    color:#888;">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Case list
            for i, (meta, doc) in enumerate(
                    zip(metadatas, documents)):
                is_scam_case = meta.get("is_scam") == "True"
                border_color = "#EF5350" if is_scam_case else "#66BB6A"
                label        = "SCAM" if is_scam_case else "SAFE"
                conf         = meta.get("confidence", 0)
                scam_type    = str(meta.get("scam_type",
                                            "unknown")).replace("_"," ").title()
                timestamp    = meta.get("timestamp","")[:10]

                with st.expander(
                    f"{'🚨' if is_scam_case else '✅'} "
                    f"{meta.get('case_id','unknown')} | "
                    f"{label} | Score: {conf}% | {scam_type}"
                ):
                    st.markdown(f"""
                    <div style="border-left:3px solid {border_color};
                                padding-left:1rem;">
                        <p><strong>Type:</strong> {scam_type}</p>
                        <p><strong>Confidence:</strong> {conf}%</p>
                        <p><strong>Date:</strong> {timestamp}</p>
                        <p><strong>Message preview:</strong><br>
                        <em>{doc[:200]}...</em></p>
                    </div>""", unsafe_allow_html=True)

    # ── TAB 3: HOW IT WORKS ───────────────────────────────────
    with tab3:
        st.markdown("#### How the System Works | كيف يعمل النظام")

        phases = [
            ("Phase 1", "Scam Classifier",
             "Single LLM agent with structured JSON output",
             "المرحلة الأولى: مصنف الاحتيال"),
            ("Phase 2", "Tool-Calling Agent",
             "5 tools: WHOIS, VirusTotal, BeautifulSoup, "
             "Email analysis, Domain mismatch",
             "المرحلة الثانية: الوكيل باستخدام الأدوات"),
            ("Phase 3", "Multi-Agent Team",
             "6 specialist agents coordinated by LangGraph — "
             "NLP, Technical, Cultural Context, Risk Scorer",
             "المرحلة الثالثة: فريق الوكلاء المتخصصين"),
            ("Phase 4", "Memory System",
             "ChromaDB vector database + sentence transformers — "
             "cosine similarity retrieval of past cases",
             "المرحلة الرابعة: نظام الذاكرة"),
            ("Phase 5", "Dashboard",
             "Streamlit web app — real-time agent progress, "
             "bilingual UI, free deployment",
             "المرحلة الخامسة: لوحة التحكم"),
        ]

        for phase, title, desc, arabic in phases:
            st.markdown(f"""
            <div class="evidence-card" style="margin:0.75rem 0;">
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div style="background:#7C4DFF; color:white;
                                padding:4px 12px; border-radius:12px;
                                font-size:0.8rem; white-space:nowrap;">
                        {phase}</div>
                    <div>
                        <strong>{title}</strong><br>
                        <span style="font-size:0.85rem;
                                     color:#666;">{desc}</span><br>
                        <span class="arabic-text">{arabic}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Built With | بُني باستخدام")
        tools = [
            ("Groq + Llama 3.3", "Free LLM API"),
            ("LangGraph",        "Agent orchestration"),
            ("ChromaDB",         "Vector database"),
            ("Sentence Transformers", "Free embeddings"),
            ("VirusTotal",       "URL security scanning"),
            ("BeautifulSoup",    "Web scraping"),
            ("Streamlit",        "Dashboard + deployment"),
        ]
        cols = st.columns(3)
        for i, (tool, desc) in enumerate(tools):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="stat-box" style="margin:0.3rem 0;">
                    <strong style="font-size:0.9rem;">{tool}</strong><br>
                    <span style="font-size:0.75rem;
                                 color:#888;">{desc}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "*Built by Sundas Mohsin Khan — UAE Data Scientist | "
            "github.com/SMKProj/agentic-antiscam-uae*")


if __name__ == "__main__":
    main()
