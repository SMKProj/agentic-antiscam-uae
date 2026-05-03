import os
import json
import time
import re
import requests
import whois
import dns.resolver
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Anti-Scam Intelligence | كاشف الاحتيال",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { text-align:center; padding:1.5rem 0 1rem; border-bottom:1px solid #e5e5e5; margin-bottom:2rem; }
    .arabic-text { font-family:'Arial',sans-serif; direction:rtl; text-align:right; color:#888; font-size:0.9rem; }
    .agent-waiting  { background:#f8f9fa; border-left:3px solid #ddd;    border-radius:6px; padding:0.6rem 1rem; margin:3px 0; font-size:0.88rem; color:#aaa; }
    .agent-running  { background:#E3F2FD; border-left:3px solid #2196F3; border-radius:6px; padding:0.6rem 1rem; margin:3px 0; font-size:0.88rem; }
    .agent-complete { background:#E8F5E9; border-left:3px solid #4CAF50; border-radius:6px; padding:0.6rem 1rem; margin:3px 0; font-size:0.88rem; }
    .verdict-scam { background:#FFEBEE; border:2px solid #EF5350; border-radius:12px; padding:1.5rem; text-align:center; }
    .verdict-safe { background:#E8F5E9; border:2px solid #66BB6A; border-radius:12px; padding:1.5rem; text-align:center; }
    .memory-card  { background:#EDE7F6; border-left:3px solid #7C4DFF; border-radius:4px; padding:0.6rem 1rem; margin:4px 0; font-size:0.83rem; }
    .red-flag     { background:#FFEBEE; color:#C62828; padding:3px 10px; border-radius:12px; font-size:0.82rem; display:inline-block; margin:3px; }
    .safe-flag    { background:#E8F5E9; color:#2E7D32; padding:3px 10px; border-radius:12px; font-size:0.82rem; display:inline-block; margin:3px; }
    .stat-box     { background:#f8f9fa; border-radius:8px; padding:1rem; text-align:center; border:1px solid #e0e0e0; }
    .evidence-card{ background:white; border:1px solid #e0e0e0; border-radius:8px; padding:1rem; margin:0.4rem 0; }
    .score-bg     { background:#e0e0e0; border-radius:10px; height:22px; width:100%; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── CREDENTIALS ───────────────────────────────────────────────
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
    VT_KEY   = st.secrets["VIRUSTOTAL_API_KEY"]
except:
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
    VT_KEY   = os.environ.get("VIRUSTOTAL_API_KEY", "")

if not GROQ_KEY:
    st.error("GROQ_API_KEY not found. Add it to Streamlit secrets.")
    st.stop()

client = Groq(api_key=GROQ_KEY)
MODEL  = "llama-3.3-70b-versatile"
MEMORY_FILE = "/tmp/scam_memory.json"


# ── LIGHTWEIGHT MEMORY SYSTEM ────────────────────────────────
# Replaces ChromaDB + sentence-transformers entirely.
# Uses sklearn TF-IDF + cosine similarity — same math,
# no Rust compilation, no GPU libraries, no heavy deps.
# You already know this from your data science background.

@st.cache_resource
def load_memory_system():

    class TFIDFMemory:
        def __init__(self, filepath):
            self.filepath   = filepath
            self.cases      = self._load()
            self.vectorizer = TfidfVectorizer(
                max_features=300,
                ngram_range=(1, 2),
                stop_words="english"
            )
            self.fitted = False
            self._refit()

        def _load(self):
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "r") as f:
                        return json.load(f)
                except:
                    return []
            return []

        def _save(self):
            with open(self.filepath, "w") as f:
                json.dump(self.cases, f)

        def _refit(self):
            if self.cases:
                corpus = [c["message"] for c in self.cases]
                self.vectorizer.fit(corpus)
                self.fitted = True
            else:
                self.fitted = False

        def count(self):
            return len(self.cases)

        def store(self, case_id, message, verdict, risks):
            self.cases = [c for c in self.cases if c["id"] != case_id]
            self.cases.append({
                "id"        : case_id,
                "message"   : message[:500],
                "is_scam"   : str(verdict.get("is_scam", False)),
                "confidence": int(verdict.get("confidence", 0)),
                "scam_type" : str(verdict.get("scam_type", "unknown")),
                "red_flags" : verdict.get("red_flags", [])[:3],
                "timestamp" : datetime.now(timezone.utc).isoformat(),
                "tech_risk" : risks.get("technical_risk", "unknown"),
                "cult_risk" : risks.get("cultural_risk",  "unknown"),
                "nlp_risk"  : risks.get("nlp_risk",       "unknown"),
            })
            self._save()
            self._refit()

        def search(self, query, n=3, threshold=0.12):
            if not self.fitted or not self.cases:
                return []
            corpus      = [c["message"] for c in self.cases]
            corpus_vecs = self.vectorizer.transform(corpus)
            query_vec   = self.vectorizer.transform([query])
            scores      = cosine_similarity(query_vec, corpus_vecs)[0]
            results = []
            for i, score in enumerate(scores):
                if score >= threshold:
                    results.append({
                        "case_id"   : self.cases[i]["id"],
                        "similarity": round(float(score), 3),
                        "message"   : self.cases[i]["message"],
                        "metadata"  : self.cases[i],
                    })
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:n]

        def get_all(self):
            return self.cases

    memory = TFIDFMemory(MEMORY_FILE)

    # Seed with known cases if empty
    if memory.count() == 0:
        seed_cases = [
            {
                "id": "seed_001",
                "message": "hralert@wadialsagroup.com ADNOC contractor "
                           "interview salary AED 18000 bring Emirates ID "
                           "pay AED 300 processing fee",
                "verdict": {"is_scam":True,"confidence":80,
                            "scam_type":"job_scam",
                            "red_flags":["ADNOC impersonation",
                                         "processing fee","unrealistic salary"]},
                "risks": {"technical_risk":"HIGH","cultural_risk":"HIGH",
                          "nlp_risk":"MEDIUM"}
            },
            {
                "id": "seed_002",
                "message": "Emirates NBD account suspended verify immediately "
                           "http://emiratesnbd-verify.tk failure 2 hours "
                           "permanent closure bank phishing",
                "verdict": {"is_scam":True,"confidence":95,
                            "scam_type":"phishing",
                            "red_flags":["suspicious URL","urgency threat",
                                         "account suspension"]},
                "risks": {"technical_risk":"VERY HIGH","cultural_risk":"HIGH",
                          "nlp_risk":"HIGH"}
            },
            {
                "id": "seed_003",
                "message": "Congratulations won AED 500000 Dubai Government "
                           "Lucky Draw send Emirates ID pay AED 250 processing "
                           "fee expires 24 hours lottery prize",
                "verdict": {"is_scam":True,"confidence":98,
                            "scam_type":"lottery_fraud",
                            "red_flags":["prize claim","processing fee",
                                         "Dubai Government impersonation"]},
                "risks": {"technical_risk":"HIGH","cultural_risk":"VERY HIGH",
                          "nlp_risk":"HIGH"}
            },
            {
                "id": "seed_004",
                "message": "ZaviyarHayat Group WAS Group FastInsu hiring "
                           "book slot first come first served "
                           "marketing@zaviyarhayatgroup.com no experience "
                           "required mass recruitment unsolicited",
                "verdict": {"is_scam":True,"confidence":80,
                            "scam_type":"job_scam",
                            "red_flags":["multiple company names",
                                         "mass recruitment","unverifiable company",
                                         "domain 64 days old"]},
                "risks": {"technical_risk":"VERY HIGH","cultural_risk":"HIGH",
                          "nlp_risk":"MEDIUM"}
            },
            {
                "id": "seed_005",
                "message": "Noon order UAE shipped expected delivery tomorrow "
                           "track customer service legitimate delivery",
                "verdict": {"is_scam":False,"confidence":95,
                            "scam_type":"not_a_scam","red_flags":[]},
                "risks": {"technical_risk":"LOW","cultural_risk":"LOW",
                          "nlp_risk":"LOW"}
            },
            {
                "id": "seed_006",
                "message": "FastInsu WAS Group walk-in interviews book slot "
                           "first come first served multiple positions "
                           "no experience recruitment spam",
                "verdict": {"is_scam":True,"confidence":75,
                            "scam_type":"job_scam",
                            "red_flags":["mass recruitment","unverifiable",
                                         "suspicious domain"]},
                "risks": {"technical_risk":"HIGH","cultural_risk":"HIGH",
                          "nlp_risk":"MEDIUM"}
            },
            {
                "id": "seed_007",
                "message": "Dear candidate open roles might interest you "
                           "visit website current openings apply directly "
                           "linkedin consulting firm professional recruiter",
                "verdict": {"is_scam":False,"confidence":88,
                            "scam_type":"not_a_scam","red_flags":[]},
                "risks": {"technical_risk":"LOW","cultural_risk":"LOW",
                          "nlp_risk":"LOW"}
            },
        ]
        for c in seed_cases:
            memory.store(c["id"], c["message"], c["verdict"], c["risks"])

    return memory


# ── TOOLS ─────────────────────────────────────────────────────
def extract_entities(message):
    emails  = re.findall(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', message)
    urls    = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', message)
    from urllib.parse import urlparse
    domains = list(set(
        [e.split("@")[1] for e in emails if "@" in e] +
        [urlparse(u).netloc for u in urls if urlparse(u).netloc]
    ))
    return {"emails": emails, "urls": urls, "domains": domains}

def whois_lookup(domain):
    try:
        w = whois.whois(domain)
        c = w.creation_date
        if isinstance(c, list): c = c[0]
        if c:
            if c.tzinfo is None: c = c.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - c).days
        else: age = None
        if age is None:  risk = "unknown"
        elif age < 30:   risk = "VERY HIGH"
        elif age < 180:  risk = "HIGH"
        elif age < 365:  risk = "MEDIUM"
        else:            risk = "LOW"
        return {"domain":domain,"age_days":age,"age_risk":risk,
                "registrar":str(w.registrar or "unknown"),
                "country":str(w.country or "unknown"),"status":"success"}
    except Exception as e:
        return {"domain":domain,"status":"failed","error":str(e),"age_risk":"unknown"}

def virustotal_scan(url):
    headers = {"x-apikey": VT_KEY}
    try:
        r = requests.post("https://www.virustotal.com/api/v3/urls",
                          headers=headers, data={"url":url}, timeout=15)
        if r.status_code != 200:
            return {"url":url,"status":"failed"}
        aid = r.json()["data"]["id"]
        time.sleep(5)
        r2  = requests.get(
            f"https://www.virustotal.com/api/v3/analyses/{aid}",
            headers=headers, timeout=15)
        stats = r2.json()["data"]["attributes"]["stats"]
        mal   = stats.get("malicious", 0)
        total = sum(stats.values())
        risk  = ("VERY HIGH" if mal>=5 else "HIGH" if mal>=2 else
                 "MEDIUM" if mal>=1 else "LOW")
        return {"url":url,"malicious":mal,"total_engines":total,
                "vt_risk":risk,"status":"success"}
    except Exception as e:
        return {"url":url,"status":"failed","error":str(e)}

def scrape_website(url):
    try:
        r    = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script","style","nav","footer"]): t.decompose()
        text = " ".join(soup.get_text().split())[:2000].lower()
        flags = {
            "login_form"       : bool(soup.find("form")),
            "urgency_language" : any(w in text for w in
                                     ["urgent","expires","act now","immediately"]),
            "prize_language"   : any(w in text for w in
                                     ["congratulations","winner","prize"]),
            "financial_request": any(w in text for w in
                                     ["bank account","wire transfer","processing fee"]),
        }
        found = [k for k,v in flags.items() if v]
        return {"url":url,"flags_found":found,
                "flags_count":len(found),"status":"success"}
    except requests.exceptions.ConnectionError:
        return {"url":url,"status":"unreachable",
                "note":"Website unreachable — strong scam signal"}
    except Exception as e:
        return {"url":url,"status":"failed","error":str(e)}

def analyze_email(email):
    if "@" not in email:
        return {"email":email,"status":"invalid"}
    prefix, domain = email.split("@", 1)
    free = ["gmail.com","yahoo.com","hotmail.com","outlook.com"]
    sus  = ["hralert","hr-alert","noreply-hr","jobs-alert",
            "marketing","alert-team","recruitment-alert"]
    is_free    = domain.lower() in free
    sus_prefix = any(p in prefix.lower() for p in sus)
    try:
        dns.resolver.resolve(domain, "MX")
        has_mx = True
    except: has_mx = False
    signals = []
    if is_free:    signals.append("free provider")
    if sus_prefix: signals.append(f"suspicious prefix: {prefix}")
    if not has_mx: signals.append("no MX records")
    risk = "HIGH" if len(signals)>=2 else "MEDIUM" if signals else "LOW"
    return {"email":email,"domain":domain,"prefix":prefix,
            "is_free_provider":is_free,"suspicious_prefix":sus_prefix,
            "has_mx_records":has_mx,"risk_signals":signals,
            "email_risk":risk,"status":"success"}

def company_existence(company, domain):
    signals = {"existence_signals":[],"absence_signals":[],"existence_risk":"LOW"}
    try:
        r    = requests.get(f"https://{domain}",
                            headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
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
        words = [w for w in company.lower().split() if len(w)>4]
        if any(w in text for w in words):
            signals["existence_signals"].append("name found on site")
        else:
            signals["absence_signals"].append("name not on site")
    except requests.exceptions.ConnectionError:
        signals["absence_signals"].append("website unreachable")
    except:
        signals["absence_signals"].append("check failed")
    absent      = len(signals["absence_signals"])
    unreachable = any("unreachable" in s for s in signals["absence_signals"])
    if unreachable and absent>=2:  signals["existence_risk"] = "HIGH"
    elif unreachable or absent>=3: signals["existence_risk"] = "HIGH"
    elif absent>=2:                signals["existence_risk"] = "MEDIUM"
    return signals


# ── LLM HELPERS ───────────────────────────────────────────────
def call_llm(system_prompt, user_message):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_message}],
        temperature=0.1, max_tokens=1200)
    return r.choices[0].message.content.strip()

def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try: return json.loads(raw.strip())
    except: return {}


# ── MEMORY BOOST ──────────────────────────────────────────────
def calc_boost(similar, base_score):
    if base_score < 45 or not similar:
        return {"boosted_score":base_score,"boost_applied":0,"boost_reason":[]}
    boost = 0
    reasons = []
    scams = [c for c in similar if c["metadata"].get("is_scam")=="True"]
    safe  = [c for c in similar if c["metadata"].get("is_scam")=="False"]
    for c in scams:
        if c["similarity"] >= 0.45:
            boost += 10
            reasons.append(f"Very similar past scam ({c['similarity']:.0%})")
        elif c["similarity"] >= 0.25:
            boost += 5
            reasons.append(f"Similar past scam ({c['similarity']:.0%})")
    if len(scams) >= 2:
        boost += 5
        reasons.append(f"{len(scams)} similar scam patterns in memory")
    for c in safe:
        if c["similarity"] >= 0.45:
            boost -= 5
            reasons.append(f"Similar safe case ({c['similarity']:.0%})")
    boost = max(-10, min(20, boost))
    return {
        "boosted_score": min(100, max(0, base_score+boost)),
        "boost_applied": boost,
        "boost_reason" : reasons,
    }


# ── MAIN ──────────────────────────────────────────────────────
def main():
    memory = load_memory_system()

    # HEADER
    st.markdown("""
    <div class="main-header">
        <h1 style="font-size:2rem;font-weight:600;margin:0;">
            🛡️ UAE Anti-Scam Intelligence System</h1>
        <p class="arabic-text">نظام الكشف عن الاحتيال الإماراتي</p>
        <p style="color:#888;font-size:0.88rem;margin-top:0.4rem;">
            6 AI agents · Real-time investigation · Memory-augmented detection
        </p>
    </div>""", unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.markdown("### 📊 System Status")
        all_cases = memory.get_all()
        total     = len(all_cases)
        scam_cnt  = sum(1 for c in all_cases if c.get("is_scam")=="True")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="stat-box">
                <div style="font-size:1.5rem;font-weight:600;color:#7C4DFF;">{total}</div>
                <div style="font-size:0.75rem;color:#888;">Cases in memory</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="stat-box">
                <div style="font-size:1.5rem;font-weight:600;color:#E53935;">{scam_cnt}</div>
                <div style="font-size:0.75rem;color:#888;">Scams stored</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🤖 Agent Pipeline")
        for num, name, arabic in [
            ("0","Memory Retrieval","نظام الذاكرة"),
            ("1","Orchestrator",    "المنسق"),
            ("2","NLP Specialist",  "تحليل النص"),
            ("3","Technical",       "التحقق التقني"),
            ("4","Cultural Context","السياق الثقافي"),
            ("5","Risk Scorer",     "تقييم المخاطر"),
            ("6","Final Verdict",   "الحكم النهائي"),
        ]:
            st.markdown(
                f'<div class="agent-waiting">Agent {num}: {name} '
                f'<span style="color:#bbb;font-size:0.78rem;">— {arabic}</span>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""### ℹ️ About
Built by **Sundas Mohsin Khan**
UAE Data Scientist

[GitHub ↗](https://github.com/SMKProj/agentic-antiscam-uae)

*5 phases · 6 agents · UAE-specific detection*""")

    # TABS
    tab1, tab2, tab3 = st.tabs([
        "🔍 Investigate | تحقق",
        "🧠 Memory | الذاكرة",
        "📖 How It Works | كيف يعمل"
    ])

    # ══════ TAB 1: INVESTIGATE ══════
    with tab1:
        st.markdown("#### Paste a suspicious message | الصق الرسالة المشبوهة")

        with st.expander("📋 Load a sample message for quick testing"):
            sample = st.selectbox("Choose a sample:", [
                "Select...",
                "UAE job scam (ADNOC impersonation)",
                "Bank phishing (Emirates NBD)",
                "Lottery fraud (Dubai Government)",
                "Legitimate message (Noon delivery)",
            ])
            samples = {
                "UAE job scam (ADNOC impersonation)":
                    "From: hralert@wadialsagroup.com\n"
                    "Subject: Urgent Interview — ADNOC Contractor\n\n"
                    "Dear Candidate, You have been selected for an interview "
                    "at ADNOC contractor Wadi Al Salam Group. "
                    "Salary AED 18,000/month tax free. Interview Monday 10am. "
                    "Bring Emirates ID and pay AED 300 processing fee.\nHR Dept",
                "Bank phishing (Emirates NBD)":
                    "Dear Emirates NBD Customer, your account requires immediate "
                    "verification. Click: http://emiratesnbd-secure-login.tk/verify "
                    "Failure to verify within 2 hours will result in suspension.",
                "Lottery fraud (Dubai Government)":
                    "Congratulations! You have won AED 500,000 in the Dubai "
                    "Government Lucky Draw. Send your Emirates ID and pay "
                    "AED 250 processing fee. Offer expires in 24 hours!",
                "Legitimate message (Noon delivery)":
                    "Hi, your Noon order #UAE-2847361 has been shipped. "
                    "Expected delivery: tomorrow between 2-6pm. "
                    "Track: https://noon.com/uae/track/2847361",
            }
            if sample != "Select...":
                st.session_state["sample_msg"] = samples.get(sample, "")

        message = st.text_area(
            "Message | الرسالة",
            value=st.session_state.get("sample_msg", ""),
            height=180,
            placeholder="Paste any suspicious SMS, email, or WhatsApp message here...\n"
                         "الصق أي رسالة مشبوهة هنا..."
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            run_btn = st.button("🔍 Investigate | تحقق",
                                type="primary", use_container_width=True)
        with col2:
            store_cb = st.checkbox("Save to memory", value=True)

        if run_btn and message.strip():
            st.markdown("---")
            st.markdown("### 🔄 Live Agent Activity | نشاط الوكلاء")

            keys   = ["mem","orc","nlp","tech","cult","risk","verd"]
            labels = [
                ("Agent 0: Memory Retrieval","نظام الذاكرة"),
                ("Agent 1: Orchestrator",    "المنسق"),
                ("Agent 2: NLP Specialist",  "تحليل النص"),
                ("Agent 3: Technical",       "التحقق التقني"),
                ("Agent 4: Cultural Context","السياق الثقافي"),
                ("Agent 5: Risk Scorer",     "تقييم المخاطر"),
                ("Agent 6: Final Verdict",   "الحكم النهائي"),
            ]
            ph = {k: st.empty() for k in keys}
            for k, (name, arabic) in zip(keys, labels):
                ph[k].markdown(
                    f'<div class="agent-waiting">⏳ {name} — {arabic}</div>',
                    unsafe_allow_html=True)

            state = {}

            # AGENT 0: Memory
            ph["mem"].markdown(
                '<div class="agent-running">🔄 Agent 0: Memory Retrieval — '
                'searching past cases...</div>', unsafe_allow_html=True)
            similar = memory.search(message, n=3, threshold=0.12)
            state["similar"] = similar
            ph["mem"].markdown(
                f'<div class="agent-complete">✅ Agent 0: Memory Retrieval — '
                f'{len(similar)} similar case(s) found</div>',
                unsafe_allow_html=True)

            # AGENT 1: Orchestrator
            ph["orc"].markdown(
                '<div class="agent-running">🔄 Agent 1: Orchestrator — '
                'extracting entities...</div>', unsafe_allow_html=True)
            entities = extract_entities(message)
            mem_ctx  = ""
            if similar:
                lines = [f"MEMORY: {len(similar)} similar past cases:"]
                for i, c in enumerate(similar, 1):
                    m = c["metadata"]
                    lines.append(
                        f"Case #{i} ({c['similarity']:.0%} similar): "
                        f"{'SCAM' if m['is_scam']=='True' else 'SAFE'} "
                        f"({m['confidence']}%)")
                mem_ctx = "\n".join(lines)
            orc_prompt = f"""
You are the lead fraud investigator.
{f'MEMORY CONTEXT:{chr(10)}{mem_ctx}{chr(10)}' if mem_ctx else ''}
IDENTITY CHECK:
1. Multiple company names in one sender → SUSPICIOUS
2. Email domain not matching claimed company → MISMATCH
3. Book a slot / first come first served → mass spam
Return ONLY JSON:
{{"claimed_identity":"...","requested_action":"...",
  "identity_flags":[],"identity_risk":"HIGH/MEDIUM/LOW",
  "memory_influenced":true,"priority":"HIGH/MEDIUM/LOW"}}"""
            orc_res = parse_json(call_llm(orc_prompt,
                                          f"Analyze:\n\n{message}"))
            state["entities"] = {
                **entities,
                "claimed_identity": orc_res.get("claimed_identity",""),
                "requested_action": orc_res.get("requested_action",""),
                "identity_flags"  : orc_res.get("identity_flags",[]),
                "identity_risk"   : orc_res.get("identity_risk","LOW"),
            }
            ph["orc"].markdown(
                f'<div class="agent-complete">✅ Agent 1: Orchestrator — '
                f'{len(entities["emails"])} email(s), '
                f'{len(entities["urls"])} URL(s) found</div>',
                unsafe_allow_html=True)

            # AGENT 2: NLP
            ph["nlp"].markdown(
                '<div class="agent-running">🔄 Agent 2: NLP Specialist — '
                'analyzing language patterns...</div>', unsafe_allow_html=True)
            nlp_prompt = """
Analyze ONLY the text. Look for:
urgency, threats, authority impersonation, reward claims,
info harvesting, generic greetings, grammar issues,
mass recruitment (book a slot, first come first served).
Return ONLY JSON:
{"urgency_found":false,"threat_found":false,"authority_claim":false,
 "reward_claim":false,"info_harvesting":false,"generic_greeting":false,
 "grammar_issues":false,"mass_recruitment":false,
 "key_phrases":[],"sentiment":"neutral","manipulation_score":0,
 "nlp_risk":"LOW","nlp_summary":"..."}"""
            nlp_res = parse_json(call_llm(nlp_prompt,
                                          f"Analyze:\n\n{message}"))
            sigs = sum([bool(nlp_res.get("urgency_found")),
                        bool(nlp_res.get("threat_found")),
                        bool(nlp_res.get("authority_claim")),
                        bool(nlp_res.get("reward_claim")),
                        bool(nlp_res.get("info_harvesting")),
                        bool(nlp_res.get("mass_recruitment")),
                        bool(nlp_res.get("generic_greeting"))])
            nlp_res["nlp_risk"]     = ("HIGH" if sigs>=4 else
                                       "MEDIUM" if sigs>=2 else "LOW")
            nlp_res["signal_count"] = sigs
            state["nlp"] = nlp_res
            ph["nlp"].markdown(
                f'<div class="agent-complete">✅ Agent 2: NLP Specialist — '
                f'{sigs} signals, risk: {nlp_res["nlp_risk"]}</div>',
                unsafe_allow_html=True)

            # AGENT 3: Technical
            ph["tech"].markdown(
                '<div class="agent-running">🔄 Agent 3: Technical — '
                'WHOIS, VirusTotal, scraper... (1-2 min)</div>',
                unsafe_allow_html=True)
            ents        = state["entities"]
            all_domains = ents.get("domains", [])
            ev = {"whois":[],"emails":[],"vt":[],"scrape":[],
                  "company":{},"tech_risk":"LOW","high_signals":0}

            for d in all_domains:
                ev["whois"].append(whois_lookup(d))
                time.sleep(1)
            for e in ents.get("emails", []):
                ev["emails"].append(analyze_email(e))
            claimed = ents.get("claimed_identity","")
            if claimed and all_domains:
                ev["company"] = company_existence(claimed, all_domains[0])
            for u in ents.get("urls", [])[:2]:
                ev["vt"].append(virustotal_scan(u))
                time.sleep(15)
            for u in ents.get("urls", [])[:1]:
                ev["scrape"].append(scrape_website(u))

            h = 0
            for w in ev["whois"]:
                if w.get("age_risk") in ["VERY HIGH","HIGH"]: h+=1
            for e in ev["emails"]:
                if e.get("email_risk")=="HIGH": h+=1
            for v in ev["vt"]:
                if v.get("vt_risk") in ["VERY HIGH","HIGH"]: h+=2
            for s in ev["scrape"]:
                if s.get("flags_count",0)>=2: h+=1
            cr = ev["company"].get("existence_risk","LOW")
            if cr=="HIGH": h+=2
            elif cr=="MEDIUM": h+=1
            if ents.get("identity_risk")=="HIGH": h+=1
            if (not ents.get("urls") and
                    nlp_res.get("nlp_risk") in ["HIGH","VERY HIGH"]):
                h+=2
            ev["tech_risk"]     = ("VERY HIGH" if h>=4 else "HIGH" if h>=3 else
                                   "MEDIUM" if h>=2 else
                                   "LOW-MEDIUM" if h>=1 else "LOW")
            ev["high_signals"]  = h
            state["tech"] = ev
            ph["tech"].markdown(
                f'<div class="agent-complete">✅ Agent 3: Technical — '
                f'{h} high-risk signals, {ev["tech_risk"]}</div>',
                unsafe_allow_html=True)

            # AGENT 4: Cultural
            ph["cult"].markdown(
                '<div class="agent-running">🔄 Agent 4: Cultural Context — '
                'checking UAE patterns...</div>', unsafe_allow_html=True)
            cult_prompt = """
UAE fraud specialist. Detect:
- Fake ADNOC/Emirates/Etisalat/bank impersonation
- Unrealistic salaries AED 15,000+ entry level
- Upfront fees before interviews
- Unsolicited recruitment (3+ flags = HIGH):
  unsolicited contact, multiple company names,
  book a slot / first come first served,
  no job requirements, no CV reference
Return ONLY JSON:
{"uae_entity_impersonated":"null","known_uae_scam_pattern":false,
 "unrealistic_offer":false,"unsolicited_recruitment":false,
 "multiple_company_names":false,"mass_recruitment_language":false,
 "cultural_red_flags":[],"cultural_risk":"LOW","cultural_summary":"..."}"""
            cult_res = parse_json(call_llm(cult_prompt,
                                           f"Analyze:\n\n{message}"))
            state["cultural"] = cult_res
            ph["cult"].markdown(
                f'<div class="agent-complete">✅ Agent 4: Cultural — '
                f'risk: {cult_res.get("cultural_risk","LOW")}, '
                f'{len(cult_res.get("cultural_red_flags",[]))} flags</div>',
                unsafe_allow_html=True)

            # AGENT 5: Risk Scorer
            ph["risk"].markdown(
                '<div class="agent-running">🔄 Agent 5: Risk Scorer — '
                'calculating weighted score...</div>', unsafe_allow_html=True)

            def r2s(r):
                return {"VERY HIGH":95,"HIGH":75,"LOW-MEDIUM":55,
                        "MEDIUM":50,"LOW":15,"unknown":30}.get(r, 30)

            tech_r = ev["tech_risk"]
            cult_r = cult_res.get("cultural_risk","LOW")
            nlp_r  = nlp_res.get("nlp_risk","LOW")
            has_tech = bool(ev.get("whois") or ev.get("emails"))

            if has_tech:
                w = {"t":0.50,"c":0.30,"n":0.20}
            else:
                w = {"t":0.20,"c":0.50,"n":0.30}

            ts   = r2s(tech_r)
            cs   = r2s(cult_r)
            ns   = r2s(nlp_r)
            base = int(ts*w["t"] + cs*w["c"] + ns*w["n"])
            mb   = calc_boost(similar, base)
            final= mb["boosted_score"]
            conf = ("VERY HIGH" if final>=85 else "HIGH" if final>=65 else
                    "MEDIUM" if final>=45 else "LOW")

            state["score"] = {
                "base":base,"final":final,"confidence":conf,
                "is_scam":final>=55,
                "breakdown":{"tech":ts,"cultural":cs,"nlp":ns},
                "risks":{"tech":tech_r,"cultural":cult_r,"nlp":nlp_r},
                "weights":{"t":int(w["t"]*100),"c":int(w["c"]*100),
                           "n":int(w["n"]*100)},
            }
            state["boost"] = mb
            ph["risk"].markdown(
                f'<div class="agent-complete">✅ Agent 5: Risk Scorer — '
                f'base:{base} boost:{mb["boost_applied"]:+d} '
                f'final:{final}/100</div>', unsafe_allow_html=True)

            # AGENT 6: Verdict
            ph["verd"].markdown(
                '<div class="agent-running">🔄 Agent 6: Final Verdict — '
                'synthesizing report...</div>', unsafe_allow_html=True)
            verd_prompt = """
Lead fraud investigator. Synthesize all findings.
Return ONLY JSON:
{"is_scam":false,"confidence":0,
 "scam_type":"not_a_scam",
 "verdict_summary":"2-3 sentences plain English",
 "key_evidence":[],"red_flags":[],"safe_indicators":[],
 "recommended_action":"clear action",
 "memory_note":"how past cases influenced verdict or null"}"""
            all_f = {
                "message" : message[:300],
                "entities": state["entities"],
                "nlp"     : state["nlp"],
                "tech"    : state["tech"],
                "cultural": state["cultural"],
                "score"   : state["score"],
                "memory"  : len(similar),
            }
            verd_res = parse_json(
                call_llm(verd_prompt, json.dumps(all_f, default=str)))
            verd_res["confidence"] = final
            verd_res["is_scam"]    = final >= 55
            state["verdict"]       = verd_res

            ph["verd"].markdown(
                '<div class="agent-complete">✅ Agent 6: Final Verdict — '
                'complete</div>', unsafe_allow_html=True)

            # Store result
            if store_cb:
                cid = f"case_{int(time.time())}"
                memory.store(cid, message, verd_res,
                             {"technical_risk":tech_r,
                              "cultural_risk":cult_r,"nlp_risk":nlp_r})

            # DISPLAY VERDICT
            st.markdown("---")
            st.markdown("### 📋 Investigation Report | تقرير التحقيق")

            is_scam = verd_res.get("is_scam", False)
            if is_scam:
                st.markdown(f"""
                <div class="verdict-scam">
                    <h2 style="color:#C62828;margin:0;">
                        🚨 SCAM DETECTED | تم اكتشاف احتيال</h2>
                    <p style="font-size:1.05rem;margin:0.5rem 0;">
                        Risk Score: {final}/100</p>
                    <p style="color:#C62828;margin:0;">
                        {verd_res.get("scam_type","").replace("_"," ").title()}
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-safe">
                    <h2 style="color:#2E7D32;margin:0;">
                        ✅ LIKELY SAFE | يبدو آمناً</h2>
                    <p style="font-size:1.05rem;margin:0.5rem 0;">
                        Risk Score: {final}/100</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_s, col_m = st.columns(2)

            with col_s:
                st.markdown("##### Score Breakdown | تفاصيل النتيجة")
                bd = state["score"]["breakdown"]
                wt = state["score"]["weights"]
                for label, score in [
                    (f"Technical ({wt['t']}%)", bd["tech"]),
                    (f"Cultural  ({wt['c']}%)", bd["cultural"]),
                    (f"NLP       ({wt['n']}%)", bd["nlp"]),
                ]:
                    color = ("#E53935" if score>=75 else
                             "#FB8C00" if score>=50 else "#43A047")
                    st.markdown(f"""
                    <div style="margin:0.4rem 0;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:0.83rem;margin-bottom:2px;">
                            <span>{label}</span>
                            <span style="font-weight:500;">{score}</span>
                        </div>
                        <div class="score-bg">
                            <div style="background:{color};border-radius:10px;
                                        height:22px;width:{score}%;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:#f5f5f5;border-radius:8px;
                            padding:0.5rem 0.75rem;font-size:0.83rem;margin-top:0.4rem;">
                    Base: {base} | Memory: {mb['boost_applied']:+d} | Final: {final}
                </div>""", unsafe_allow_html=True)

            with col_m:
                st.markdown("##### Memory Matches | تطابق الذاكرة")
                if similar:
                    for c in similar:
                        meta = c["metadata"]
                        isc  = meta.get("is_scam")=="True"
                        bg   = "#FFEBEE" if isc else "#E8F5E9"
                        st.markdown(f"""
                        <div class="memory-card" style="background:{bg};">
                            <strong>{c['case_id']}</strong><br>
                            Similarity: {c['similarity']:.0%} |
                            {'SCAM' if isc else 'SAFE'} |
                            Score: {meta.get('confidence')}%
                        </div>""", unsafe_allow_html=True)
                    for r in mb.get("boost_reason",[]):
                        st.markdown(f"📈 *{r}*")
                else:
                    st.info("No similar past cases found")

            # Evidence tabs
            st.markdown("##### Evidence Details | تفاصيل الأدلة")
            et1, et2, et3, et4 = st.tabs([
                "🔬 Technical","💬 NLP","🇦🇪 Cultural","📝 Summary"])

            with et1:
                for wr in ev.get("whois",[]):
                    rc = ("#E53935" if wr.get("age_risk") in
                          ["VERY HIGH","HIGH"] else "#43A047")
                    st.markdown(f"""
                    <div class="evidence-card">
                        <strong>WHOIS: {wr.get('domain')}</strong><br>
                        Age: {wr.get('age_days')} days |
                        <span style="color:{rc};">{wr.get('age_risk')}</span> |
                        Registrar: {wr.get('registrar','?')} |
                        Country: {wr.get('country','?')}
                    </div>""", unsafe_allow_html=True)
                for e in ev.get("emails",[]):
                    st.markdown(f"""
                    <div class="evidence-card">
                        <strong>Email: {e.get('email')}</strong><br>
                        Risk: {e.get('email_risk')} |
                        Signals: {', '.join(e.get('risk_signals',[])) or 'none'}
                    </div>""", unsafe_allow_html=True)
                comp = ev.get("company",{})
                if comp:
                    st.markdown(f"""
                    <div class="evidence-card">
                        <strong>Company existence</strong><br>
                        Risk: {comp.get('existence_risk')} |
                        Issues: {', '.join(comp.get('absence_signals',[]))}
                    </div>""", unsafe_allow_html=True)

            with et2:
                nlp = state["nlp"]
                found_f = [f for f,k in [
                    ("Urgency language","urgency_found"),
                    ("Threat language","threat_found"),
                    ("Authority claim","authority_claim"),
                    ("Reward claim","reward_claim"),
                    ("Info harvesting","info_harvesting"),
                    ("Mass recruitment","mass_recruitment"),
                    ("Generic greeting","generic_greeting"),
                ] if nlp.get(k)]
                st.markdown(
                    f"**NLP Risk:** {nlp.get('nlp_risk')} | "
                    f"**Signals:** {nlp.get('signal_count',0)}")
                st.markdown(
                    f"**Key phrases:** {', '.join(nlp.get('key_phrases',[]))}")
                if found_f:
                    st.markdown("".join(
                        [f'<span class="red-flag">{f}</span>' for f in found_f]),
                        unsafe_allow_html=True)

            with et3:
                cult = state["cultural"]
                st.markdown(
                    f"**Cultural Risk:** {cult.get('cultural_risk','LOW')}")
                if cult.get("uae_entity_impersonated") and \
                   cult.get("uae_entity_impersonated") != "null":
                    st.warning(
                        f"UAE Entity Impersonated: "
                        f"{cult.get('uae_entity_impersonated')}")
                flags = cult.get("cultural_red_flags",[])
                if flags:
                    st.markdown("".join(
                        [f'<span class="red-flag">{f}</span>' for f in flags]),
                        unsafe_allow_html=True)

            with et4:
                verd = state["verdict"]
                st.markdown(f"**Summary:** {verd.get('verdict_summary','')}")
                st.markdown("---")
                cr, cs2 = st.columns(2)
                with cr:
                    st.markdown("**🚩 Red Flags:**")
                    for f in verd.get("red_flags",[]):
                        st.markdown(f'<span class="red-flag">{f}</span>',
                                    unsafe_allow_html=True)
                with cs2:
                    st.markdown("**✅ Safe Indicators:**")
                    for f in verd.get("safe_indicators",[]):
                        st.markdown(f'<span class="safe-flag">{f}</span>',
                                    unsafe_allow_html=True)
                st.markdown("---")
                st.info(f"💡 **Action:** {verd.get('recommended_action','')}")
                if verd.get("memory_note"):
                    st.markdown(f"🧠 **Memory:** {verd.get('memory_note')}")

        elif run_btn and not message.strip():
            st.warning("Please paste a message. الرجاء لصق رسالة للتحليل")

    # ══════ TAB 2: MEMORY ══════
    with tab2:
        st.markdown("#### Memory Database | قاعدة بيانات الذاكرة")
        cases = memory.get_all()
        total = len(cases)
        if total == 0:
            st.info("No cases stored yet. Run an investigation first.")
        else:
            scams = sum(1 for c in cases if c.get("is_scam")=="True")
            safe  = total - scams
            avg_c = sum(int(c.get("confidence",0)) for c in cases)/total
            c1,c2,c3,c4 = st.columns(4)
            for col,label,val,color in [
                (c1,"Total",total,"#7C4DFF"),
                (c2,"Scams",scams,"#E53935"),
                (c3,"Safe",safe,"#43A047"),
                (c4,"Avg Score",f"{avg_c:.0f}%","#1976D2"),
            ]:
                with col:
                    st.markdown(f"""<div class="stat-box">
                        <div style="font-size:1.5rem;font-weight:600;color:{color};">{val}</div>
                        <div style="font-size:0.75rem;color:#888;">{label}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("---")
            for case in reversed(cases):
                isc   = case.get("is_scam")=="True"
                icon  = "🚨" if isc else "✅"
                label = "SCAM" if isc else "SAFE"
                with st.expander(
                    f"{icon} {case.get('id','?')} | {label} | "
                    f"Score: {case.get('confidence')}% | "
                    f"{str(case.get('scam_type','?')).replace('_',' ').title()}"
                ):
                    st.markdown(f"""
                    <div style="border-left:3px solid {'#EF5350' if isc else '#66BB6A'};
                                padding-left:1rem;">
                        <p><strong>Confidence:</strong> {case.get('confidence')}%</p>
                        <p><strong>Date:</strong> {case.get('timestamp','')[:10]}</p>
                        <p><strong>Preview:</strong><br>
                        <em>{case.get('message','')[:200]}...</em></p>
                    </div>""", unsafe_allow_html=True)

    # ══════ TAB 3: HOW IT WORKS ══════
    with tab3:
        st.markdown("#### How the System Works | كيف يعمل النظام")
        for phase,title,desc,arabic in [
            ("Phase 1","Scam Classifier",
             "Single LLM agent with structured JSON output",
             "المرحلة الأولى: مصنف الاحتيال"),
            ("Phase 2","Tool-Calling Agent",
             "5 tools: WHOIS · VirusTotal · BeautifulSoup · Email · Domain mismatch",
             "المرحلة الثانية: الوكيل باستخدام الأدوات"),
            ("Phase 3","Multi-Agent Team",
             "6 specialist agents — NLP · Technical · Cultural · Risk Scorer",
             "المرحلة الثالثة: فريق الوكلاء المتخصصين"),
            ("Phase 4","Memory System",
             "TF-IDF cosine similarity store — lightweight, no heavy dependencies",
             "المرحلة الرابعة: نظام الذاكرة"),
            ("Phase 5","Dashboard",
             "Streamlit — bilingual UI · real-time agents · free deployment",
             "المرحلة الخامسة: لوحة التحكم"),
        ]:
            st.markdown(f"""
            <div class="evidence-card" style="margin:0.6rem 0;">
                <div style="display:flex;align-items:center;gap:1rem;">
                    <div style="background:#7C4DFF;color:white;padding:3px 10px;
                                border-radius:12px;font-size:0.8rem;white-space:nowrap;">
                        {phase}</div>
                    <div>
                        <strong>{title}</strong><br>
                        <span style="font-size:0.83rem;color:#666;">{desc}</span><br>
                        <span class="arabic-text">{arabic}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Tech Stack | التقنيات المستخدمة")
        tools = [
            ("Groq + Llama 3.3","Free LLM API"),
            ("LangGraph","Agent orchestration"),
            ("TF-IDF + Cosine","Lightweight memory"),
            ("VirusTotal","URL security"),
            ("BeautifulSoup","Web scraping"),
            ("Streamlit","Dashboard & deploy"),
        ]
        cols = st.columns(3)
        for i,(tool,desc) in enumerate(tools):
            with cols[i%3]:
                st.markdown(f"""
                <div class="stat-box" style="margin:0.3rem 0;">
                    <strong style="font-size:0.88rem;">{tool}</strong><br>
                    <span style="font-size:0.75rem;color:#888;">{desc}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "*Built by Sundas Mohsin Khan — UAE Data Scientist | "
            "[github.com/SMKProj](https://github.com/SMKProj)*")


if __name__ == "__main__":
    main()
