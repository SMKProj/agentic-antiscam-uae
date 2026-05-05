# ============================================================
# UAE Anti-Scam Intelligence System — Phase 5 Dashboard
# Author: Sundus Mohsin Khan | github.com/SMKProj
# ============================================================
import os, json, time, re
import numpy as np
import requests
import whois
import dns.resolver
import streamlit as st
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── CREDENTIALS (module level — available everywhere) ─────────
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
    VT_KEY   = st.secrets["VIRUSTOTAL_API_KEY"]
except Exception:
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
    VT_KEY   = os.environ.get("VIRUSTOTAL_API_KEY", "")

if not GROQ_KEY:
    st.error("GROQ_API_KEY not found. Add it to Streamlit secrets.")
    st.stop()

groq_client = Groq(api_key=GROQ_KEY)

st.set_page_config(
    page_title="UAE Anti-Scam | كاشف الاحتيال",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
* { box-sizing: border-box; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 860px !important;
    background: #f7f7f5;
}

/* ── Header card ── */
.hdr {
    background: #ffffff;
    border: 1px solid #e4e2dc;
    border-radius: 12px;
    padding: 1.2rem 1.5rem 1rem;
    text-align: center;
    margin-bottom: 10px;
}
.hdr-title { font-size: 22px; font-weight: 600; color: #1a1a1a; }
.hdr-ar    { font-size: 13px; color: #888; direction: rtl; margin-top: 3px; }
.hdr-sub   { font-size: 12px; color: #aaa; margin-top: 4px; }

/* ── Stats row ── */
.stats-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    border: 1px solid #e4e2dc;
    border-radius: 12px;
    overflow: hidden;
    background: #ffffff;
    margin-bottom: 10px;
}
.stat { padding: 12px; text-align: center; }
.stat + .stat { border-left: 1px solid #e4e2dc; }
.stat-n { font-size: 22px; font-weight: 600; }
.stat-l { font-size: 11px; color: #aaa; margin-top: 2px; }

/* ── Input card ── */
.input-label-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    background: #ffffff;
    border: 1px solid #e4e2dc;
    border-radius: 12px 12px 0 0;
    padding: 10px 14px 6px;
    border-bottom: none;
}
.input-label    { font-size: 13px; font-weight: 500; color: #444; }
.input-label-ar { font-size: 11px; color: #bbb; direction: rtl; }

/* Override Streamlit textarea */
div[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1px solid #e4e2dc !important;
    border-radius: 0 0 0 0 !important;
    border-top: none !important;
    font-size: 13px !important;
    color: #555 !important;
    resize: none !important;
}

/* ── Verdict cards ── */
.verdict-scam {
    background: #fff5f5;
    border: 1px solid #f5a0a0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    height: 100%;
}
.verdict-safe {
    background: #f4fbf6;
    border: 1px solid #8fd4a8;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    height: 100%;
}
.v-icon  { font-size: 20px; margin-bottom: 5px; }
.v-title { font-size: 15px; font-weight: 600; margin-bottom: 2px; }
.v-ar    { font-size: 11px; color: #888; direction: rtl; margin-bottom: 6px; }
.v-score { font-size: 11px; color: #888; margin-bottom: 6px; }
.pbar-bg   { background: #e8e8e8; border-radius: 20px; height: 7px; }
.pbar-fill { border-radius: 20px; height: 7px; }

/* ── Action card ── */
.action-card {
    background: #ffffff;
    border: 1px solid #e4e2dc;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 12px;
    color: #555;
    line-height: 1.7;
    height: 100%;
}
.action-title { font-size: 12px; font-weight: 600; color: #333; margin-bottom: 6px; }
.action-ar    { font-size: 11px; color: #bbb; direction: rtl; margin-top: 6px; }

/* ── Analysis panels ── */
.panel {
    background: #ffffff;
    border: 1px solid #e4e2dc;
    border-radius: 12px;
    overflow: hidden;
}
.panel-hdr {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 14px;
    border-bottom: 1px solid #e4e2dc;
    font-size: 12px;
    font-weight: 600;
    color: #333;
    background: #fafaf8;
}
.panel-hdr-ar { font-size: 10px; color: #bbb; direction: rtl; }
.panel-body   { padding: 6px 14px 8px; }
.ev-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #f0ede8;
    font-size: 12px;
}
.ev-row:last-child { border-bottom: none; }
.ev-key { color: #777; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 500;
    white-space: nowrap;
}
.b-red    { background: #FEECEC; color: #B02020; }
.b-amber  { background: #FEF3E2; color: #8A5000; }
.b-green  { background: #EDFBF1; color: #1A6B35; }
.b-purple { background: #F0EEFF; color: #4B3BAA; }
.b-gray   { background: #F2F2F2; color: #666; }

/* ── Flag pills row ── */
.flags {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    padding: 8px 14px;
    border-top: 1px solid #f0ede8;
    background: #fafaf8;
}

/* ── Agent strip ── */
.agents-wrap {
    background: #ffffff;
    border: 1px solid #e4e2dc;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 10px;
}
.agents-hdr {
    padding: 9px 14px;
    border-bottom: 1px solid #e4e2dc;
    font-size: 12px;
    font-weight: 600;
    color: #555;
    background: #fafaf8;
}
.agents-grid { display: grid; grid-template-columns: repeat(7,1fr); }
.ag {
    padding: 8px 6px;
    text-align: center;
    font-size: 10px;
    border-right: 1px solid #e4e2dc;
    line-height: 1.45;
}
.ag:last-child { border-right: none; }
.ag-done { background: #edfbf1; color: #1a6b35; }
.ag-run  { background: #e8f4ff; color: #1a5fa8; }
.ag-wait { background: #f7f7f5; color: #bbb; }
.ag-name { font-weight: 600; margin-bottom: 2px; }
.ag-status { font-size: 9px; }

/* ── Summary card ── */
.summary-card {
    background: #ffffff;
    border: 1px solid #e4e2dc;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 12px;
    color: #555;
    line-height: 1.7;
    margin-bottom: 10px;
}
.summary-title { font-size: 12px; font-weight: 600; color: #333; margin-bottom: 5px; }

/* ── Footer ── */
.footer-bar {
    text-align: center;
    padding-top: 10px;
    border-top: 1px solid #e4e2dc;
    margin-top: 4px;
}
.footer-bar p { font-size: 11px; color: #bbb; }
.footer-bar a { color: #6b8cff; text-decoration: none; }

/* ── Streamlit button overrides ── */
div[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border: 1px solid #e4e2dc !important;
}
div[data-testid="column"]:first-child div[data-testid="stButton"] > button {
    color: #4B3BAA !important;
}

/* Gap helper */
.gap { margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"
MEMORY_FILE = "/tmp/scam_memory.json"
SUSPICIOUS_TLDS = [
    ".co", ".tk", ".ml", ".ga", ".cf", ".gq",
    ".xyz", ".top", ".click", ".link", ".work", ".loan"
]
SUSPICIOUS_PREFIXES = [
    "hralert", "hr-alert", "noreply-hr", "jobs-alert",
    "company", "info", "hello", "hi", "contact",
    "admin", "support", "team", "career", "marketing", "office"
]
FREE_PROVIDERS = [
    "gmail.com", "yahoo.com", "hotmail.com",
    "outlook.com", "live.com", "protonmail.com", "icloud.com"
]


# ── BADGE HELPERS ─────────────────────────────────────────────
def b(text, cls):
    return f'<span class="badge {cls}">{text}</span>'

def risk_cls(risk):
    return ("b-red" if risk in ["VERY HIGH", "HIGH"]
            else "b-amber" if risk in ["MEDIUM", "LOW-MEDIUM"]
            else "b-green")

def rbadge(risk):
    return b(risk, risk_cls(risk))

def yesno(val, yes_cls="b-red", no_cls="b-green"):
    return b("Yes", yes_cls) if val else b("No", no_cls)


# ── EMBEDDER ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2"), "st"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(
            max_features=300, ngram_range=(1, 2), stop_words="english"
        ), "tfidf"


def encode_text(text, embedder, etype):
    if etype == "st":
        vec = embedder.encode([text])[0]
    else:
        try:
            vec = embedder.transform([text]).toarray()[0]
        except Exception:
            embedder.fit([text])
            vec = embedder.transform([text]).toarray()[0]
    n = np.linalg.norm(vec)
    return (vec / n if n > 0 else vec).tolist()


# ── SCAM MEMORY ───────────────────────────────────────────────
class ScamMemory:
    def __init__(self, filepath):
        self.filepath = filepath
        self.cases    = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.cases, f)

    def count(self):
        return len(self.cases)

    def get_all(self):
        return list(self.cases)

    def store(self, case_id, message, verdict, risks, embedder, etype):
        self.cases = [c for c in self.cases if c.get("id") != case_id]
        self.cases.append({
            "id"        : case_id,
            "message"   : message[:500],
            "embedding" : encode_text(message, embedder, etype),
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

    def search(self, query_vec, n=3, threshold=0.35):
        candidates = [c for c in self.cases if c.get("embedding")]
        if not candidates:
            return []
        qv     = np.array(query_vec).reshape(1, -1)
        sv     = np.array([c["embedding"] for c in candidates])
        scores = cosine_similarity(qv, sv)[0]
        hits   = [
            {"case_id"   : candidates[i]["id"],
             "similarity": round(float(scores[i]), 3),
             "metadata"  : candidates[i]}
            for i in range(len(candidates))
            if scores[i] >= threshold
        ]
        hits.sort(key=lambda x: x["similarity"], reverse=True)
        return hits[:n]


@st.cache_resource(show_spinner="Initialising memory…")
def get_memory():
    return ScamMemory(MEMORY_FILE)


def seed_memory(mem, embedder, etype):
    seeds = [
        {"id": "seed_001",
         "msg": "hralert@wadialsagroup.com ADNOC contractor interview "
                "salary AED 18000 Emirates ID pay AED 300 processing fee UAE job",
         "v": {"is_scam": True, "confidence": 80, "scam_type": "job_scam",
               "red_flags": ["ADNOC impersonation", "processing fee",
                              "unrealistic salary"]},
         "r": {"technical_risk": "HIGH",
                "cultural_risk": "HIGH", "nlp_risk": "MEDIUM"}},

        {"id": "seed_002",
         "msg": "Emirates NBD account suspended verify immediately "
                "http://emiratesnbd-verify.tk 2 hours permanent closure phishing",
         "v": {"is_scam": True, "confidence": 95, "scam_type": "phishing",
               "red_flags": ["suspicious URL", "urgency", "account suspension"]},
         "r": {"technical_risk": "VERY HIGH",
                "cultural_risk": "HIGH", "nlp_risk": "HIGH"}},

        {"id": "seed_003",
         "msg": "Congratulations won AED 500000 Dubai Government Lucky Draw "
                "send Emirates ID pay AED 250 processing fee expires 24 hours lottery",
         "v": {"is_scam": True, "confidence": 98, "scam_type": "lottery_fraud",
               "red_flags": ["prize claim", "processing fee",
                              "Dubai Government impersonation"]},
         "r": {"technical_risk": "HIGH",
                "cultural_risk": "VERY HIGH", "nlp_risk": "HIGH"}},

        {"id": "seed_004",
         "msg": "ZaviyarHayat WAS Group FastInsu hiring book slot "
                "first come first served marketing@zaviyarhayatgroup.com "
                "no experience required mass recruitment unsolicited UAE",
         "v": {"is_scam": True, "confidence": 80, "scam_type": "job_scam",
               "red_flags": ["multiple company names", "mass recruitment",
                              "new domain"]},
         "r": {"technical_risk": "VERY HIGH",
                "cultural_risk": "HIGH", "nlp_risk": "MEDIUM"}},

        {"id": "seed_005",
         "msg": "Noon order UAE shipped delivery tomorrow track "
                "https://noon.com customer service legitimate package",
         "v": {"is_scam": False, "confidence": 95, "scam_type": "not_a_scam",
               "red_flags": []},
         "r": {"technical_risk": "LOW",
                "cultural_risk": "LOW", "nlp_risk": "LOW"}},

        {"id": "seed_006",
         "msg": "FastInsu WAS Group walk-in interviews book slot "
                "first come first served multiple positions no experience UAE job",
         "v": {"is_scam": True, "confidence": 75, "scam_type": "job_scam",
               "red_flags": ["mass recruitment", "unverifiable", "suspicious domain"]},
         "r": {"technical_risk": "HIGH",
                "cultural_risk": "HIGH", "nlp_risk": "MEDIUM"}},

        {"id": "seed_007",
         "msg": "open roles visit website current openings apply directly "
                "linkedin consulting firm professional recruiter established company",
         "v": {"is_scam": False, "confidence": 88, "scam_type": "not_a_scam",
               "red_flags": []},
         "r": {"technical_risk": "LOW",
                "cultural_risk": "LOW", "nlp_risk": "LOW"}},

        {"id": "seed_008",
         "msg": "came across your professional profile impressed background "
                "expanding team Al Teeba LLC company@teeba.co alteeballc.co "
                "website unreachable new domain unsolicited recruitment UAE",
         "v": {"is_scam": True, "confidence": 70, "scam_type": "job_scam",
               "red_flags": ["unsolicited", "website unreachable",
                              "new .co domain", "generic company@ email"]},
         "r": {"technical_risk": "HIGH",
                "cultural_risk": "MEDIUM", "nlp_risk": "LOW"}},

        {"id": "seed_009",
         "msg": "unsolicited job offer came across profile HR manager "
                "company email unreachable website new domain .co .tk "
                "introductory chat career goals no requirements listed",
         "v": {"is_scam": True, "confidence": 68, "scam_type": "job_scam",
               "red_flags": ["unsolicited", "website unreachable",
                              "suspicious domain"]},
         "r": {"technical_risk": "HIGH",
                "cultural_risk": "MEDIUM", "nlp_risk": "LOW"}},
    ]
    for s in seeds:
        mem.store(s["id"], s["msg"], s["v"], s["r"], embedder, etype)


# ── INVESTIGATION TOOLS ───────────────────────────────────────
def extract_entities(msg):
    emails  = re.findall(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', msg)
    urls    = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', msg)
    bare    = re.findall(r'\bwww\.[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b', msg)
    from urllib.parse import urlparse
    domains = list(set(
        [e.split("@")[1] for e in emails if "@" in e] +
        [urlparse(u).netloc for u in urls if urlparse(u).netloc] +
        [b.replace("www.", "", 1) for b in bare]
    ))
    return {"emails": emails, "urls": urls, "domains": domains}


def whois_lookup(domain):
    try:
        w = whois.whois(domain)
        c = w.creation_date
        if isinstance(c, list): c = c[0]
        if c:
            if c.tzinfo is None:
                c = c.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - c).days
        else:
            age = None
        if age is None:  ar = "unknown"
        elif age < 30:   ar = "VERY HIGH"
        elif age < 180:  ar = "HIGH"
        elif age < 365:  ar = "MEDIUM"
        elif age < 730:  ar = "LOW-MEDIUM"
        else:            ar = "LOW"
        tld_sus = any(domain.endswith(t) for t in SUSPICIOUS_TLDS)
        if tld_sus and ar in ["LOW", "LOW-MEDIUM"]:
            ar = "MEDIUM"
        return {"domain": domain, "age_days": age, "age_risk": ar,
                "tld_suspicious": tld_sus,
                "registrar": str(w.registrar or "unknown"),
                "country": str(w.country or "unknown"),
                "status": "success"}
    except Exception:
        return {"domain": domain, "status": "failed", "age_risk": "unknown",
                "tld_suspicious": any(domain.endswith(t) for t in SUSPICIOUS_TLDS)}


def virustotal_scan(url):
    headers = {"x-apikey": VT_KEY}
    try:
        r = requests.post("https://www.virustotal.com/api/v3/urls",
                          headers=headers, data={"url": url}, timeout=15)
        if r.status_code != 200:
            return {"url": url, "status": "failed", "vt_risk": "unknown"}
        aid = r.json()["data"]["id"]
        time.sleep(5)
        r2  = requests.get(
            f"https://www.virustotal.com/api/v3/analyses/{aid}",
            headers=headers, timeout=15)
        stats = r2.json()["data"]["attributes"]["stats"]
        mal   = stats.get("malicious", 0)
        risk  = ("VERY HIGH" if mal >= 5 else "HIGH" if mal >= 2
                 else "MEDIUM" if mal >= 1 else "LOW")
        return {"url": url, "malicious": mal, "vt_risk": risk, "status": "success"}
    except Exception:
        return {"url": url, "status": "failed", "vt_risk": "unknown"}


def scrape_website(url):
    try:
        r    = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer"]): t.decompose()
        text = " ".join(soup.get_text().split())[:2000].lower()
        flags = {
            "login_form"       : bool(soup.find("form")),
            "urgency_language" : any(w in text for w in
                                     ["urgent", "expires", "act now", "immediately"]),
            "prize_language"   : any(w in text for w in
                                     ["congratulations", "winner", "prize"]),
            "financial_request": any(w in text for w in
                                     ["bank account", "wire transfer", "processing fee"]),
        }
        found = [k for k, v in flags.items() if v]
        return {"url": url, "flags_found": found,
                "flags_count": len(found), "status": "success"}
    except requests.exceptions.ConnectionError:
        return {"url": url, "status": "unreachable"}
    except Exception:
        return {"url": url, "status": "failed"}


def analyze_email(email):
    if "@" not in email:
        return {"email": email, "email_risk": "LOW", "status": "invalid"}
    prefix, domain = email.split("@", 1)
    is_free    = domain.lower() in FREE_PROVIDERS
    sus_prefix = any(p == prefix.lower() or prefix.lower().startswith(p)
                     for p in SUSPICIOUS_PREFIXES)
    tld_sus    = any(domain.endswith(t) for t in SUSPICIOUS_TLDS)
    try:
        dns.resolver.resolve(domain, "MX")
        has_mx = True
    except Exception:
        has_mx = False
    signals = []
    if is_free:    signals.append("free email provider")
    if sus_prefix: signals.append(f"generic prefix: {prefix}")
    if tld_sus:    signals.append(f"suspicious TLD")
    if not has_mx: signals.append("no MX records")
    risk = ("HIGH"   if len(signals) >= 2 else
            "MEDIUM" if len(signals) == 1 else "LOW")
    return {"email": email, "domain": domain, "prefix": prefix,
            "is_free": is_free, "sus_prefix": sus_prefix,
            "tld_sus": tld_sus, "has_mx": has_mx,
            "signals": signals, "email_risk": risk, "status": "success"}


def company_existence(company, domain):
    res = {"existence_signals": [], "absence_signals": [],
           "existence_risk": "LOW"}
    try:
        r    = requests.get(f"https://{domain}",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text().lower()
        if any(w in text for w in ["about us", "our team", "contact"]):
            res["existence_signals"].append("has business content")
        else:
            res["absence_signals"].append("lacks business content")
        if any(w in text for w in ["careers", "jobs", "vacancies"]):
            res["existence_signals"].append("has careers section")
        else:
            res["absence_signals"].append("no careers section")
        words = [w for w in company.lower().split() if len(w) > 4]
        if any(w in text for w in words):
            res["existence_signals"].append("company name on site")
        else:
            res["absence_signals"].append("company name not found")
    except requests.exceptions.ConnectionError:
        res["absence_signals"].append("website unreachable")
    except Exception:
        res["absence_signals"].append("check failed")
    absent  = len(res["absence_signals"])
    unreach = any("unreachable" in s for s in res["absence_signals"])
    if unreach:        res["existence_risk"] = "HIGH"
    elif absent >= 3:  res["existence_risk"] = "HIGH"
    elif absent >= 2:  res["existence_risk"] = "MEDIUM"
    return res


def r2s(r):
    return {"VERY HIGH": 95, "HIGH": 75, "LOW-MEDIUM": 55,
            "MEDIUM": 50, "LOW": 15, "unknown": 30}.get(r, 30)


# ── LLM ───────────────────────────────────────────────────────
def call_llm(sys_p, usr_p):
    r = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": sys_p},
                  {"role": "user",   "content": usr_p}],
        temperature=0.1, max_tokens=1200)
    return r.choices[0].message.content.strip()


def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return {}


# ── AGENT STRIP HTML ──────────────────────────────────────────
AGENT_NAMES = ["Memory", "Orchestrator", "NLP",
               "Technical", "Cultural", "Risk scorer", "Verdict"]


def agent_strip_html(done: int, running: str, running_status: str) -> str:
    cells = ""
    for name in AGENT_NAMES:
        if AGENT_NAMES.index(name) < done:
            cls    = "ag-done"
            status = "✓ done"
        elif name == running:
            cls    = "ag-run"
            status = running_status
        else:
            cls    = "ag-wait"
            status = "waiting"
        cells += (f'<div class="ag {cls}">'
                  f'<div class="ag-name">{name}</div>'
                  f'<div class="ag-status">{status}</div>'
                  f'</div>')
    return (f'<div class="agents-wrap">'
            f'<div class="agents-hdr">'
            f'Live agent activity | نشاط الوكلاء المباشر</div>'
            f'<div class="agents-grid">{cells}</div>'
            f'</div>')


# ── MAIN APP ──────────────────────────────────────────────────
def main():
    # Load cached resources
    embedder, etype = load_embedder()
    memory          = get_memory()



    # Seed on first run
    if memory.count() == 0:
        seed_memory(memory, embedder, etype)

    # Stats
    all_cases = memory.get_all()
    total     = len(all_cases)
    scam_cnt  = sum(1 for c in all_cases if c.get("is_scam") == "True")
    safe_cnt  = total - scam_cnt

    # ── HEADER ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hdr">
      <div style="font-size:26px;margin-bottom:4px;">🛡️</div>
      <div class="hdr-title">UAE Anti-Scam Intelligence System</div>
      <div class="hdr-ar">نظام الكشف عن الاحتيال الإماراتي</div>
      <div class="hdr-sub">6 AI agents · Semantic memory · Real-time investigation</div>
    </div>
    <div class="stats-row">
      <div class="stat">
        <div class="stat-n" style="color:#5B4ECC;">{total}</div>
        <div class="stat-l">Cases in memory</div>
      </div>
      <div class="stat">
        <div class="stat-n" style="color:#B02020;">{scam_cnt}</div>
        <div class="stat-l">Scams stored</div>
      </div>
      <div class="stat">
        <div class="stat-n" style="color:#1A6B35;">{safe_cnt}</div>
        <div class="stat-l">Safe messages</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT ─────────────────────────────────────────────────
    st.markdown("""
    <div class="input-label-row">
      <span class="input-label">Your message | رسالتك</span>
      <span class="input-label-ar">الصق أي رسالة مشبوهة هنا</span>
    </div>
    """, unsafe_allow_html=True)

    message = st.text_area(
        label="msg", label_visibility="collapsed",
        height=120,
        placeholder="Paste any suspicious SMS, email, or WhatsApp message here…"
    )

    c1, c2 = st.columns(2)
    with c1:
        run_btn   = st.button("🔍 Investigate | تحقق",
                              type="primary", use_container_width=True)
    with c2:
        clear_btn = st.button("✕  Clear | مسح", use_container_width=True)

    st.markdown("""
    <div style="font-size:11px;color:#bbb;text-align:center;margin:4px 0 10px;">
      Results are automatically saved to improve future detections ·
      النتائج تُحفظ تلقائياً لتحسين الكشف
    </div>
    """, unsafe_allow_html=True)

    if clear_btn:
        st.rerun()

    if not run_btn or not message.strip():
        st.markdown("""
        <div class="footer-bar">
          <p>Built by Sundus Mohsin Khan · UAE Data Scientist ·
          <a href="https://github.com/SMKProj/agentic-antiscam-uae">GitHub ↗</a></p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── AGENT STRIP PLACEHOLDER ───────────────────────────────
    agent_ph = st.empty()

    def upd(done, running, status):
        agent_ph.markdown(agent_strip_html(done, running, status),
                          unsafe_allow_html=True)

    upd(0, "Memory", "searching…")

    # ════════════════════════════════════════════════════════
    # AGENT 0 — Memory retrieval
    # ════════════════════════════════════════════════════════
    qvec    = encode_text(message, embedder, etype)
    similar = memory.search(qvec, n=3, threshold=0.35)
    upd(1, "Orchestrator", "extracting…")

    # ════════════════════════════════════════════════════════
    # AGENT 1 — Orchestrator
    # ════════════════════════════════════════════════════════
    entities = extract_entities(message)
    mem_ctx  = ""
    if similar:
        lines = [f"MEMORY: {len(similar)} similar past cases:"]
        for i, c in enumerate(similar, 1):
            m = c["metadata"]
            lines.append(
                f"Case #{i} ({c['similarity']:.0%} similar): "
                f"{'SCAM' if m['is_scam']=='True' else 'SAFE'} "
                f"({m['confidence']}%) — {m.get('scam_type','?')}"
            )
        mem_ctx = "\n".join(lines)

    orc_p = f"""You are the lead fraud investigator.
{f'MEMORY:{chr(10)}{mem_ctx}{chr(10)}' if mem_ctx else ''}
IDENTITY CHECK — flag all that apply:
1. Multiple company names in one sender identity → SUSPICIOUS
2. Email domain does not match claimed company → MISMATCH
3. Generic prefix (company@ info@ hello@ contact@) → SUSPICIOUS
4. Suspicious TLD (.co .tk .xyz) impersonating .com → SUSPICIOUS
5. Unsolicited contact with no reference to how they found you → FLAG
6. "Book a slot" / "first come first served" → MASS SPAM
Return ONLY JSON:
{{"claimed_identity":"...","requested_action":"...",
  "identity_flags":["list each flag found"],
  "identity_risk":"HIGH/MEDIUM/LOW",
  "unsolicited":true/false,
  "priority":"HIGH/MEDIUM/LOW"}}"""

    orc  = parse_json(call_llm(orc_p, f"Analyze:\n\n{message}"))
    ents = {
        **entities,
        "claimed_identity": orc.get("claimed_identity", ""),
        "requested_action": orc.get("requested_action", ""),
        "identity_flags"  : orc.get("identity_flags", []),
        "identity_risk"   : orc.get("identity_risk", "LOW"),
        "unsolicited"     : orc.get("unsolicited", False),
    }
    upd(2, "NLP", "analyzing…")

    # ════════════════════════════════════════════════════════
    # AGENT 2 — NLP
    # ════════════════════════════════════════════════════════
    nlp_p = """Analyze ONLY the text for manipulation patterns. Look for:
1.Urgency language  2.Threat language  3.Authority impersonation
4.Reward claims  5.Info harvesting  6.Generic greeting (Dear Candidate)
7.Grammar issues  8.Mass recruitment (book a slot / first come first served)
9.Vague flattery (impressed by your background with no specifics)
Return ONLY JSON:
{"urgency_found":false,"threat_found":false,"authority_claim":false,
 "reward_claim":false,"info_harvesting":false,"generic_greeting":false,
 "grammar_issues":false,"mass_recruitment":false,"vague_flattery":false,
 "key_phrases":[],"sentiment":"neutral","manipulation_score":0,
 "nlp_risk":"LOW","nlp_summary":"..."}"""

    nlp = parse_json(call_llm(nlp_p, f"Analyze:\n\n{message}"))
    sigs = sum([bool(nlp.get("urgency_found")),
                bool(nlp.get("threat_found")),
                bool(nlp.get("authority_claim")),
                bool(nlp.get("reward_claim")),
                bool(nlp.get("info_harvesting")),
                bool(nlp.get("mass_recruitment")),
                bool(nlp.get("generic_greeting")),
                bool(nlp.get("vague_flattery"))])
    nlp["nlp_risk"]     = ("HIGH"       if sigs >= 4 else
                           "MEDIUM"     if sigs >= 2 else
                           "LOW-MEDIUM" if sigs == 1 else "LOW")
    nlp["signal_count"] = sigs
    upd(3, "Technical", "WHOIS + scan…")

    # ════════════════════════════════════════════════════════
    # AGENT 3 — Technical
    # ════════════════════════════════════════════════════════
    all_d = ents.get("domains", [])
    ev    = {"whois": [], "emails": [], "vt": [],
             "scrape": [], "company": {}}

    for d in all_d:
        ev["whois"].append(whois_lookup(d))
        time.sleep(1)
    for e in ents.get("emails", []):
        ev["emails"].append(analyze_email(e))
    claimed = ents.get("claimed_identity", "")
    if claimed and all_d:
        ev["company"] = company_existence(claimed, all_d[0])
    for u in ents.get("urls", [])[:2]:
        ev["vt"].append(virustotal_scan(u))
        time.sleep(15)
    for u in ents.get("urls", [])[:1]:
        ev["scrape"].append(scrape_website(u))

    # Signal aggregation
    h = 0
    for w in ev["whois"]:
        ar = w.get("age_risk", "LOW")
        if ar == "VERY HIGH": h += 3
        elif ar == "HIGH":    h += 2
        elif ar == "MEDIUM":  h += 1
        if w.get("tld_suspicious"): h += 1
    for e in ev["emails"]:
        er = e.get("email_risk", "LOW")
        if er == "HIGH":   h += 2
        elif er == "MEDIUM": h += 1
        if e.get("tld_sus"): h += 1
    for v in ev["vt"]:
        vr = v.get("vt_risk", "LOW")
        if vr == "VERY HIGH": h += 4
        elif vr == "HIGH":    h += 3
        elif vr == "MEDIUM":  h += 1
    for s in ev["scrape"]:
        if s.get("status") == "unreachable":     h += 2
        elif s.get("flags_count", 0) >= 2:       h += 2
    cr = ev["company"].get("existence_risk", "LOW")
    if cr == "HIGH":   h += 3
    elif cr == "MEDIUM": h += 1
    ir = ents.get("identity_risk", "LOW")
    if ir == "HIGH":   h += 2
    elif ir == "MEDIUM": h += 1
    if (not ents.get("urls") and
            nlp.get("nlp_risk") in ["HIGH", "VERY HIGH"]):
        h += 2
    tech_r = ("VERY HIGH" if h >= 6 else "HIGH" if h >= 4 else
              "MEDIUM"    if h >= 2 else "LOW-MEDIUM" if h >= 1 else "LOW")
    ev["tech_risk"] = tech_r
    ev["h"]         = h
    upd(4, "Cultural", "UAE patterns…")

    # ════════════════════════════════════════════════════════
    # AGENT 4 — Cultural context
    # ════════════════════════════════════════════════════════
    cult_p = """UAE fraud specialist. Detect:
- Fake ADNOC/Emirates/Etisalat/du/DEWA/banks impersonation
- Unrealistic salaries AED 15,000+ for entry-level roles
- Upfront fees before any interview
- Unsolicited recruitment (3+ of these = HIGH risk):
  · unsolicited contact, vague flattery with no specifics
  · new/suspicious domain (.co .tk .xyz)
  · website unreachable · no job requirements listed
  · introductory chat only, no actual job description
Return ONLY JSON:
{"uae_entity_impersonated":"null",
 "known_uae_scam_pattern":false,
 "unrealistic_offer":false,
 "unsolicited_recruitment":false,
 "unsolicited_flags_count":0,
 "vague_flattery_recruitment":false,
 "suspicious_domain_recruitment":false,
 "multiple_company_names":false,
 "mass_recruitment_language":false,
 "targets_expat_community":false,
 "cultural_red_flags":[],
 "cultural_risk":"LOW",
 "cultural_summary":"..."}"""

    cult = parse_json(call_llm(cult_p, f"Analyze:\n\n{message}"))
    cult_r = cult.get("cultural_risk", "LOW")

    # Compensating control with cultural
    if (not ents.get("urls") and
            nlp.get("nlp_risk") in ["HIGH", "VERY HIGH"] and
            cult_r in ["HIGH", "VERY HIGH"]):
        h += 1
        tech_r = ("VERY HIGH" if h >= 6 else "HIGH" if h >= 4 else
                  "MEDIUM"    if h >= 2 else "LOW-MEDIUM" if h >= 1 else "LOW")
        ev["tech_risk"] = tech_r
        ev["h"]         = h
    upd(5, "Risk scorer", "scoring…")

    # ════════════════════════════════════════════════════════
    # AGENT 5 — Risk scorer
    # ════════════════════════════════════════════════════════
    nlp_r    = nlp.get("nlp_risk", "LOW")
    has_tech = bool(ev.get("whois") or ev.get("emails"))
    w        = ({"t": 0.50, "c": 0.30, "n": 0.20} if has_tech
                else {"t": 0.20, "c": 0.50, "n": 0.30})
    ts   = r2s(tech_r)
    cs   = r2s(cult_r)
    ns   = r2s(nlp_r)
    base = int(ts * w["t"] + cs * w["c"] + ns * w["n"])

    # Memory boost (internal — not shown to user)
    boost       = 0
    scam_hits   = [s for s in similar if s["metadata"].get("is_scam") == "True"]
    safe_hits   = [s for s in similar if s["metadata"].get("is_scam") == "False"]
    if base >= 40:
        for c in scam_hits:
            sim = c["similarity"]
            if sim >= 0.80:   boost += 12
            elif sim >= 0.60: boost += 8
            elif sim >= 0.40: boost += 4
        if len(scam_hits) >= 2: boost += 5
        for c in safe_hits:
            if c["similarity"] >= 0.75: boost -= 8
        boost = max(-15, min(25, boost))
    final = min(100, max(0, base + boost))
    conf  = ("VERY HIGH" if final >= 85 else "HIGH"   if final >= 65
             else "MEDIUM"   if final >= 45 else "LOW")
    is_scam = final >= 50
    upd(6, "Verdict", "writing…")

    # ════════════════════════════════════════════════════════
    # AGENT 6 — Final verdict
    # ════════════════════════════════════════════════════════
    verd_p = """Lead fraud investigator. Synthesize all agent findings.
Return ONLY JSON:
{"is_scam":false,"confidence":0,
 "scam_type":"not_a_scam",
 "verdict_summary":"2-3 plain English sentences",
 "key_evidence":[],"red_flags":[],"safe_indicators":[],
 "recommended_action":"specific action for the recipient"}"""

    all_f = {
        "message" : message[:300],
        "entities": ents,
        "nlp"     : nlp,
        "tech"    : ev,
        "cultural": cult,
        "score"   : {"base": base, "final": final,
                     "tech_r": tech_r, "cult_r": cult_r, "nlp_r": nlp_r},
    }
    verd = parse_json(call_llm(verd_p, json.dumps(all_f, default=str)))
    verd["confidence"] = final
    verd["is_scam"]    = is_scam

    # All agents done
    upd(7, "", "")

    # Store result
    cid = f"case_{int(time.time())}"
    memory.store(cid, message, verd,
                 {"technical_risk": tech_r,
                  "cultural_risk": cult_r, "nlp_risk": nlp_r},
                 embedder, etype)

    # ════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ════════════════════════════════════════════════════════
    v_color = "#B02020" if is_scam else "#1A6B35"
    v_cls   = "verdict-scam" if is_scam else "verdict-safe"
    v_icon  = "🚨" if is_scam else "✅"
    v_title = ("Scam detected | تم اكتشاف احتيال" if is_scam
               else "Likely safe | يبدو آمناً")
    v_ar    = ("تم تحديد هذا كاحتيال" if is_scam
               else "هذه الرسالة تبدو آمنة")
    stype   = str(verd.get("scam_type", "")).replace("_", " ").title()
    bar_clr = "#B02020" if is_scam else "#1A6B35"
    action  = verd.get("recommended_action",
                       "Verify independently before responding.")

    cv, ca = st.columns(2)
    with cv:
        st.markdown(f"""
        <div class="{v_cls}">
          <div class="v-icon">{v_icon}</div>
          <div class="v-title" style="color:{v_color};">{v_title}</div>
          <div class="v-ar">{v_ar}</div>
          <div class="v-score">Risk score: {final} / 100 · {stype}</div>
          <div class="pbar-bg">
            <div class="pbar-fill"
                 style="width:{final}%;background:{bar_clr};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with ca:
        st.markdown(f"""
        <div class="action-card">
          <div class="action-title">💡 What to do | ماذا تفعل</div>
          {action}
          <div class="action-ar">
            اتبع التعليمات أعلاه قبل التفاعل مع هذه الرسالة.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # ── THREE ANALYSIS PANELS ─────────────────────────────────
    ct, cc, cn = st.columns(3)

    # ── Technical ─────────────────────────────────────────────
    with ct:
        rows = ""
        for wr in ev.get("whois", []):
            ar   = wr.get("age_risk", "unknown")
            age  = wr.get("age_days", "?")
            tld  = wr.get("tld_suspicious", False)
            rows += (f'<div class="ev-row"><span class="ev-key">Domain age</span>'
                     f'{b(f"{age} days", risk_cls(ar))}</div>')
            rows += (f'<div class="ev-row"><span class="ev-key">Domain TLD</span>'
                     f'{b(".co suspicious", "b-red") if tld else b("Normal", "b-green")}</div>')
        for e in ev.get("emails", []):
            rows += (f'<div class="ev-row"><span class="ev-key">Email prefix</span>'
                     f'{b(f"{e.get(chr(112)+(chr(114)+chr(101))+(chr(102)+chr(105))+(chr(120))+chr(64))!r} generic" if e.get("sus_prefix") else "OK", "b-amber" if e.get("sus_prefix") else "b-green")}</div>')
        comp = ev.get("company", {})
        if comp:
            cr_   = comp.get("existence_risk", "LOW")
            unr   = any("unreachable" in s for s in comp.get("absence_signals", []))
            rows += (f'<div class="ev-row"><span class="ev-key">Website</span>'
                     f'{b("Unreachable", "b-red") if unr else b(cr_, risk_cls(cr_))}</div>')
            rows += (f'<div class="ev-row"><span class="ev-key">Company exists</span>'
                     f'{b("Not verified", "b-red") if cr_=="HIGH" else b("Partial", "b-amber") if cr_=="MEDIUM" else b("Verified", "b-green")}</div>')
        if ev.get("vt"):
            for v in ev["vt"]:
                vr = v.get("vt_risk", "unknown")
                rows += (f'<div class="ev-row"><span class="ev-key">URL scan</span>'
                         f'{b(vr, risk_cls(vr))}</div>')
        else:
            rows += (f'<div class="ev-row"><span class="ev-key">URL scan</span>'
                     f'{b("Clean", "b-green")}</div>')

        # Tech flags
        tf = []
        for wr in ev.get("whois", []):
            if wr.get("tld_suspicious"): tf.append(".co suspicious")
            if wr.get("age_risk") in ["HIGH","VERY HIGH"]:
                tf.append(f"New domain ({wr.get('age_days','?')}d)")
        if comp.get("existence_risk") == "HIGH":
            tf.append("Site unreachable")
        for e in ev.get("emails", []):
            if e.get("sus_prefix"):
                tf.append(f"Generic email")
        flags_html = "".join([f'<span class="badge b-red">{f}</span>' for f in tf[:4]])

        st.markdown(f"""
        <div class="panel">
          <div class="panel-hdr">
            <span>Technical evidence | الأدلة التقنية</span>
            {rbadge(tech_r)}
          </div>
          <div class="panel-body">{rows}</div>
          <div class="flags">
            {flags_html if flags_html else b("No flags","b-green")}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Cultural ───────────────────────────────────────────────
    with cc:
        uae_ent = cult.get("uae_entity_impersonated", "null")
        uae_ent = None if str(uae_ent).lower() in ["null","none",""] else uae_ent
        cf = cult.get("cultural_red_flags", [])[:4]
        cf_html = "".join([f'<span class="badge b-red">{f}</span>' for f in cf])

        st.markdown(f"""
        <div class="panel">
          <div class="panel-hdr">
            <span>Cultural context | السياق الثقافي</span>
            {rbadge(cult_r)}
          </div>
          <div class="panel-body">
            <div class="ev-row">
              <span class="ev-key">UAE entity impersonated</span>
              {b(uae_ent,"b-red") if uae_ent else b("None","b-green")}
            </div>
            <div class="ev-row">
              <span class="ev-key">Unsolicited contact</span>
              {yesno(cult.get("unsolicited_recruitment"))}
            </div>
            <div class="ev-row">
              <span class="ev-key">Unrealistic offer</span>
              {yesno(cult.get("unrealistic_offer"))}
            </div>
            <div class="ev-row">
              <span class="ev-key">Multiple company names</span>
              {yesno(cult.get("multiple_company_names"))}
            </div>
            <div class="ev-row">
              <span class="ev-key">Mass recruitment</span>
              {yesno(cult.get("mass_recruitment_language"),"b-amber","b-green")}
            </div>
            <div class="ev-row">
              <span class="ev-key">Known UAE pattern</span>
              {b("Possible","b-amber") if cult.get("known_uae_scam_pattern") else b("None","b-green")}
            </div>
          </div>
          <div class="flags">
            {cf_html if cf_html else b("No flags","b-green")}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Language / NLP ─────────────────────────────────────────
    with cn:
        phrases = nlp.get("key_phrases", [])[:4]
        ph_html = "".join([f'<span class="badge b-amber">{p}</span>'
                           for p in phrases])
        sent    = str(nlp.get("sentiment", "neutral")).title()

        st.markdown(f"""
        <div class="panel">
          <div class="panel-hdr">
            <span>Language analysis | تحليل اللغة</span>
            {rbadge(nlp_r)}
          </div>
          <div class="panel-body">
            <div class="ev-row">
              <span class="ev-key">Urgency language</span>
              {yesno(nlp.get("urgency_found"))}
            </div>
            <div class="ev-row">
              <span class="ev-key">Threat language</span>
              {yesno(nlp.get("threat_found"))}
            </div>
            <div class="ev-row">
              <span class="ev-key">Vague flattery</span>
              {yesno(nlp.get("vague_flattery"))}
            </div>
            <div class="ev-row">
              <span class="ev-key">Generic greeting</span>
              {yesno(nlp.get("generic_greeting"),"b-amber","b-green")}
            </div>
            <div class="ev-row">
              <span class="ev-key">Info harvesting</span>
              {yesno(nlp.get("info_harvesting"))}
            </div>
            <div class="ev-row">
              <span class="ev-key">Sentiment</span>
              {b(sent,"b-purple")}
            </div>
          </div>
          <div class="flags">
            {ph_html if ph_html else b("No phrases flagged","b-green")}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # ── SUMMARY ───────────────────────────────────────────────
    summary = verd.get("verdict_summary", "")
    if summary:
        st.markdown(f"""
        <div class="summary-card">
          <div class="summary-title">📋 Summary | الملخص</div>
          {summary}
        </div>
        """, unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────
    st.markdown("""
    <div class="footer-bar">
      <p>Built by Sundus Mohsin Khan · UAE Data Scientist ·
      <a href="https://github.com/SMKProj/agentic-antiscam-uae">GitHub ↗</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
