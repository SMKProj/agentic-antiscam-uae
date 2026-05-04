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
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="UAE Anti-Scam Intelligence | كاشف الاحتيال",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 820px !important; }
.hdr { text-align: center; padding: 1rem 1.25rem 0.85rem; border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-lg); margin-bottom: 10px; background: var(--color-background-primary); }
.hdr-title { font-size: 20px; font-weight: 500; color: var(--color-text-primary); }
.hdr-ar { font-size: 12px; color: var(--color-text-secondary); direction: rtl; margin-top: 2px; }
.hdr-sub { font-size: 11px; color: var(--color-text-tertiary); margin-top: 3px; }
.stats-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0; margin-bottom: 10px; border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-lg); overflow: hidden; background: var(--color-background-primary); }
.stat { padding: 10px 12px; text-align: center; }
.stat + .stat { border-left: 0.5px solid var(--color-border-tertiary); }
.stat-n { font-size: 20px; font-weight: 500; }
.stat-l { font-size: 10px; color: var(--color-text-tertiary); margin-top: 1px; }
.input-card { border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-lg); background: var(--color-background-primary); margin-bottom: 10px; overflow: hidden; }
.input-label { font-size: 12px; font-weight: 500; color: var(--color-text-secondary); padding: 10px 14px 2px; }
.input-ar { font-size: 10px; color: var(--color-text-tertiary); direction: rtl; padding: 0 14px 6px; }
.verdict-scam { border: 0.5px solid #f5a0a0; border-radius: var(--border-radius-lg); background: #fff5f5; padding: 1rem 1.25rem; }
.verdict-safe { border: 0.5px solid #8fd4a8; border-radius: var(--border-radius-lg); background: #f4fbf6; padding: 1rem 1.25rem; }
.v-title { font-size: 15px; font-weight: 500; margin-bottom: 2px; }
.v-ar { font-size: 11px; color: var(--color-text-secondary); direction: rtl; margin-bottom: 6px; }
.v-score { font-size: 11px; color: var(--color-text-secondary); margin-bottom: 5px; }
.progress-bg { background: var(--color-background-secondary); border-radius: 20px; height: 7px; }
.progress-fill { border-radius: 20px; height: 7px; }
.action-card { border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-lg); background: var(--color-background-primary); padding: 10px 14px; font-size: 12px; color: var(--color-text-secondary); line-height: 1.6; height: 100%; }
.action-title { font-size: 11px; font-weight: 500; color: var(--color-text-primary); margin-bottom: 5px; }
.action-ar { font-size: 10px; color: var(--color-text-tertiary); direction: rtl; margin-top: 4px; }
.panel { border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-lg); background: var(--color-background-primary); overflow: hidden; height: 100%; }
.panel-hdr { padding: 8px 12px; border-bottom: 0.5px solid var(--color-border-tertiary); font-size: 11px; font-weight: 500; display: flex; justify-content: space-between; align-items: center; }
.panel-hdr-ar { font-size: 9px; color: var(--color-text-tertiary); direction: rtl; }
.panel-body { padding: 8px 12px; }
.ev-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 0.5px solid var(--color-border-tertiary); font-size: 11px; }
.ev-row:last-child { border-bottom: none; }
.ev-key { color: var(--color-text-secondary); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 10px; font-weight: 500; }
.b-red    { background: #FCEBEB; color: #A32D2D; }
.b-amber  { background: #FAEEDA; color: #633806; }
.b-green  { background: #EAF3DE; color: #27500A; }
.b-purple { background: #EEEDFE; color: #3C3489; }
.flags { display: flex; flex-wrap: wrap; gap: 4px; padding: 7px 12px; border-top: 0.5px solid var(--color-border-tertiary); }
.agents-card { border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-lg); background: var(--color-background-primary); overflow: hidden; margin-bottom: 10px; }
.agents-hdr { padding: 8px 12px; border-bottom: 0.5px solid var(--color-border-tertiary); font-size: 11px; font-weight: 500; color: var(--color-text-secondary); }
.agents-grid { display: grid; grid-template-columns: repeat(7, 1fr); }
.ag { padding: 8px 6px; text-align: center; font-size: 10px; border-right: 0.5px solid var(--color-border-tertiary); line-height: 1.4; }
.ag:last-child { border-right: none; }
.ag-done { background: var(--color-background-success); color: var(--color-text-success); }
.ag-run  { background: var(--color-background-info); color: var(--color-text-info); }
.ag-wait { background: var(--color-background-secondary); color: var(--color-text-tertiary); }
.ag-name { font-weight: 500; margin-bottom: 2px; }
.ag-status { font-size: 9px; opacity: .85; }
.footer { text-align: center; padding-top: 8px; border-top: 0.5px solid var(--color-border-tertiary); }
.footer p { font-size: 10px; color: var(--color-text-tertiary); }
.footer a { color: var(--color-text-info); }
.sec-gap { margin-bottom: 10px; }
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

client      = Groq(api_key=GROQ_KEY)
MODEL       = "llama-3.3-70b-versatile"
MEMORY_FILE = "/tmp/scam_memory.json"

SUSPICIOUS_TLDS = [".co",".tk",".ml",".ga",".cf",".gq",".xyz",
                   ".top",".click",".link",".work",".loan"]


# ── EMBEDDER ──────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, "st"
    except:
        from sklearn.feature_extraction.text import TfidfVectorizer
        model = TfidfVectorizer(max_features=300, ngram_range=(1,2),
                                stop_words="english")
        return model, "tfidf"

def encode(text, embedder, etype):
    if etype == "st":
        vec = embedder.encode([text])[0]
    else:
        try: vec = embedder.transform([text]).toarray()[0]
        except:
            embedder.fit([text])
            vec = embedder.transform([text]).toarray()[0]
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()


# ── MEMORY ────────────────────────────────────────────────────
@st.cache_resource
def load_memory():
    class Memory:
        def __init__(self, fp):
            self.fp    = fp
            self.cases = self._load()
        def _load(self):
            if os.path.exists(self.fp):
                try:
                    with open(self.fp) as f: return json.load(f)
                except: return []
            return []
        def _save(self):
            with open(self.fp,"w") as f: json.dump(self.cases, f)
        def count(self): return len(self.cases)
        def store(self, cid, msg, verdict, risks, emb, et):
            self.cases = [c for c in self.cases if c["id"] != cid]
            self.cases.append({
                "id":cid, "message":msg[:500],
                "embedding": encode(msg, emb, et),
                "is_scam":str(verdict.get("is_scam",False)),
                "confidence":int(verdict.get("confidence",0)),
                "scam_type":str(verdict.get("scam_type","unknown")),
                "red_flags":verdict.get("red_flags",[])[:3],
                "timestamp":datetime.now(timezone.utc).isoformat(),
            })
            self._save()
             """Searches by pre-computed vector — embedder called outside."""
        def search(self, query_vec, n=3, threshold=0.35):
            cases = [c for c in self.cases if c.get("embedding")]
            if not cases:
                return []
            qv = np.array(query_vec).reshape(1, -1)
            sv = np.array([c["embedding"] for c in cases])
            sc = cosine_similarity(qv, sv)[0]
            res = [
                {"case_id"   : cases[i]["id"],"similarity": round(float(sc[i]), 3), "metadata"  : cases[i]}
            for i in range(len(cases)) if sc[i] >= threshold
            ]
            res.sort(key=lambda x: x["similarity"], reverse=True)
        return res[:n]
    def get_all(self): return self.cases
    return Memory(MEMORY_FILE)

def seed_memory(mem, emb, et):
    seeds = [
        {"id":"seed_001",
         "msg":"hralert@wadialsagroup.com ADNOC contractor interview salary AED 18000 "
               "bring Emirates ID pay AED 300 processing fee urgent job offer UAE",
         "verdict":{"is_scam":True,"confidence":80,"scam_type":"job_scam",
                    "red_flags":["ADNOC impersonation","processing fee","unrealistic salary"]},
         "risks":{"technical_risk":"HIGH","cultural_risk":"HIGH","nlp_risk":"MEDIUM"}},
        {"id":"seed_002",
         "msg":"Emirates NBD account suspended verify immediately "
               "http://emiratesnbd-verify.tk failure 2 hours permanent closure",
         "verdict":{"is_scam":True,"confidence":95,"scam_type":"phishing",
                    "red_flags":["suspicious URL","urgency","account suspension"]},
         "risks":{"technical_risk":"VERY HIGH","cultural_risk":"HIGH","nlp_risk":"HIGH"}},
        {"id":"seed_003",
         "msg":"Congratulations won AED 500000 Dubai Government Lucky Draw "
               "send Emirates ID pay AED 250 processing fee expires 24 hours",
         "verdict":{"is_scam":True,"confidence":98,"scam_type":"lottery_fraud",
                    "red_flags":["prize claim","processing fee","Dubai Government impersonation"]},
         "risks":{"technical_risk":"HIGH","cultural_risk":"VERY HIGH","nlp_risk":"HIGH"}},
        {"id":"seed_004",
         "msg":"ZaviyarHayat Group WAS Group FastInsu hiring book slot "
               "first come first served marketing@zaviyarhayatgroup.com "
               "no experience required mass recruitment unsolicited job UAE",
         "verdict":{"is_scam":True,"confidence":80,"scam_type":"job_scam",
                    "red_flags":["multiple company names","mass recruitment","new domain"]},
         "risks":{"technical_risk":"VERY HIGH","cultural_risk":"HIGH","nlp_risk":"MEDIUM"}},
        {"id":"seed_005",
         "msg":"Noon order UAE shipped expected delivery tomorrow track "
               "https://noon.com customer service legitimate package",
         "verdict":{"is_scam":False,"confidence":95,"scam_type":"not_a_scam",
                    "red_flags":[]},
         "risks":{"technical_risk":"LOW","cultural_risk":"LOW","nlp_risk":"LOW"}},
        {"id":"seed_006",
         "msg":"FastInsu WAS Group walk-in interviews book slot "
               "first come first served multiple positions no experience UAE",
         "verdict":{"is_scam":True,"confidence":75,"scam_type":"job_scam",
                    "red_flags":["mass recruitment","unverifiable","suspicious domain"]},
         "risks":{"technical_risk":"HIGH","cultural_risk":"HIGH","nlp_risk":"MEDIUM"}},
        {"id":"seed_007",
         "msg":"Dear candidate open roles visit website current openings "
               "apply directly linkedin consulting firm professional recruiter",
         "verdict":{"is_scam":False,"confidence":88,"scam_type":"not_a_scam",
                    "red_flags":[]},
         "risks":{"technical_risk":"LOW","cultural_risk":"LOW","nlp_risk":"LOW"}},
        {"id":"seed_008",
         "msg":"I came across your professional profile impressed your background "
               "expanding team Al Teeba LLC company@teeba.co alteeballc.co "
               "website unreachable new domain unsolicited recruitment UAE",
         "verdict":{"is_scam":True,"confidence":70,"scam_type":"job_scam",
                    "red_flags":["unsolicited","website unreachable","new .co domain",
                                 "generic company@ email"]},
         "risks":{"technical_risk":"HIGH","cultural_risk":"MEDIUM","nlp_risk":"LOW"}},
        {"id":"seed_009",
         "msg":"unsolicited job offer came across profile HR manager "
               "company email unreachable website new domain .co .tk "
               "introductory chat career goals no requirements listed",
         "verdict":{"is_scam":True,"confidence":68,"scam_type":"job_scam",
                    "red_flags":["unsolicited","website unreachable","suspicious domain"]},
         "risks":{"technical_risk":"HIGH","cultural_risk":"MEDIUM","nlp_risk":"LOW"}},
    ]
    for s in seeds:
        mem.store(s["id"], s["msg"], s["verdict"], s["risks"], emb, et)


# ── TOOLS ─────────────────────────────────────────────────────
def extract_entities(msg):
    emails  = re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', msg)
    urls    = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', msg)
    bare    = re.findall(r'\bwww\.[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b', msg)
    from urllib.parse import urlparse
    domains = list(set(
        [e.split("@")[1] for e in emails if "@" in e] +
        [urlparse(u).netloc for u in urls if urlparse(u).netloc] +
        [b.replace("www.","",1) for b in bare]
    ))
    return {"emails":emails,"urls":urls,"domains":domains}

def whois_lookup(domain):
    try:
        w = whois.whois(domain)
        c = w.creation_date
        if isinstance(c,list): c = c[0]
        if c:
            if c.tzinfo is None: c = c.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - c).days
        else: age = None
        if age is None:  ar = "unknown"
        elif age < 30:   ar = "VERY HIGH"
        elif age < 180:  ar = "HIGH"
        elif age < 365:  ar = "MEDIUM"
        elif age < 730:  ar = "LOW-MEDIUM"
        else:            ar = "LOW"
        tld_sus = any(domain.endswith(t) for t in SUSPICIOUS_TLDS)
        if tld_sus and ar in ["LOW","LOW-MEDIUM"]: ar = "MEDIUM"
        return {"domain":domain,"age_days":age,"age_risk":ar,
                "tld_suspicious":tld_sus,
                "registrar":str(w.registrar or "unknown"),
                "country":str(w.country or "unknown"),"status":"success"}
    except Exception as e:
        return {"domain":domain,"status":"failed","age_risk":"unknown",
                "tld_suspicious":any(domain.endswith(t) for t in SUSPICIOUS_TLDS)}

def virustotal_scan(url):
    headers = {"x-apikey": VT_KEY}
    try:
        r = requests.post("https://www.virustotal.com/api/v3/urls",
                          headers=headers,data={"url":url},timeout=15)
        if r.status_code != 200: return {"url":url,"status":"failed","vt_risk":"unknown"}
        aid = r.json()["data"]["id"]
        time.sleep(5)
        r2  = requests.get(f"https://www.virustotal.com/api/v3/analyses/{aid}",
                           headers=headers,timeout=15)
        stats = r2.json()["data"]["attributes"]["stats"]
        mal   = stats.get("malicious",0)
        risk  = ("VERY HIGH" if mal>=5 else "HIGH" if mal>=2 else
                 "MEDIUM" if mal>=1 else "LOW")
        return {"url":url,"malicious":mal,"vt_risk":risk,"status":"success"}
    except Exception as e:
        return {"url":url,"status":"failed","vt_risk":"unknown"}

def scrape_website(url):
    try:
        r    = requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=10)
        soup = BeautifulSoup(r.text,"html.parser")
        for t in soup(["script","style","nav","footer"]): t.decompose()
        text = " ".join(soup.get_text().split())[:2000].lower()
        flags = {
            "login_form"       :bool(soup.find("form")),
            "urgency_language" :any(w in text for w in ["urgent","expires","act now"]),
            "prize_language"   :any(w in text for w in ["congratulations","winner","prize"]),
            "financial_request":any(w in text for w in ["bank account","processing fee"]),
        }
        found = [k for k,v in flags.items() if v]
        return {"url":url,"flags_found":found,"flags_count":len(found),"status":"success"}
    except requests.exceptions.ConnectionError:
        return {"url":url,"status":"unreachable"}
    except Exception as e:
        return {"url":url,"status":"failed"}

def analyze_email(email):
    if "@" not in email: return {"email":email,"status":"invalid","email_risk":"LOW"}
    prefix,domain = email.split("@",1)
    free = ["gmail.com","yahoo.com","hotmail.com","outlook.com","live.com"]
    sus  = ["hralert","hr-alert","noreply-hr","jobs-alert","company",
            "info","hello","hi","contact","admin","support","team","career","marketing"]
    is_free    = domain.lower() in free
    sus_prefix = any(p == prefix.lower() or prefix.lower().startswith(p) for p in sus)
    tld_sus    = any(domain.endswith(t) for t in SUSPICIOUS_TLDS)
    try:
        dns.resolver.resolve(domain,"MX")
        has_mx = True
    except: has_mx = False
    signals = []
    if is_free:    signals.append("free email provider")
    if sus_prefix: signals.append(f"generic prefix: {prefix}")
    if tld_sus:    signals.append(f"suspicious TLD: .{domain.split('.')[-1]}")
    if not has_mx: signals.append("no MX records")
    risk = "HIGH" if len(signals)>=2 else "MEDIUM" if signals else "LOW"
    return {"email":email,"domain":domain,"prefix":prefix,
            "is_free":is_free,"sus_prefix":sus_prefix,"tld_sus":tld_sus,
            "has_mx":has_mx,"signals":signals,"email_risk":risk,"status":"success"}

def company_existence(company, domain):
    res = {"existence_signals":[],"absence_signals":[],"existence_risk":"LOW"}
    try:
        r    = requests.get(f"https://{domain}",
                            headers={"User-Agent":"Mozilla/5.0"},timeout=8)
        soup = BeautifulSoup(r.text,"html.parser")
        text = soup.get_text().lower()
        if any(w in text for w in ["about us","our team","contact"]):
            res["existence_signals"].append("has business content")
        else: res["absence_signals"].append("lacks business content")
        if any(w in text for w in ["careers","jobs","vacancies"]):
            res["existence_signals"].append("has careers section")
        else: res["absence_signals"].append("no careers section")
        words = [w for w in company.lower().split() if len(w)>4]
        if any(w in text for w in words):
            res["existence_signals"].append("company name found")
        else: res["absence_signals"].append("company name not found")
    except requests.exceptions.ConnectionError:
        res["absence_signals"].append("website unreachable")
    except: res["absence_signals"].append("check failed")
    absent = len(res["absence_signals"])
    unreach = any("unreachable" in s for s in res["absence_signals"])
    if unreach:        res["existence_risk"] = "HIGH"
    elif absent >= 3:  res["existence_risk"] = "HIGH"
    elif absent >= 2:  res["existence_risk"] = "MEDIUM"
    return res

def r2s(r):
    return {"VERY HIGH":95,"HIGH":75,"LOW-MEDIUM":55,
            "MEDIUM":50,"LOW":15,"unknown":30}.get(r,30)


# ── LLM ───────────────────────────────────────────────────────
def call_llm(sys_p, usr_p):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":sys_p},
                  {"role":"user","content":usr_p}],
        temperature=0.1, max_tokens=1200)
    return r.choices[0].message.content.strip()

def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try: return json.loads(raw.strip())
    except: return {}


# ── BADGE HTML ────────────────────────────────────────────────
def badge(text, cls):
    return f'<span class="badge {cls}">{text}</span>'

def risk_badge(risk):
    cls = ("b-red" if risk in ["VERY HIGH","HIGH"] else
           "b-amber" if risk in ["MEDIUM","LOW-MEDIUM"] else "b-green")
    return badge(risk, cls)

def bool_badge(val, yes_cls="b-red", no_cls="b-green"):
    return badge("Yes", yes_cls) if val else badge("No", no_cls)


# ── MAIN ──────────────────────────────────────────────────────
def main():
    embedder, etype = load_embedder()
    memory          = load_memory()

    if memory.count() == 0:
        with st.spinner("Setting up memory…"):
            seed_memory(memory, embedder, etype)

    cases     = memory.get_all()
    total     = len(cases)
    scam_cnt  = sum(1 for c in cases if c.get("is_scam")=="True")
    safe_cnt  = total - scam_cnt

    # ── HEADER ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hdr">
      <div style="font-size:22px;margin-bottom:4px;">🛡️</div>
      <div class="hdr-title">UAE Anti-Scam Intelligence System</div>
      <div class="hdr-ar">نظام الكشف عن الاحتيال الإماراتي</div>
      <div class="hdr-sub">6 AI agents · Semantic memory · Real-time investigation</div>
    </div>
    <div class="stats-row">
      <div class="stat">
        <div class="stat-n" style="color:#534AB7;">{total}</div>
        <div class="stat-l">Cases in memory</div>
      </div>
      <div class="stat">
        <div class="stat-n" style="color:#A32D2D;">{scam_cnt}</div>
        <div class="stat-l">Scams stored</div>
      </div>
      <div class="stat">
        <div class="stat-n" style="color:#0F6E56;">{safe_cnt}</div>
        <div class="stat-l">Safe messages</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT ─────────────────────────────────────────────────
    st.markdown("""
    <div class="input-card">
      <div class="input-label">Your message | رسالتك</div>
      <div class="input-ar">الصق أي رسالة مشبوهة هنا</div>
    </div>
    """, unsafe_allow_html=True)

    message = st.text_area(
        label="message",
        label_visibility="collapsed",
        height=130,
        placeholder="Paste any suspicious SMS, email, or WhatsApp message here…\nالصق أي رسالة مشبوهة هنا…"
    )

    col1, col2 = st.columns([3,1])
    with col1:
        run_btn = st.button("🔍 Investigate | تحقق", type="primary",
                            use_container_width=True)
    with col2:
        clear_btn = st.button("✕ Clear | مسح", use_container_width=True)

    if clear_btn:
        st.rerun()

    st.markdown("""
    <div style="font-size:11px;color:var(--color-text-tertiary);
                text-align:center;margin:4px 0 10px;">
      Results are automatically saved to improve future detections ·
      النتائج تُحفظ تلقائياً لتحسين الكشف
    </div>
    """, unsafe_allow_html=True)

    if not run_btn or not message.strip():
        # Footer when idle
        st.markdown("""
        <div class="footer" style="margin-top:2rem;">
          <p>Built by Sundas Mohsin Khan · UAE Data Scientist ·
          <a href="https://github.com/SMKProj/agentic-antiscam-uae">GitHub ↗</a></p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── AGENT ACTIVITY (live) ─────────────────────────────────
    def agent_grid(statuses):
        names   = ["Memory","Orchestrator","NLP","Technical","Cultural","Risk scorer","Verdict"]
        classes = statuses
        cells   = ""
        for name, cls, status in zip(names, classes[0], classes[1]):
            cells += (f'<div class="ag {cls}">'
                      f'<div class="ag-name">{name}</div>'
                      f'<div class="ag-status">{status}</div>'
                      f'</div>')
        return (f'<div class="agents-card">'
                f'<div class="agents-hdr">Live agent activity | نشاط الوكلاء المباشر</div>'
                f'<div class="agents-grid">{cells}</div>'
                f'</div>')

    WAIT = "ag-wait"
    DONE = "ag-done"
    RUN  = "ag-run"

    agents_ph = st.empty()

    def update_agents(done_count, running_name, running_status):
        names    = ["Memory","Orchestrator","NLP","Technical","Cultural","Risk scorer","Verdict"]
        cls_list = []
        sts_list = []
        for i, name in enumerate(names):
            if i < done_count:
                cls_list.append(DONE)
                sts_list.append("✓ done")
            elif name == running_name:
                cls_list.append(RUN)
                sts_list.append(running_status)
            else:
                cls_list.append(WAIT)
                sts_list.append("waiting")
        agents_ph.markdown(agent_grid([cls_list, sts_list]),
                           unsafe_allow_html=True)

    update_agents(0, "Memory", "searching…")

    # ── RUN INVESTIGATION ─────────────────────────────────────
    state = {}

    # Agent 0: Memory
    # Compute vector first, then search
    query_vec = encode(message, embedder, etype)
    similar   = memory.search(query_vec, n=3, threshold=0.35)
    state["similar"] = similar
    update_agents(1, "Orchestrator", "extracting…")

    # Agent 1: Orchestrator
    entities = extract_entities(message)
    mem_ctx  = ""
    if similar:
        lines = [f"MEMORY: {len(similar)} similar past cases:"]
        for i,c in enumerate(similar,1):
            m = c["metadata"]
            lines.append(f"Case #{i} ({c['similarity']:.0%}): "
                         f"{'SCAM' if m['is_scam']=='True' else 'SAFE'} "
                         f"({m['confidence']}%) — {m.get('scam_type','?')}")
        mem_ctx = "\n".join(lines)

    orc_p = f"""
You are the lead fraud investigator.
{f'MEMORY:{chr(10)}{mem_ctx}{chr(10)}' if mem_ctx else ''}
IDENTITY CHECK — flag all that apply:
1. Multiple company names in one sender → SUSPICIOUS
2. Email domain does not match claimed company → MISMATCH
3. Generic prefix (company@, info@, hello@, contact@) → SUSPICIOUS
4. Suspicious TLD (.co .tk .xyz) impersonating .com → SUSPICIOUS
5. Unsolicited contact, no reference to how they found you → FLAG
6. "Book a slot" / "first come first served" → MASS SPAM
Return ONLY JSON:
{{"claimed_identity":"...","requested_action":"...",
  "identity_flags":["list each flag found"],
  "identity_risk":"HIGH/MEDIUM/LOW",
  "unsolicited":true/false,
  "priority":"HIGH/MEDIUM/LOW"}}"""
    orc_res = parse_json(call_llm(orc_p, f"Analyze:\n\n{message}"))
    state["entities"] = {
        **entities,
        "claimed_identity": orc_res.get("claimed_identity",""),
        "requested_action": orc_res.get("requested_action",""),
        "identity_flags"  : orc_res.get("identity_flags",[]),
        "identity_risk"   : orc_res.get("identity_risk","LOW"),
        "unsolicited"     : orc_res.get("unsolicited",False),
    }
    update_agents(2, "NLP", "analyzing…")

    # Agent 2: NLP
    nlp_p = """
Analyze ONLY the text. Look for:
1.Urgency language 2.Threat language 3.Authority impersonation
4.Reward claims 5.Info harvesting 6.Generic greeting
7.Grammar issues 8.Mass recruitment (book a slot, first come first served)
9.Vague flattery (impressed by your background with no specifics)
Return ONLY JSON:
{"urgency_found":false,"threat_found":false,"authority_claim":false,
 "reward_claim":false,"info_harvesting":false,"generic_greeting":false,
 "grammar_issues":false,"mass_recruitment":false,"vague_flattery":false,
 "key_phrases":[],"sentiment":"neutral","manipulation_score":0,
 "nlp_risk":"LOW","nlp_summary":"..."}"""
    nlp_res = parse_json(call_llm(nlp_p, f"Analyze:\n\n{message}"))
    sigs = sum([bool(nlp_res.get("urgency_found")),
                bool(nlp_res.get("threat_found")),
                bool(nlp_res.get("authority_claim")),
                bool(nlp_res.get("reward_claim")),
                bool(nlp_res.get("info_harvesting")),
                bool(nlp_res.get("mass_recruitment")),
                bool(nlp_res.get("generic_greeting")),
                bool(nlp_res.get("vague_flattery"))])
    nlp_res["nlp_risk"]     = ("HIGH" if sigs>=4 else "MEDIUM" if sigs>=2 else
                               "LOW-MEDIUM" if sigs==1 else "LOW")
    nlp_res["signal_count"] = sigs
    state["nlp"] = nlp_res
    update_agents(3, "Technical", "WHOIS + scan…")

    # Agent 3: Technical
    ents   = state["entities"]
    all_d  = ents.get("domains",[])
    ev     = {"whois":[],"emails":[],"vt":[],"scrape":[],"company":{}}

    for d in all_d:
        ev["whois"].append(whois_lookup(d))
        time.sleep(1)
    for e in ents.get("emails",[]):
        ev["emails"].append(analyze_email(e))
    claimed = ents.get("claimed_identity","")
    if claimed and all_d:
        ev["company"] = company_existence(claimed, all_d[0])
    for u in ents.get("urls",[])[:2]:
        ev["vt"].append(virustotal_scan(u))
        time.sleep(15)
    for u in ents.get("urls",[])[:1]:
        ev["scrape"].append(scrape_website(u))

    # Signal aggregation
    h = 0
    for w in ev["whois"]:
        ar = w.get("age_risk","LOW")
        if ar == "VERY HIGH": h+=3
        elif ar == "HIGH":    h+=2
        elif ar == "MEDIUM":  h+=1
        if w.get("tld_suspicious"): h+=1
    for e in ev["emails"]:
        er = e.get("email_risk","LOW")
        if er == "HIGH":   h+=2
        elif er == "MEDIUM": h+=1
        if e.get("tld_sus"): h+=1
    for v in ev["vt"]:
        vr = v.get("vt_risk","LOW")
        if vr == "VERY HIGH": h+=4
        elif vr == "HIGH":    h+=3
        elif vr == "MEDIUM":  h+=1
    for s in ev["scrape"]:
        if s.get("status")=="unreachable": h+=2
        elif s.get("flags_count",0)>=2:    h+=2
    cr = ev["company"].get("existence_risk","LOW")
    if cr=="HIGH": h+=3
    elif cr=="MEDIUM": h+=1
    if ents.get("identity_risk")=="HIGH":   h+=2
    elif ents.get("identity_risk")=="MEDIUM": h+=1
    if (not ents.get("urls") and
        nlp_res.get("nlp_risk") in ["HIGH","VERY HIGH"]): h+=2

    tech_risk = ("VERY HIGH" if h>=6 else "HIGH" if h>=4 else
                 "MEDIUM" if h>=2 else "LOW-MEDIUM" if h>=1 else "LOW")
    ev["tech_risk"] = tech_risk
    ev["h"]         = h
    state["tech"]   = ev
    update_agents(4, "Cultural", "checking UAE…")

    # Agent 4: Cultural
    cult_p = """
UAE fraud specialist. Detect:
- Fake ADNOC/Emirates/Etisalat/du/banks
- Unrealistic salaries AED 15,000+ entry level
- Upfront fees before interviews
- Unsolicited recruitment (3+ flags = HIGH):
  unsolicited contact, vague flattery, new/suspicious domain,
  website unreachable, no job requirements, introductory chat only
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
    cult_res = parse_json(call_llm(cult_p, f"Analyze:\n\n{message}"))
    state["cultural"] = cult_res

    # Re-aggregate with cultural
    if (not ents.get("urls") and
        nlp_res.get("nlp_risk") in ["HIGH","VERY HIGH"] and
        cult_res.get("cultural_risk") in ["HIGH","VERY HIGH"]):
        h += 1
        tech_risk = ("VERY HIGH" if h>=6 else "HIGH" if h>=4 else
                     "MEDIUM" if h>=2 else "LOW-MEDIUM" if h>=1 else "LOW")
        ev["tech_risk"] = tech_risk
        ev["h"]         = h
    update_agents(5, "Risk scorer", "scoring…")

    # Agent 5: Risk scorer
    tech_r = ev["tech_risk"]
    cult_r = cult_res.get("cultural_risk","LOW")
    nlp_r  = nlp_res.get("nlp_risk","LOW")
    has_tech = bool(ev.get("whois") or ev.get("emails"))
    w = {"t":0.50,"c":0.30,"n":0.20} if has_tech else {"t":0.20,"c":0.50,"n":0.30}
    ts   = r2s(tech_r)
    cs   = r2s(cult_r)
    ns   = r2s(nlp_r)
    base = int(ts*w["t"] + cs*w["c"] + ns*w["n"])

    # Memory boost (internal only — not shown to user)
    boost = 0
    scam_matches = [s for s in similar if s["metadata"].get("is_scam")=="True"]
    safe_matches = [s for s in similar if s["metadata"].get("is_scam")=="False"]
    if base >= 40:
        for c in scam_matches:
            if c["similarity"] >= 0.80: boost += 12
            elif c["similarity"] >= 0.60: boost += 8
            elif c["similarity"] >= 0.40: boost += 4
        if len(scam_matches) >= 2: boost += 5
        for c in safe_matches:
            if c["similarity"] >= 0.75: boost -= 8
        boost = max(-15, min(25, boost))
    final = min(100, max(0, base + boost))
    conf  = ("VERY HIGH" if final>=85 else "HIGH" if final>=65 else
             "MEDIUM" if final>=45 else "LOW")
    state["score"] = {"base":base,"final":final,"confidence":conf,
                      "is_scam":final>=50,
                      "ts":ts,"cs":cs,"ns":ns,"w":w,
                      "tech_r":tech_r,"cult_r":cult_r,"nlp_r":nlp_r}
    update_agents(6, "Verdict", "writing…")

    # Agent 6: Verdict
    verd_p = """
Lead fraud investigator. Write the final verdict.
Return ONLY JSON:
{"is_scam":false,"confidence":0,
 "scam_type":"not_a_scam",
 "verdict_summary":"2-3 plain English sentences",
 "key_evidence":[],"red_flags":[],"safe_indicators":[],
 "recommended_action":"specific action for the recipient"}"""
    all_f = {
        "message" :message[:300],
        "entities":state["entities"],
        "nlp"     :state["nlp"],
        "tech"    :state["tech"],
        "cultural":state["cultural"],
        "score"   :state["score"],
    }
    verd_res = parse_json(call_llm(verd_p, json.dumps(all_f, default=str)))
    verd_res["confidence"] = final
    verd_res["is_scam"]    = final >= 50
    state["verdict"]       = verd_res

    # All done
    update_agents(7, "", "")

    # Store in memory
    cid = f"case_{int(time.time())}"
    memory.store(cid, message, verd_res,
             {"technical_risk":tech_r,"cultural_risk":cult_r,"nlp_risk":nlp_r},
             embedder, etype)

    # ── DISPLAY ───────────────────────────────────────────────
    is_scam = verd_res.get("is_scam", False)
    score   = state["score"]

    # Verdict + Action row
    v_cls   = "verdict-scam" if is_scam else "verdict-safe"
    v_color = "#A32D2D" if is_scam else "#0F6E56"
    v_icon  = "🚨" if is_scam else "✅"
    v_text  = "Scam detected | تم اكتشاف احتيال" if is_scam else "Likely safe | يبدو آمناً"
    v_ar    = ("تم تحديد هذا كاحتيال" if is_scam else "هذه الرسالة تبدو آمنة")
    scam_type = str(verd_res.get("scam_type","")).replace("_"," ").title()
    bar_color = "#A32D2D" if is_scam else "#0F6E56"

    action_text = verd_res.get("recommended_action",
                               "Verify independently before responding.")

    c_verd, c_act = st.columns(2)
    with c_verd:
        st.markdown(f"""
        <div class="{v_cls}">
          <div style="font-size:22px;margin-bottom:4px;">{v_icon}</div>
          <div class="v-title" style="color:{v_color};">{v_text}</div>
          <div class="v-ar">{v_ar}</div>
          <div class="v-score">Risk score: {final} / 100 · {scam_type}</div>
          <div class="progress-bg" style="margin-top:6px;">
            <div class="progress-fill" style="width:{final}%;background:{bar_color};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c_act:
        st.markdown(f"""
        <div class="action-card">
          <div class="action-title">💡 What to do | ماذا تفعل</div>
          {action_text}
          <div class="action-ar">اتبع التعليمات أعلاه قبل التفاعل مع هذه الرسالة.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-gap"></div>', unsafe_allow_html=True)

    # ── THREE PANELS ──────────────────────────────────────────
    c_tech, c_cult, c_nlp = st.columns(3)

    # ── Technical panel ───────────────────────────────────────
    with c_tech:
        # WHOIS summary
        whois_rows = ""
        for wr in ev.get("whois",[]):
            ar       = wr.get("age_risk","unknown")
            age_b    = badge(f"{wr.get('age_days','?')} days", risk_badge_cls(ar))
            tld_warn = badge("Suspicious TLD","b-red") if wr.get("tld_suspicious") else badge("Normal TLD","b-green")
            whois_rows += (f'<div class="ev-row">'
                           f'<span class="ev-key">Domain age</span>{age_b}</div>'
                           f'<div class="ev-row">'
                           f'<span class="ev-key">Domain TLD</span>{tld_warn}</div>')

        # Email rows
        email_rows = ""
        for e in ev.get("emails",[]):
            ep = badge(f"{e.get('prefix','?')}@ — {e.get('email_risk','?')}",
                       "b-red" if e.get("email_risk")=="HIGH" else
                       "b-amber" if e.get("email_risk")=="MEDIUM" else "b-green")
            email_rows += (f'<div class="ev-row">'
                           f'<span class="ev-key">Email prefix</span>{ep}</div>')

        # Company
        comp        = ev.get("company",{})
        comp_risk   = comp.get("existence_risk","LOW")
        comp_b      = badge("Unreachable" if "unreachable" in
                            " ".join(comp.get("absence_signals",[])) else comp_risk,
                            "b-red" if comp_risk=="HIGH" else
                            "b-amber" if comp_risk=="MEDIUM" else "b-green")
        company_row = (f'<div class="ev-row">'
                       f'<span class="ev-key">Company website</span>{comp_b}</div>')

        # VT
        vt_rows = ""
        if ev.get("vt"):
            for v in ev["vt"]:
                vr  = v.get("vt_risk","unknown")
                vb  = badge(vr, "b-red" if vr in ["VERY HIGH","HIGH"] else
                            "b-amber" if vr=="MEDIUM" else "b-green")
                vt_rows += (f'<div class="ev-row">'
                            f'<span class="ev-key">URL scan</span>{vb}</div>')
        else:
            vt_rows = (f'<div class="ev-row">'
                       f'<span class="ev-key">URL scan</span>'
                       f'{badge("No URLs","b-purple")}</div>')

        # Red flags for tech
        tech_flags = []
        for wr in ev.get("whois",[]):
            if wr.get("tld_suspicious"): tech_flags.append("Suspicious TLD")
            ar = wr.get("age_risk","LOW")
            if ar in ["HIGH","VERY HIGH"]: tech_flags.append(f"New domain ({wr.get('age_days','?')}d)")
        if comp.get("existence_risk")=="HIGH":
            tech_flags.append("Website unreachable")
        for e in ev.get("emails",[]):
            if e.get("email_risk") in ["HIGH","MEDIUM"]:
                tech_flags.append(f"Generic prefix: {e.get('prefix','?')}@")

        flags_html = "".join([f'<span class="badge b-red">{f}</span>' for f in tech_flags[:4]])

        st.markdown(f"""
        <div class="panel">
          <div class="panel-hdr">
            <span>Technical evidence | الأدلة التقنية</span>
            {risk_badge(tech_r)}
          </div>
          <div class="panel-body">
            {whois_rows}{email_rows}{company_row}{vt_rows}
          </div>
          <div class="flags">{flags_html if flags_html else badge("No flags","b-green")}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Cultural panel ────────────────────────────────────────
    with c_cult:
        uae_ent   = cult_res.get("uae_entity_impersonated","null")
        uae_ent   = None if str(uae_ent).lower() in ["null","none",""] else uae_ent
        ent_b     = badge(uae_ent,"b-red") if uae_ent else badge("None","b-green")
        unsol_b   = bool_badge(cult_res.get("unsolicited_recruitment"))
        unreal_b  = bool_badge(cult_res.get("unrealistic_offer"))
        multi_b   = bool_badge(cult_res.get("multiple_company_names"))
        mass_b    = bool_badge(cult_res.get("mass_recruitment_language"))
        expat_b   = bool_badge(cult_res.get("targets_expat_community"),
                               "b-amber","b-green")

        cult_flags = cult_res.get("cultural_red_flags",[])[:4]
        cult_flags_html = "".join(
            [f'<span class="badge b-red">{f}</span>' for f in cult_flags])

        st.markdown(f"""
        <div class="panel">
          <div class="panel-hdr">
            <span>Cultural context | السياق الثقافي</span>
            {risk_badge(cult_r)}
          </div>
          <div class="panel-body">
            <div class="ev-row"><span class="ev-key">UAE entity impersonated</span>{ent_b}</div>
            <div class="ev-row"><span class="ev-key">Unsolicited contact</span>{unsol_b}</div>
            <div class="ev-row"><span class="ev-key">Unrealistic offer</span>{unreal_b}</div>
            <div class="ev-row"><span class="ev-key">Multiple company names</span>{multi_b}</div>
            <div class="ev-row"><span class="ev-key">Mass recruitment language</span>{mass_b}</div>
            <div class="ev-row"><span class="ev-key">Targets expat community</span>{expat_b}</div>
          </div>
          <div class="flags">{cult_flags_html if cult_flags_html else badge("No flags","b-green")}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Language / NLP panel ──────────────────────────────────
    with c_nlp:
        nlp = state["nlp"]
        urg_b  = bool_badge(nlp.get("urgency_found"))
        thr_b  = bool_badge(nlp.get("threat_found"))
        flat_b = bool_badge(nlp.get("vague_flattery"))
        greet_b= bool_badge(nlp.get("generic_greeting"),"b-amber","b-green")
        harv_b = bool_badge(nlp.get("info_harvesting"))
        sent   = nlp.get("sentiment","neutral").title()
        sent_b = badge(sent, "b-purple")

        nlp_phrases = nlp.get("key_phrases",[])[:4]
        nlp_flags_html = "".join(
            [f'<span class="badge b-amber">{p}</span>' for p in nlp_phrases])

        st.markdown(f"""
        <div class="panel">
          <div class="panel-hdr">
            <span>Language analysis | تحليل اللغة</span>
            {risk_badge(nlp_r)}
          </div>
          <div class="panel-body">
            <div class="ev-row"><span class="ev-key">Urgency language</span>{urg_b}</div>
            <div class="ev-row"><span class="ev-key">Threat language</span>{thr_b}</div>
            <div class="ev-row"><span class="ev-key">Vague flattery</span>{flat_b}</div>
            <div class="ev-row"><span class="ev-key">Generic greeting</span>{greet_b}</div>
            <div class="ev-row"><span class="ev-key">Info harvesting</span>{harv_b}</div>
            <div class="ev-row"><span class="ev-key">Sentiment</span>{sent_b}</div>
          </div>
          <div class="flags">{nlp_flags_html if nlp_flags_html else badge("No phrases flagged","b-green")}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-gap"></div>', unsafe_allow_html=True)

    # ── VERDICT SUMMARY ───────────────────────────────────────
    summary = verd_res.get("verdict_summary","")
    if summary:
        st.markdown(f"""
        <div style="background:var(--color-background-secondary);
                    border:0.5px solid var(--color-border-tertiary);
                    border-radius:var(--border-radius-lg);
                    padding:10px 14px;font-size:12px;
                    color:var(--color-text-secondary);
                    line-height:1.6;margin-bottom:10px;">
          <span style="font-size:11px;font-weight:500;
                       color:var(--color-text-primary);">
            📋 Summary | الملخص</span><br><br>
          {summary}
        </div>
        """, unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
      <p>Built by Sundas Mohsin Khan · UAE Data Scientist ·
      <a href="https://github.com/SMKProj/agentic-antiscam-uae">GitHub ↗</a></p>
    </div>
    """, unsafe_allow_html=True)


# ── HELPER ────────────────────────────────────────────────────
def risk_badge_cls(risk):
    return ("b-red" if risk in ["VERY HIGH","HIGH"] else
            "b-amber" if risk in ["MEDIUM","LOW-MEDIUM"] else "b-green")

def risk_badge(risk):
    return f'<span class="badge {risk_badge_cls(risk)}">{risk}</span>'

def bool_badge(val, yes_cls="b-red", no_cls="b-green"):
    return f'<span class="badge {yes_cls}">Yes</span>' if val else f'<span class="badge {no_cls}">No</span>'

def badge(text, cls_or_html):
    if cls_or_html.startswith("<"):
        return cls_or_html
    return f'<span class="badge {cls_or_html}">{text}</span>'


if __name__ == "__main__":
    main()
