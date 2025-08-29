import os
import time
import csv
import io
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

import streamlit as st

# -----------------------------
# Configuration / Secrets
# -----------------------------
SERPER_KEY = st.secrets.get("SERPER_KEY", os.getenv("SERPER_KEY", ""))
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", ""))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0 Safari/537.36"
}

# -----------------------------
# Helpers
# -----------------------------
def require_password():
    """Simple optional lock for internal use."""
    if not APP_PASSWORD:
        return True
    with st.sidebar:
        pwd = st.text_input("App password", type="password")
        if st.button("Unlock"):
            st.session_state["_authed"] = (pwd == APP_PASSWORD)
        if st.session_state.get("_authed"):
            st.success("Unlocked")
            return True
        else:
            st.stop()

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def extract_visible_text(html: str) -> str:
    """Extract readable text using BS4 with built-in parser."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return normalize_whitespace(text)

def extract_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc

def similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio (0..1). Works well, fast, stdlib only."""
    a = normalize_whitespace(a)
    b = normalize_whitespace(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(1, chunk_size - overlap)
    return chunks

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_url(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.ok and "text" in r.headers.get("Content-Type", ""):
            return extract_visible_text(r.text)
    except Exception:
        pass
    return ""

@st.cache_data(show_spinner=False, ttl=1800)
def serper_search(query: str, num: int = 5):
    """
    Use serper.dev (Google SERP API).
    """
    if not SERPER_KEY:
        raise RuntimeError("SERPER_KEY missing. Add it in Streamlit Secrets.")

    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": num, "gl": "us", "hl": "en"}
    headers = {
        "X-API-KEY": SERPER_KEY,
        "Content-Type": "application/json",
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        if res.ok:
            data = res.json()
            results = []
            # pull organic results if present
            for item in data.get("organic", [])[:num]:
                link = item.get("link")
                title = item.get("title")
                if link:
                    results.append({"url": link, "title": title or ""})
            return results
    except Exception:
        pass
    return []

def best_match_for_chunk(chunk: str, results: list, min_len: int = 300):
    """
    Fetch pages and compute similarity against the chunk.
    Return the best match record or None.
    """
    best = None
    best_score = 0.0

    # Skip very small chunks (avoid false positives)
    if len(chunk) < min_len:
        return None

    for r in results:
        url = r["url"]
        page_text = fetch_url(url)
        if not page_text:
            continue

        score = similarity(chunk[:4000], page_text[:20000])  # caps for speed
        if score > best_score:
            best_score = score
            best = {
                "url": url,
                "title": r.get("title", ""),
                "domain": extract_domain(url),
                "similarity": round(score * 100, 2),
            }

    return best

def to_csv(rows: list) -> bytes:
    """Create a CSV from a list of dicts (no pandas needed)."""
    if not rows:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Plagiarism Checker", page_icon="🕵️", layout="wide")

require_password()

st.title("🕵️ Plagiarism Checker (lightweight)")
st.caption("No `lxml`, no `tldextract`, no `pandas` — fast deploy & run.")

col1, col2 = st.columns([2, 1])
with col1:
    text_input = st.text_area(
        "Paste text to check",
        height=220,
        placeholder="Paste up to ~50k characters…"
    )
    uploaded = st.file_uploader("…or upload a .txt file", type=["txt"])

with col2:
    chunk_size = st.number_input("Chunk size (words)", 400, 3000, 1200, 50)
    overlap = st.number_input("Overlap (words)", 0, 1000, 200, 50)
    results_per_query = st.slider("Search results per chunk", 2, 10, 5)
    sim_threshold = st.slider("Flag if similarity ≥ (%)", 20, 100, 45)
    rate_delay = st.number_input("Delay between searches (sec)", 0.0, 5.0, 0.6, 0.1)

# Load text
if uploaded and not text_input:
    try:
        text_input = uploaded.read().decode("utf-8", errors="ignore")
    except Exception:
        st.error("Could not read the file as UTF-8.")

run = st.button("Run check", type="primary", use_container_width=True)

if run:
    if not SERPER_KEY:
        st.error("SERPER_KEY missing. Add it in **Manage app → Settings → Secrets**.")
        st.stop()

    if not text_input or len(text_input.strip()) < 40:
        st.warning("Please paste more text or upload a .txt file.")
        st.stop()

    chunks = chunk_text(text_input.strip(), chunk_size=chunk_size, overlap=overlap)
    st.write(f"**Split into {len(chunks)} chunk(s).**")

    progress = st.progress(0, text="Searching…")
    findings = []

    for idx, chunk in enumerate(chunks, start=1):
        # Use beginning of chunk to form query
        few_words = " ".join(chunk.split()[:12])
        results = serper_search(few_words, num=results_per_query)

        match = best_match_for_chunk(chunk, results)
        if match and match["similarity"] >= sim_threshold:
            match_row = {
                "chunk": idx,
                "similarity_%": match["similarity"],
                "domain": match["domain"],
                "title": match["title"],
                "url": match["url"],
            }
            findings.append(match_row)

        progress.progress(idx / len(chunks), text=f"Processed {idx}/{len(chunks)}")
        time.sleep(rate_delay)

    progress.empty()

    st.subheader("Results")
    if not findings:
        st.success("No suspicious matches above the threshold were found.")
    else:
        # Show table (no pandas required)
        st.dataframe(findings, use_container_width=True, hide_index=True)
        csv_bytes = to_csv(findings)
        st.download_button(
            "Download CSV",
            csv_bytes,
            "plagiarism_results.csv",
            "text/csv",
            use_container_width=True,
        )

    st.info(
        "Tip: Increase *results per chunk* and lower the *threshold* if you want a broader scan. "
        "Decrease chunk size if matches are short and scattered."
    )
