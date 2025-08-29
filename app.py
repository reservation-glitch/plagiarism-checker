# SPDX-License-Identifier: MIT
# Plagiarism Checker — Serper (Streamlit Secrets), page fetching + dual similarity

import re
import time
import logging
from typing import List, Dict

import requests
import streamlit as st
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("plagcheck")

SERPER_ENDPOINT = "https://google.serper.dev/search"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

# -----------------------------
# Text utilities
# -----------------------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\u200b", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_html(html: str) -> str:
    """Return visible text from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = clean_text(text)
    return text

def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def make_ngrams(tokens: List[str], n: int = 5) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    stride = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), stride):
        part = words[i : i + size]
        if not part:
            break
        chunks.append(" ".join(part))
    return chunks

def make_query_from_chunk(chunk: str, query_words: int = 12) -> str:
    return " ".join(chunk.split()[:query_words])

# -----------------------------
# Serper & fetching
# -----------------------------
def serper_search(api_key: str, query: str, count: int = 10) -> List[Dict]:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": min(count, 20), "hl": "en", "gl": "us"}
    try:
        r = requests.post(SERPER_ENDPOINT, headers=headers, json=payload, timeout=12)
        if r.status_code == 401:
            return [{"auth_error": True, "detail": r.text}]
        if r.status_code == 429:
            return [{"rate_limited": True}]
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic", [])
        return [{"title": o.get("title", ""), "snippet": o.get("snippet", ""), "url": o.get("link", "")}
                for o in organic]
    except Exception as e:
        LOGGER.exception("Serper error: %s", e)
        return []

def fetch_page_text(url: str, timeout: float = 12.0, max_chars: int = 60_000) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        txt = strip_html(r.text)
        # keep it bounded (enough for similarity, faster)
        return txt[:max_chars]
    except Exception:
        return ""

# -----------------------------
# Core scan
# -----------------------------
def scan_text(
    text: str,
    api_key: str,
    *,
    chunk_size: int,
    overlap: int,
    results_per_chunk: int,
    threshold_pct: int,
    delay_sec: float,
    retries: int,
    ngram_n: int,
    require_both_signals: bool,
) -> List[Dict]:
    """Return list of matches with similarity scores."""
    results = []
    chunks = chunk_text(clean_text(text), size=chunk_size, overlap=overlap)
    if not chunks:
        return results

    thresh = threshold_pct / 100.0
    prog = st.progress(0.0)

    for idx, chunk in enumerate(chunks, start=1):
        query = make_query_from_chunk(chunk)
        # - Try Serper (with backoff for 429)
        tries = 0
        while True:
            sres = serper_search(api_key, query, count=results_per_chunk)
            if sres and sres[0].get("auth_error"):
                st.error("Authentication error: check your Serper API key in **Secrets**")
                return results
            if sres and sres[0].get("rate_limited"):
                time.sleep(delay_sec + 0.6)
                tries += 1
                if tries > retries:
                    sres = []
                    break
                continue
            break

        # compare against each page
        chunk_tokens = tokenize_words(chunk)
        chunk_ngrams = make_ngrams(chunk_tokens, n=ngram_n)

        for hit in sres:
            url = hit.get("url") or ""
            if not url:
                continue
            page_text = fetch_page_text(url)
            if not page_text:
                # fall back to snippet if page blocked
                page_text = hit.get("snippet", "")

            page_tokens = tokenize_words(page_text)
            page_ngrams = make_ngrams(page_tokens, n=ngram_n)

            # two signals
            j_sim = jaccard(chunk_ngrams, page_ngrams)            # 0..1
            s_sim = seq_ratio(chunk, page_text)                   # 0..1

            # Either signal above threshold (or require both if toggled)
            if (require_both_signals and j_sim >= thresh and s_sim >= thresh) or \
               (not require_both_signals and (j_sim >= thresh or s_sim >= thresh)):
                results.append({
                    "chunk": idx,
                    "url": url,
                    "title": hit.get("title", ""),
                    "jaccard_pct": round(j_sim * 100, 1),
                    "seqmatch_pct": round(s_sim * 100, 1),
                    "preview": chunk[:180] + "…"
                })

        prog.progress(idx / len(chunks))
        time.sleep(delay_sec)

    prog.progress(1.0)
    return results

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Plagiarism Checker", page_icon="🧭", layout="wide")
st.title("🧭 Plagiarism Checker (Serper + Page Fetching)")

# Aggressive defaults to catch more
with st.sidebar:
    st.markdown("### Scan Settings")
    mode = st.radio("Preset", ["Aggressive (recommended)", "Balanced", "Custom"], index=0)

    if mode == "Aggressive (recommended)":
        chunk_size = 30
        overlap = 15
        results_per_chunk = 10
        threshold_pct = 40
        delay_sec = 1.0
    elif mode == "Balanced":
        chunk_size = 80
        overlap = 30
        results_per_chunk = 8
        threshold_pct = 50
        delay_sec = 0.8
    else:  # Custom
        chunk_size = st.slider("Chunk size (words)", 20, 2000, 80)
        overlap = st.slider("Overlap (words)", 0, 500, 30)
        results_per_chunk = st.slider("Search results per chunk", 1, 20, 8)
        threshold_pct = st.slider("Flag if similarity ≥ (%)", 10, 95, 50)
        delay_sec = st.slider("Delay between searches (sec)", 0.1, 2.0, 0.8)

    retries = st.slider("Retries on 429", 0, 5, 2)
    ngram_n = st.slider("n-gram size", 2, 10, 5)
    require_both_signals = st.checkbox("Require BOTH signals (Jaccard & SequenceMatcher) to exceed threshold",
                                       value=False, help="If unchecked, either signal can trigger a flag.")

# Text input
col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Paste text to check", height=280, placeholder="Paste up to ~12,000 words…")
with col2:
    up = st.file_uploader("…or upload .txt", type=["txt"])
    if up and not text:
        text = up.read().decode(errors="ignore")

# API key from secrets
try:
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
except Exception:
    SERPER_API_KEY = None

if st.button("Run check", type="primary"):
    if not text.strip():
        st.warning("Please paste some text or upload a file.")
        st.stop()
    if not SERPER_API_KEY:
        st.error("No Serper API key found in **Secrets**. Add `SERPER_API_KEY = \"...\"`.")
        st.stop()

    st.info(f"Scanning ~{len(text.split())} words • mode: **{mode}**")
    hits = scan_text(
        text=text,
        api_key=SERPER_API_KEY,
        chunk_size=chunk_size,
        overlap=overlap,
        results_per_chunk=results_per_chunk,
        threshold_pct=threshold_pct,
        delay_sec=delay_sec,
        retries=retries,
        ngram_n=ngram_n,
        require_both_signals=require_both_signals,
    )

    st.subheader("Results")
    if not hits:
        st.success("No suspicious matches found.")
    else:
        # Sort by stronger signal
        hits = sorted(hits, key=lambda x: (x["jaccard_pct"] + x["seqmatch_pct"]), reverse=True)
        for h in hits:
            with st.expander(f"Chunk {h['chunk']} • Jaccard {h['jaccard_pct']}% • SeqMatch {h['seqmatch_pct']}% • {h['title']}"):
                st.write(h["url"])
                st.write("Preview of your text:\n\n> " + h["preview"])
