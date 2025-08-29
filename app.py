# SPDX-License-Identifier: MIT
# Plagiarism Checker - Streamlit App
# Provider-agnostic search (Serper.dev or Bing Search v7)
# Author: You
# ----------------------------------

import os
import re
import json
import time
import math
import logging
from typing import List, Dict, Tuple

import requests
import streamlit as st
from bs4 import BeautifulSoup

# --------------------------
# Defaults / constants
# --------------------------
BING_ENDPOINT_DEFAULT = "https://api.bing.microsoft.com/v7.0/search"
SERPER_ENDPOINT_DEFAULT = "https://google.serper.dev/search"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)

logging.basicConfig(level=logging.INFO)


# --------------------------
# Text utilities
# --------------------------
def clean_text(s: str) -> str:
    """Minimal normalization for comparison."""
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\u200b", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_words(s: str) -> List[str]:
    """Very simple tokenization: letters/numbers only, lowercased."""
    s = s.lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    return tokens


def make_ngrams(tokens: List[str], n: int = 5) -> List[Tuple[str, ...]]:
    if n <= 1:
        return [(t,) for t in tokens]
    return [tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


def jaccard_similarity(ngrams_a, ngrams_b) -> float:
    """Jaccard similarity on n-gram sets [0..1]."""
    set_a, set_b = set(ngrams_a), set(ngrams_b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text (by words) into chunks of size 'chunk_size' with 'overlap' words overlap.
    """
    tokens = text.split()
    if not tokens:
        return []

    if chunk_size <= 0:
        return [" ".join(tokens)]

    chunks = []
    i = 0
    stride = max(1, chunk_size - max(0, overlap))
    while i < len(tokens):
        chunk_tokens = tokens[i : i + chunk_size]
        if not chunk_tokens:
            break
        chunks.append(" ".join(chunk_tokens))
        i += stride
        if i >= len(tokens):
            break

    return chunks


def query_from_chunk(chunk: str, query_words: int = 12) -> str:
    """
    Construct a search query from the first 'query_words' words of a chunk.
    Keep it relatively short so providers return matches instead of 'too long queries'.
    """
    words = chunk.split()
    return " ".join(words[: max(1, query_words)])


# --------------------------
# Fetch remote page text
# --------------------------
def fetch_page_text(url: str, timeout: int = 12, max_chars: int = 80_000) -> str:
    """
    Fetch HTML and return visible text for deeper comparison.
    """
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        html = r.text[: max_chars * 3]  # cut before parsing
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = clean_text(text)[:max_chars]
        return text
    except Exception as e:
        logging.warning("fetch_page_text error for %s: %s", url, e)
        return ""


# --------------------------
# Provider: Serper (Google)
# --------------------------
def serper_search(api_key: str, query: str, endpoint: str, count: int = 10) -> List[Dict]:
    """
    Query Serper (Google) and return normalized results
    with keys: name, snippet, url
    """
    endpoint = (endpoint or SERPER_ENDPOINT_DEFAULT).strip()
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": max(1, min(count, 20)), "hl": "en", "gl": "us"}

    try:
        r = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=12)
        if r.status_code in (401, 403):
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            return [{"auth_error": True, "detail": str(detail)}]
        if r.status_code == 429:
            return [{"rate_limited": True}]
        r.raise_for_status()

        data = r.json() or {}
        organic = data.get("organic", []) or []
        results = []
        for item in organic:
            results.append(
                {
                    "name": item.get("title") or "",
                    "snippet": item.get("snippet") or "",
                    "url": item.get("link") or "",
                }
            )
        return results
    except Exception as e:
        logging.exception("Serper error: %s", e)
        return []


# --------------------------
# Provider: Bing
# --------------------------
def bing_search(api_key: str, query: str, endpoint: str, count: int = 10) -> List[Dict]:
    """
    Query Bing Web Search v7.
    Return normalized results: name, snippet, url
    """
    endpoint = (endpoint or BING_ENDPOINT_DEFAULT).strip()
    if not endpoint.rstrip("/").endswith("search"):
        endpoint = endpoint.rstrip("/") + "/v7.0/search"

    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": max(1, min(count, 50)),
        "mkt": "en-US",
        "safeSearch": "Off",
        "textDecorations": False,
        "textFormat": "Raw",
    }

    try:
        r = requests.get(endpoint, headers=headers, params=params, timeout=12)
        if r.status_code in (401, 403):
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            return [{"auth_error": True, "detail": str(detail)}]
        if r.status_code == 429:
            return [{"rate_limited": True}]
        r.raise_for_status()

        data = r.json() or {}
        web_pages = data.get("webPages", {}).get("value", []) or []
        results = []
        for item in web_pages:
            results.append(
                {
                    "name": item.get("name") or "",
                    "snippet": item.get("snippet") or "",
                    "url": item.get("url") or "",
                }
            )
        return results
    except Exception as e:
        logging.exception("Bing error: %s", e)
        return []


# --------------------------
# Main scan logic
# --------------------------
def scan_text(
    text: str,
    provider: str,
    api_key: str,
    endpoint: str,
    *,
    chunk_size: int,
    overlap: int,
    results_per_chunk: int,
    threshold_pct: int,
    delay_sec: float,
    retries: int,
    ngram_n: int,
    fetch_full_pages: bool,
    confirm_threshold_pct: int,
) -> List[Dict]:
    """
    Process text in chunks, query search, compute similarities,
    return list of suspicious matches (dicts).
    """
    suspicious = []
    chunks = split_into_chunks(clean_text(text), chunk_size, overlap)
    if not chunks:
        return suspicious

    threshold = threshold_pct / 100.0
    confirm_threshold = confirm_threshold_pct / 100.0

    # choose the function
    def do_search(q, k):
        if provider.startswith("Serper"):
            return serper_search(api_key, q, endpoint, count=k)
        else:
            return bing_search(api_key, q, endpoint, count=k)

    st.write(f"Scanning **{len(chunks)}** chunk(s)…")
    prog = st.progress(0)
    for idx, chunk in enumerate(chunks, start=1):
        # 1) form query
        query = query_from_chunk(chunk, query_words=12)
        # 2) call search with backoff
        attempt = 0
        while True:
            results = do_search(query, results_per_chunk)
            if results and isinstance(results[0], dict) and results[0].get("auth_error"):
                detail = results[0].get("detail", "")
                st.error(
                    f"Authentication error from {provider}. "
                    f"Check your API key / subscription. Details: {detail}"
                )
                return suspicious
            if results and isinstance(results[0], dict) and results[0].get("rate_limited"):
                # rate limit – back off
                time.sleep(max(1.2, delay_sec))
                attempt += 1
                if attempt > retries:
                    break
                continue
            break

        # 3) compare snippet similarity → optional full page confirmation
        chunk_tokens = tokenize_words(chunk)
        chunk_ngrams = make_ngrams(chunk_tokens, n=ngram_n)

        best_for_chunk = None
        for r in results or []:
            name = clean_text(r.get("name", ""))
            snippet = clean_text(r.get("snippet", ""))
            url = r.get("url", "")

            snippet_tokens = tokenize_words(snippet)
            snippet_ngrams = make_ngrams(snippet_tokens, n=ngram_n)
            sim_snippet = jaccard_similarity(chunk_ngrams, snippet_ngrams)

            # keep if snippet already clears the threshold
            if sim_snippet >= threshold:
                info = {
                    "chunk_index": idx,
                    "chunk_preview": chunk[:140] + ("…" if len(chunk) > 140 else ""),
                    "url": url,
                    "title": name,
                    "from": "snippet",
                    "similarity": round(sim_snippet * 100, 1),
                }
                best_for_chunk = (sim_snippet, info)
                # still do a confirm check if asked
            elif fetch_full_pages and sim_snippet >= (threshold * 0.6):
                # borderline on snippet → fetch page for confirmation
                page_text = fetch_page_text(url)
                if page_text:
                    page_tokens = tokenize_words(page_text)
                    page_ngrams = make_ngrams(page_tokens, n=ngram_n)
                    sim_page = jaccard_similarity(chunk_ngrams, page_ngrams)
                    if sim_page >= confirm_threshold:
                        info = {
                            "chunk_index": idx,
                            "chunk_preview": chunk[:140] + ("…" if len(chunk) > 140 else ""),
                            "url": url,
                            "title": name,
                            "from": "page",
                            "similarity": round(sim_page * 100, 1),
                        }
                        best_for_chunk = (sim_page, info)

        if best_for_chunk:
            suspicious.append(best_for_chunk[1])

        # progress + pacing
        prog.progress(min(100, int(idx / len(chunks) * 100)))
        time.sleep(max(0.0, delay_sec))

    prog.progress(100)
    return suspicious


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Plagiarism Checker", page_icon="🧭", layout="wide")
st.title("🧭 Plagiarism Checker (Web Search)")

st.markdown(
    """
Paste text below or upload a `.txt` file. The app splits it into chunks and checks each
chunk against the web using **Serper (Google)** or **Bing**. It computes a similarity score
based on n-gram Jaccard overlap.  
Use **Aggressive** settings for best recall; it will be slower and use more API calls.
"""
)

# Input area
col_a, col_b = st.columns([2, 1])
with col_a:
    text_input = st.text_area("Text to check", height=260, placeholder="Paste up to ~15,000 words…")
    up = st.file_uploader("…or upload a .txt file", type=["txt"])
    if up and not text_input:
        text_input = up.read().decode(errors="ignore")

with col_b:
    st.subheader("Search Provider")
    provider = st.selectbox("Provider", ["Serper (Google) - Recommended", "Bing"], index=0)

    if provider.startswith("Serper"):
        api_key = st.text_input(
            "Serper API Key",
            type="password",
            value=os.getenv("SERPER_API_KEY", ""),
            help="Get one at https://serper.dev",
        )
        endpoint = st.text_input(
            "Serper endpoint",
            value=os.getenv("SERPER_ENDPOINT", SERPER_ENDPOINT_DEFAULT),
        )
    else:
        api_key = st.text_input(
            "Bing API Key",
            type="password",
            value=os.getenv("BING_API_KEY", ""),
            help="Bing Web Search v7",
        )
        endpoint = st.text_input(
            "Bing endpoint",
            value=os.getenv("BING_ENDPOINT", BING_ENDPOINT_DEFAULT),
        )

st.divider()

# Controls
with st.sidebar:
    st.header("Settings")

    mode = st.radio("Preset", ["Balanced", "Aggressive (catch more, slower)"], index=0)

    if mode.startswith("Aggressive"):
        chunk_size = st.number_input("Chunk size (words)", 10, 5000, 30, step=2)
        overlap = st.number_input("Overlap (words)", 0, 500, 14, step=1)
        results_per_chunk = st.slider("Search results per chunk", 1, 20, 10, 1)
        threshold_pct = st.slider("Flag if similarity ≥ (%)", 10, 95, 42, 1)
        delay_sec = st.number_input("Delay between searches (sec)", 0.1, 3.0, 1.1, step=0.05)
    else:
        chunk_size = st.number_input("Chunk size (words)", 10, 5000, 120, step=10)
        overlap = st.number_input("Overlap (words)", 0, 500, 30, step=2)
        results_per_chunk = st.slider("Search results per chunk", 1, 20, 6, 1)
        threshold_pct = st.slider("Flag if similarity ≥ (%)", 10, 95, 55, 1)
        delay_sec = st.number_input("Delay between searches (sec)", 0.1, 3.0, 0.7, step=0.05)

    ngram_n = st.slider("n-gram size", 2, 8, 5, 1)
    retries = st.slider("Retries on 429/5xx", 0, 5, 2, 1)

    st.markdown("### Confirmation (optional)")
    fetch_full_pages = st.checkbox(
        "Fetch web pages for confirmation (slower, more accurate)", value=True
    )
    confirm_threshold_pct = st.slider("Confirm if similarity ≥ (%)", 10, 95, 45, 1)

    st.caption(
        "Tip: smaller chunks + larger overlap + more results per chunk + lower threshold "
        "→ catches more but costs more and runs slower."
    )

st.divider()

if st.button("Run check", type="primary", use_container_width=True):
    if not text_input or len(text_input.strip()) < 20:
        st.warning("Please paste some text or upload a `.txt` file.")
        st.stop()
    if not api_key:
        st.error("Please provide a valid API key for the selected provider.")
        st.stop()

    n_words = len(text_input.split())
    st.write(f"Input length: **{n_words}** words")

    matches = scan_text(
        text=text_input,
        provider="Serper" if provider.startswith("Serper") else "Bing",
        api_key=api_key,
        endpoint=endpoint,
        chunk_size=int(chunk_size),
        overlap=int(overlap),
        results_per_chunk=int(results_per_chunk),
        threshold_pct=int(threshold_pct),
        delay_sec=float(delay_sec),
        retries=int(retries),
        ngram_n=int(ngram_n),
        fetch_full_pages=bool(fetch_full_pages),
        confirm_threshold_pct=int(confirm_threshold_pct),
    )

    st.subheader("Results")
    if not matches:
        st.success("No suspicious matches above your threshold were found.")
        st.info(
            "Try **Aggressive** preset or manually set: chunk≈30, overlap≈12–15, "
            "results per chunk=10–20, threshold≈40–45%, delay≈1.0–1.2 s."
        )
    else:
        matches = sorted(matches, key=lambda x: -x["similarity"])
        for m in matches:
            with st.expander(
                f'Chunk {m["chunk_index"]} • {m["similarity"]}% • {m["title"][:80]}'
            ):
                st.write(f"**URL:** {m['url']}")
                st.write(f"**Detected from:** {m['from']}")
                st.write(f"**Chunk preview:** {m['chunk_preview']}")
        st.success(f"Found **{len(matches)}** suspicious match(es).")


st.caption(
    "This tool estimates similarity using public search results. "
    "It is not a definitive legal plagiarism determination."
)
