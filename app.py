# SPDX-License-Identifier: MIT
# Plagiarism Checker - Streamlit App (Secrets-enabled)

import os
import re
import json
import time
import logging
from typing import List, Dict, Tuple

import requests
import streamlit as st
from bs4 import BeautifulSoup

# --------------------------
# Defaults
# --------------------------
SERPER_ENDPOINT_DEFAULT = "https://google.serper.dev/search"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
logging.basicConfig(level=logging.INFO)

# --------------------------
# Text utilities
# --------------------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\u200b", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_words(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())

def make_ngrams(tokens, n=5):
    return [tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]

def jaccard_similarity(a, b) -> float:
    set_a, set_b = set(a), set(b)
    return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0

def split_into_chunks(text: str, chunk_size: int, overlap: int):
    tokens = text.split()
    if not tokens:
        return []
    stride = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i : i + chunk_size]
        if not chunk_tokens:
            break
        chunks.append(" ".join(chunk_tokens))
    return chunks

def query_from_chunk(chunk: str, query_words: int = 12) -> str:
    return " ".join(chunk.split()[:query_words])

# --------------------------
# Serper API
# --------------------------
def serper_search(api_key: str, query: str, endpoint: str, count: int = 10):
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": min(count, 20), "hl": "en", "gl": "us"}
    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=12)
        if r.status_code == 401:
            return [{"auth_error": True, "detail": r.text}]
        if r.status_code == 429:
            return [{"rate_limited": True}]
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic", [])
        return [{"name": o.get("title",""), "snippet": o.get("snippet",""), "url": o.get("link","")} for o in organic]
    except Exception as e:
        logging.exception("Serper error: %s", e)
        return []

# --------------------------
# Scan text
# --------------------------
def scan_text(text, api_key, endpoint, *, chunk_size, overlap, results_per_chunk, threshold_pct, delay_sec, retries, ngram_n):
    matches = []
    chunks = split_into_chunks(clean_text(text), chunk_size, overlap)
    threshold = threshold_pct / 100.0
    prog = st.progress(0)

    for idx, chunk in enumerate(chunks, start=1):
        query = query_from_chunk(chunk)
        attempt = 0
        while True:
            results = serper_search(api_key, query, endpoint, results_per_chunk)
            if results and results[0].get("auth_error"):
                st.error("Auth error: check your Serper API key in Streamlit Secrets")
                return matches
            if results and results[0].get("rate_limited"):
                time.sleep(delay_sec + 0.5)
                attempt += 1
                if attempt > retries:
                    break
                continue
            break

        chunk_ngrams = make_ngrams(tokenize_words(chunk), n=ngram_n)
        for r in results:
            snippet_ngrams = make_ngrams(tokenize_words(r.get("snippet","")), n=ngram_n)
            sim = jaccard_similarity(chunk_ngrams, snippet_ngrams)
            if sim >= threshold:
                matches.append({
                    "chunk": idx,
                    "preview": chunk[:140]+"…",
                    "url": r["url"],
                    "title": r["name"],
                    "similarity": round(sim*100,1)
                })

        prog.progress(idx/len(chunks))
        time.sleep(delay_sec)

    prog.progress(1.0)
    return matches

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Plagiarism Checker", page_icon="🧭", layout="wide")
st.title("🧭 Plagiarism Checker")

text_input = st.text_area("Paste text here", height=260)
upload = st.file_uploader("…or upload a .txt file", type=["txt"])
if upload and not text_input:
    text_input = upload.read().decode(errors="ignore")

chunk_size = st.sidebar.slider("Chunk size", 20, 2000, 120)
overlap = st.sidebar.slider("Overlap", 0, 500, 30)
results_per_chunk = st.sidebar.slider("Results per chunk", 1, 20, 6)
threshold_pct = st.sidebar.slider("Flag if similarity ≥ (%)", 10, 95, 55)
delay_sec = st.sidebar.slider("Delay (sec)", 0.1, 2.0, 0.7)
retries = st.sidebar.slider("Retries on 429", 0, 5, 2)
ngram_n = st.sidebar.slider("n-gram size", 2, 8, 5)

# 🔑 Fetch API key directly from secrets
api_key = st.secrets["SERPER_API_KEY"]
endpoint = SERPER_ENDPOINT_DEFAULT

if st.button("Run check"):
    if not text_input.strip():
        st.warning("Please enter some text")
        st.stop()
    if not api_key:
        st.error("No Serper API key found. Please set it in Streamlit Secrets.")
        st.stop()

    st.write(f"Checking {len(text_input.split())} words...")
    results = scan_text(
        text=text_input,
        api_key=api_key,
        endpoint=endpoint,
        chunk_size=chunk_size,
        overlap=overlap,
        results_per_chunk=results_per_chunk,
        threshold_pct=threshold_pct,
        delay_sec=delay_sec,
        retries=retries,
        ngram_n=ngram_n,
    )

    st.subheader("Results")
    if not results:
        st.success("No suspicious matches found.")
    else:
        for r in results:
            with st.expander(f"Chunk {r['chunk']} • {r['similarity']}% • {r['title']}"):
                st.write(f"URL: {r['url']}")
                st.write(f"Preview: {r['preview']}")
