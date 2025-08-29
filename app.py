import os
import time
import re
import html
import math
import logging
from urllib.parse import quote_plus

import requests
import streamlit as st
from bs4 import BeautifulSoup

# ---------------------------
# CONFIG / DEFAULTS (Aggressive)
# ---------------------------
DEFAULT_CHUNK_SIZE = 30          # words
DEFAULT_OVERLAP = 12             # words
DEFAULT_RESULTS_PER_CHUNK = 10   # try 20 if you want even more recall
DEFAULT_THRESHOLD = 0.42         # 42% similarity
DEFAULT_DELAY = 1.1              # seconds between web queries

BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

# ---------------------------
# SIMPLE TOKEN HELPERS
# ---------------------------
_word_re = re.compile(r"[A-Za-z0-9']+")

def normalize_text(txt: str) -> str:
    # unescape & compress whitespace
    t = html.unescape(txt or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str) -> list:
    return [w.lower() for w in _word_re.findall(text)]

def jaccard_similarity(a_tokens: list, b_tokens: list) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union

# ---------------------------
# CHUNKING
# ---------------------------
def make_chunks(text: str, chunk_size: int, overlap: int) -> list:
    tokens = tokenize(text)
    if not tokens:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, max(1, len(tokens) - chunk_size + 1), step):
        piece = tokens[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
    # edge case: if text shorter than chunk size, just one chunk
    if not chunks and tokens:
        chunks.append(" ".join(tokens))
    return chunks

# ---------------------------
# BING WEB SEARCH
# ---------------------------
def bing_search(api_key: str, query: str, count: int = 10) -> list:
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": max(1, min(count, 50)),  # bing caps count
        "mkt": "en-US",
        "safeSearch": "Off",
        "textDecorations": False,
        "textFormat": "Raw",
    }
    try:
        r = requests.get(BING_ENDPOINT, headers=headers, params=params, timeout=12)
        if r.status_code == 429:
            # rate-limited
            return [{"rate_limited": True}]
        r.raise_for_status()
        data = r.json()
        web_pages = data.get("webPages", {}).get("value", [])
        return [
            {
                "name": item.get("name") or "",
                "snippet": item.get("snippet") or "",
                "url": item.get("url") or "",
            }
            for item in web_pages
        ]
    except Exception as e:
        logging.exception("Bing search error: %s", e)
        return []

# ---------------------------
# FETCH PAGE TEXT
# ---------------------------
def fetch_page_text(url: str, max_chars: int = 120_000) -> str:
    if not url.startswith("http"):
        return ""
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        html_text = r.text[:max_chars]
        soup = BeautifulSoup(html_text, "html.parser")
        # remove scripts/styles/navs
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        # join paragraphs
        text = " ".join(x.get_text(separator=" ", strip=True) for x in soup.find_all(["p", "article", "div"]))
        return normalize_text(text)[:max_chars]
    except Exception:
        return ""

# ---------------------------
# SCAN LOGIC
# ---------------------------
def scan_text(api_key: str,
              full_text: str,
              chunk_size: int,
              overlap: int,
              results_per_chunk: int,
              threshold: float,
              delay: float):
    matches = []
    chunks = make_chunks(full_text, chunk_size, overlap)

    st.write(f"**Chunks:** {len(chunks)}  •  **Chunk size:** {chunk_size} words  •  **Overlap:** {overlap}")
    if not chunks:
        return matches

    total = len(chunks)
    progress = st.progress(0, text="Scanning…")

    for i, chunk in enumerate(chunks, start=1):
        # Query: we use the chunk itself (short enough)
        q = chunk
        results = bing_search(api_key, q, count=results_per_chunk)

        # Rate limit handling
        if results and isinstance(results[0], dict) and results[0].get("rate_limited"):
            time.sleep(max(2.0, delay * 2))
            results = bing_search(api_key, q, count=results_per_chunk)

        chunk_tokens = chunk.split()
        for item in results:
            url = item.get("url", "")
            # first compare with the snippet to short-circuit
            snippet_text = normalize_text(f"{item.get('name','')} {item.get('snippet','')}")
            sim1 = jaccard_similarity(chunk_tokens, tokenize(snippet_text))
            best_sim = sim1
            best_text = "snippet"

            # If snippet weak, try full page
            if best_sim < threshold and url:
                page_text = fetch_page_text(url)
                if page_text:
                    sim2 = jaccard_similarity(chunk_tokens, tokenize(page_text))
                    if sim2 > best_sim:
                        best_sim = sim2
                        best_text = "page"

            if best_sim >= threshold:
                matches.append({
                    "similarity": best_sim,
                    "url": url,
                    "used": best_text,
                    "excerpt": item.get("snippet", "")[:300],
                })

        progress.progress(i / total, text=f"Scanning chunk {i}/{total}")
        time.sleep(max(0.1, delay))

    # sort high → low
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Web Plagiarism Checker", page_icon="🔎", layout="wide")

st.title("🔎 Web Plagiarism Checker (Aggressive Mode Defaults)")

api_key = st.text_input(
    "Bing Web Search API Key",
    value=os.getenv("BING_API_KEY", ""),
    type="password",
    help="Paste your Bing Web Search v7 key."
)

text = st.text_area("Paste the text to check", height=300, placeholder="Paste up to ~12–15k words…")

with st.expander("Scan settings", expanded=True):
    cols = st.columns(5)
    with cols[0]:
        chunk_size = st.number_input("Chunk size (words)", min_value=10, max_value=200, value=DEFAULT_CHUNK_SIZE, step=2)
    with cols[1]:
        overlap = st.number_input("Overlap (words)", min_value=0, max_value=100, value=DEFAULT_OVERLAP, step=1)
    with cols[2]:
        results_per_chunk = st.slider("Search results per chunk", 1, 20, DEFAULT_RESULTS_PER_CHUNK)
    with cols[3]:
        threshold_perc = st.slider("Flag if similarity ≥ (%)", 20, 90, int(DEFAULT_THRESHOLD * 100))
        threshold = threshold_perc / 100.0
    with cols[4]:
        delay = st.number_input("Delay between searches (sec)", min_value=0.1, max_value=2.5, value=DEFAULT_DELAY, step=0.1)

run = st.button("Run Check", type="primary", disabled=not (api_key and text.strip()))

st.divider()

if run:
    with st.spinner("Working…"):
        results = scan_text(
            api_key=api_key,
            full_text=text,
            chunk_size=int(chunk_size),
            overlap=int(overlap),
            results_per_chunk=int(results_per_chunk),
            threshold=float(threshold),
            delay=float(delay),
        )

    if not results:
        st.success("No suspicious matches **above your threshold** were found.")
        st.info("Tip: Decrease the threshold, reduce chunk size, or increase results per chunk for a broader scan.")
    else:
        st.subheader(f"Matches found ({len(results)})")
        for m in results[:300]:  # don't render infinite
            st.markdown(
                f"- **Similarity:** {m['similarity']:.0%}  •  **Source:** [{m['url']}]({m['url']})  "
                f"• *(matched on {m['used']})*\n\n"
                f"  > {m['excerpt']}"
            )
