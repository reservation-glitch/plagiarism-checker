# SPDX-License-Identifier: MIT
# Plagiarism Checker (Serper + page fetch) with robust progress and rule-based paraphraser
import re
import time
import logging
from typing import List, Dict, Tuple

import requests
import streamlit as st
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("plagcheck")

SERPER_ENDPOINT = "https://google.serper.dev/search"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# ==========================
# Text utilities
# ==========================
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\u200b", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return clean_text(text)

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

def chunk_text(text: str, size: int, overlap: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Returns chunks AND the word-span (start_idx, end_idx) for each chunk,
    so we can map back to the original text for paraphrasing.
    """
    words = text.split()
    if not words:
        return [], []
    stride = max(1, size - overlap)
    chunks = []
    spans = []
    for i in range(0, len(words), stride):
        start = i
        end = min(i + size, len(words))
        part = words[start:end]
        if not part:
            break
        chunks.append(" ".join(part))
        spans.append((start, end))  # inclusive/exclusive word indices
        if end == len(words):
            break
    return chunks, spans

def make_query_from_chunk(chunk: str, query_words: int = 12) -> str:
    return " ".join(chunk.split()[:query_words])

# ==========================
# Serper + fetch
# ==========================
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
        return [
            {
                "title": o.get("title", ""),
                "snippet": o.get("snippet", ""),
                "url": o.get("link", ""),
            }
            for o in organic
        ]
    except Exception as e:
        LOG.exception("Serper error: %s", e)
        return []

def fetch_page_text(url: str, timeout: float = 12.0, max_chars: int = 60_000) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        txt = strip_html(r.text)
        return txt[:max_chars]
    except Exception as e:
        LOG.debug("Fetch fail %s: %s", url, e)
        return ""

# ==========================
# Rule-based Paraphraser (fast, dependency-light)
# ==========================
SYN_MAP = {
    "important": "crucial",
    "very": "highly",
    "really": "truly",
    "many": "numerous",
    "big": "large",
    "small": "compact",
    "help": "assist",
    "show": "demonstrate",
    "use": "utilize",
    "make": "create",
    "get": "obtain",
    "good": "beneficial",
    "bad": "harmful",
    "because": "since",
    "therefore": "thus",
    "but": "however",
    "also": "additionally",
    "so": "hence",
    "begin": "commence",
    "start": "initiate",
    "end": "conclude",
}

def simple_syn_swap(text: str) -> str:
    def repl(m):
        w = m.group(0)
        lw = w.lower()
        if lw in SYN_MAP:
            repl_word = SYN_MAP[lw]
            # preserve capitalization
            if w[0].isupper():
                repl_word = repl_word.capitalize()
            return repl_word
        return w
    return re.sub(r"\b[A-Za-z]+\b", repl, text)

def reorder_commas(text: str) -> str:
    """Swap the first two comma-separated clauses if present."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) >= 3:
        parts[0], parts[1] = parts[1], parts[0]
        return ", ".join(parts)
    return text

def smooth_fillers(text: str) -> str:
    text = re.sub(r"\b(in order to)\b", "to", text, flags=re.I)
    text = re.sub(r"\b(a lot of)\b", "many", text, flags=re.I)
    text = re.sub(r"\b(kind of|sort of)\b", "rather", text, flags=re.I)
    return text

def paraphrase_chunk(chunk: str) -> str:
    """A quick, safe paraphrase (does not aim for creativity)."""
    chunk = smooth_fillers(chunk)
    chunk = reorder_commas(chunk)
    chunk = simple_syn_swap(chunk)
    return chunk

def merge_and_paraphrase(original_text: str, word_spans: List[Tuple[int, int]], flagged_indices: List[int]) -> str:
    """
    Replace flagged chunk spans (by chunk index) with paraphrased text,
    avoiding repeated replacement on overlaps.
    """
    words = original_text.split()
    if not words:
        return original_text

    # Merge overlapping spans for flagged chunks
    flagged_spans = [word_spans[i] for i in flagged_indices]
    if not flagged_spans:
        return original_text

    flagged_spans.sort(key=lambda x: x[0])
    merged = []
    cur_s, cur_e = flagged_spans[0]
    for s, e in flagged_spans[1:]:
        if s <= cur_e:  # overlap/adjacent
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # Replace each merged span with paraphrased version of that segment
    out_words = []
    last = 0
    for s, e in merged:
        # append clean segment before
        out_words.extend(words[last:s])
        segment = " ".join(words[s:e])
        para = paraphrase_chunk(segment)
        out_words.extend(para.split())
        last = e

    # tail
    out_words.extend(words[last:])
    return " ".join(out_words)

# ==========================
# Core scan (robust)
# ==========================
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
) -> Tuple[List[Dict], str]:
    """Return (matches, paraphrased_text)."""
    matches: List[Dict] = []
    cleaned = clean_text(text)
    chunks, spans = chunk_text(cleaned, size=chunk_size, overlap=overlap)
    if not chunks:
        return matches, cleaned

    thresh = threshold_pct / 100.0
    prog = st.progress(0.0)

    # Keep track of which chunks are truly flagged
    flagged_chunk_indices: List[int] = []

    total = len(chunks)
    for i, chunk in enumerate(chunks):
        # Always move the progress in a finally block so it never freezes at 80%
        try:
            query = make_query_from_chunk(chunk)

            # Resilient Serper with backoff
            tries = 0
            while True:
                sres = serper_search(api_key, query, count=results_per_chunk)
                if sres and sres[0].get("auth_error"):
                    st.error("Authentication error: check your Serper API key in **Secrets**.")
                    return matches, cleaned
                if sres and sres[0].get("rate_limited"):
                    tries += 1
                    time.sleep(delay_sec + 0.6)
                    if tries > retries:
                        sres = []
                        break
                    continue
                break

            # Prepare chunk tokens/ngrams once
            chunk_tokens = tokenize_words(chunk)
            chunk_ngrams = make_ngrams(chunk_tokens, n=ngram_n)

            local_flagged = False

            for hit in sres:
                url = hit.get("url") or ""
                if not url:
                    continue

                page_text = fetch_page_text(url)
                if not page_text:
                    page_text = hit.get("snippet", "")

                # Similarity signals in separate try so one broken page won't kill the chunk
                try:
                    page_tokens = tokenize_words(page_text)
                    page_ngrams = make_ngrams(page_tokens, n=ngram_n)

                    j_sim = jaccard(chunk_ngrams, page_ngrams)
                    s_sim = seq_ratio(chunk, page_text)

                    if (require_both_signals and j_sim >= thresh and s_sim >= thresh) or \
                       (not require_both_signals and (j_sim >= thresh or s_sim >= thresh)):
                        matches.append({
                            "chunk": i + 1,
                            "url": url,
                            "title": hit.get("title", ""),
                            "jaccard_pct": round(j_sim * 100, 1),
                            "seqmatch_pct": round(s_sim * 100, 1),
                            "preview": chunk[:180] + "…"
                        })
                        local_flagged = True
                except Exception as sim_err:
                    LOG.debug("Similarity error for %s: %s", url, sim_err)
                    continue

            if local_flagged:
                flagged_chunk_indices.append(i)

            time.sleep(delay_sec)

        except Exception as loop_err:
            LOG.debug("Chunk loop error idx=%s: %s", i, loop_err)
            # continue with next chunk
        finally:
            prog.progress((i + 1) / total)

    # Build paraphrased version with only flagged spans replaced
    paraphrased = merge_and_paraphrase(cleaned, spans, flagged_chunk_indices)
    return matches, paraphrased

# ==========================
# UI
# ==========================
st.set_page_config(page_title="Plagiarism Checker", page_icon="🧭", layout="wide")
st.title("🧭 Plagiarism Checker (Serper + full-page matching)")

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
    else:
        chunk_size = st.slider("Chunk size (words)", 20, 2000, 80)
        overlap = st.slider("Overlap (words)", 0, 500, 30)
        results_per_chunk = st.slider("Search results per chunk", 1, 20, 8)
        threshold_pct = st.slider("Flag if similarity ≥ (%)", 10, 95, 50)
        delay_sec = st.slider("Delay between searches (sec)", 0.1, 2.0, 0.8)

    retries = st.slider("Retries on 429", 0, 5, 2)
    ngram_n = st.slider("n-gram size", 2, 10, 5)
    require_both_signals = st.checkbox(
        "Require BOTH signals (Jaccard & SequenceMatcher) to exceed threshold",
        value=False
    )

col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Paste text to check", height=280, placeholder="Paste up to ~12,000 words…")
with col2:
    up = st.file_uploader("…or upload .txt", type=["txt"])
    if up and not text:
        text = up.read().decode(errors="ignore")

# API key from Secrets
try:
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
except Exception:
    SERPER_API_KEY = None

if st.button("Run check", type="primary"):
    if not text.strip():
        st.warning("Please paste some text or upload a file.")
        st.stop()
    if not SERPER_API_KEY:
        st.error("No Serper API key found. Add it to **Secrets** as `SERPER_API_KEY`.")
        st.stop()

    st.info(f"Scanning ~{len(text.split())} words • mode: **{mode}**")
    hits, paraphrased = scan_text(
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

    # 1) Matches
    st.subheader("Results")
    if not hits:
        st.success("No suspicious matches found.")
    else:
        hits = sorted(hits, key=lambda x: (x["jaccard_pct"] + x["seqmatch_pct"]), reverse=True)
        for h in hits:
            with st.expander(
                f"Chunk {h['chunk']} • Jaccard {h['jaccard_pct']}% • SeqMatch {h['seqmatch_pct']}% • {h['title']}"
            ):
                st.write(h["url"])
                st.write("Preview of your text:\n\n> " + h["preview"])

    # 2) Paraphrase (separate section)
    st.subheader("Paraphrased output (only flagged spans changed)")
    st.caption("Use this as a draft; please review for meaning and tone before final use.")
    st.text_area("Paraphrased text", value=paraphrased, height=260)
    st.download_button("Download paraphrased.txt", paraphrased, file_name="paraphrased.txt")
