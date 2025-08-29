import os
import re
import time
from pathlib import Path
from typing import List, Set, Tuple, Dict

import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import tldextract

# --------------------- Text + shingles ---------------------
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

def make_shingles(words: List[str], k: int) -> Set[Tuple[str, ...]]:
    if len(words) < k:
        return set()
    return {tuple(words[i:i+k]) for i in range(len(words) - k + 1)}

def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def containment(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    denom = min(len(a), len(b))
    return len(a & b) / denom if denom else 0.0

def split_into_word_chunks(text: str, chunk_words: int = 2500, overlap: int = 150) -> List[str]:
    """Split very long documents into overlapping word chunks so we can handle 10k+ words."""
    words = tokenize(text)
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        block = words[i:i+chunk_words]
        chunks.append(" ".join(block))
        if i + chunk_words >= len(words):
            break
        i += chunk_words - overlap
    return chunks

def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join(p for p in [ext.domain, ext.suffix] if p)

def pick_queries_from_text(text: str, max_queries: int = 8) -> List[str]:
    """Pick short sentences or word windows (8–25 words) as search queries."""
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    cands = []
    for s in sentences:
        n = len(WORD_RE.findall(s))
        if 8 <= n <= 25:
            cands.append(s)
    if len(cands) < max_queries:
        w = tokenize(text)
        for i in range(0, len(w), 18):
            chunk = " ".join(w[i:i+18])
            if len(chunk.split()) >= 8:
                cands.append(chunk)
            if len(cands) >= max_queries * 2:
                break
    seen, out = set(), []
    for c in cands:
        k = " ".join(WORD_RE.findall(c.lower()))
        if k not in seen:
            out.append(c)
            seen.add(k)
        if len(out) >= max_queries:
            break
    return out

def compare_texts(doc_text: str, other_text: str, k: int):
    A = make_shingles(tokenize(doc_text), k)
    B = make_shingles(tokenize(other_text), k)
    return {
        "jaccard": jaccard(A, B),
        "containment": containment(A, B),
        "overlap": len(A & B),
        "doc_shingles": len(A),
        "other_shingles": len(B),
    }

# --------------------- Fetch & cache pages ---------------------
@st.cache_data(ttl=3600)
def fetch_url_text(url: str, timeout: int = 12) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PlagCheck/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
            return ""
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
            tag.decompose()
        txt = soup.get_text(separator=" ")
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return ""

# --------------------- Serper.dev search ---------------------
@st.cache_data(ttl=3600)
def serper_search(query: str, api_key: str, num: int = 5):
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    data = {"q": query}
    try:
        r = requests.post("https://google.serper.dev/search", headers=headers, json=data, timeout=10)
        r.raise_for_status()
        res = r.json()
        urls = [item["link"] for item in res.get("organic", [])[:num]]
        return urls
    except Exception as e:
        return []

def require_password():
    pwd = st.secrets.get("APP_PASSWORD")
    if not pwd:
        return True
    if st.session_state.get("authed"):
        return True
    user = st.text_input("Enter app password", type="password")
    if st.button("Unlock"):
        if user == pwd:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

# --------------------- Streamlit app ---------------------
def main():
    st.set_page_config(page_title="Web Plagiarism Checker", page_icon="🔎", layout="wide")
    require_password()

    st.title("🔎 Web Plagiarism Checker (Serper.dev)")
    st.caption("Handles long documents by chunking; uses n-gram similarity (Jaccard & Containment).")

    with st.sidebar:
        st.header("Settings")
        k = st.slider("Shingle size (n-gram)", 3, 8, 5, 1)
        threshold = st.slider("Show matches ≥", 0.05, 0.90, 0.35, 0.05)
        topn = st.slider("Top results per query", 1, 10, 5, 1)
        max_queries_total = st.slider("Max total queries (cap)", 2, 50, 16, 1)
        delay = st.slider("Delay between API calls (sec)", 0.2, 2.0, 0.6, 0.1)

        st.markdown("---")
        st.subheader("Large text")
        chunk_words = st.slider("Words per chunk", 1500, 4000, 2500, 100)
        overlap_words = st.slider("Chunk overlap (words)", 0, 400, 150, 10)

        st.markdown("---")
        st.subheader("API key")
        # Preferred: from secrets
        serper_key = st.secrets.get("SERPER_KEY", "")

        # If you *really* want to hard-code (NOT recommended), uncomment the next line:
        # serper_key = "c1fff5f9a884af5bc6d3466a719bab4c3e4f2689"

        if not serper_key:
            st.warning("Set SERPER_KEY in Streamlit Secrets (recommended).")

    st.subheader("1) Paste text or upload file")
    col1, col2 = st.columns(2)
    with col1:
        text = st.text_area("Paste text here", height=220, placeholder="Paste your content here…")
    with col2:
        f = st.file_uploader("…or upload a .txt / .md", type=["txt", "md"])
        if f and not text.strip():
            text = f.read().decode("utf-8", errors="ignore")

    st.subheader("2) (Optional) Compare with local files")
    local_files = st.file_uploader("Upload multiple .txt/.md", type=["txt", "md"], accept_multiple_files=True)

    if st.button("Run Check", type="primary", disabled=not text.strip()):
        if not text.strip():
            st.error("Please provide text.")
            st.stop()

        # Split long texts so 10k+ words are handled
        chunks = split_into_word_chunks(text, chunk_words=chunk_words, overlap=overlap_words)
        st.write(f"Document split into **{len(chunks)}** chunk(s).")

        # Build search queries from each chunk (but cap total)
        all_queries: List[str] = []
        for ch in chunks:
            per_chunk = max(2, max_queries_total // max(1, len(chunks)))
            all_queries.extend(pick_queries_from_text(ch, max_queries=per_chunk))
        all_queries = all_queries[:max_queries_total]
        st.write(f"Prepared **{len(all_queries)}** search queries.")

        # Web search + compare
        web_rows = []
        seen_domains = set()
        if not serper_key:
            st.error("No SERPER_KEY set. Add it in secrets or hard-code (not recommended).")
        else:
            progress = st.progress(0)
            found_urls = []
            for i, q in enumerate(all_queries, start=1):
                urls = serper_search(q, serper_key, num=topn)
                for u in urls:
                    d = extract_domain(u)
                    if d not in seen_domains:
                        seen_domains.add(d)
                        found_urls.append(u)
                progress.progress(min(i / len(all_queries), 1.0))
                time.sleep(delay)

            st.write(f"Fetched **{len(found_urls)}** unique domains.")
            progress2 = st.progress(0)
            tmp = []
            for i, u in enumerate(found_urls, start=1):
                page_text = fetch_url_text(u)
                if page_text and len(page_text) > 400:
                    scores = compare_texts(text, page_text, k)
                    tmp.append({
                        "url": u,
                        "domain": extract_domain(u),
                        "jaccard": scores["jaccard"],
                        "containment": scores["containment"],
                        "overlap_shingles": scores["overlap"],
                    })
                progress2.progress(min(i / len(found_urls), 1.0))
                time.sleep(delay)
            web_rows = sorted(tmp, key=lambda r: (r["containment"], r["jaccard"]), reverse=True)

        # Local compare
        local_rows = []
        if local_files:
            tmp = []
            for lf in local_files:
                try:
                    other = lf.read().decode("utf-8", errors="ignore")
                    scores = compare_texts(text, other, k)
                    tmp.append({
                        "file": lf.name,
                        "jaccard": scores["jaccard"],
                        "containment": scores["containment"],
                        "overlap_shingles": scores["overlap"],
                    })
                except Exception:
                    pass
            local_rows = sorted(tmp, key=lambda r: (r["containment"], r["jaccard"]), reverse=True)

        # Results
        st.markdown("### Web matches")
        if web_rows:
            df = pd.DataFrame(web_rows)
            df = df[(df["jaccard"] >= threshold) | (df["containment"] >= threshold)]
            if len(df):
                df = df[["jaccard", "containment", "overlap_shingles", "domain", "url"]]
                st.dataframe(df, use_container_width=True,
                             column_config={"url": st.column_config.LinkColumn("url")})
            else:
                st.info("No web pages above threshold.")
        else:
            st.info("No web matches found (or key missing).")

        st.markdown("### Local matches")
        if local_rows:
            df2 = pd.DataFrame(local_rows)
            df2 = df2[(df2["jaccard"] >= threshold) | (df2["containment"] >= threshold)]
            if len(df2):
                st.dataframe(df2, use_container_width=True)
            else:
                st.info("No local files above threshold.")
        else:
            st.caption("No local files uploaded.")

if __name__ == "__main__":
    main()
