from __future__ import annotations
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.utils.config import load_config, Config
# Optional deps (handled gracefully where used)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # type: ignore

# -----------------------------------------------------------------------------
# Globals / caches
# -----------------------------------------------------------------------------
_CFG: Optional[Config] = None
_DENSE_INDEX: Optional["DenseIndex"] = None
_DOCS: Optional[List[Dict[str, Any]]] = None
_BM25: Optional["BM25Okapi"] = None  # type: ignore
# Tavily client cache (safe micro-speedup)
_TAVILY_CLIENT = None

def _get_cfg() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())

# -----------------------------------------------------------------------------
# PDF loading + chunking
# -----------------------------------------------------------------------------
def load_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """Load PDFs into per-page docs: {source, page, text}."""
    docs: List[Dict[str, Any]] = []
    try:
        import pdfplumber
    except Exception:
        return docs

    for p in pdf_paths or []:
        path = str(p)
        if not path or not os.path.exists(path):
            continue
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    txt = (page.extract_text() or "").strip()
                    if txt:
                        docs.append({"source": os.path.basename(path), "page": i, "text": txt})
        except Exception:
            continue
    return docs

def chunk_docs(docs: List[Dict[str, Any]], chunk_size: int = 900, chunk_overlap: int = 120) -> List[Dict[str, Any]]:
    """Chunk per-page docs into overlapping character windows."""
    out: List[Dict[str, Any]] = []
    cs = max(200, int(chunk_size or 900))
    ov = max(0, int(chunk_overlap or 0))
    step = max(50, cs - ov)

    for d in docs or []:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        src = d.get("source", "unknown.pdf")
        page = d.get("page", "?")

        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + cs)
            chunk = text[start:end].strip()
            if chunk:
                out.append({"source": src, "page": page, "start": start, "end": end, "text": chunk})
            if end >= n:
                break
            start += step

    return out

# -----------------------------------------------------------------------------
# Dense index (persisted)
# -----------------------------------------------------------------------------
class DenseIndex:
    def __init__(self, persist_dir: str, embedding_model: str):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")

        self.persist_dir = Path(persist_dir)
        self.embedding_model = embedding_model
        self.docs_path = self.persist_dir / "docs.jsonl"
        self.emb_path = self.persist_dir / "embeddings.npy"

        self.embedder = SentenceTransformer(embedding_model)
        self._docs: Optional[List[Dict[str, Any]]] = None
        self._embeddings: Optional[np.ndarray] = None

    @property
    def docs(self) -> List[Dict[str, Any]]:
        if self._docs is None:
            self.load()
        return self._docs or []

    def _load_docs(self) -> None:
        docs: List[Dict[str, Any]] = []
        if self.docs_path.exists():
            with self.docs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            docs.append(json.loads(line))
                        except Exception:
                            continue
        self._docs = docs

    def _load_embeddings(self) -> None:
        if self.emb_path.exists():
            self._embeddings = np.load(self.emb_path)
        else:
            self._embeddings = np.zeros((0, 384), dtype="float32")

    def build(self, chunks: List[Dict[str, Any]]) -> int:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        with self.docs_path.open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")

        texts = [c.get("text", "") for c in chunks]
        embs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings = embs.astype("float32")
        np.save(self.emb_path, self._embeddings)
        self._docs = chunks
        return len(chunks)

    def load(self) -> None:
        self._load_docs()
        self._load_embeddings()

    def search_dense(self, query: str, top_k: int) -> Dict[int, float]:
        if self._embeddings is None or self._docs is None:
            self.load()
        if self._embeddings is None or self._docs is None or len(self._embeddings) == 0:
            return {}
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = np.dot(self._embeddings, q_emb)
        top_k = min(max(int(top_k), 1), len(scores))
        idxs = np.argpartition(-scores, top_k - 1)[:top_k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return {int(i): float(scores[i]) for i in idxs}

def _get_dense_index() -> DenseIndex:
    global _DENSE_INDEX
    if _DENSE_INDEX is None:
        cfg = _get_cfg()
        _DENSE_INDEX = DenseIndex(cfg.rag.persist_dir, cfg.rag.embedding_model)
    return _DENSE_INDEX

def _get_docs() -> List[Dict[str, Any]]:
    global _DOCS
    if _DOCS is None:
        _DOCS = _get_dense_index().docs
    return _DOCS

def _ensure_bm25() -> Optional["BM25Okapi"]:
    global _BM25
    if _BM25 is None:
        if BM25Okapi is None:
            return None
        docs = _get_docs()
        if not docs:
            return None
        corpus = [_tokenize(d.get("text", "")) for d in docs]
        _BM25 = BM25Okapi(corpus)
    return _BM25

def hybrid_retrieve(question: str, cfg: Config) -> List[Dict[str, Any]]:
    """Hybrid BM25 + dense retrieval."""
    docs = _get_docs()
    if not docs:
        return []
    top_k = max(1, int(cfg.retrieval.top_k or 8))
    mode = (cfg.retrieval.mode or "hybrid").lower()

    dense_scores = _get_dense_index().search_dense(question, top_k * 2) if mode in ("dense", "hybrid") else {}

    bm25_scores: Dict[int, float] = {}
    if mode in ("bm25", "hybrid"):
        bm25 = _ensure_bm25()
        if bm25:
            q_tokens = _tokenize(question)
            scores_arr = bm25.get_scores(q_tokens)
            if len(scores_arr) > 0:
                top_idxs = np.argpartition(-scores_arr, min(top_k * 2, len(scores_arr)) - 1)[: top_k * 2]
                bm25_scores = {int(i): float(scores_arr[i]) for i in top_idxs if scores_arr[i] > 0}

    if mode == "hybrid":
        alpha = float(getattr(cfg.retrieval, "hybrid_alpha", 0.6))
        all_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        fused = {i: alpha * dense_scores.get(i, 0) + (1 - alpha) * bm25_scores.get(i, 0) for i in all_ids}
    elif mode == "bm25":
        fused = bm25_scores
    else:
        fused = dense_scores

    sorted_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]
    return [docs[i] for i in sorted_ids if i < len(docs)]

# -----------------------------------------------------------------------------
# Whitelisted web search
# -----------------------------------------------------------------------------
def _get_tavily_client():
    global _TAVILY_CLIENT
    if _TAVILY_CLIENT is None:
        import tavily
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        _TAVILY_CLIENT = tavily.TavilyClient(api_key=api_key)
    return _TAVILY_CLIENT

def _in_whitelist(url: str, wl: List[str]) -> bool:
    import urllib.parse
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    for d in (wl or []):
        dd = (d or "").lower().lstrip(".")
        if host == dd or host.endswith("." + dd):
            return True
    return False


def _web_search_whitelisted_resources(question: str, cfg: Config) -> Dict[str, Any]:
    """
    FAST: fetch-only whitelisted web resources (NO evidence-judge LLM call, NO web-summary LLM call).
    Returns the same resource objects as _web_search_whitelisted().
    """
    if not cfg.web.allow_web_fallback:
        return {"resources": [], "citations": [], "has_sources": False, "reason": "Web fallback disabled."}

    try:
        import tavily  # noqa: F401
    except ImportError:
        return {"resources": [], "citations": [], "has_sources": False, "reason": "tavily not installed."}

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"resources": [], "citations": [], "has_sources": False, "reason": "Missing TAVILY_API_KEY."}

    wl = list(getattr(cfg.web, "domain_whitelist", []) or [])
    max_results = int(getattr(cfg.web, "tavily_top_k", 8) or 8)

    try:
        client = _get_tavily_client()
        results = client.search(
            query=question,
            search_depth="advanced",
            topic="general",
            include_raw_content=False,
            include_answer=False,
            max_results=max_results,
            include_domains=wl if wl else None,
        )
    except Exception as e:
        return {"resources": [], "citations": [], "has_sources": False, "reason": f"Search failed: {e}"}

    raw_results = (results or {}).get("results", []) or []
    raw_results = [r for r in raw_results if _in_whitelist(r.get("url") or "", wl)]

    resources, citations = [], []
    for i, r in enumerate(raw_results, 1):
        url = r.get("url")
        title = r.get("title") or url or "Web Source"
        snippet = _clean(r.get("content") or r.get("raw_content") or "")
        if len(snippet) > 700:
            snippet = snippet[:700] + "."
        resources.append(
            {"kind": "web", "rank": i, "title": title, "url": url, "snippet": snippet, "citation": url or title}
        )
        if url:
            citations.append(url)

    if not resources:
        return {"resources": [], "citations": [], "has_sources": False, "reason": "No results found in whitelisted domains."}

    return {"resources": resources, "citations": citations, "has_sources": True, "reason": ""}

# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------
def build_index_from_files(pdf_paths: List[str]) -> Dict[str, Any]:
    """Build index from PDFs with diagnostics (prevents 'empty success')."""
    cfg = _get_cfg()

    # keep only valid existing paths
    pdf_paths = [str(p) for p in (pdf_paths or []) if p and os.path.exists(str(p))]
    if not pdf_paths:
        return {"ok": False, "count": 0, "pages_with_text": 0, "persist_dir": cfg.rag.persist_dir,
                "reason": "No valid PDF files found (wrong folder/path or missing files)."}

    # Detect missing dependency early (load_pdfs() currently hides this)
    try:
        import pdfplumber  # noqa: F401
    except Exception as e:
        return {"ok": False, "count": 0, "pages_with_text": 0, "persist_dir": cfg.rag.persist_dir,
                "reason": f"pdfplumber is not installed or failed to import: {e}"}

    docs = load_pdfs(pdf_paths)
    pages_with_text = len(docs)

    if pages_with_text == 0:
        return {"ok": False, "count": 0, "pages_with_text": 0, "persist_dir": cfg.rag.persist_dir,
                "reason": ("Found PDFs, but extracted 0 text pages. "
                           "Common causes: scanned/image PDFs, encrypted PDFs, or extraction failing.")}

    chunks = chunk_docs(docs, cfg.rag.chunk_size, cfg.rag.chunk_overlap)
    if not chunks:
        return {"ok": False, "count": 0, "pages_with_text": pages_with_text, "persist_dir": cfg.rag.persist_dir,
                "reason": "0 chunks produced (unexpected). Check chunk_size/chunk_overlap and PDF text."}

    di = DenseIndex(cfg.rag.persist_dir, cfg.rag.embedding_model)
    n = di.build(chunks)

    global _DENSE_INDEX, _DOCS, _BM25
    _DENSE_INDEX = None
    _DOCS = None
    _BM25 = None

    return {"ok": True, "count": n, "pages_with_text": pages_with_text, "persist_dir": cfg.rag.persist_dir,
            "reason": ""}
