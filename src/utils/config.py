# src/utils/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class AppConfig:
    name: str = "Medical Knowledge & Healthcare Assistant (Pro)"
    model: str = "llama3.2"
    temperature: float = 0.1
    top_k: int = 4
    max_new_tokens: int = 400


@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    vector_store: str = "sklearn"
    persist_dir: str = "data/processed/index"

    # legacy
    hybrid: bool = True
    hybrid_alpha: float = 0.6
    rerank: bool = False
    rerank_model: Optional[str] = None

    max_context_chars: Optional[int] = 12000

    # Speed/quality control for "evidence judge"
    # - "adaptive": heuristic first, LLM only when borderline (recommended)
    # - "always": always use LLM judge (slower)
    # - "never": never use LLM judge (fastest, slightly less strict)
    judge_mode: str = "adaptive"


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"     # dense / bm25 / hybrid
    top_k: int = 8
    hybrid_alpha: float = 0.6

    # IMPORTANT: default False to avoid accidental extra LLM call when YAML isn't loaded.
    use_multi_query: bool = False
    num_query_variants: int = 3

    # Optional rerank
    rerank: bool = False
    rerank_model: Optional[str] = None


@dataclass
class WebConfig:
    allow_web_fallback: bool = True
    tavily_top_k: int = 6
    domain_whitelist: Optional[List[str]] = None

    # Similar judge control for web answerability
    web_judge_mode: str = "adaptive"


@dataclass
class SafetyConfig:
    show_disclaimer: bool = True
    redact_pii: bool = False


@dataclass
class Config:
    app: AppConfig
    rag: RAGConfig
    retrieval: RetrievalConfig
    web: WebConfig
    safety: SafetyConfig


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str = "config.yaml") -> Config:
    data = _read_yaml(Path(path))

    app_data = data.get("app", {}) or {}
    rag_data = data.get("rag", {}) or {}
    retrieval_data = data.get("retrieval", {}) or {}
    web_data = data.get("web", {}) or {}
    safety_data = data.get("safety", {}) or {}

    app = AppConfig(
        name=app_data.get("name", AppConfig.name),
        model=app_data.get("model", AppConfig.model),
        temperature=float(app_data.get("temperature", AppConfig.temperature)),
        top_k=int(app_data.get("top_k", AppConfig.top_k)),
        max_new_tokens=int(app_data.get("max_new_tokens", AppConfig.max_new_tokens)),
    )

    rag = RAGConfig(
        embedding_model=rag_data.get("embedding_model", RAGConfig.embedding_model),
        chunk_size=int(rag_data.get("chunk_size", RAGConfig.chunk_size)),
        chunk_overlap=int(rag_data.get("chunk_overlap", RAGConfig.chunk_overlap)),
        vector_store=rag_data.get("vector_store", RAGConfig.vector_store),
        persist_dir=rag_data.get("persist_dir", RAGConfig.persist_dir),
        hybrid=bool(rag_data.get("hybrid", RAGConfig.hybrid)),
        hybrid_alpha=float(rag_data.get("hybrid_alpha", RAGConfig.hybrid_alpha)),
        rerank=bool(rag_data.get("rerank", RAGConfig.rerank)),
        rerank_model=rag_data.get("rerank_model", RAGConfig.rerank_model),
        max_context_chars=rag_data.get("max_context_chars", RAGConfig.max_context_chars),
        judge_mode=str(rag_data.get("judge_mode", RAGConfig.judge_mode)),
    )

    retrieval = RetrievalConfig(
        mode=retrieval_data.get("mode", RetrievalConfig.mode),
        top_k=int(retrieval_data.get("top_k", RetrievalConfig.top_k)),
        hybrid_alpha=float(retrieval_data.get("hybrid_alpha", RetrievalConfig.hybrid_alpha)),
        use_multi_query=bool(retrieval_data.get("use_multi_query", RetrievalConfig.use_multi_query)),
        num_query_variants=int(retrieval_data.get("num_query_variants", RetrievalConfig.num_query_variants)),
        rerank=bool(retrieval_data.get("rerank", RetrievalConfig.rerank)),
        rerank_model=retrieval_data.get("rerank_model", RetrievalConfig.rerank_model),
    )

    web = WebConfig(
        allow_web_fallback=bool(web_data.get("allow_web_fallback", WebConfig.allow_web_fallback)),
        tavily_top_k=int(web_data.get("tavily_top_k", WebConfig.tavily_top_k)),
        domain_whitelist=web_data.get("domain_whitelist", WebConfig.domain_whitelist),
        web_judge_mode=str(web_data.get("web_judge_mode", WebConfig.web_judge_mode)),
    )

    safety = SafetyConfig(
        show_disclaimer=bool(safety_data.get("show_disclaimer", SafetyConfig.show_disclaimer)),
        redact_pii=bool(safety_data.get("redact_pii", SafetyConfig.redact_pii)),
    )

    return Config(app=app, rag=rag, retrieval=retrieval, web=web, safety=safety)
