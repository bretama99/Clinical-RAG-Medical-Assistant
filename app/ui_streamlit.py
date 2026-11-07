from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.config import load_config
from src.llm.ollama_client import diagnose_ollama, ask_ollama
from src.rag.pipeline import hybrid_retrieve, _web_search_whitelisted_resources, _clean
from src.agents.multi_agent import agentic_answer
from src.rag.verifier import verify_answer, filter_pdf_chunks, filter_web_results
from src.prompting.prompts import SIMPLE_SYSTEM, SIMPLE_USER, NOT_FOUND_MSG

st.set_page_config(page_title="Medical RAG Assistant", layout="wide")

def _find_pdfs(pdf_dir: Path) -> list[str]:
    return [str(p) for p in sorted(pdf_dir.glob("*.pdf"))]

def _build_index(cfg, pdf_dir: Path) -> tuple[bool, str]:
    try:
        import src.rag.pipeline as pipeline
        pdfs = _find_pdfs(pdf_dir)

        if hasattr(pipeline, "build_index_from_files") and callable(pipeline.build_index_from_files):
            if not pdfs:
                return False, f"No PDFs found in {pdf_dir}"

            stats = pipeline.build_index_from_files(pdfs) or {}
            if not stats.get("ok"):
                return False, (
                    f"Index build failed (empty index). "
                    f"PDFs: {len(pdfs)}, pages_with_text: {stats.get('pages_with_text', 0)}. "
                    f"Reason: {stats.get('reason', '(unknown)')}"
                )

            return True, (
                f"Index built ✅ PDFs: {len(pdfs)}, pages_with_text: {stats.get('pages_with_text', 0)}, "
                f"chunks: {stats.get('count', 0)} → {stats.get('persist_dir')}"
            )

        return False, "No index builder found in src.rag.pipeline."
    except Exception as e:
        return False, f"Index build failed: {e}"

def _evidence_pack(question: str, also_web: bool, cfg) -> dict:
    raw_chunks = hybrid_retrieve(question, cfg) or []
    pdf_chunks = filter_pdf_chunks(question, raw_chunks, max_keep=cfg.retrieval.top_k)

    web_results = []
    if also_web:
        # ✅ speed: fetch-only resources (no extra LLM calls), identical resources output
        web = _web_search_whitelisted_resources(question, cfg) or {}
        web_results = web.get("resources") or []
        web_results = filter_web_results(question, web_results, max_keep=cfg.web.tavily_top_k if hasattr(cfg, "web") else 16)

    return {"raw_pdf_n": len(raw_chunks), "pdf_chunks": pdf_chunks, "web_results": web_results}

def _chunks_to_pdf_evidence(chunks: list[dict]) -> tuple[list[dict], str]:
    res, parts = [], []
    for ch in chunks:
        src, page = ch.get("source", "unknown.pdf"), ch.get("page", "?")
        txt = (ch.get("text", "") or "").strip()
        if not txt:
            continue
        res.append({"source": src, "page": page, "snippet": txt[:500] + ("…" if len(txt) > 500 else "")})
        parts.append(f"[{src}:p{page}] {txt}")
    return res, _clean("\n\n".join(parts))[:12000] or "NO_PDF_EVIDENCE"

def _results_to_web_evidence(results: list[dict]) -> tuple[list[dict], str]:
    res, parts = [], []
    for r in results:
        snip = (r.get("snippet") or r.get("content") or "").strip()
        if not snip:
            continue
        res.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": snip[:600] + ("…" if len(snip) > 600 else ""),
            }
        )
        parts.append(snip)
    return res, _clean("\n\n".join(parts))[:12000] or "NO_WEB_EVIDENCE"

def _strict_finalize_answer(question: str, answer: str, pdf_chunks: list[dict], web_results: list[dict]) -> tuple[str, dict]:
    v = verify_answer(question, answer, pdf_chunks=pdf_chunks, web_results=web_results)
    # hard-enforce: if not supported → NOT_FOUND
    if not v.get("ok"):
        return NOT_FOUND_MSG, v
    return answer, v

def main():
    cfg = load_config()
    st.title(cfg.app.name)
    st.caption("Informational support using uploaded PDFs + whitelisted medical sites. Not medical advice.")
    diag = diagnose_ollama()
    ollama_ok = bool(diag.get("running"))
    pdf_dir = ROOT / "data" / "raw"

    with st.sidebar:
        st.header("Settings")
        # ✅ Build button restored
        if st.button("🔨 Build / Rebuild PDF Index", use_container_width=True):
            with st.spinner("Building PDF index..."):
                ok, msg = _build_index(cfg, pdf_dir)
            (st.success if ok else st.error)(msg)
        st.divider()
        mode = st.radio("Mode", ["Simple", "Agentic"], index=0)
        also_web = st.checkbox("Also search whitelisted web", value=True)
        st.markdown("**Difference:**")
        st.write("- **Simple**: 1 retrieval → 1 answer (fast).")
        st.write("- **Agentic**: decompose → critique → improve (deeper, slower).")
        st.divider()
        if ollama_ok:
            st.success("Ollama: running")
        else:
            st.warning("Ollama: not reachable (answering disabled). You can still build the index.")
            
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "✅ Answer Verifier", "📚 Index / Logs"])

    st.session_state.setdefault("last_question", "")
    st.session_state.setdefault("last_answer", "")
    st.session_state.setdefault("last_pdf_chunks", [])
    st.session_state.setdefault("last_web_results", [])
    st.session_state.setdefault("agentic_debug", None)

    with tab1:
        q = st.text_input(
            "Ask a medical/health question",
            value="",
            placeholder="e.g. common symptoms of iron-deficiency anemia",
        )
        run = st.button("Run", type="primary", disabled=not ollama_ok)
        if run and q.strip():
            q = q.strip()
            with st.spinner("Retrieving evidence."):
                pack = _evidence_pack(q, also_web, cfg)
            pdf_chunks = pack["pdf_chunks"]
            web_results = pack["web_results"]
            st.session_state["last_question"] = q
            st.session_state["last_pdf_chunks"] = pdf_chunks
            st.session_state["last_web_results"] = web_results
            # ✅ HARD GATE: if nothing relevant, stop here (NO LLM CALL)
            if not pdf_chunks and not web_results:
                st.markdown("## Answer")
                st.warning(NOT_FOUND_MSG)
                st.session_state["last_answer"] = NOT_FOUND_MSG
                st.markdown("## Evidence found")
                st.write("No relevant evidence after filtering.")
                return
            pdf_res, pdf_evidence = _chunks_to_pdf_evidence(pdf_chunks)
            web_res, web_evidence = _results_to_web_evidence(web_results)
            with st.spinner("Generating answer."):
                if mode == "simple":
                    draft = ask_ollama(
                        system=SIMPLE_SYSTEM,
                        user=SIMPLE_USER.format(question=q, pdf_evidence=pdf_evidence, web_evidence=web_evidence),
                        model=cfg.app.model,
                        temperature=cfg.app.temperature,
                        num_predict=cfg.app.max_new_tokens,
                        raise_on_fail=False,
                    ).strip()
                    final, v = _strict_finalize_answer(q, draft, pdf_chunks, web_results)
                else:
                    # ✅ pass cfg (avoid reload), no behavior change
                    agent = agentic_answer(
                        q,
                        also_search_web=also_web,
                        provided_pdf_chunks=pdf_chunks,
                        provided_web_results=web_results,
                        cfg=cfg,
                    )
                    draft = (agent.get("answer") or "").strip()
                    final, v = _strict_finalize_answer(q, draft, pdf_chunks, web_results)
                    st.session_state["agentic_debug"] = agent

            st.markdown("## Answer")
            st.write(final)
            st.session_state["last_answer"] = final

            # Show agentic details clearly
            if mode == "agentic" and st.session_state.get("agentic_debug"):
                a = st.session_state["agentic_debug"]
                with st.expander("🧠 Agentic details"):
                    if a.get("subquestions"):
                        st.write("**Subquestions:**", a["subquestions"])
                    if a.get("critic_review"):
                        st.write("**Critic review:**")
                        st.write(a["critic_review"])
            st.markdown("## Evidence found")
            st.caption(
                f"Searched {pack['raw_pdf_n']} PDF chunks → kept {len(pdf_chunks)} relevant. "
                f"Kept {len(web_results)} relevant web results."
            )

            st.markdown("### PDFs")
            if not pdf_res:
                st.write("No relevant PDF evidence.")
            else:
                for r in pdf_res:
                    st.write(f"- **{r['source']}** p{r['page']}: {r['snippet']}")
            st.markdown("### Whitelisted Web")
            if not web_res:
                st.write("No relevant web evidence.")
            else:
                for r in web_res:
                    st.write(f"- **{r['title'] or 'Source'}** ({r['url']}): {r['snippet']}")
            with st.expander("✅ Verification (strict)"):
                st.write(f"OK: {v.get('ok')} | support: {v.get('support')} | reason: {v.get('reason')}")

    with tab2:
        qv = st.text_input("Question", value=st.session_state.get("last_question", ""))
        av = st.text_area("Answer", value=st.session_state.get("last_answer", ""), height=160)
        if st.button("Verify answer"):
            pdf_chunks = st.session_state.get("last_pdf_chunks", []) or []
            web_results = st.session_state.get("last_web_results", []) or []
            v = verify_answer(qv.strip(), av.strip(), pdf_chunks=pdf_chunks, web_results=web_results)
            st.markdown("## 📋 Verification Results")
            st.write(f"**OK:** {v.get('ok')}")
            st.write(f"**Support:** {v.get('support')}")
            st.write(f"**Reason:** {v.get('reason')}")
            for b in (v.get("best_sources") or []):
                st.write(f"- **{b['kind']}** {b['source']} (score {b['score']}): {b['snippet']}")

    with tab3:
        st.markdown("## PDF Index")
        st.write(f"PDF folder: `{pdf_dir}`")
        pdfs = _find_pdfs(pdf_dir)
        st.write(f"Found **{len(pdfs)}** PDFs.")
        if pdfs:
            with st.expander("Show PDF list"):
                for p in pdfs[:200]:
                    st.write(f"- {Path(p).name}")

        st.markdown("---")
        st.markdown("## Ollama diagnostics")
        st.json(diag)

if __name__ == "__main__":
    main()
