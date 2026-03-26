"""Streamlit frontend for the Agentic RAG system."""

import streamlit as st

from config.settings import settings
from src.graph.builder import build_graph
from src.ingestion.pipeline import run_ingestion
from src.utils.citations import format_citation

# Page config
st.set_page_config(page_title="Agentic RAG", page_icon="?", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Chat", "Ingestion", "Debug"])

# LLM provider toggle
st.sidebar.divider()
st.sidebar.subheader("LLM Backend")
provider = st.sidebar.radio(
    "Provider",
    ["Anthropic Claude", "Local (Qwen)"],
    index=0 if settings.llm_provider == "anthropic" else 1,
)
# Apply the selection to settings so client.py picks it up
settings.llm_provider = "anthropic" if provider == "Anthropic Claude" else "local"
if settings.llm_provider == "local":
    st.sidebar.caption(f"Model: {settings.local_llm_model}")
    st.sidebar.caption(f"URL: {settings.local_llm_base_url}")
else:
    st.sidebar.caption(f"Model: {settings.anthropic_model}")

# Build graph once per session
if "graph" not in st.session_state:
    with st.spinner("Compiling RAG pipeline..."):
        st.session_state.graph = build_graph()

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "thread_counter" not in st.session_state:
    st.session_state.thread_counter = 0


def _run_query(question: str) -> dict:
    """Invoke the LangGraph pipeline and return the full state."""
    st.session_state.thread_counter += 1
    thread_id = f"streamlit-{st.session_state.thread_counter}"
    config = {"configurable": {"thread_id": thread_id}}

    result = st.session_state.graph.invoke(
        {"original_query": question, "messages": []},
        config=config,
    )
    return result


def _confidence_color(confidence) -> str:
    """Map confidence level to a display color."""
    if confidence is None:
        return "gray"
    val = confidence.value if hasattr(confidence, "value") else str(confidence)
    return {"high": "green", "medium": "orange", "low": "red"}.get(val, "gray")


def _confidence_label(confidence) -> str:
    if confidence is None:
        return "unknown"
    return confidence.value if hasattr(confidence, "value") else str(confidence)


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------
if page == "Chat":
    st.title("Legal Document Q&A")

    # Display conversation history
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            conf = entry.get("confidence")
            if conf:
                color = _confidence_color(conf)
                st.markdown(f"**Confidence:** :{color}[{_confidence_label(conf)}]")
            if entry.get("citations"):
                with st.expander("Citations"):
                    for cite in entry["citations"]:
                        if hasattr(cite, "section_ref"):
                            ref, quote = cite.section_ref, cite.quote
                            conf_val = cite.confidence.value if hasattr(cite.confidence, "value") else str(cite.confidence)
                        else:
                            ref = cite.get("section_ref", "")
                            quote = cite.get("quote", "")
                            conf_val = cite.get("confidence", "")
                        st.markdown(f"- {format_citation(ref, quote, conf_val)}")

    # Input
    question = st.chat_input("Ask a question about the legal documents...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and reasoning..."):
                try:
                    result = _run_query(question)
                    st.session_state.last_result = result

                    answer = result.get("synthesis", "No answer generated.")
                    confidence = result.get("confidence")
                    citations = result.get("citations", [])

                    st.write(answer)

                    if confidence:
                        color = _confidence_color(confidence)
                        st.markdown(f"**Confidence:** :{color}[{_confidence_label(confidence)}]")

                    if citations:
                        with st.expander("Citations"):
                            for cite in citations:
                                if hasattr(cite, "section_ref"):
                                    ref, quote = cite.section_ref, cite.quote
                                    conf_val = cite.confidence.value if hasattr(cite.confidence, "value") else str(cite.confidence)
                                else:
                                    ref = cite.get("section_ref", "")
                                    quote = cite.get("quote", "")
                                    conf_val = cite.get("confidence", "")
                                st.markdown(f"- {format_citation(ref, quote, conf_val)}")

                    # Save to history
                    st.session_state.history.append({
                        "question": question,
                        "answer": answer,
                        "confidence": confidence,
                        "citations": citations,
                    })

                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")


# ---------------------------------------------------------------------------
# Ingestion page
# ---------------------------------------------------------------------------
elif page == "Ingestion":
    st.title("Document Ingestion")
    st.write("Point to a directory of legal documents (PDF, HTML, TXT) to ingest them into the system.")

    col1, col2 = st.columns([3, 1])
    with col1:
        input_dir = st.text_input("Document directory path", placeholder="/path/to/documents")
    with col2:
        collection = st.text_input("Collection name", value="legal_docs")

    if st.button("Run Ingestion", type="primary", disabled=not input_dir):
        with st.spinner("Running ingestion pipeline..."):
            try:
                stats = run_ingestion(input_dir, collection)

                st.success("Ingestion complete!")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Files Found", stats["files_found"])
                col_b.metric("Documents Parsed", stats["documents_parsed"])
                col_c.metric("Chunks Created", stats["chunks"])
                col_d.metric("Loaded to Qdrant", stats["loaded"])

                graph_stats = stats.get("graph", {})
                if graph_stats:
                    st.subheader("Knowledge Graph")
                    gc1, gc2 = st.columns(2)
                    gc1.metric("Nodes Created", graph_stats.get("nodes_created", 0))
                    gc2.metric("Edges Created", graph_stats.get("edges_created", 0))

            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")


# ---------------------------------------------------------------------------
# Debug page
# ---------------------------------------------------------------------------
elif page == "Debug":
    st.title("Debug / Retrieval View")

    result = st.session_state.last_result
    if result is None:
        st.info("Run a query on the Chat page first. Debug details will appear here.")
    else:
        # Query classification
        with st.expander("Query Classification", expanded=True):
            st.write(f"**Original query:** {result.get('original_query', 'N/A')}")
            query_type = result.get("query_type")
            if query_type:
                qt_val = query_type.value if hasattr(query_type, "value") else str(query_type)
                st.write(f"**Query type:** {qt_val}")

        # Retrieval plan
        plan = result.get("retrieval_plan")
        if plan:
            with st.expander("Retrieval Strategy", expanded=True):
                if hasattr(plan, "primary_strategy"):
                    st.write(f"**Primary strategy:** {plan.primary_strategy.value}")
                    if plan.secondary_strategies:
                        secondary = ", ".join(s.value for s in plan.secondary_strategies)
                        st.write(f"**Secondary strategies:** {secondary}")
                    if plan.expected_cross_refs:
                        st.write(f"**Expected cross-refs:** {', '.join(plan.expected_cross_refs)}")
                else:
                    st.json(plan if isinstance(plan, dict) else plan.model_dump())

        # Retrieved chunks
        retrieved = result.get("retrieved_results", [])
        if retrieved:
            with st.expander(f"Retrieved Chunks ({sum(len(r.chunks) for r in retrieved if hasattr(r, 'chunks'))} total)"):
                for rr_idx, rr in enumerate(retrieved):
                    if hasattr(rr, "strategy_used"):
                        st.markdown(f"**Result set {rr_idx + 1}** - strategy: `{rr.strategy_used.value}`")
                    chunks = rr.chunks if hasattr(rr, "chunks") else rr.get("chunks", [])
                    for chunk in chunks:
                        if hasattr(chunk, "content"):
                            score = chunk.score
                            ref = chunk.section_ref
                            text = chunk.content[:300]
                        else:
                            score = chunk.get("score", 0)
                            ref = chunk.get("section_ref", "")
                            text = chunk.get("content", "")[:300]
                        st.markdown(f"- **[{ref}]** (score: {score:.3f}): {text}...")
                    gaps = rr.coverage_gaps if hasattr(rr, "coverage_gaps") else rr.get("coverage_gaps", [])
                    if gaps:
                        st.warning(f"Coverage gaps: {', '.join(gaps)}")

        # Cross-references
        resolved_refs = result.get("resolved_cross_refs", [])
        pending_refs = result.get("pending_cross_refs", [])
        if resolved_refs or pending_refs:
            with st.expander(f"Cross-References ({len(resolved_refs)} resolved, {len(pending_refs)} pending)"):
                for xref in resolved_refs:
                    if hasattr(xref, "source_section"):
                        st.markdown(f"- {xref.source_section} -> {xref.target_section}: {xref.reference_text}")
                    else:
                        st.markdown(f"- {xref.get('source_section', '')} -> {xref.get('target_section', '')}")
                for xref in pending_refs:
                    if hasattr(xref, "source_section"):
                        st.markdown(f"- (pending) {xref.source_section} -> {xref.target_section}")
                    else:
                        st.markdown(f"- (pending) {xref.get('source_section', '')} -> {xref.get('target_section', '')}")

        # Confidence and iteration info
        with st.expander("Reasoning Summary"):
            confidence = result.get("confidence")
            if confidence:
                conf_val = confidence.value if hasattr(confidence, "value") else str(confidence)
                st.write(f"**Final confidence:** {conf_val}")
            st.write(f"**Iterations used:** {result.get('iteration_count', 'N/A')}")
            st.write(f"**Max iterations:** {result.get('max_iterations', 'N/A')}")

            synthesis = result.get("synthesis", "")
            if synthesis:
                st.markdown("**Synthesized answer:**")
                st.markdown(synthesis)
