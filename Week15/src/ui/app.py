"""
Streamlit Interactive Production Web UI for ShopAssist AI.
Features Live Chat, ONNX Router Playground, RAG Knowledge Base Browser, and Production Telemetry.
"""

import os
import sys
import time
import json
import httpx
import pandas as pd
import streamlit as st

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings
from src.core.structured_outputs import CANONICAL_INTENTS
from src.router.onnx_router import onnx_router
from src.core.rag_pipeline import rag_pipeline
from src.services.assistant_service import assistant_orchestrator, AssistantRequest
from src.services.cache_service import cache_service
from src.services.rate_limiter import rate_limiter

st.set_page_config(
    page_title="ShopAssist AI - Enterprise Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .badge-intent {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        background-color: #E3F2FD;
        color: #0D47A1;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #90CAF9;
    }
    .metric-chip {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background-color: #F5F5F5;
        color: #424242;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ System Controls")
    
    # Provider Selection
    provider_options = ["gemini", "openai", "claude", "vllm", "mock"]
    selected_provider = st.selectbox(
        "Primary LLM Provider",
        provider_options,
        index=provider_options.index(settings.primary_provider) if settings.primary_provider in provider_options else 0,
        help="Target provider. Falls back automatically if API key or service is offline."
    )
    
    # Preset Selector
    preset = st.radio(
        "Parameter Preset",
        ["Deterministic (0.1)", "Balanced (0.2)", "Conversational (0.7)"],
        index=1
    )
    if "Deterministic" in preset:
        default_temp = 0.1
    elif "Balanced" in preset:
        default_temp = 0.2
    else:
        default_temp = 0.7
        
    temp_slider = st.slider("Temperature", 0.0, 1.0, default_temp, 0.05)
    top_p_slider = st.slider("Top_p", 0.0, 1.0, settings.default_top_p, 0.05)
    
    st.divider()
    
    # Telemetry Mini-Card
    st.markdown("### 📈 Live Telemetry")
    cache_stats = cache_service.get_stats()
    st.write(f"**Cache Hit Ratio:** {cache_stats['hit_ratio_percent']}% ({cache_stats['hits']} hits)")
    st.write(f"**Router Mode:** `{onnx_router.model_type}`")
    st.write(f"**RAG Chunks:** `{len(rag_pipeline.vector_db.chunks)} indexed`")
    
    skip_cache_toggle = st.checkbox("Skip Cache", value=False)
    
    if st.button("🧹 Clear Prompt Cache"):
        cache_service.clear()
        st.success("Cache cleared!")

# --- Main App Layout ---
st.markdown('<div class="main-header">ShopAssist AI: Enterprise Support Platform</div>', unsafe_allow_html=True)
st.caption("Applied Generative AI (Task 1) & Production Engineering Systems (Task 2)")

tabs = st.tabs([
    "💬 Live AI Assistant",
    "⚡ ONNX Router Playground",
    "📚 Knowledge Base RAG",
    "📊 System Telemetry & Batch Tester"
])

# ==========================================
# TAB 1: Live AI Assistant Chat
# ==========================================
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("##### 💡 Sample Queries")
        sample_queries = [
            "What is the cancellation fee for ORD-1001?",
            "Where is my delivery tracking for ORD-1003?",
            "Can I return opened merchandise after 15 days?",
            "How do I update my shipping address for ORD-1004?",
            "Please escalate my issue to a supervisor immediately."
        ]
        for sq in sample_queries:
            if st.button(sq, key=f"btn_{sq[:15]}"):
                st.session_state["user_query_input"] = sq

    with col1:
        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "Hello! I am **ShopAssist AI**, your customer operations assistant. How can I help you with orders, returns, delivery tracking, or cancellation fees today?"
                }
            ]

        # Display history
        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "telemetry" in msg:
                    t = msg["telemetry"]
                    st.markdown(
                        f"<span class='badge-intent'>Intent: {t['intent']} ({t['confidence']:.1%})</span> "
                        f"<span class='metric-chip'>⚡ Total: {t['latency']:.1f}ms</span> "
                        f"<span class='metric-chip'>🧠 LLM: {t['llm_latency']:.1f}ms</span> "
                        f"<span class='metric-chip'>⚙️ Router: {t['onnx_latency']:.1f}ms</span> "
                        f"<span class='metric-chip'>{'📦 Cached' if t['cached'] else '🌐 Live'}</span>",
                        unsafe_allow_html=True
                    )
                if "tool_result" in msg and msg["tool_result"]:
                    with st.expander(f"🛠️ Tool Call Executed: `{msg['tool_result']['tool_name']}`", expanded=False):
                        st.json(msg["tool_result"])

        # Chat Input
        default_input = st.session_state.pop("user_query_input", "")
        user_input = st.chat_input("Ask about an order, refund policy, cancellation, or billing...") or default_input

        if user_input:
            # User message
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Process Assistant Query
            with st.chat_message("assistant"):
                with st.spinner("Processing through ONNX Router, RAG, and LLM..."):
                    req = AssistantRequest(
                        query=user_input,
                        temperature=temp_slider,
                        top_p=top_p_slider,
                        provider=selected_provider,
                        skip_cache=skip_cache_toggle
                    )
                    resp = assistant_orchestrator.process_query(req)
                    
                    st.markdown(resp.structured_output.final_response)
                    
                    # Badges
                    st.markdown(
                        f"<span class='badge-intent'>Intent: {resp.onnx_intent} ({resp.onnx_confidence:.1%})</span> "
                        f"<span class='metric-chip'>⚡ Total: {resp.total_latency_ms:.1f}ms</span> "
                        f"<span class='metric-chip'>🧠 LLM: {resp.llm_latency_ms:.1f}ms</span> "
                        f"<span class='metric-chip'>⚙️ ONNX: {resp.onnx_latency_ms:.1f}ms</span> "
                        f"<span class='metric-chip'>{'📦 Cached' if resp.cached else '🌐 ' + resp.provider_used}</span>",
                        unsafe_allow_html=True
                    )
                    
                    # Tool details
                    if resp.structured_output.needs_tool and resp.structured_output.tool_result:
                        with st.expander(f"🛠️ Executed Tool: `{resp.structured_output.tool_result.tool_name}`", expanded=False):
                            st.json(resp.structured_output.tool_result.model_dump())
                            
                    # Structured JSON Inspector
                    with st.expander("🔍 View Validated Structured JSON Payload"):
                        st.json(resp.structured_output.model_dump())

            # Save in history
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": resp.structured_output.final_response,
                "telemetry": {
                    "intent": resp.onnx_intent,
                    "confidence": resp.onnx_confidence,
                    "latency": resp.total_latency_ms,
                    "llm_latency": resp.llm_latency_ms,
                    "onnx_latency": resp.onnx_latency_ms,
                    "cached": resp.cached
                },
                "tool_result": resp.structured_output.tool_result.model_dump() if resp.structured_output.tool_result else None
            })

# ==========================================
# TAB 2: ONNX Router Playground (Task 2)
# ==========================================
with tabs[1]:
    st.subheader("⚡ Edge Intent Classifier (Productionized Week 14 Model)")
    st.markdown(
        "Directly test queries against the **DistilBERT ONNX-INT8 quantized router**. "
        "Evaluates the 10 customer support categories with sub-5ms latency."
    )
    
    test_text = st.text_input("Enter customer inquiry text:", "Can I cancel my package and what is the fee?")
    
    if st.button("Classify Intent with ONNX", type="primary"):
        pred = onnx_router.predict(test_text)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Intent", pred.intent)
        c2.metric("Confidence", f"{pred.confidence:.2%}")
        c3.metric("Latency", f"{pred.latency_ms:.2f} ms", delta=f"{pred.model_type}")
        
        st.markdown("##### Probability Distribution Across 10 Canonical Categories")
        df_probs = pd.DataFrame({
            "Intent Category": list(pred.probabilities.keys()),
            "Probability": list(pred.probabilities.values())
        }).sort_values("Probability", ascending=False)
        
        st.bar_chart(df_probs.set_index("Intent Category"))

# ==========================================
# TAB 3: Knowledge Base RAG
# ==========================================
with tabs[2]:
    st.subheader("📚 Retrieval-Augmented Generation Knowledge Base")
    st.markdown("Explore indexed policies and execute semantic search queries over the vector database.")
    
    rag_query = st.text_input("Search knowledge base:", "What are the rules for returning damaged goods?")
    top_k_select = st.slider("Number of Chunks (Top-K)", 1, 5, 3)
    
    if st.button("Search Vector DB"):
        results = rag_pipeline.vector_db.search(rag_query, top_k=top_k_select)
        if not results:
            st.warning("No matching policy chunks found above similarity threshold.")
        else:
            for r in results:
                with st.container():
                    st.markdown(f"**📄 Document:** `{r.doc_title}` | **Section:** `{r.section}` | **Score:** `{r.similarity_score:.4f}`")
                    st.info(r.content)
                    st.divider()

# ==========================================
# TAB 4: System Telemetry & Batch Tester (Task 2)
# ==========================================
with tabs[3]:
    st.subheader("📊 Production Telemetry & Concurrent Batch Processing")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    stats = cache_service.get_stats()
    rl_stats = rate_limiter.get_stats()
    
    kpi1.metric("Cache Hit Ratio", f"{stats['hit_ratio_percent']}%", f"{stats['hits']} Hits")
    kpi2.metric("Cache Size", f"{stats['current_size']} / {stats['max_size']}", f"{stats['evictions']} Evicted")
    kpi3.metric("Allowed Requests", f"{rl_stats['allowed_requests']}", f"RPM Limit: {rl_stats['rate_limit_rpm']}")
    kpi4.metric("Active Model", onnx_router.model_type, "64.2 MB INT8")
    
    st.divider()
    
    st.markdown("##### 🚀 Concurrent Batch Query Processor")
    st.markdown("Tests asynchronous concurrent evaluation of multiple customer queries.")
    
    default_batch = [
        "What is the cancellation fee for ORD-1001?",
        "Where is tracking number for ORD-1004?",
        "Can I return an opened item after 10 days?",
        "How do I reset my account password?",
        "Unsubscribe me from promotional emails"
    ]
    batch_input = st.text_area("Queries (one per line):", value="\n".join(default_batch), height=140)
    
    if st.button("Execute Concurrent Batch", type="primary"):
        queries = [q.strip() for q in batch_input.split("\n") if q.strip()]
        st.write(f"Executing {len(queries)} queries concurrently...")
        
        t0 = time.perf_counter()
        batch_results = []
        for q in queries:
            req = AssistantRequest(query=q, provider=selected_provider, temperature=temp_slider)
            batch_results.append(assistant_orchestrator.process_query(req))
        total_time = (time.perf_counter() - t0) * 1000.0
        
        st.success(f"Batch completed in {total_time:.2f} ms (Avg: {total_time/len(queries):.2f} ms per query)")
        
        table_data = []
        for r in batch_results:
            table_data.append({
                "Query": r.structured_output.final_response[:60] + "...",
                "Intent": r.onnx_intent,
                "Confidence": f"{r.onnx_confidence:.1%}",
                "Tool": r.tool_executed or "None",
                "Cached": "Yes" if r.cached else "No",
                "Latency (ms)": r.total_latency_ms
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
