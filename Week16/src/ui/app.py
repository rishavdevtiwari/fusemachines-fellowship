"""
Interactive Streamlit Web Dashboard for ShopAssist AI Agent (Week 16).
Features:
1. Live Dispute Resolution & Step-by-Step Agentic Trajectory Inspector
2. Multi-Agent vs. Single-Agent Comparative Architecture Playground
3. Real-Time Evaluation Harness Suite & Failure Taxonomy Dashboard
4. Failure Injection Laboratory (Tool Outages, Malformed Data, Timeouts)
5. Corporate Policy Knowledge Base Browser & Telemetry
"""

import os
import sys
import time
import json
from pathlib import Path
import streamlit as st

# Add Week16 to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.services.agentic_service import agentic_service, DisputeResolutionRequest
from src.evaluation.eval_harness import EvaluationHarness
from src.evaluation.failure_injection import run_failure_injection_suite
from src.core.rag_pipeline import rag_pipeline
from src.services.cache_service import cache_service

st.set_page_config(
    page_title="ShopAssist AI - Agentic Dispute Resolution",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .step-box {
        background-color: #F3F4F6;
        border-left: 4px solid #4F46E5;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }
    .role-badge {
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        text-transform: uppercase;
    }
    .badge-coord { background-color: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }
    .badge-inv { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-ver { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/ShopAssist_AI-Week_16_Agentic-4F46E5?style=for-the-badge", use_container_width=True)
    st.markdown("### ⚙️ Agent Configuration")

    pattern_choice = st.radio(
        "Agent Architecture Pattern:",
        ["multi_agent", "single_agent"],
        format_func=lambda x: "Specialized Multi-Agent" if x == "multi_agent" else "Monolithic Single-Agent Baseline",
        help="Choose between coordinated multi-agent specialists or monolithic baseline loop."
    )

    provider_choice = st.selectbox(
        "LLM Provider:",
        ["gemini", "openai", "claude", "vllm", "mock"],
        index=4 if not settings.gemini_api_key else 0,
        help="Select foundation model inference backend or deterministic offline engine."
    )

    max_iter = st.slider("Max Iterations Guardrail:", min_value=1, max_value=6, value=4)

    st.markdown("---")
    st.markdown("### 🧪 Failure Injection Testing")
    failure_sim = st.selectbox(
        "Simulate Upstream Failure:",
        ["none", "tool_unavailable", "malformed_rag", "timeout"],
        format_func=lambda x: {
            "none": "Normal Operation (No Failure)",
            "tool_unavailable": "Tool Database Outage (HTTP 503)",
            "malformed_rag": "Malformed / Corrupted RAG",
            "timeout": "Gateway Timeout (HTTP 504)"
        }[x]
    )
    settings.failure_simulation_mode = failure_sim

    st.markdown("---")
    if st.button("🗑️ Clear Response Cache"):
        cache_service.clear()
        st.sidebar.success("Cache cleared!")

# -----------------------------------------------------------------------------
# MAIN CONTENT HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">ShopAssist AI: Agentic Dispute Resolution</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Week 16: Autonomous Multi-Step Reasoning, Cross-Source Verification, Policy Arbitration & Evaluation Harness</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "💬 Live Dispute Resolution & Trajectory",
    "⚖️ Multi-Agent vs Baseline Comparison",
    "📊 Evaluation Harness Dashboard",
    "⚠️ Failure Injection Lab",
    "📚 Knowledge Base & Policies"
])

# -----------------------------------------------------------------------------
# TAB 1: LIVE DISPUTE RESOLUTION
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown("#### Autonomous Customer Case Arbitration")
    st.write("Submit complex dispute queries involving order records, courier tracking, and corporate return exceptions.")

    # Quick prompt buttons
    st.markdown("**Example Test Inquiries:**")
    c1, c2, c3, c4 = st.columns(4)
    query_input = ""
    if c1.button("📦 Damaged Return Dispute"):
        query_input = "I received order ORD-1003 11 days ago, but the package arrived cracked and damaged. Am I entitled to a full refund or will I be charged a return shipping fee?"
    if c2.button("⏱️ 30-Min Cancellation"):
        query_input = "Can I cancel my order ORD-1002 and how much fee will I be charged?"
    if c3.button("❓ Missing Order ID"):
        query_input = "I want to cancel my recent purchase and get my money back."
    if c4.button("🚨 Supervisor Escalation"):
        query_input = "This is unacceptable, I was double charged for order ORD-1001 and your warehouse refuses to answer. Connect me to a human supervisor immediately!"

    user_query = st.text_area("Customer Query:", value=query_input or "I received order ORD-1003 11 days ago, but the package arrived cracked and damaged. Am I entitled to a full refund or will I be charged a return shipping fee?", height=80)

    if st.button("🚀 Run Agentic Dispute Resolution", type="primary"):
        with st.spinner("Agentic Loop Reasoning in Progress..."):
            req = DisputeResolutionRequest(
                query=user_query,
                pattern=pattern_choice,
                max_iterations=max_iter,
                skip_cache=True
            )
            resp = agentic_service.resolve(req)
            payload = resp.resolution

        # Telemetry Bar
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Iterations Required", f"{payload.iterations_count} / {max_iter}")
        m2.metric("Total Tokens", f"{payload.total_tokens}")
        m3.metric("Coordination Tokens", f"{payload.coordination_tokens}")
        m4.metric("Stopping Condition", payload.stopping_condition)
        m5.metric("Latency", f"{resp.total_service_time_ms:.1f} ms")

        # Step Trajectory Inspector
        st.markdown("---")
        st.markdown("### 🔍 Step-by-Step Trajectory Inspector")
        for step in payload.trajectory:
            role_badge = f'<span class="role-badge badge-coord">{step.agent_role}</span>'
            if step.agent_role == "investigator":
                role_badge = f'<span class="role-badge badge-inv">{step.agent_role}</span>'
            elif step.agent_role == "verifier":
                role_badge = f'<span class="role-badge badge-ver">{step.agent_role}</span>'

            with st.expander(f"Step {step.step_number} | {step.agent_role.upper()} | Action: {step.action}", expanded=True):
                st.markdown(f"**Agent:** {role_badge}", unsafe_allow_html=True)
                st.markdown(f"**Thought:** *{step.thought}*")
                if step.action_input:
                    st.json({"action_input": step.action_input})
                if step.observation_compacted:
                    st.info(f"**Compacted Observation:** {step.observation_compacted}")
                if step.observation_raw:
                    with st.expander("Show Raw Tool Output (Compacted & Cleared)"):
                        st.json(step.observation_raw)
                st.caption(f"Tokens consumed: {step.tokens_used} | Latency: {step.latency_ms} ms")

        # Working Memory (Scratchpad)
        st.markdown("---")
        st.markdown("### 🧠 External Working Memory (Investigation Scratchpad)")
        sp = payload.scratchpad
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"**Dispute Goal:** {sp.dispute_goal}")
            st.markdown(f"**Target Order ID:** `{sp.order_id or 'Unspecified'}`")
            st.markdown(f"**Fee Waiver Applied:** `{'✅ YES' if sp.fee_waiver_applied else '❌ NO'}`")
            st.markdown(f"**Calculated Penalty:** `${sp.calculated_penalty:.2f}`")
            st.markdown(f"**Estimated Refund:** `${sp.refund_amount:.2f}`")
        with sc2:
            st.markdown("**Verified Factual Claims:**")
            for f in sp.verified_facts:
                st.markdown(f"- {f}")
            if sp.exceptions_identified:
                st.markdown("**Policy Exceptions / Waivers:**")
                for e in sp.exceptions_identified:
                    st.markdown(f"- 🏷️ {e}")

        # Final Response
        st.markdown("---")
        st.markdown("### 📢 Final Authoritative Customer Response")
        st.success(payload.final_customer_response)

# -----------------------------------------------------------------------------
# TAB 2: COMPARATIVE BENCHMARKING
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown("#### Architectural Comparison: Multi-Agent System vs. Single-Agent Baseline")
    st.write("Execute the same inquiry across both architectures side-by-side to visualize the coordination cost and context isolation benefits.")

    comp_query = st.text_input("Comparison Query:", value="I received order ORD-1003 11 days ago, but the package arrived cracked and damaged. Am I entitled to a full refund or will I be charged a return shipping fee?")

    if st.button("⚡ Run Side-by-Side Comparison"):
        with st.spinner("Benchmarking both patterns..."):
            m_resp = agentic_service.resolve(DisputeResolutionRequest(query=comp_query, pattern="multi_agent", skip_cache=True))
            s_resp = agentic_service.resolve(DisputeResolutionRequest(query=comp_query, pattern="single_agent", skip_cache=True))

        col_m, col_s = st.columns(2)
        with col_m:
            st.markdown("### 🤖 Multi-Agent System")
            st.metric("Total Tokens", m_resp.resolution.total_tokens)
            st.metric("Coordination Overhead Tokens", m_resp.resolution.coordination_tokens)
            st.metric("Trajectory Steps", m_resp.resolution.iterations_count)
            st.metric("Fee Waiver Caught?", "✅ YES" if m_resp.resolution.scratchpad.fee_waiver_applied else "❌ NO")
            st.success(m_resp.resolution.final_customer_response)

        with col_s:
            st.markdown("### 👤 Single-Agent Baseline")
            st.metric("Total Tokens", s_resp.resolution.total_tokens)
            st.metric("Coordination Overhead Tokens", 0)
            st.metric("Trajectory Steps", s_resp.resolution.iterations_count)
            st.metric("Fee Waiver Caught?", "✅ YES" if s_resp.resolution.scratchpad.fee_waiver_applied else "❌ NO (Self-Verification Paradox)")
            st.info(s_resp.resolution.final_customer_response)

# -----------------------------------------------------------------------------
# TAB 3: EVALUATION HARNESS
# -----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("#### Evaluation Harness Built From Scratch")
    st.write("Evaluates the agentic system across canonical test scenarios measuring completion rate, tool correctness, trajectory length, and failure classifications.")

    if st.button("▶️ Execute Full Evaluation Suite"):
        with st.spinner("Running evaluation harness across test scenarios..."):
            harness = EvaluationHarness()
            comp_data = harness.compare_patterns()

        m_sum = comp_data["multi_agent"]["summary"]
        s_sum = comp_data["single_agent_baseline"]["summary"]

        st.markdown("### 🏆 Summary Scorecard")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Task Completion Rate", f"{m_sum['completion_rate_pct']}%", f"+{comp_data['comparison']['completion_rate_delta_pct']}% vs baseline")
        sc2.metric("Tool Correctness", f"{m_sum['tool_correctness_rate_pct']}%", f"+{comp_data['comparison']['tool_correctness_delta_pct']}% vs baseline")
        sc3.metric("Avg Trajectory Length", f"{m_sum['avg_trajectory_length']} steps")
        sc4.metric("Coordination Token Overhead", f"+{comp_data['comparison']['coordination_token_overhead_pct']}%")

        st.markdown("---")
        st.markdown("### 📋 Failure Taxonomy Breakdown")
        f_cols = st.columns(3)
        f_cols[0].metric("Hard Failures", m_sum["hard_failures"], "Crashes / Exceptions")
        f_cols[1].metric("Soft Failures", m_sum["soft_failures"], "Missed policy / tool errors")
        f_cols[2].metric("Cascading Soft Failures", m_sum["cascading_soft_failures"], "Compounding reasoning errors")

        st.markdown("---")
        st.markdown("### 🔬 Query Trajectory Breakdown (Multi-Agent System)")
        cases = comp_data["multi_agent"]["cases"]
        st.dataframe(cases)

# -----------------------------------------------------------------------------
# TAB 4: FAILURE INJECTION LAB
# -----------------------------------------------------------------------------
with tabs[3]:
    st.markdown("#### Additional Requirement 3: Failure Injection Test Suite")
    st.write("Intentionally injects hardware/database outages, malformed context chunks, and gateway timeouts to verify robust graceful degradation.")

    if st.button("🧪 Run Automated Failure Injection Tests"):
        with st.spinner("Injecting failures into operational tools and RAG pipeline..."):
            f_report = run_failure_injection_suite()

        for name, data in f_report.items():
            with st.expander(f"Failure Mode: {name.upper()}", expanded=True):
                st.write(f"**Failure Injected:** {data.get('failure_injected')}")
                st.write(f"**Recognized by Agent:** {data.get('recognized_by_agent', True)}")
                st.write(f"**Hallucination Avoided:** {data.get('hallucination_avoided', True)}")
                st.write(f"**Degraded Safely:** {data.get('degraded_safely', True)}")
                st.success(f"**Agent Response:** {data.get('final_response')}")

# -----------------------------------------------------------------------------
# TAB 5: KNOWLEDGE BASE BROWSER
# -----------------------------------------------------------------------------
with tabs[4]:
    st.markdown("#### Indexed Corporate Knowledge Base Policies")
    st.write(f"Total chunks indexed: **{len(rag_pipeline.chunks)}** across Markdown documents.")

    kb_files = list(settings.kb_dir.glob("*.md"))
    selected_doc = st.selectbox("Select Document:", [f.name for f in kb_files])

    doc_path = settings.kb_dir / selected_doc
    if doc_path.exists():
        with open(doc_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
