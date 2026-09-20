"""
High-Resolution Architecture Diagram Generator for ShopAssist AI (Week 16).
Generates:
1. architecture_agentic_loop.png: Visualizes the autonomous iterative loop and stopping conditions.
2. architecture_multi_agent.png: Visualizes the multi-agent coordination structure, specialists, and scratchpad context engineering.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DOCS_DIR = Path(__file__).resolve().parent

def generate_agentic_loop_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(15, 9), dpi=200)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    # Header
    ax.text(7.5, 9.5, "Week 16: Autonomous Agentic Loop Architecture", ha="center", va="center", fontsize=18, fontweight="bold", color="#1E1B4B")
    ax.text(7.5, 9.1, "Dynamic Multi-Step Reasoning, Cross-Source Verification, Compaction & Stopping Conditions", ha="center", va="center", fontsize=11, color="#475569")

    def draw_node(x, y, w, h, title, subtitle, color, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", linewidth=2, edgecolor=border, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#1E1B4B")
        if subtitle:
            for i, line in enumerate(subtitle):
                ax.text(x + 0.25, y + h - 0.7 - (i * 0.28), f"• {line}", ha="left", va="center", fontsize=8.2, color="#334155")

    # 1. Customer Query Input
    draw_node(0.5, 5.0, 2.5, 3.2, "1. User Query", ["Customer Inquiry / Dispute", "Order ID or Missing Details", "Disputed Fee or Defect", "Multi-Constraint Intent"], "#EEF2FF", "#4338CA")

    # 2. Coordinator & Scratchpad
    draw_node(3.6, 4.5, 3.4, 4.2, "2. Coordinator Agent", [
        "Structured External Notes",
        "DisputeScratchpad State",
        "Iteration Counter (<=4)",
        "Evaluates Intermediate Evidence",
        "Selects Next Dynamic Action:",
        " [INVESTIGATE | AUDIT | FINISH]"
    ], "#ECFDF5", "#059669")

    # 3. Investigator Tool Specialist
    draw_node(7.7, 5.8, 3.2, 3.2, "3. Investigator Specialist", [
        "Selects Operational Tool",
        "check_order_status()",
        "calculate_cancellation_fee()",
        "check_refund_eligibility()",
        "verify_courier_tracking()",
        "Compacts Raw Output"
    ], "#FEF3C7", "#D97706")

    # 4. Policy Auditor Verifier
    draw_node(7.7, 1.2, 3.2, 3.8, "4. Policy Verifier Specialist", [
        "Audits Facts vs. Guidelines",
        "RAG Policy Vector Search",
        "Detects Fee Waivers (Damaged)",
        "Overcomes Self-Verification",
        "Cross-Source Fact Checking",
        "Sanitizes Retrieval Context"
    ], "#FDF2F8", "#DB2777")

    # 5. Context Engineering Layer
    draw_node(3.6, 0.8, 3.4, 3.0, "Context Engineering", [
        "Tool Result Compaction",
        "Pruning Raw JSON Tool Dumps",
        "Structured External Notes Block",
        "Eliminates Context Saturation",
        "60%+ Token Reduction / Turn"
    ], "#F1F5F9", "#64748B")

    # 6. Stopping Conditions & Delivery
    draw_node(11.6, 4.5, 3.0, 4.2, "5. Stopping & Synthesis", [
        "Guardrail Conditions Checked:",
        "• Task Completed Successfully",
        "• Clarification Requested",
        "• Supervisor Escalation Triggered",
        "• Max Iterations Bound (4 steps)",
        "Authoritative Response Synthesis"
    ], "#F0FDF4", "#16A34A")

    # Connecting Arrows
    arrow = dict(arrowstyle="->", lw=2, color="#475569")
    ax.annotate("", xy=(3.6, 6.6), xytext=(3.0, 6.6), arrowprops=arrow)
    ax.annotate("", xy=(7.7, 7.4), xytext=(7.0, 7.4), arrowprops=arrow)
    ax.annotate("", xy=(7.0, 6.0), xytext=(7.7, 6.0), arrowprops=dict(arrowstyle="->", lw=2, color="#059669", linestyle="--"))
    ax.annotate("", xy=(7.7, 3.1), xytext=(5.3, 4.5), arrowprops=arrow)
    ax.annotate("", xy=(5.3, 4.5), xytext=(7.7, 2.5), arrowprops=dict(arrowstyle="->", lw=2, color="#DB2777", linestyle="--"))
    ax.annotate("", xy=(11.6, 6.6), xytext=(7.0, 6.6), arrowprops=arrow)
    ax.annotate("", xy=(5.3, 4.5), xytext=(5.3, 3.8), arrowprops=dict(arrowstyle="<->", lw=1.8, color="#64748B", linestyle=":"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Agentic loop diagram generated at: {output_path}")

def generate_multi_agent_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(15, 9), dpi=200)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    # Header
    ax.text(7.5, 9.5, "Week 16: Multi-Agent Coordination Structure", ha="center", va="center", fontsize=18, fontweight="bold", color="#0F172A")
    ax.text(7.5, 9.1, "Specialized Agent Roles, Context Isolation, Working Memory Scratchpad & Anti-Failure Protection", ha="center", va="center", fontsize=11, color="#475569")

    def draw_node(x, y, w, h, title, subtitle, color, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", linewidth=2, edgecolor=border, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")
        if subtitle:
            for i, line in enumerate(subtitle):
                ax.text(x + 0.25, y + h - 0.7 - (i * 0.28), f"• {line}", ha="left", va="center", fontsize=8.2, color="#334155")

    # Router Tier
    draw_node(0.5, 4.2, 2.5, 3.6, "Edge Intent Router", [
        "ONNX INT8 Quantized Model",
        "Sub-5ms Latency",
        "10 Canonical Support Intents",
        "Confidence & Telemetry",
        "Fast Offline Inference"
    ], "#F0F9FF", "#0284C7")

    # Coordinator
    draw_node(3.6, 3.5, 3.4, 5.0, "Coordinator Agent", [
        "Role: Chief Dispute Conductor",
        "Manages InvestigationScratchpad",
        "Tracks Goal & Verified Facts",
        "Dispatches Sub-Tasks",
        "Detects Early Termination:",
        " - Clarification Requested",
        " - Human Escalation Triggered",
        " - Maximum Steps Guardrail",
        "Mitigates Sequential Bottleneck"
    ], "#EEF2FF", "#4F46E5")

    # Investigator
    draw_node(7.8, 5.5, 3.2, 3.6, "Investigator Agent", [
        "Specialization: Data Retrieval",
        "Context Isolation: Tool Output",
        "check_order_status()",
        "calculate_cancellation_fee()",
        "check_refund_eligibility()",
        "verify_courier_tracking()",
        "Tool Result Compactor"
    ], "#ECFDF5", "#059669")

    # Policy Verifier
    draw_node(7.8, 1.0, 3.2, 3.8, "Policy Verifier Agent", [
        "Specialization: Compliance Audit",
        "Mitigates Self-Verification Paradox",
        "RAG Semantic Search",
        "Damaged Goods Fee Waiver Rule",
        "Cancellation Window Policy",
        "Sanitizes Malformed Retrieval",
        "Independent Claim Verification"
    ], "#FEF2F2", "#DC2626")

    # Storage & DBs
    draw_node(11.6, 5.5, 2.8, 3.6, "Operational DBs", [
        "sample_orders.json",
        "FedEx / UPS Carrier APIs",
        "Live Shipment Telemetry",
        "Simulated Failure Modes:",
        " - Tool Unavailable (503)",
        " - Gateway Timeout (504)"
    ], "#FFFBEB", "#D97706")

    # Policy KB
    draw_node(11.6, 1.0, 2.8, 3.8, "Knowledge Base RAG", [
        "Return & Refund Policy",
        "Cancellation Fee Policy",
        "Shipping & Delivery Guide",
        "Account Security FAQ",
        "Dense Embedding Engine",
        "In-Memory Cosine Vector DB"
    ], "#F5F3FF", "#7C3AED")

    # Connecting arrows
    arrow = dict(arrowstyle="->", lw=2, color="#475569")
    ax.annotate("", xy=(3.6, 6.0), xytext=(3.0, 6.0), arrowprops=arrow)
    ax.annotate("", xy=(7.8, 7.3), xytext=(7.0, 7.3), arrowprops=arrow)
    ax.annotate("", xy=(7.0, 6.3), xytext=(7.8, 6.3), arrowprops=dict(arrowstyle="->", lw=2, color="#059669", linestyle="--"))
    ax.annotate("", xy=(7.8, 2.9), xytext=(7.0, 4.5), arrowprops=arrow)
    ax.annotate("", xy=(7.0, 4.0), xytext=(7.8, 2.0), arrowprops=dict(arrowstyle="->", lw=2, color="#DC2626", linestyle="--"))
    ax.annotate("", xy=(11.6, 7.3), xytext=(11.0, 7.3), arrowprops=arrow)
    ax.annotate("", xy=(11.6, 2.9), xytext=(11.0, 2.9), arrowprops=arrow)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Multi-agent diagram generated at: {output_path}")

if __name__ == "__main__":
    generate_agentic_loop_diagram(str(DOCS_DIR / "architecture_agentic_loop.png"))
    generate_multi_agent_diagram(str(DOCS_DIR / "architecture_multi_agent.png"))
