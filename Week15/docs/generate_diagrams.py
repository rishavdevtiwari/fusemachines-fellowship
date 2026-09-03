"""
Generates high-resolution architecture diagrams for Task 1 and Task 2.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_task1_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    
    # Title
    ax.text(7, 9.4, "Task 1: Applied AI Assistant Architecture", ha="center", va="center", fontsize=18, fontweight="bold", color="#1A237E")
    ax.text(7, 9.0, "End-to-End LLM Integration, Prompt Engineering, RAG Pipeline & Tool Calling", ha="center", va="center", fontsize=11, color="#455A64")

    # Boxes helper
    def draw_box(x, y, w, h, title, items, color, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", linewidth=2, edgecolor=border, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#0D47A1")
        for i, item in enumerate(items):
            ax.text(x + 0.3, y + h - 0.75 - (i * 0.32), f"• {item}", ha="left", va="center", fontsize=8.5, color="#212121")

    # 1. User & Input
    draw_box(0.5, 5.5, 2.5, 2.8, "1. User Query", ["Customer Input", "Order Inquiries", "Policy Questions", "JSON Output Request"], "#E3F2FD", "#1565C0")

    # 2. Prompt Engineering
    draw_box(3.5, 5.5, 3.2, 2.8, "2. Prompt Engineering", ["System Prompt & Persona", "Anti-Hallucination Rules", "Tuned Temp (0.2), Top_p (0.95)", "Few-Shot Schema Context"], "#E8F5E9", "#2E7D32")

    # 3. RAG Pipeline
    draw_box(3.5, 1.5, 3.2, 3.2, "3. RAG Pipeline", ["Knowledge Base Markdown Docs", "Recursive Chunking (500 chars)", "Dense Semantic Embeddings", "Cosine Vector Database", "Top-K Policy Retrieval"], "#FFF3E0", "#E65100")

    # 4. Tool Calling Engine
    draw_box(7.3, 1.5, 3.2, 3.2, "4. External Tools", ["check_order_status()", "calculate_cancellation_fee()", "check_refund_eligibility()", "escalate_to_human()", "Real-Time DB Inspection"], "#F3E5F5", "#6A1B9A")

    # 5. LLM Gateway & Providers
    draw_box(7.3, 5.5, 3.2, 2.8, "5. LLM Provider Gateway", ["Google Gemini (2.5 Flash)", "OpenAI (GPT-4o-mini)", "Anthropic Claude 3.5", "Local vLLM (Mistral-7B)", "Graceful Provider Fallback"], "#E0F7FA", "#00838F")

    # 6. Structured JSON & Delivery
    draw_box(11.0, 3.5, 2.6, 3.5, "6. Structured Output", ["Pydantic Schema Validation", "Automated JSON Self-Repair", "Extracted Intent & Confidence", "Action & Tool Results", "Polished Markdown Response"], "#FFEBEE", "#C62828")

    # Flow arrows
    arrow_style = dict(arrowstyle="->", lw=2, color="#37474F")
    ax.annotate("", xy=(3.5, 6.9), xytext=(3.0, 6.9), arrowprops=arrow_style)
    ax.annotate("", xy=(5.1, 4.7), xytext=(5.1, 5.5), arrowprops=arrow_style)
    ax.annotate("", xy=(7.3, 6.9), xytext=(6.7, 6.9), arrowprops=arrow_style)
    ax.annotate("", xy=(8.9, 4.7), xytext=(8.9, 5.5), arrowprops=arrow_style)
    ax.annotate("", xy=(11.0, 6.9), xytext=(10.5, 6.9), arrowprops=arrow_style)
    ax.annotate("", xy=(11.0, 3.1), xytext=(10.5, 3.1), arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Task 1 diagram saved to: {output_path}")

def generate_task2_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(15, 9), dpi=200)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    
    # Title
    ax.text(7.5, 9.4, "Task 2: Production Engineering & AI Systems Architecture", ha="center", va="center", fontsize=18, fontweight="bold", color="#004D40")
    ax.text(7.5, 9.0, "High-Throughput Asynchronous FastAPI, ONNX Intent Router, Multi-Tier Caching, & Container Deployment", ha="center", va="center", fontsize=11, color="#455A64")

    def draw_box(x, y, w, h, title, items, color, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", linewidth=2, edgecolor=border, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#004D40")
        for i, item in enumerate(items):
            ax.text(x + 0.3, y + h - 0.75 - (i * 0.32), f"• {item}", ha="left", va="center", fontsize=8.5, color="#212121")

    # 1. UI & Client Layer
    draw_box(0.5, 4.5, 2.5, 3.8, "Client Layer", ["Streamlit Web UI (8501)", "REST API Clients", "Concurrent Batch Requests", "Live Router Telemetry", "Admin Performance Dashboard"], "#E0F2F1", "#00695C")

    # 2. API Gateway & Traffic Management
    draw_box(3.5, 4.5, 3.2, 3.8, "FastAPI Async Gateway", ["Asynchronous Non-Blocking I/O", "Sliding Window Rate Limiter", "429 Too Many Requests Protect", "POST /api/v1/chat, /batch", "GET /api/v1/health, /metrics"], "#E1F5FE", "#0277BD")

    # 3. In-Memory Cache
    draw_box(7.2, 5.5, 3.2, 2.8, "Multi-Tier Caching", ["Thread-Safe LRU + TTL Cache", "Deterministic SHA256 Key", "Sub-1ms Latency Cache Hits", "Eliminates Duplicate LLM Calls", "Real-Time Hit Ratio Metrics"], "#FFF8E1", "#F57F17")

    # 4. Tier 1: ONNX Intent Router
    draw_box(7.2, 1.2, 3.2, 3.8, "Tier 1: ONNX Router", ["Week 14 DistilBERT Classifier", "Exported to ONNX (Opset 17)", "INT8 Dynamic Quantization", "74.9% Compression (64 MB)", "Sub-5ms Latency on CPU", "10 Category Probabilities"], "#E8EAF6", "#283593")

    # 5. Resilience Mesh & Providers
    draw_box(11.0, 4.5, 3.5, 3.8, "Reliability & LLM Mesh", ["Circuit Breakers (Per Provider)", "Exponential Backoff + Jitter", "Primary: Google Gemini / OpenAI", "Fallback: vLLM (Mistral-7B)", "Safety Net: Deterministic Fallback", "Graceful Error Degradation"], "#FCE4EC", "#AD1457")

    # 6. Deployment Topologies
    draw_box(3.5, 0.8, 6.9, 1.8, "Production Deployment & Orchestration", [
        "Multi-Stage Dockerfile (Non-Root Runner) | Docker Compose (API:8000 + UI:8501 + vLLM)",
        "Cloud Blueprints: AWS ECS Fargate, GCP Cloud Run (Serverless), Azure Container Apps"
    ], "#ECEFF1", "#37474F")

    # Arrows
    arrow_style = dict(arrowstyle="->", lw=2, color="#37474F")
    ax.annotate("", xy=(3.5, 6.4), xytext=(3.0, 6.4), arrowprops=arrow_style)
    ax.annotate("", xy=(7.2, 6.9), xytext=(6.7, 6.9), arrowprops=arrow_style)
    ax.annotate("", xy=(7.2, 3.1), xytext=(6.7, 5.0), arrowprops=arrow_style)
    ax.annotate("", xy=(11.0, 6.4), xytext=(10.4, 6.4), arrowprops=arrow_style)
    ax.annotate("", xy=(11.0, 3.1), xytext=(10.4, 3.1), arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Task 2 diagram saved to: {output_path}")

if __name__ == "__main__":
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    generate_task1_diagram(os.path.join(docs_dir, "architecture_task1.png"))
    generate_task2_diagram(os.path.join(docs_dir, "architecture_task2.png"))
