"""
High-Speed ONNX Runtime Intent Router for ShopAssist AI (Week 16).
Serves the productionized Week 14 DistilBERT model with sub-5ms latency,
providing edge classification across 10 canonical customer support intents.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pydantic import BaseModel

from src.config import settings

class IntentPrediction(BaseModel):
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    model_type: str

class ONNXIntentRouter:
    """
    Sub-5ms edge intent classification router powered by ONNX Runtime.
    """
    def __init__(self):
        self.session = None
        self.tokenizer = None
        self.labels: List[str] = []
        self.id2label: Dict[int, str] = {}
        self.model_type = "heuristic_fallback"
        self._initialize_session()

    def _initialize_session(self):
        if settings.metadata_path.exists():
            try:
                with open(settings.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.labels = meta.get("labels", [])
                    self.id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}
            except Exception as e:
                print(f"Warning: Could not read metadata: {e}")

        if not self.labels:
            self.labels = [
                "ACCOUNT", "CANCELLATION_FEE", "DELIVERY", "FEEDBACK",
                "INVOICE", "NEWSLETTER", "ORDER", "PAYMENT", "REFUND", "SHIPPING_ADDRESS"
            ]
            self.id2label = {i: label for i, label in enumerate(self.labels)}

        target_path = settings.onnx_int8_path if settings.use_quantized_onnx and settings.onnx_int8_path.exists() else settings.onnx_fp32_path

        if target_path.exists():
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = settings.onnx_threads

                self.session = ort.InferenceSession(
                    str(target_path),
                    sess_options,
                    providers=["CPUExecutionProvider"]
                )
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

                self.model_type = "ONNX-INT8-Quantized" if target_path == settings.onnx_int8_path else "ONNX-FP32"
                print(f"ONNX Intent Router loaded successfully: {self.model_type} ({target_path.name})")
            except Exception as e:
                print(f"Notice: Using fast heuristic router ({e})")
                self.session = None
        else:
            print(f"ONNX model file not found at {target_path}. Using fast heuristic router.")

    def predict(self, text: str) -> IntentPrediction:
        t0 = time.perf_counter()
        
        if self.session and self.tokenizer:
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="np",
                    truncation=True,
                    max_length=64,
                    padding="max_length"
                )
                ort_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64)
                }
                outputs = self.session.run(None, ort_inputs)
                logits = outputs[0][0]

                # Softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                pred_idx = int(np.argmax(probs))
                pred_intent = self.id2label.get(pred_idx, "ORDER")
                confidence = float(probs[pred_idx])

                prob_dict = {self.id2label.get(i, f"CLASS_{i}"): round(float(p), 4) for i, p in enumerate(probs)}
                latency = (time.perf_counter() - t0) * 1000.0

                return IntentPrediction(
                    intent=pred_intent,
                    confidence=round(confidence, 4),
                    probabilities=prob_dict,
                    latency_ms=round(latency, 2),
                    model_type=self.model_type
                )
            except Exception as e:
                print(f"ONNX inference failed: {e}. Falling back to rule engine.")

        return self._heuristic_predict(text, t0)

    def _heuristic_predict(self, text: str, t0: float) -> IntentPrediction:
        q = text.lower()
        if any(k in q for k in ["cancel", "cancellation", "fee", "penalty"]):
            intent = "CANCELLATION_FEE"
        elif any(k in q for k in ["return", "refund", "money back"]):
            intent = "REFUND"
        elif any(k in q for k in ["track", "delivery", "courier", "shipping", "fedex", "ups"]):
            intent = "DELIVERY"
        elif any(k in q for k in ["address", "street", "reroute", "shipping address"]):
            intent = "SHIPPING_ADDRESS"
        elif any(k in q for k in ["invoice", "receipt", "tax", "vat"]):
            intent = "INVOICE"
        elif any(k in q for k in ["password", "login", "account", "2fa"]):
            intent = "ACCOUNT"
        elif any(k in q for k in ["newsletter", "unsubscribe", "promo", "marketing"]):
            intent = "NEWSLETTER"
        elif any(k in q for k in ["pay", "card", "billing", "checkout", "paypal"]):
            intent = "PAYMENT"
        elif any(k in q for k in ["feedback", "complaint", "review", "terrible", "great"]):
            intent = "FEEDBACK"
        else:
            intent = "ORDER"

        probs = {lbl: (0.85 if lbl == intent else round(0.15 / 9, 4)) for lbl in self.labels}
        latency = (time.perf_counter() - t0) * 1000.0

        return IntentPrediction(
            intent=intent,
            confidence=0.85,
            probabilities=probs,
            latency_ms=round(latency, 2),
            model_type="Heuristic-FastRule"
        )

onnx_router = ONNXIntentRouter()
