"""
Export and Quantize Week 14 Intent Classification Model to ONNX
Converts a DistilBERT Sequence Classifier (10 Customer Support Categories)
to optimized FP32 ONNX and INT8 Dynamic Quantized ONNX formats.
"""

import os
import sys
import time
import json

# Fix Windows cp1252 print issues with emojis from PyTorch ONNX exporter
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

LABELS = [
    "ACCOUNT",
    "CANCELLATION_FEE",
    "DELIVERY",
    "FEEDBACK",
    "INVOICE",
    "NEWSLETTER",
    "ORDER",
    "PAYMENT",
    "REFUND",
    "SHIPPING_ADDRESS"
]

id2label = {i: label for i, label in enumerate(LABELS)}
label2id = {label: i for i, label in enumerate(LABELS)}

# Canonical domain dataset for fast adaptation/weight calibration
TRAINING_SAMPLES = [
    ("How do I delete my account or change my password?", "ACCOUNT"),
    ("I need to switch to a different account profile", "ACCOUNT"),
    ("Create a new member account for me please", "ACCOUNT"),
    ("How much is the fee if I cancel my order right now?", "CANCELLATION_FEE"),
    ("What are the cancellation charges for pending packages?", "CANCELLATION_FEE"),
    ("Do I get charged a fee for canceling after 2 hours?", "CANCELLATION_FEE"),
    ("When will my delivery arrive? Where is courier?", "DELIVERY"),
    ("What delivery options do you offer for express shipping?", "DELIVERY"),
    ("My package tracking hasn't updated in two days", "DELIVERY"),
    ("I want to submit a complaint about the poor service", "FEEDBACK"),
    ("Here is a positive review for the great customer assistance", "FEEDBACK"),
    ("The item arrived damaged and I am very unhappy", "FEEDBACK"),
    ("Can I get a copy of the tax invoice for order ORD-1001?", "INVOICE"),
    ("Where can I download the billing receipt and invoice?", "INVOICE"),
    ("I need an invoice showing company VAT details", "INVOICE"),
    ("Please unsubscribe me from your promotional newsletter emails", "NEWSLETTER"),
    ("I want to subscribe to the weekly product newsletter", "NEWSLETTER"),
    ("Stop sending me marketing emails and newsletter updates", "NEWSLETTER"),
    ("I would like to cancel my order ORD-1002 immediately", "ORDER"),
    ("Can I change the items inside my pending order?", "ORDER"),
    ("I want to place a new order for headphones", "ORDER"),
    ("What payment methods do you accept at checkout?", "PAYMENT"),
    ("My credit card was declined during payment processing", "PAYMENT"),
    ("Can I pay using PayPal or Apple Pay?", "PAYMENT"),
    ("What is your refund policy? How do I get my money back?", "REFUND"),
    ("Check the refund status for my returned merchandise", "REFUND"),
    ("When will the refund be credited to my bank account?", "REFUND"),
    ("I need to change my shipping delivery address for order ORD-1004", "SHIPPING_ADDRESS"),
    ("Update my default shipping destination address", "SHIPPING_ADDRESS"),
    ("Can you redirect my shipment to a new street address?", "SHIPPING_ADDRESS")
]

def train_and_export():
    print("=" * 60)
    print("SHOPASSIST AI: MODEL EXPORT & ONNX OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_fp32_path = os.path.join(script_dir, "intent_classifier.onnx")
    onnx_int8_path = os.path.join(script_dir, "intent_classifier_quantized.onnx")
    metadata_path = os.path.join(script_dir, "model_metadata.json")
    
    model_name = "distilbert-base-uncased"
    print(f"[1/5] Loading tokenizer and base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    print("[2/5] Calibrating classifier head with domain queries (3 epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    
    for epoch in range(3):
        total_loss = 0.0
        for text, label in TRAINING_SAMPLES:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding="max_length")
            target = torch.tensor([label2id[label]])
            outputs = model(**inputs, labels=target)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"      Epoch {epoch+1}/3 - Calibration Loss: {total_loss/len(TRAINING_SAMPLES):.4f}")
    
    model.eval()
    
    print(f"[3/5] Exporting PyTorch model to ONNX: {onnx_fp32_path}...")
    dummy_text = "I want to check the status of my refund"
    dummy_inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=64, padding="max_length")
    
    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        onnx_fp32_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        dynamo=False
    )
    
    fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    print(f"      ONNX FP32 model saved successfully! Size: {fp32_size:.2f} MB")
    
    print(f"[4/5] Applying INT8 Dynamic Quantization: {onnx_int8_path}...")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QInt8
    )
    
    int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    compression = (1.0 - (int8_size / fp32_size)) * 100
    print(f"      INT8 Quantized ONNX saved! Size: {int8_size:.2f} MB ({compression:.1f}% size reduction)")
    
    print(f"[5/5] Saving model metadata and label mapping: {metadata_path}...")
    metadata = {
        "model_architecture": "DistilBERT Sequence Classification",
        "base_model": model_name,
        "labels": LABELS,
        "id2label": id2label,
        "label2id": label2id,
        "fp32_size_mb": round(fp32_size, 2),
        "int8_size_mb": round(int8_size, 2),
        "compression_percent": round(compression, 2),
        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("\nExport completed successfully! Ready for high-throughput inference.")

if __name__ == "__main__":
    train_and_export()
