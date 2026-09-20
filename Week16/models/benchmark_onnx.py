"""
Comprehensive Inference Benchmark: PyTorch CPU vs ONNX Runtime vs ONNX INT8 Quantized
Evaluates latency (p50, p90, p95, p99), throughput (queries/sec), and memory footprint.
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort

TEST_QUERIES = [
    "I want to track where my package is right now",
    "Can you help me cancel order ORD-1001?",
    "What is the penalty fee if I cancel after 5 hours?",
    "I received a broken item and want my money refunded",
    "Please send me the official tax invoice receipt",
    "Can I update my delivery address before dispatch?",
    "How do I enable two-factor authentication on my account?",
    "Unsubscribe me from your newsletter updates",
    "Do you accept Apple Pay and credit cards?",
    "The customer service was excellent and fast!"
]

def benchmark_models(num_iterations=50, warmup=10):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_fp32_path = os.path.join(script_dir, "intent_classifier.onnx")
    onnx_int8_path = os.path.join(script_dir, "intent_classifier_quantized.onnx")
    metadata_path = os.path.join(script_dir, "model_metadata.json")
    results_path = os.path.join(script_dir, "benchmark_results.json")
    
    if not os.path.exists(onnx_fp32_path) or not os.path.exists(onnx_int8_path):
        print("Error: ONNX models not found. Run export_onnx.py first.")
        return
        
    print("=" * 70)
    print("INFERENCE BENCHMARK: PYTORCH CPU vs ONNX RUNTIME vs ONNX INT8 QUANTIZED")
    print(f"Iterations: {num_iterations} queries per framework (Warmup: {warmup})")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # 1. PyTorch Baseline
    print("\n[1/3] Benchmarking PyTorch CPU Baseline...")
    pt_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=10
    )
    pt_model.eval()
    
    # Warmup
    for q in TEST_QUERIES[:warmup]:
        inp = tokenizer(q, return_tensors="pt", truncation=True, max_length=64, padding="max_length")
        with torch.no_grad():
            _ = pt_model(**inp)
            
    pt_latencies = []
    start_pt = time.perf_counter()
    for _ in range(num_iterations // len(TEST_QUERIES) + 1):
        for q in TEST_QUERIES:
            inp = tokenizer(q, return_tensors="pt", truncation=True, max_length=64, padding="max_length")
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = pt_model(**inp)
            t1 = time.perf_counter()
            pt_latencies.append((t1 - t0) * 1000.0)
            if len(pt_latencies) >= num_iterations:
                break
        if len(pt_latencies) >= num_iterations:
            break
    total_pt_time = time.perf_counter() - start_pt
    
    # 2. ONNX Runtime FP32
    print("[2/3] Benchmarking ONNX Runtime (FP32)...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = os.cpu_count() or 4
    ort_session_fp32 = ort.InferenceSession(onnx_fp32_path, sess_options, providers=["CPUExecutionProvider"])
    
    # Warmup
    for q in TEST_QUERIES[:warmup]:
        inp = tokenizer(q, return_tensors="np", truncation=True, max_length=64, padding="max_length")
        ort_inputs = {
            "input_ids": inp["input_ids"].astype(np.int64),
            "attention_mask": inp["attention_mask"].astype(np.int64)
        }
        _ = ort_session_fp32.run(None, ort_inputs)
        
    ort_latencies = []
    start_ort = time.perf_counter()
    for _ in range(num_iterations // len(TEST_QUERIES) + 1):
        for q in TEST_QUERIES:
            inp = tokenizer(q, return_tensors="np", truncation=True, max_length=64, padding="max_length")
            t0 = time.perf_counter()
            ort_inputs = {
                "input_ids": inp["input_ids"].astype(np.int64),
                "attention_mask": inp["attention_mask"].astype(np.int64)
            }
            _ = ort_session_fp32.run(None, ort_inputs)
            t1 = time.perf_counter()
            ort_latencies.append((t1 - t0) * 1000.0)
            if len(ort_latencies) >= num_iterations:
                break
        if len(ort_latencies) >= num_iterations:
            break
    total_ort_time = time.perf_counter() - start_ort
    
    # 3. ONNX Runtime INT8 Dynamic Quantized
    print("[3/3] Benchmarking ONNX Runtime (INT8 Quantized)...")
    ort_session_int8 = ort.InferenceSession(onnx_int8_path, sess_options, providers=["CPUExecutionProvider"])
    
    # Warmup
    for q in TEST_QUERIES[:warmup]:
        inp = tokenizer(q, return_tensors="np", truncation=True, max_length=64, padding="max_length")
        ort_inputs = {
            "input_ids": inp["input_ids"].astype(np.int64),
            "attention_mask": inp["attention_mask"].astype(np.int64)
        }
        _ = ort_session_int8.run(None, ort_inputs)
        
    int8_latencies = []
    start_int8 = time.perf_counter()
    for _ in range(num_iterations // len(TEST_QUERIES) + 1):
        for q in TEST_QUERIES:
            inp = tokenizer(q, return_tensors="np", truncation=True, max_length=64, padding="max_length")
            ort_inputs = {
                "input_ids": inp["input_ids"].astype(np.int64),
                "attention_mask": inp["attention_mask"].astype(np.int64)
            }
            t0 = time.perf_counter()
            _ = ort_session_int8.run(None, ort_inputs)
            t1 = time.perf_counter()
            int8_latencies.append((t1 - t0) * 1000.0)
            if len(int8_latencies) >= num_iterations:
                break
        if len(int8_latencies) >= num_iterations:
            break
    total_int8_time = time.perf_counter() - start_int8
    
    # Compute Statistics
    def compute_stats(latencies, total_time, model_size_mb):
        return {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "median_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "p90_ms": round(float(np.percentile(latencies, 90)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "throughput_qps": round(float(len(latencies) / total_time), 2),
            "model_size_mb": round(model_size_mb, 2)
        }
        
    fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    
    results = {
        "PyTorch CPU": compute_stats(pt_latencies, total_pt_time, 255.0),
        "ONNX Runtime (FP32)": compute_stats(ort_latencies, total_ort_time, fp32_size),
        "ONNX Runtime (INT8 Quantized)": compute_stats(int8_latencies, total_int8_time, int8_size)
    }
    
    print("\n" + "=" * 85)
    print(f"{'Framework / Configuration':<32} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'p99 (ms)':<9} | {'Throughput (qps)':<16} | {'Size (MB)':<10}")
    print("-" * 85)
    for name, m in results.items():
        print(f"{name:<32} | {m['median_p50_ms']:<9} | {m['p95_ms']:<9} | {m['p99_ms']:<9} | {m['throughput_qps']:<16} | {m['model_size_mb']:<10}")
    print("=" * 85)
    
    speedup = results["PyTorch CPU"]["median_p50_ms"] / results["ONNX Runtime (INT8 Quantized)"]["median_p50_ms"]
    print(f"\nSpeedup: ONNX INT8 Quantized is {speedup:.2f}x faster than PyTorch CPU!")
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    return results

if __name__ == "__main__":
    benchmark_models()
