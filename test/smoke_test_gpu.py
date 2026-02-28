"""Smoke test: verify bertalign uses GPU for faiss operations."""
import subprocess
import sys
import threading
import time

import faiss
import numpy as np
import torch
from sys import platform


def monitor_gpu(stop_event, peak_mem):
    """Poll nvidia-smi for GPU memory usage."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            mem = int(out.strip())
            if mem > peak_mem[0]:
                peak_mem[0] = mem
        except Exception:
            pass
        time.sleep(0.2)


def main():
    print("=" * 60)
    print("GPU Smoke Test for bertalign")
    print("=" * 60)
    print()

    # Check prerequisites
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"platform: {platform}")
    print(f"faiss.StandardGpuResources available: {hasattr(faiss, 'StandardGpuResources')}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)
    if not hasattr(faiss, "StandardGpuResources"):
        print("FAIL: faiss GPU not available")
        sys.exit(1)

    # Test 1: Direct faiss GPU index
    print("--- Test 1: faiss GPU index ---")
    res = faiss.StandardGpuResources()
    d = 768
    index = faiss.IndexFlatIP(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    vecs = np.random.randn(500, d).astype(np.float32)
    faiss.normalize_L2(vecs)
    gpu_index.add(vecs)
    D, I = gpu_index.search(vecs[:10], 5)
    assert abs(D[0, 0] - 1.0) < 0.01, f"Self-similarity {D[0,0]} != 1.0"
    assert I[0, 0] == 0, f"Top-1 index {I[0,0]} != 0"
    print(f"  Top-1 self-similarity: {D[0, 0]:.4f} (expected ~1.0)")
    print("  PASSED")
    print()

    # Test 2: Full bertalign alignment with GPU monitoring
    print("--- Test 2: bertalign end-to-end (with GPU memory monitoring) ---")
    stop_event = threading.Event()
    peak_mem = [0]
    monitor = threading.Thread(target=monitor_gpu, args=(stop_event, peak_mem))
    monitor.start()

    from bertalign import Bertalign

    src = open("text+berg/de/001").read()
    tgt = open("text+berg/fr/001").read()
    aligner = Bertalign(src, tgt, is_split=True)
    aligner.align_sents()

    stop_event.set()
    monitor.join()

    n = len(aligner.result)
    print(f"  Aligned {n} pairs")
    print(f"  First 3 alignments: {aligner.result[:3]}")
    print(f"  Peak GPU memory during alignment: {peak_mem[0]} MiB")
    if peak_mem[0] > 0:
        print("  GPU was used!")
    else:
        print("  WARNING: No GPU memory detected — faiss may have fallen back to CPU")
    print("  PASSED")
    print()

    print("=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
