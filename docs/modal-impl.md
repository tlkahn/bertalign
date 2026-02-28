# Modal GPU Backend — Implementation Details

This document describes the architecture and implementation of `bertalign/modal_gpu.py`, which offloads bertalign's full alignment pipeline to a cloud GPU via [Modal](https://modal.com).

## Architecture

```
Local Mac                              Modal Cloud (L40S GPU)
─────────                              ──────────────────────
Bertalign(src, tgt)                    BertalignService
  └─ .align_sents()  ─── RPC ───────>   └─ .align()
                                             ├─ LaBSE encode (GPU)
                                             ├─ faiss top-k search (GPU)
                                             ├─ DP first pass (numba)
                                             └─ DP second pass (numba)
  .result             <── serialized ──  return {result, src_sents, tgt_sents}
  .src_sents
  .tgt_sents
  └─ .print_sents()  (local)
```

The entire compute-heavy pipeline runs on Modal. The local machine only needs the `modal` Python package — no torch, faiss, numba, or sentence-transformers.

## Module Structure

`bertalign/modal_gpu.py` contains four components:

### 1. Modal Image

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2.2", "numba>=0.64", "torch>=2.5",
        "sentence-transformers>=3.0", "faiss-gpu-cu12>=1.9",
        "sentence-splitter>=1.4", "googletrans==4.0.0rc1",
    )
    .add_local_python_source("bertalign")
)
```

- **Python 3.11**: Chosen for broad torch/faiss wheel availability.
- **`add_local_python_source("bertalign")`**: Copies the local `bertalign/` package into the container image, excluding `__pycache__`, `.git`, etc. This means any local code changes are picked up on the next `modal run`.
- **`faiss-gpu-cu12` (not `[fix-cuda]`)**: The `[fix-cuda]` extra downgrades `nvidia-cuda-runtime` to 12.1, which conflicts with PyTorch's CUDA 12.x libs. The base package works with Modal's system CUDA.

### 2. `BertalignService` — GPU Service Class

```python
@app.cls(
    gpu="L40S",
    image=image,
    volumes={"/root/.cache/huggingface": model_volume},
    scaledown_window=300,
    timeout=600,
)
class BertalignService:
    @modal.enter()
    def load_model(self): ...

    @modal.method()
    def align(self, src_text, tgt_text, ...) -> dict: ...
```

Key design decisions:

| Setting | Value | Rationale |
|---|---|---|
| `gpu` | `"L40S"` | 48GB VRAM, good price/performance for inference |
| `scaledown_window` | `300` (5 min) | Keeps container warm between documents during batch runs |
| `timeout` | `600` (10 min) | Generous limit for large documents |
| Volume mount | `/root/.cache/huggingface` | Caches LaBSE weights (~1.8GB) across cold starts |

**`@modal.enter()` — `load_model()`**

Runs once when a container starts. Imports `bertalign`, which triggers the eager LaBSE model load in `bertalign/__init__.py`:

```python
# bertalign/__init__.py (upstream behavior)
model_name = "LaBSE"
model = Encoder(model_name)  # loads ~1.8GB model
```

After loading, `model_volume.commit()` persists the downloaded weights to the volume. On subsequent cold starts, the model loads from the volume (~10s) instead of re-downloading (~30-60s).

**`@modal.method()` — `align()`**

Runs the full bertalign pipeline and returns a serializable dict:

```python
{
    "result": [[src_ids, tgt_ids], ...],   # list of [list[int], list[int]]
    "src_sents": ["sentence 1", ...],       # split source sentences
    "tgt_sents": ["sentence 1", ...],       # split target sentences
}
```

Numpy integers (`np.int64`) are cast to plain `int` via `list(map(int, ...))` for clean Modal serialization.

### 3. `Bertalign` — Local Proxy Class

A drop-in replacement for `bertalign.Bertalign` that delegates to the remote service:

```python
from bertalign.modal_gpu import Bertalign

aligner = Bertalign(src, tgt)
aligner.align_sents()    # RPC to Modal
aligner.print_sents()    # local, uses cached src_sents/tgt_sents
```

The proxy stores `result`, `src_sents`, and `tgt_sents` locally after the remote call, enabling `print_sents()` to work without another round-trip.

### 4. `align_remote()` — Functional API

A simpler interface that returns only the alignment indices:

```python
from bertalign.modal_gpu import align_remote

result = align_remote(src, tgt, is_split=False)
# [([0], [0]), ([1], [1]), ...]
```

## Cold Start Behavior

| Scenario | Time | What happens |
|---|---|---|
| First ever call (no volume) | ~60s | Download LaBSE from HuggingFace Hub, save to volume |
| Cold start (volume exists) | ~10s | Load LaBSE from volume |
| Warm container (within 5 min) | ~1s | Model already in memory, no load needed |

For batch processing (e.g. the 7-document Text+Berg benchmark), `scaledown_window=300` ensures the container stays warm between documents, so only the first document pays the cold start cost.

## Dependency Summary

**Local machine** (Mac / no GPU):
- `modal>=0.73` — that's it

**Modal container** (cloud GPU):
- `numpy`, `numba`, `torch`, `sentence-transformers`, `faiss-gpu-cu12`, `sentence-splitter`, `googletrans`
- Plus the `bertalign` package itself (injected via `add_local_python_source`)

## Known Issues and Mitigations

| Issue | Status | Mitigation |
|---|---|---|
| `googletrans` HTTP flakiness in container | Works in practice | Only used for language detection display; alignment works regardless |
| `bertalign/__init__.py` eager model load | By design | Fine for Modal (`@modal.enter()`). Don't `import bertalign` locally — only import `bertalign.modal_gpu` |
| `add_local_python_source` doesn't include `pyproject.toml` | Not needed | All deps are installed via `pip_install()` in the image definition |
| faiss CUDA version conflicts | Resolved | Use `faiss-gpu-cu12` base package, never `[fix-cuda]` extra |

## Usage from CLI

```bash
# Run the built-in example (Three Body Problem zh→en)
modal run bertalign/modal_gpu.py

# Check Modal dashboard for GPU usage, logs, and container lifecycle
# https://modal.com/apps
```
