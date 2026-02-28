# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bertalign[modal]",
#     "requests",
# ]
# [tool.uv.sources]
# bertalign = { path = "..", editable = true }
# ///
"""
Bertalign GPU → Alignment API: end-to-end example.

Aligns a Chinese-English text pair on a Modal cloud GPU (LaBSE + faiss),
then imports the result into the alignment_api Rails backend and queries it.

Prerequisites:
    modal setup          # one-time auth (if first time)
    cd ~/Projects/bertalign && modal run bertalign/modal_gpu.py  # warm up

Usage (uv — no manual venv or pip install needed):
    # With the Rails server running (rails server -p 3000):
    cd ~/Projects/alignment_api
    uv run --with modal --with requests \
           --extra-search-path ~/Projects/bertalign \
           docs/bertalign-gpu-example.py

    The --with flags install modal and requests into an ephemeral
    environment. --extra-search-path makes the local bertalign
    package importable without sys.path hacks.

Alternative (plain Python):
    pip install modal requests
    PYTHONPATH=~/Projects/bertalign python docs/bertalign-gpu-example.py
"""

import json
import requests
from bertalign.modal_gpu import Bertalign

API = "http://localhost:3000/api/v1"

# ── 1. Source texts ──────────────────────────────────────────

src = """\
两年以后，大兴安岭。
"顺山倒咧——"
随着这声嘹亮的号子，一棵如巴特农神庙的巨柱般高大的落叶松轰然倒下，叶文洁感到大地抖动了一下。\
"""

tgt = """\
Two years later, the Greater Khingan Mountains.
"Timber!"
Following the loud chant, a large Dahurian larch, thick as the columns of the Parthenon, fell with a thump, and Ye Wenjie felt the earth quake.\
"""

# ── 2. Align on GPU ─────────────────────────────────────────

print("Aligning on Modal GPU...")
aligner = Bertalign(src, tgt, is_split=True)
aligner.align_sents()

print(f"  {len(aligner.src_sents)} source sentences")
print(f"  {len(aligner.tgt_sents)} target sentences")
print(f"  {len(aligner.result)} alignment pairs")
for s, t in aligner.result:
    print(f"    src{list(s)} -> tgt{list(t)}")

# ── 3. Import into alignment_api ─────────────────────────────

print("\nImporting into alignment_api...")
resp = requests.post(f"{API}/import", json={
    "src_lang": "zh",
    "tgt_lang": "en",
    "src_name": "three_body_zh.txt",
    "tgt_name": "three_body_en.txt",
    "name": "zh-en Three Body Problem excerpt",
    "src_sents": list(aligner.src_sents),
    "tgt_sents": list(aligner.tgt_sents),
    "alignment_result": [
        {"src_indices": list(s), "tgt_indices": list(t)}
        for s, t in aligner.result
    ],
})
resp.raise_for_status()
alignment = resp.json()
alignment_id = alignment["id"]
print(f"  Created alignment #{alignment_id} ({alignment['aligned_pairs_count']} pairs)")

# ── 4. Query the API ─────────────────────────────────────────

print("\nAligned pairs:")
resp = requests.get(f"{API}/alignments/{alignment_id}/aligned_pairs")
for p in resp.json():
    print(f"  [{p['position']}] ({p['alignment_type']})")
    print(f"    ZH: {p['text_from']}")
    print(f"    EN: {p['text_to']}")

print("\nStats:")
resp = requests.get(f"{API}/alignments/{alignment_id}/stats")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

print("\nSearch for '大兴安岭':")
resp = requests.get(
    f"{API}/alignments/{alignment_id}/aligned_pairs/search",
    params={"q": "大兴安岭"},
)
for p in resp.json():
    print(f"  [{p['position']}] {p['text_from'][:40]}... ↔ {p['text_to'][:40]}...")

print("\nTMX export (first 300 chars):")
resp = requests.get(
    f"{API}/alignments/{alignment_id}/export",
    params={"format_type": "tmx"},
)
print(resp.text[:300])

print("\nDone.")
