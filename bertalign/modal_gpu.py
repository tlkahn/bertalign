"""Modal GPU backend for Bertalign.

Offloads the full pipeline (LaBSE encode → faiss search → DP alignment)
to a cloud GPU via Modal. No torch/faiss/numba needed locally.

Usage:
    # Smoke test
    modal run bertalign/modal_gpu.py

    # From Python — same API as bertalign.Bertalign
    from bertalign.modal_gpu import Bertalign
    aligner = Bertalign(src, tgt)
    aligner.align_sents()
    aligner.print_sents()
"""

from __future__ import annotations

import modal

app = modal.App("bertalign-gpu")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2.2",
        "numba>=0.64",
        "torch>=2.5",
        "sentence-transformers>=3.0",
        # Do NOT use faiss-gpu-cu12[fix-cuda] — it conflicts with
        # PyTorch's CUDA runtime.
        "faiss-gpu-cu12>=1.9",
        "sentence-splitter>=1.4",
        "googletrans==4.0.0rc1",
    )
    .add_local_python_source("bertalign")
)

model_volume = modal.Volume.from_name("bertalign-labse-cache", create_if_missing=True)


@app.cls(
    gpu="L40S",
    image=image,
    volumes={"/root/.cache/huggingface": model_volume},
    scaledown_window=300,
    timeout=600,
)
class BertalignService:

    @modal.enter()
    def load_model(self):
        """Import bertalign to trigger LaBSE download + load."""
        import bertalign  # noqa: F401

        model_volume.commit()

    @modal.method()
    def align(
        self,
        src_text: str,
        tgt_text: str,
        is_split: bool = True,
        max_align: int = 5,
        top_k: int = 3,
        win: int = 5,
        skip: float = -0.1,
        margin: bool = True,
        len_penalty: bool = True,
    ) -> dict:
        from bertalign import Bertalign

        aligner = Bertalign(
            src_text,
            tgt_text,
            max_align=max_align,
            top_k=top_k,
            win=win,
            skip=skip,
            margin=margin,
            len_penalty=len_penalty,
            is_split=is_split,
        )
        aligner.align_sents()
        return {
            "result": [[list(map(int, b[0])), list(map(int, b[1]))] for b in aligner.result],
            "src_sents": list(aligner.src_sents),
            "tgt_sents": list(aligner.tgt_sents),
        }


def align_remote(src_text: str, tgt_text: str, **kwargs) -> list[tuple[list[int], list[int]]]:
    """Call bertalign on Modal GPU from local code. Returns raw alignment indices."""
    service = BertalignService()
    data = service.align.remote(src_text, tgt_text, **kwargs)
    return [(tuple(b[0]), tuple(b[1])) for b in data["result"]]


class Bertalign:
    """Local proxy that mirrors the bertalign.Bertalign API but runs on Modal GPU."""

    def __init__(self, src: str, tgt: str, is_split: bool = False, **kwargs):
        self._src = src
        self._tgt = tgt
        self._is_split = is_split
        self._kwargs = kwargs

    def align_sents(self):
        service = BertalignService()
        data = service.align.remote(
            self._src, self._tgt, is_split=self._is_split, **self._kwargs
        )
        self.result = data["result"]
        self.src_sents = data["src_sents"]
        self.tgt_sents = data["tgt_sents"]

    def print_sents(self):
        for src_ids, tgt_ids in self.result:
            src_line = " ".join(self.src_sents[src_ids[0] : src_ids[-1] + 1]) if src_ids else ""
            tgt_line = " ".join(self.tgt_sents[tgt_ids[0] : tgt_ids[-1] + 1]) if tgt_ids else ""
            print(src_line + "\n" + tgt_line + "\n")


@app.local_entrypoint()
def main(eval: bool = False):
    """Usage: modal run bertalign/modal_gpu.py [--eval]"""
    if eval:
        _run_eval()
        return

    src = """两年以后，大兴安岭。
"顺山倒咧——"
随着这声嘹亮的号子，一棵如巴特农神庙的巨柱般高大的落叶松轰然倒下，叶文洁感到大地抖动了一下。她拿起斧头和短锯，开始去除巨大树身上的枝丫。每到这时，她总觉得自己是在为一个巨人整理遗体。她甚至常常有这样的想象：这巨人就是自己的父亲。两年前那个凄惨的夜晚，她在太平间为父亲整理遗容时的感觉就在这时重现。巨松上那绽开的树皮，似乎就是父亲躯体上累累的伤痕。
内蒙古生产建设兵团的六个师四十一个团十多万人就分布在这辽阔的森林和草原之间。刚从城市来到这陌生的世界时，很多兵团知青都怀着一个浪漫的期望：当苏修帝国主义的坦克集群越过中蒙边境时，他们将飞快地武装起来，用自己的血肉构成共和国的第一道屏障。事实上，这也确实是兵团组建时的战略考虑之一。但他们渴望的战争就像草原天边那跑死马的远山，清晰可见，但到不了眼前，于是他们只有垦荒、放牧和砍伐。这些曾在"大串联"中燃烧青春的年轻人很快发现，与这广阔天地相比，内地最大的城市不过是个羊圈；在这寒冷无际的草原和森林间，燃烧是无意义的，一腔热血喷出来，比一堆牛粪凉得更快，还不如后者有使用价值。但燃烧是他们的命运，他们是燃烧的一代。于是，在他们的油锯和电锯下，大片的林海化为荒山秃岭；在他们的拖拉机和康拜因（联合收割机）下，大片的草原被犁成粮田，然后变成沙漠。
叶文洁看到的砍伐只能用疯狂来形容，高大挺拔的兴安岭落叶松、四季常青的樟子松、亭亭玉立的白桦、耸入云天的山杨、西伯利亚冷杉，以及黑桦、柞树、山榆、水曲柳、钻天柳、蒙古栎，见什么伐什么，几百把油锯如同一群钢铁蝗虫，她的连队所过之处，只剩下一片树桩。
整理好的落叶松就要被履带拖拉机拖走了，在树干另一头，叶文洁轻轻抚摸了一下那崭新的锯断面，她常常下意识地这么做，总觉得那是一处巨大的伤口，似乎能感到大树的剧痛。她突然看到，在不远处树桩的锯断面上，也有一只在轻轻抚摸的手，那手传达出的心灵的颤抖，与她产生了共振。那手虽然很白皙，但能够看出是属于男性的。叶文洁抬头，看到抚摸树桩的人是白沐霖，一个戴眼镜的瘦弱青年，他是兵团《大生产报》的记者，前天刚到连队来采访。叶文洁看过他写的文章，文笔很好，其中有一种与这个粗放环境很不协调的纤细和敏感，令她很难忘。"""

    tgt = """Two years later, the Greater Khingan Mountains
"Tim-ber…"
Following the loud chant, a large Dahurian larch, thick as the columns of the Parthenon, fell with a thump, and Ye Wenjie felt the earth quake.
She picked up her ax and saw and began to clear the branches from the trunk. Every time she did this, she felt as though she were cleaning the corpse of a giant. Sometimes she even imagined the giant was her father. The feelings from that terrible night two years ago when she cleaned her father's body in the mortuary would resurface, and the splits and cracks in the larch bark seemed to turn into the old scars and new wounds covering her father.
Over one hundred thousand people from the six divisions and forty-one regiments of the Inner Mongolia Production and Construction Corps were scattered among the vast forests and grasslands. When they first left the cities and arrived at this unfamiliar wilderness, many of the corps' "educated youths"—young college students who no longer had schools to go to—had cherished a romantic wish: When the tank clusters of the Soviet Revisionist Imperialists rolled over the Sino-Mongolian border, they would arm themselves and make their own bodies the first barrier in the Republic's defense. Indeed, this expectation was one of the strategic considerations motivating the creation of the Production and Construction Corps.
But the war they craved was like a mountain at the other end of the grassland: clearly visible, but as far away as a mirage. So they had to content themselves with clearing fields, grazing animals, and chopping down trees.
Soon, the young men and women who had once expended their youthful energy on tours to the holy sites of the Chinese Revolution discovered that, compared to the huge sky and open air of Inner Mongolia, the biggest cities in China's interior were nothing more than sheep pens. Stuck in the middle of the cold, endless expanse of forests and grasslands, their burning ardor was meaningless. Even if they spilled all of their blood, it would cool faster than a pile of cow dung, and not be as useful.
But burning was their fate; they were the generation meant to be consumed by fire.
And so, under their chain saws, vast seas of forests turned into barren ridges and denuded hills. Under their tractors and combine harvesters, vast tracts of grasslands became grain fields, then deserts.
Ye Wenjie could only describe the deforestation that she witnessed as madness. The tall Dahurian larch, the evergreen Scots pine, the slim and straight white birch, the cloud-piercing Korean aspen, the aromatic Siberian fir, along with black birch, oak, mountain elm, Chosenia arbutifolia—whatever they laid eyes on, they cut down. Her company wielded hundreds of chain saws like a swarm of steel locusts, and after they passed, only stumps were left.
The fallen Dahurian larch, now bereft of branches, was ready to be taken away by tractor. Ye gently caressed the freshly exposed cross section of the felled trunk. She did this often, as though such surfaces were giant wounds, as though she could feel the tree's pain. Suddenly, she saw another hand lightly stroking the matching surface of the stump a few feet away. The tremors in that hand revealed a heart that resonated with hers.
Though the hand was pale, she could tell it belonged to a man.
She looked up. It was Bai Mulin. A slender, delicate man who wore glasses, he was a reporter for the Great Production News, the corps' newspaper. He had arrived the day before yesterday to gather news about her company. Ye remembered reading his articles, which were written in a beautiful style, sensitive and fine, ill suited to the rough-hewn environment."""

    aligner = Bertalign(src, tgt)
    aligner.align_sents()
    aligner.print_sents()


def _run_eval():
    """Run Text+Berg benchmark evaluation."""
    import importlib.util
    import os
    import sys

    # Import eval.py directly to avoid bertalign/__init__.py (which eagerly
    # loads torch/LaBSE — not available locally).
    spec = importlib.util.spec_from_file_location(
        "bertalign_eval", os.path.join(os.path.dirname(__file__), "eval.py")
    )
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    read_alignments = eval_mod.read_alignments
    score_multiple = eval_mod.score_multiple
    log_final_scores = eval_mod.log_final_scores

    src_dir = "text+berg/de"
    tgt_dir = "text+berg/fr"
    gold_dir = "text+berg/gold"

    test_alignments = []
    gold_alignments = []
    for file in sorted(os.listdir(src_dir)):
        src_file = os.path.join(src_dir, file)
        tgt_file = os.path.join(tgt_dir, file)
        src_text = open(src_file, "rt", encoding="utf-8").read()
        tgt_text = open(tgt_file, "rt", encoding="utf-8").read()

        print(f"Aligning {src_file} → {tgt_file} ...", file=sys.stderr)
        aligner = Bertalign(src_text, tgt_text, is_split=True)
        aligner.align_sents()
        test_alignments.append(aligner.result)

        gold_file = os.path.join(gold_dir, file)
        gold_alignments.append(read_alignments(gold_file))

    scores = score_multiple(gold_list=gold_alignments, test_list=test_alignments)
    log_final_scores(scores)
