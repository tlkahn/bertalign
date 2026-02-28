"""
Bertalign initialization
"""

__author__ = "Jason (bfsujason@163.com)"
__version__ = "1.1.0"

# See other cross-lingual embedding models at
# https://www.sbert.net/docs/pretrained_models.html

model_name = "LaBSE"
_model = None


def _get_model():
    global _model
    if _model is None:
        from bertalign.encoder import Encoder
        _model = Encoder(model_name)
    return _model


def __getattr__(name):
    if name == "Bertalign":
        from bertalign.aligner import Bertalign
        return Bertalign
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
