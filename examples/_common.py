from itertools import product
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

MOMENTUM_GRID = [
    {"lookback": lb, "skip": sk, "quantile": q}
    for lb, sk, q in product((63, 126, 189, 252, 378), (0, 5, 21), (0.1, 0.2, 0.3))
]


def fmt(d: dict) -> str:
    out = []
    for k, v in d.items():
        if isinstance(v, tuple):
            out.append(f"{k}: [{v[0]:.2f}, {v[1]:.2f}]")
        elif isinstance(v, float):
            out.append(f"{k}: {v:.3f}")
        else:
            out.append(f"{k}: {v}")
    return "\n  ".join(out)
