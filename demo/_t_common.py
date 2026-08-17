"""Shared setup + compact output helpers for the T-series demos (English, data-first)."""

from __future__ import annotations

import os
import sys

W = 98


def setup_env() -> None:
    """Offline HF + sane threads + silence logs. MUST run before importing torch.

    Offline mode is not optional here: the server cannot reach huggingface.co, and
    transformers retries HEAD requests 5x per config file (measured: RWKV stalled for
    minutes). Weights are already in the local HF cache.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if not os.environ.get("OMP_NUM_THREADS", "").strip().isdigit():
        os.environ["OMP_NUM_THREADS"] = "4"

    import warnings
    warnings.filterwarnings("ignore")
    import logging
    for n in ("transformers", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(n).setLevel(logging.ERROR)


def add_project_to_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def say(*a, **k) -> None:
    k.setdefault("flush", True)
    print(*a, **k)


def line(ch: str = "-") -> None:
    say(ch * W)


def title(t: str) -> None:
    say()
    say(t)
    line("=")


def sub(t: str) -> None:
    say()
    say(t)
    line("-")


def check(ok: bool, msg: str) -> bool:
    say(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    return ok


def trunc(items, head: int = 5, tail: int = 3) -> str:
    items = list(items)
    if len(items) <= head + tail:
        return ", ".join(map(str, items))
    return (", ".join(map(str, items[:head])) + f", ...({len(items)-head-tail} more)..., "
            + ", ".join(map(str, items[-tail:])))


def trunc_words(words, head: int = 5, tail: int = 3) -> str:
    words = list(words)
    if len(words) <= head + tail:
        return " ".join(words)
    return (" ".join(words[:head]) + f" [...{len(words)-head-tail}w...] "
            + " ".join(words[-tail:]))
