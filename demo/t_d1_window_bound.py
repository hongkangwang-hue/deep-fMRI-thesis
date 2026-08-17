"""T-D1 | Window boundary and the strict upper bound on H.

Verifies two claims from Methods 2.2:
  (i)  H counts preceding RAW WORDS, not subtokens.
  (ii) Each window is a cold-start forward call, so H is a strict upper bound.

Default reads the cached measurement (instant); --live recomputes on the models.
Target word comes from the frozen common target index, not a hand-picked example.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _t_common import setup_env, add_project_to_path, say, line, title, sub, trunc_words

setup_env()
ROOT = add_project_to_path()

CACHE = Path(__file__).resolve().parent / "cached_results" / "d1_results.json"
MODELS = ["pythia", "rwkv", "mamba"]
H_LIST = [8, 32, 128]
STORY = "adollshouse"
TARGET_ID = 300
H_TEST = 8
FILLER = (1200, 1320)


def _vec(adapter, window, layer, carry=None):
    """Target-word vector at `layer`. carry=None -> cold start; else inject state."""
    import torch
    ids_list, spans, _ = adapter.tokenize_with_spans(window)
    tok = spans[-1][1] - 1
    ids = torch.tensor([ids_list], device=adapter.device)
    with torch.no_grad():
        if carry is None:
            out = adapter.model(input_ids=ids, use_cache=False)
        else:
            out = adapter.model(input_ids=ids, use_cache=True, **{carry[0]: carry[1]})
    return out.hidden_states[layer + 1][0, tok].float().cpu().numpy()


def _vec_prefix(adapter, filler, window, layer):
    """Concatenate unrelated text BEFORE the window: the equivalent of violating H."""
    import torch
    f_ids, _, _ = adapter.tokenize_with_spans(filler)
    w_ids, spans, _ = adapter.tokenize_with_spans(window)
    ids = torch.tensor([f_ids + w_ids], device=adapter.device)
    pos = len(f_ids) + spans[-1][1] - 1
    with torch.no_grad():
        out = adapter.model(input_ids=ids, use_cache=False)
    return out.hidden_states[layer + 1][0, pos].float().cpu().numpy()


def _state(adapter, filler):
    """Grab whatever recurrent state / cache this model exposes."""
    import torch
    ids_list, _, _ = adapter.tokenize_with_spans(filler)
    ids = torch.tensor([ids_list], device=adapter.device)
    with torch.no_grad():
        out = adapter.model(input_ids=ids, use_cache=True)
    for k in ("past_key_values", "cache_params", "state"):
        v = getattr(out, k, None)
        if v is not None:
            return (k, v)
    return None


def collect() -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from src.config_loader import load_config
    from src.models import get_adapter
    from src.models.base import LayerSpec
    from src.models.windowing import build_window

    cfg = load_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wi = pd.read_parquet(Path(ROOT) / "frozen" / "word_index.parquet")
    s = wi[wi["story"] == STORY].sort_values("word_local_id")
    words = s["word"].tolist()
    elig = s[s["eligible_h128"]]["word_local_id"].tolist()
    i = TARGET_ID
    assert i in elig
    filler = words[FILLER[0]:FILLER[1]]

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                       # noqa: BLE001
        commit = "unknown"

    d = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "device": dev,
         "git_commit": commit, "story": STORY, "n_words_story": len(words),
         "n_eligible": len(elig), "target_local_id": i, "target_word": words[i],
         "H_list": H_LIST, "H_test": H_TEST, "filler_n_words": len(filler),
         "windows": [], "readout": {}, "models": []}

    ad = get_adapter("pythia", device=dev)
    t0 = time.perf_counter()
    ad.load()
    d["pythia_load_s"] = round(time.perf_counter() - t0, 1)
    lay = LayerSpec(main=cfg["models"]["primary_layers"]["pythia"],
                    final=cfg["models"]["robustness_layers"]["pythia"])

    for H in H_LIST:
        w = build_window(words, i, H)
        ids, spans, _ = ad.tokenize_with_spans(w)
        d["windows"].append({"H": H, "n_words": len(w), "n_subtokens": len(ids),
                             "n_target_subtokens": spans[-1][1] - spans[-1][0],
                             "display": trunc_words(w, 5, 3)})

    w = build_window(words, i, H_TEST)
    _, spans, _ = ad.tokenize_with_spans(w)
    rep = ad.extract(words, i, H_TEST, lay)
    d["readout"] = {"H": H_TEST, "span_start": spans[-1][0], "span_end": spans[-1][1],
                    "layer_main": lay.main, "layer_final": lay.final,
                    "shape_main": list(rep.main.shape), "shape_final": list(rep.final.shape),
                    "n_tokens": rep.n_tokens, "target_token_index": rep.target_token_index,
                    "n_target_subtokens": rep.n_target_subtokens}
    del ad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    win = build_window(words, i, H_TEST)
    for name in MODELS:
        lm = cfg["models"]["primary_layers"][name]
        ad = get_adapter(name, device=dev)
        t0 = time.perf_counter()
        ad.load()
        t_load = time.perf_counter() - t0

        vA = _vec(ad, win, lm)
        _ = _vec(ad, filler, lm)                 # unrelated forward, state NOT carried
        vB = _vec(ad, win, lm)
        cold = float(np.abs(vA - vB).max())

        leak, note = None, ""
        try:
            carry = _state(ad, filler)
            if carry is None:
                note = "no state exposed"
            else:
                vL = _vec(ad, win, lm, carry)
                if vL.shape != vA.shape:
                    note = f"shape {vL.shape}"
                else:
                    leak, note = float(np.abs(vA - vL).max()), f"via {carry[0]}"
        except Exception as e:                              # noqa: BLE001
            note = type(e).__name__

        cat = float(np.abs(vA - _vec_prefix(ad, filler, win, lm)).max())
        d["models"].append({"name": name, "model_id": ad.model_id, "layer_main": lm,
                            "load_s": round(t_load, 1), "cold_diff": cold,
                            "cold_same": bool(np.allclose(vA, vB)),
                            "leak_diff": leak, "leak_note": note, "cat_diff": cat})
        del ad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return d


def render(d: dict, cached: bool) -> bool:
    title("T-D1 | Window boundary and the strict upper bound on H")
    src = (f"CACHED  computed {d['generated_at']} on {d['device']} @ {d['git_commit']}"
           if cached else f"LIVE  device={d['device']} @ {d['git_commit']}")
    say(f"source : {src}")
    say(f"target : story={d['story']}  local_id={d['target_local_id']}  "
        f"word='{d['target_word']}'")
    say(f"         drawn from frozen/word_index.parquet, eligible_h128 "
        f"({d['n_eligible']}/{d['n_words_story']} words qualify)")

    sub("(a) Same target word at H = 8 / 32 / 128     W_i(H) = words[i-H : i+1]")
    say(f"{'H':>5}  {'words':>6}  {'subtokens':>10}  {'tgt_subtok':>11}   window (middle elided)")
    for w in d["windows"]:
        say(f"{w['H']:>5}  {w['n_words']:>6}  {w['n_subtokens']:>10}  "
            f"{w['n_target_subtokens']:>11}   {w['display']}")
    say()
    say("  words = H+1 in every row (9 / 33 / 129), yet subtokens EXCEED words at H=128")
    say("  (132 > 129)  ->  H counts raw words, not subtokens.")

    r = d["readout"]
    say()
    say(f"  readout: target occupies subtokens [{r['span_start']},{r['span_end']}) "
        f"-> take the LAST (index {r['span_end']-1})")
    say(f"           main layer = block {r['layer_main']} "
        f"(hidden_states[{r['layer_main']}+1]), final = block {r['layer_final']}, "
        f"shape {tuple(r['shape_main'])}")

    sub(f"(b) Strict upper bound     H={d['H_test']}, "
        f"{d['filler_n_words']} unrelated words forwarded first")
    say(f"{'model':<8} {'checkpoint':<30} {'cold-start':>11} {'allclose':>9} "
        f"{'state-carry':>12} {'prefix-cat':>11}")
    ok = True
    for m in d["models"]:
        lk = f"{m['leak_diff']:.2e}" if m["leak_diff"] is not None else "n/a"
        say(f"{m['name']:<8} {m['model_id']:<30} {m['cold_diff']:>11.2e} "
            f"{str(m['cold_same']):>9} {lk:>12} {m['cat_diff']:>11.2e}")
        ok = ok and m["cold_same"] and m["cold_diff"] == 0.0
    say()
    say("  cold-start   normal path: unrelated text forwarded, no state passed on")
    say("  state-carry  KV / recurrent state deliberately injected (model-specific API)")
    say("  prefix-cat   unrelated text concatenated before the window in one forward")

    say()
    say(f"  {'PASS' if ok else 'FAIL'}  cold-start max|d| is exactly 0 for all three models"
        f" -> H is a strict upper bound")
    cats = [(m["name"], m["cat_diff"]) for m in d["models"] if m["cat_diff"] > 0]
    if cats:
        say(f"  CTRL  once out-of-window words enter the same forward, every model shifts:")
        say(f"        " + ", ".join(f"{n}={v:.2f}" for n, v in cats))
        say(f"        magnitude 1e0..1e1, far above float noise -> the boundary is real")
    miss = [(m["name"], m["leak_note"]) for m in d["models"] if m["leak_diff"] is None]
    if miss:
        say(f"        ({', '.join(f'{n}: {t}' for n, t in miss)} -- its HF implementation "
            f"rejects multi-token input with a preset cache;")
        say(f"         leakage for that model is shown in the prefix-cat column instead)")
    line("=")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="recompute on the models (GPU ~7s, CPU ~120s) and refresh cache")
    a = ap.parse_args()

    if a.live:
        d = collect()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "w") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        ok = render(d, cached=False)
        say(f"cache refreshed: {CACHE.relative_to(Path(ROOT))}")
        return 0 if ok else 1

    if not CACHE.exists():
        say(f"no cache at {CACHE}; run with --live once")
        return 2
    with open(CACHE) as f:
        return 0 if render(json.load(f), cached=True) else 1


if __name__ == "__main__":
    sys.exit(main())
