"""
M1 — 特征缓存读写（story × model × H × layer × word_id）

布局：每个 (model, story, H) 一个 .npz，内含按 word_id 升序排列的主层/最终层
表示、word_id 顺序、is_unk，以及一段 JSON metadata（含 revision、层号、
hidden 宽度、代码版本与内容 hash）。缓存目录默认在 cache/features（已 gitignore，
可重新提取）。

content_hash 让正式运行能写 run manifest，并在复用缓存时校验未被悄悄改动。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CACHE_SCHEMA_VERSION = 1


def cache_path(cache_dir: str | Path, model: str, story: str, H: int) -> Path:
    return Path(cache_dir) / model / f"{story}_H{H}.npz"


def _content_hash(
    word_ids: np.ndarray,
    main: np.ndarray,
    final: np.ndarray,
    meta_core: dict,
) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(word_ids).tobytes())
    h.update(np.ascontiguousarray(main).tobytes())
    h.update(np.ascontiguousarray(final).tobytes())
    h.update(json.dumps(meta_core, sort_keys=True).encode())
    return h.hexdigest()


def save_features(
    cache_dir: str | Path,
    model: str,
    story: str,
    H: int,
    word_ids: np.ndarray,
    main: np.ndarray,
    final: np.ndarray,
    is_unk: np.ndarray,
    meta: dict,
) -> Path:
    """保存一个 (model, story, H) 的双层特征。

    main/final 形状均为 (n_targets, hidden)，行顺序与 word_ids 一致。meta 至少
    应含 model_id/revision/layer_main/layer_final/hidden_width/code_version。
    """
    word_ids = np.asarray(word_ids, dtype=np.int64)
    main = np.asarray(main, dtype=np.float32)
    final = np.asarray(final, dtype=np.float32)
    is_unk = np.asarray(is_unk, dtype=bool)

    n = len(word_ids)
    if not (main.shape[0] == final.shape[0] == is_unk.shape[0] == n):
        raise ValueError(
            f"行数不一致: word_ids={n}, main={main.shape[0]}, "
            f"final={final.shape[0]}, is_unk={is_unk.shape[0]}"
        )
    # 注意：主层与最终层宽度可以不同（AWD-LSTM 主层=1152、最终层=400），
    # 因此这里分别记录，不做等宽断言。
    meta_core = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": model,
        "story": story,
        "H": H,
        "n_targets": n,
        "hidden_width_main": int(main.shape[1]),
        "hidden_width_final": int(final.shape[1]),
        **{k: meta[k] for k in sorted(meta)},
    }
    meta_core["content_hash"] = _content_hash(word_ids, main, final, meta_core)

    out = cache_path(cache_dir, model, story, H)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        word_ids=word_ids,
        main=main,
        final=final,
        is_unk=is_unk,
        meta=np.array(json.dumps(meta_core)),
    )
    return out


def load_features(cache_dir: str | Path, model: str, story: str, H: int) -> dict:
    """加载并校验 content_hash，返回 dict（含 word_ids/main/final/is_unk/meta）。"""
    path = cache_path(cache_dir, model, story, H)
    with np.load(path, allow_pickle=False) as z:
        word_ids = z["word_ids"]
        main = z["main"]
        final = z["final"]
        is_unk = z["is_unk"]
        meta = json.loads(str(z["meta"]))

    stored = meta.pop("content_hash")
    recomputed = _content_hash(word_ids, main, final, meta)
    if stored != recomputed:
        raise ValueError(
            f"缓存内容 hash 不匹配（可能被改动）: {path}\n"
            f"  stored={stored[:16]}... recomputed={recomputed[:16]}..."
        )
    meta["content_hash"] = stored
    return {
        "word_ids": word_ids,
        "main": main,
        "final": final,
        "is_unk": is_unk,
        "meta": meta,
    }
