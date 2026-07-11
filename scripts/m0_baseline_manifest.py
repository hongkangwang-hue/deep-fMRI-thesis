"""
M0（三被试扩展）—— 登记 UTS03 已验证基线的只读锚定信息。

不修改、不重算任何 UTS03 冻结产物，只对已存在的冻结文件做 sha256 registration，
并从 frozen/word_index.parquet 的 eligible_h128 列派生刺激侧目标集合的 hash
（该集合与被试无关，三被试共享，见 milestone M0 的"设计思路"）。

语言特征缓存（cache/features/）本身不在本机（gitignored、只存在于计算服务器），
其内容完整性由 src/models/feature_cache.py 的逐文件 content_hash 在每次
save/load 时自动校验，不需要在这里另算一个目录级全局 hash。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINE_FILES = [
    "config/config.yaml",
    "frozen/analysis_spec.yaml",
    "frozen/contrast_registry.yaml",
    "frozen/roi_spec.json",
    "frozen/fold_split.json",
    "frozen/story_manifest.csv",
    "frozen/word_index.parquet",
    "frozen/voxel_mask_UTS03.json",
    "frozen/roi_columns_UTS03.json",
    "frozen/m2c_reference_validation.yaml",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def eligible_h128_hash() -> dict:
    import pandas as pd

    df = pd.read_parquet(REPO_ROOT / "frozen/word_index.parquet")
    elig = df.loc[df["eligible_h128"], ["story", "word_id"]].sort_values(
        ["story", "word_id"]
    )
    h = hashlib.sha256()
    h.update(elig.to_csv(index=False).encode())
    return {
        "n_eligible_targets": int(len(elig)),
        "n_stories": int(elig["story"].nunique()),
        "hash": h.hexdigest(),
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    file_hashes = {}
    for rel in BASELINE_FILES:
        p = REPO_ROOT / rel
        file_hashes[rel] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    manifest = {
        "purpose": (
            "UTS03 已验证基线的只读锚定登记（三被试扩展 M0 deliverable #2）。"
            "本文件登记的是基线的现状指纹，不是新的冻结决定；UTS03 的任何规则"
            "改动都应回到各自的冻结文件，不应改这里。"
        ),
        "baseline_git_tag": "uts03-graduation-baseline",
        "baseline_tag_commit": git("rev-list", "-n", "1", "uts03-graduation-baseline"),
        "manifest_generated_at_commit": git("rev-parse", "HEAD"),
        "config_and_frozen_file_hashes_sha256": file_hashes,
        "eligible_h128_target_set": eligible_h128_hash(),
        "language_feature_cache": {
            "location": "cache/features/ (gitignored, lives on compute server only)",
            "integrity_mechanism": (
                "src/models/feature_cache.py: 每个 (model, story, H) 缓存文件在"
                " save_features 时写入内容 sha256 (content_hash)，load_features"
                " 每次读取都会重新计算并比对，不一致直接抛错。因此复用缓存的完整性"
                " 由代码在每次使用时自动校验，不依赖此处另算一个目录级全局 hash。"
            ),
            "extraction_verified": (
                "M1b（2026-07-01，服务器）：4 模型（pythia/mamba/rwkv/awd_lstm）"
                "×3 个 H（8/32/128）= 12 组合，各 83 个 CV 故事全部提取完成，"
                "批量提取 vs 逐窗口提取数值比对 max|Δ|<3e-4，全部通过。"
            ),
            "reused_by_three_subject_extension": True,
        },
    }

    out_path = REPO_ROOT / "frozen" / "uts03_baseline_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")
    print(json.dumps(manifest["eligible_h128_target_set"], indent=2))


if __name__ == "__main__":
    main()
