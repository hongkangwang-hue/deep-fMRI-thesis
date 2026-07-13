"""
诊断脚本 —— AWD-LSTM 上下文窗口/记忆衰减完整核实（非冻结矩阵，纯诊断）。

一次运行回答 6 类问题（见每个 SECTION 标注是否需要新的模型推理）：
  1. 窗口是否真实不同（零计算，读窗口构造代码本身）
  2. 原始 LSTM 特征比较，PCA/插值/Ridge 之前（读已有缓存，零新推理）
  3. 缓存/文件互异性检查（读 content_hash 元数据，零新推理）
  4. 差异在哪一步消失：LSTM 原始输出 → TR 插值后 → PCA 后（插值复用缓存零新
     推理；PCA 需要一次真实的 StandardScaler+PCA-100 拟合，CPU、轻量、非 Ridge）
  5. 按距离分箱的敏感度（1-8/9-32/33-64/65-128 词），需要新的 AWD-LSTM 前向
     （轻量：单故事、300 目标、5 个额外 H 点）

**不写入 frozen/、不写入任何正式结果目录、不影响 M1-M6 任何已冻结产物**——
纯诊断，输出只打印到终端。

用法（服务器，需 fastai+torch+sklearn）：
  python scripts/diag_awd_lstm_context_decay.py --story souls --n-targets 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config                     # noqa: E402
from src.models import get_adapter                             # noqa: E402
from src.models.base import LayerSpec                          # noqa: E402
from src.models.windowing import iter_story_targets, build_window  # noqa: E402
from src.models.feature_cache import cache_path, load_features  # noqa: E402
from src.fmri.trfile import load_respdict                      # noqa: E402
from src.ridge.assemble import assemble_story                  # noqa: E402

FINE_H_GRID = [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128]
BIN_EDGES = [0, 8, 32, 64, 128]           # 划出 1-8 / 9-32 / 33-64 / 65-128 四个区间
BIN_LABELS = ["1-8词", "9-32词", "33-64词", "65-128词"]


def _diff_stats(A: np.ndarray, B: np.ndarray) -> dict:
    c = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    d = np.abs(A - B)
    return {
        "corr": c,
        "mean_abs_diff": float(d.mean()),
        "max_abs_diff": float(d.max()),
        "elementwise_identical": bool(np.array_equal(A, B)),
    }


def section1_window_facts():
    print("\n" + "=" * 70)
    print("[1] 窗口是否真实不同（零计算，直接核对窗口构造代码本身）")
    print("=" * 70)
    dummy = [f"w{i}" for i in range(140)]
    i = 135
    for H in (8, 32, 128):
        w = build_window(dummy, i, H)
        print(f"  H={H:>3}: 输入 token 数={len(w)}（=H+1），"
              f"窗口起止=位置[{i-H},{i}]（0-based，含端点），"
              f"末词={w[-1]}（=目标词本身）")
    print("  三者窗口起点不同、终点相同（右对齐、嵌套结构：H128窗口 ⊃ H32窗口 ⊃ H8窗口）")
    print("  reset_state: extract_batch() 每个 batch 前向前显式调用 self.reset_state()")
    print("    → self.encoder.reset() 清空各层 hidden/cell state 为零")
    print("    （src/models/base.py:156, src/models/awd_lstm_adapter.py:90-92）")
    print("  批内序列间天然独立（LSTM 按 batch 维并行处理，与 reset 无关，是矩阵运算的结构性质）；")
    print("  reset 的作用是保证【跨调用】（上一个 H/上一个故事/上一个 batch）不会残留状态污染本次前向。")


def section2_raw_feature_compare(cache_dir, model, story):
    print("\n" + "=" * 70)
    print("[2] 原始 LSTM 特征比较（PCA/插值/Ridge 之前，读已缓存文件，零新推理）")
    print("=" * 70)
    f = {H: load_features(cache_dir, model, story, H) for H in (8, 32, 128)}
    assert np.array_equal(f[8]["word_ids"], f[32]["word_ids"]) and \
           np.array_equal(f[32]["word_ids"], f[128]["word_ids"]), \
           "三个 H 的目标词集合不一致，无法逐元素比较"
    X = {H: f[H]["main"].astype(np.float64) for H in (8, 32, 128)}
    print(f"  目标词数（三者一致）: {X[8].shape[0]}，特征维度: {X[8].shape[1]}")
    r_8_32 = _diff_stats(X[8], X[32])
    r_32_128 = _diff_stats(X[32], X[128])
    for name, r in (("H8  vs H32 ", r_8_32), ("H32 vs H128", r_32_128)):
        print(f"  {name}: corr={r['corr']:.6f}  mean|diff|={r['mean_abs_diff']:.3e}  "
              f"max|diff|={r['max_abs_diff']:.3e}  逐元素完全相同={r['elementwise_identical']}")
    print("  判读：H32/H128 的 max|diff| 若落在 float32 舍入误差量级（~1e-6~1e-7），")
    print("  即为『数值上等同但并非严格逐位相同』；『逐元素完全相同=False』是预期结果，")
    print("  不代表存在差异信号，只说明两次独立前向计算存在浮点噪声。")
    return X


def section3_cache_identity(cache_dir, model, story):
    print("\n" + "=" * 70)
    print("[3] 缓存/文件互异性检查（读 content_hash 元数据，零新推理）")
    print("=" * 70)
    hashes = {}
    for H in (8, 32, 128):
        p = cache_path(cache_dir, model, story, H)
        f = load_features(cache_dir, model, story, H)
        hashes[H] = f["meta"]["content_hash"]
        print(f"  H={H:>3}: 缓存文件={p.name}  content_hash={hashes[H][:16]}...")
    all_diff = len(set(hashes.values())) == 3
    print(f"  三个 H 的 content_hash 两两互异: {all_diff}"
          f"（content_hash 由特征内容本身算出，若 H32/H128 误读同一份文件，"
          f"hash 会完全相同——{'排除' if all_diff else '⚠️未排除'}误读同一份特征的可能）")
    return hashes


def section4_pca_stage(cache_dir, model, cfg, subject, pca_max_stories):
    import gc
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print("\n" + "=" * 70)
    print("[4] 差异在哪一步消失：LSTM原始输出 → TR插值后 → PCA后")
    print("=" * 70)
    print("  （管线真实顺序：原始输出 → Lanczos TR插值[assemble.py] → PCA-100[pipeline.py] → FIR → Ridge；")
    print("   与常见直觉顺序不同，插值先于PCA，此处按真实顺序核实）")

    paths, ds = cfg["paths"], cfg["datasets"]
    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    fold = fold_split["folds"]["fold_0"]
    respdict = load_respdict(ds["respdict"])
    word_index = pd.read_parquet(Path(paths["frozen_dir"]) / "word_index.parquet")

    # -- 4a. TR 插值后（单故事，复用已缓存特征，零新模型推理）--
    test_story = fold["test_stories"][0]
    sd32 = assemble_story(test_story, model, 32, "main", subject, cache_dir,
                          ds["data_dir"], respdict, word_index)
    sd128 = assemble_story(test_story, model, 128, "main", subject, cache_dir,
                           ds["data_dir"], respdict, word_index)
    Xtest32_f32 = sd32.X.astype(np.float32)          # 后面投影用，float32 与正式管线一致
    Xtest128_f32 = sd128.X.astype(np.float32)
    r_interp = _diff_stats(sd32.X.astype(np.float64), sd128.X.astype(np.float64))
    print(f"\n  [4a] TR 插值后（故事={test_story}，形状={sd32.X.shape}）:")
    print(f"       H32 vs H128: corr={r_interp['corr']:.6f}  "
          f"mean|diff|={r_interp['mean_abs_diff']:.3e}  max|diff|={r_interp['max_abs_diff']:.3e}")
    del sd32, sd128

    # -- 4b. PCA 后（StandardScaler+PCA-100，CPU轻量拟合，非Ridge）--
    # 内存教训（上一版在此处被 OOM Killed）：awd_lstm 主层 1152 维，比其余模型的
    # 400 维宽近 3 倍；float64 是正式管线 float32 的两倍内存；且原代码把 H=32、
    # H=128 两份完整训练矩阵同时留在内存里。改为：float32、每个 H 用完立刻释放、
    # 训练故事数可控（--pca-max-stories，默认远小于 fold 全部 55 个故事，诊断
    # 用途不需要复现冻结 fold 的精确 PCA 对象，只需验证 evr@100 与投影后是否
    # 仍数值一致）。
    train_stories = fold["train_stories"][:pca_max_stories]
    print(f"\n  [4b] PCA-100 拟合（训练故事取 {len(train_stories)}/{len(fold['train_stories'])}"
          f"，H32/H128 各自独立拟合、依次处理不同时占内存，float32）...")
    seed = cfg["seeds"]["pca"]
    evr = {}
    Z = {}
    for H in (32, 128):
        parts = [assemble_story(s, model, H, "main", subject, cache_dir,
                                ds["data_dir"], respdict, word_index).X.astype(np.float32)
                for s in train_stories]
        Xtr = np.concatenate(parts, axis=0)
        del parts
        scaler = StandardScaler().fit(Xtr)
        pca = PCA(n_components=100, svd_solver="full", random_state=seed).fit(scaler.transform(Xtr))
        del Xtr
        gc.collect()
        evr[H] = float(pca.explained_variance_ratio_.sum())
        Xtest = Xtest32_f32 if H == 32 else Xtest128_f32
        Z[H] = pca.transform(scaler.transform(Xtest)).astype(np.float64)
        del scaler, pca
        gc.collect()
        print(f"       H={H}: evr@100={evr[H]:.6f}")
    print(f"       evr@100 差值 |H32-H128| = {abs(evr[32]-evr[128]):.3e}")

    r_pca = _diff_stats(Z[32], Z[128])
    print(f"       测试故事 {test_story} 投影到各自PCA空间后: corr={r_pca['corr']:.6f}  "
          f"mean|diff|={r_pca['mean_abs_diff']:.3e}  max|diff|={r_pca['max_abs_diff']:.3e}")
    print("\n  结论：若 [2]（原始输出）已在 float32 精度下无差异，[4a]（线性插值，保幂等）、")
    print("  [4b]（PCA，基于协方差结构拟合）理应同样无实质差异——以上数字用于直接核验，非推断。")


def section5_distance_binned_sensitivity(cache_dir, model, story, n_targets, device, cfg):
    print("\n" + "=" * 70)
    print("[5] 按距离分箱的敏感度（需要新的 AWD-LSTM 前向：额外 H 点，轻量）")
    print("=" * 70)
    word_index = pd.read_parquet(Path(cfg["paths"]["frozen_dir"]) / "word_index.parquet")
    story_targets = {s: (w, e) for s, w, e in iter_story_targets(word_index)}
    words, eligible = story_targets[story]
    targets = eligible[:n_targets]
    print(f"  故事={story}  取目标={len(targets)}/{len(eligible)}  额外 H 网格={FINE_H_GRID}")

    layers = LayerSpec(main=cfg["models"]["primary_layers"]["awd_lstm"],
                       final=cfg["models"]["robustness_layers"]["awd_lstm"])
    adapter = get_adapter("awd_lstm", device=device)
    adapter.load()

    X = {}
    for H in FINE_H_GRID:
        reps = adapter.extract_batch(words, targets, H, layers, batch_size=len(targets))
        X[H] = np.stack([r.main for r in reps]).astype(np.float64)

    print("\n  相邻网格点差异（衰减曲线是否平滑单调，非台阶跳变）:")
    for a, b in zip(FINE_H_GRID[:-1], FINE_H_GRID[1:]):
        c = np.corrcoef(X[a].ravel(), X[b].ravel())[0, 1]
        print(f"    corr(X_{a:>3}, X_{b:>3}) = {c:.6f}")

    print(f"\n  四个距离区间对隐藏状态的平均影响（区间边界差值的 mean|diff|，越大=该区间贡献越大）:")
    for (lo, hi), label in zip(zip(BIN_EDGES[:-1], BIN_EDGES[1:]), BIN_LABELS):
        d = np.abs(X[hi] - X[lo]).mean()
        print(f"    第{lo+1}-{hi}词（{label}）: mean|X_{hi}-X_{lo}| = {d:.3e}")
    print("\n  判读：数值应逐区间递减（越远的历史区间，新增贡献越小），且 65-128 区间应趋近于零"
          "（与[2]中H32≈H128的发现一致：128词窗口比32词窗口多出的96个更早的词，"
          "对目标词表示已无可测量贡献）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", default="souls")
    ap.add_argument("--n-targets", type=int, default=300)
    ap.add_argument("--subject", default="UTS03",
                    help="仅用于满足 assemble_story 的 Y 加载接口，本诊断不使用/不依赖 Y 的正确性")
    ap.add_argument("--device", default="auto",
                    help="auto=探测 torch.cuda.is_available() 自动选择；"
                         "AWD-LSTM 仅44M参数、300目标×14个H点，CPU也能轻松跑完")
    ap.add_argument("--model", default="awd_lstm")
    ap.add_argument("--pca-max-stories", type=int, default=15,
                    help="4b 步 PCA 拟合用的训练故事数上限（诊断用途，"
                         "不追求复现冻结fold的精确PCA对象，只验证evr@100是否一致；"
                         "awd_lstm主层1152维，故事数太多在部分容器上会被OOM killer杀掉）")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[diag] --device auto → 探测到 {device}"
              f"（GPU 不可用时自动退回 CPU，AWD-LSTM 规模小，CPU 完全可行）")

    cfg = load_config()
    cache_dir = cfg["paths"]["cache_dir"]

    section1_window_facts()
    section2_raw_feature_compare(cache_dir, args.model, args.story)
    section3_cache_identity(cache_dir, args.model, args.story)
    section4_pca_stage(cache_dir, args.model, cfg, args.subject, args.pca_max_stories)
    section5_distance_binned_sensitivity(cache_dir, args.model, args.story,
                                         args.n_targets, device, cfg)

    print("\n" + "=" * 70)
    print("[完成] 以上全部为诊断输出，未写入 frozen/ 或任何正式结果目录。")
    print("=" * 70)


if __name__ == "__main__":
    main()
