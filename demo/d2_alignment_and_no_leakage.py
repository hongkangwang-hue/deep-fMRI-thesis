"""D2 — 时间对齐的形状变换 + 无泄漏的可执行检查

【证明论文中的哪句声明】
  PPT Slide 6 横幅：“No held-out story contributes to feature transformation,
  hyperparameter selection or model fitting.”
  论文 2.3 节末尾对 leakage 的文字辩护（引 Moscovich & Rosset；Kapoor & Narayanan）。

【对应论文章节】2.1 节（词→TR 对齐与 FIR）、2.3 节（交叉验证与防泄漏）
【PPT 播放位置】Slide 6 之后

【为什么需要这个演示】
  论文用文字声明“没有泄漏”，但读者无法验证。本演示把这句声明拆成可执行检查：
  (a) 对一个真实故事逐步打印每一步之后的 shape，让观众看到数据被如何塑形；
  (b) 用三层证据检验防泄漏——真实 fold 划分断言、**运行时拦截** scaler/PCA/ridge 的
      fit 调用、以及当初全量运行时落盘的审计记录。每条都标注证据来源，不伪造断言。

【数据来源】
  - 词级特征：cache/features/（M1 已提取好的真实缓存，不重新跑模型）
  - fold 划分：frozen/fold_split.json（M0 冻结）
  - 审计记录：results/m4_full_matrix/{被试}/cells/*.json（M4 全量运行时真实落盘）

【实现说明（对应原代码位置）】
  (a) 全部调用现有函数，未复制逻辑：
      src/models/feature_cache.py::load_features      词级特征
      src/ridge/assemble.py::_word_times              词中点时间戳（私有函数，仅读取）
      src/fmri/trfile.py::story_tr_times/trimmed_tr_times   TR 时间轴
      src/fmri/alignment.py::word_to_tr               Lanczos 重采样
      src/fmri/alignment.py::apply_fir                FIR 四延迟展开
      src/fmri/mask.py::common_scoring_mask           评分掩码
  (b) 运行时拦截用 monkeypatch 包住 sklearn 的 StandardScaler.fit / PCA.fit，
      再调用真实的 src/ridge/pipeline.py::run_fold（未修改其代码）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _demo_common import (  # noqa: E402
    setup_env, add_project_to_path, say, rule, header, section, check, truncate_list,
)

setup_env()
ROOT = add_project_to_path()

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

from src.config_loader import load_config                              # noqa: E402
from src.models.feature_cache import load_features                     # noqa: E402
from src.ridge.assemble import _word_times, assemble_all               # noqa: E402
from src.fmri.trfile import (                                          # noqa: E402
    story_tr_times, trimmed_tr_times, TRIM_FIRST, TRIM_LAST, SIMULATE_PAD,
)
from src.fmri.alignment import word_to_tr, apply_fir                   # noqa: E402
from src.fmri.mask import common_scoring_mask                          # noqa: E402
from src.ridge.pipeline import (                                       # noqa: E402
    run_fold, numpy_ridgecv_solver, DELAYS_S, TR_SECONDS, AFTER_S, PCA_K, INNER_FOLDS,
)

SUBJECT = "UTS03"
DEMO_STORY = "adollshouse"
DEMO_MODEL = "pythia"
DEMO_H = 8
MINI_VOXELS = 150      # (b) 运行时拦截只用少量体素，保证秒级完成


def part_a(cfg) -> None:
    section("(a) 单个故事的形状变换链：词级特征 → TR 级 → FIR → 评分掩码")

    # 路径键与 scripts/m4_pythia.py:78-85 一致：数据集路径在 cfg["datasets"] 下
    paths, ds = cfg["paths"], cfg["datasets"]
    wi = pd.read_parquet(Path(paths["frozen_dir"]) / "word_index.parquet")
    with open(ds["respdict"]) as f:
        respdict = json.load(f)

    feat = load_features(paths["cache_dir"], DEMO_MODEL, DEMO_STORY, DEMO_H)
    word_ids = feat["word_ids"]
    vecs = feat["main"].astype(np.float64)

    say(f"故事 = {DEMO_STORY}   模型 = {DEMO_MODEL}   H = {DEMO_H}   被试 = {SUBJECT}")
    say(f"（词级特征读自 cache/features/，M1 已提取，本演示不重跑模型）")
    say()
    say(f"{'步骤':<38} {'shape / 数值':<26} 说明")
    say("-" * 96)

    say(f"{'1. 词级特征（缓存）':<38} {str(vecs.shape):<26} "
        f"{len(word_ids)} 个合格目标词 × 768 维")

    data_times = _word_times(wi, word_ids)
    order = np.argsort(data_times)
    data_times, vecs = data_times[order], vecs[order]
    say(f"{'2. 词中点时间戳 (onset+offset)/2':<38} {str(data_times.shape):<26} "
        f"范围 {data_times.min():.1f}s ~ {data_times.max():.1f}s（不等间隔）")

    n_resps = respdict[DEMO_STORY]
    tr_full = story_tr_times(n_resps)
    say(f"{'3. TR 时间轴（去 pad）':<38} {str(tr_full.shape):<26} "
        f"respdict={n_resps} − pad {SIMULATE_PAD} 帧"
        f"({SIMULATE_PAD * TR_SECONDS:.0f}s) = {len(tr_full)}")

    X_full = word_to_tr(vecs, data_times, tr_full)
    say(f"{'4. Lanczos 重采样 词→TR':<38} {str(X_full.shape):<26} "
        f"不等间隔词 → 等间隔 TR 网格（TR={TR_SECONDS:.0f}s）")

    X = X_full[TRIM_FIRST: len(X_full) - TRIM_LAST]
    trt = trimmed_tr_times(n_resps)
    say(f"{'5. 裁边 [10:-5]':<38} {str(X.shape):<26} "
        f"再去头 {TRIM_FIRST} 帧({TRIM_FIRST * TR_SECONDS:.0f}s)、"
        f"去尾 {TRIM_LAST} 帧({TRIM_LAST * TR_SECONDS:.0f}s)")

    # 与论文 "removed the first 15 and last 5 TRs" 的对账：单步是 20s，**累计**才是 30s
    head_total = SIMULATE_PAD + TRIM_FIRST
    say(f"{'   ↳ 相对原始响应的累计裁切':<38} "
        f"{f'{n_resps}−{SIMULATE_PAD}−{TRIM_FIRST}−{TRIM_LAST}={X.shape[0]}':<26} "
        f"累计去头 {head_total} 帧({head_total * TR_SECONDS:.0f}s)、"
        f"去尾 {TRIM_LAST} 帧({TRIM_LAST * TR_SECONDS:.0f}s)")

    # 硬断言：X / Y / tr_times 三者行数必须相等（原代码 assemble_story 内的同一检查）
    ok_rows = (X.shape[0] == len(trt))
    say(f"{'   ↳ 与 TR 时间轴行数一致性':<38} {str(X.shape[0]) + ' == ' + str(len(trt)):<26} "
        f"{'一致（assemble 内为硬断言）' if ok_rows else '不一致！'}")

    # PCA 位置：真实管线是 折内 scaler→PCA→FIR（PCA 在 FIR 之前，不是之后）
    say(f"{'6. [折内] scaler + PCA':<38} {'(T, ' + str(PCA_K) + ')':<26} "
        f"降到 {PCA_K} 维，仅用训练折拟合（见 (b)）")

    Z_fake = np.zeros((X.shape[0], PCA_K))     # 只为演示形状，不做真实 PCA
    Xf, valid = apply_fir(Z_fake, delays_s=DELAYS_S, tr=TR_SECONDS)
    say(f"{'7. FIR 四延迟展开 (2/4/6/8s)':<38} {str(Xf.shape):<26} "
        f"维度 ×{len(DELAYS_S)} = {PCA_K}×{len(DELAYS_S)}")

    say(f"{'   ↳ FIR 有效 TR（边缘置无效）':<38} "
        f"{str(int(valid.sum())) + ' / ' + str(len(valid)):<26} "
        f"前 {max(int(round(d/TR_SECONDS)) for d in DELAYS_S)} 帧因零填充无效")

    mask = common_scoring_mask(trt, valid, after_s=AFTER_S)
    say(f"{'8. 评分掩码 ∩ 故事开头>100s':<38} "
        f"{str(int(mask.sum())) + ' / ' + str(len(mask)):<26} "
        f"最终参与评分的 TR 数")
    say("-" * 96)
    say(f"总结：{len(word_ids)} 个词级向量 → {X.shape[0]} 个 TR 行 → "
        f"{Xf.shape[1]} 列设计矩阵 → {int(mask.sum())} 个 TR 参与打分")


def part_b(cfg) -> bool:
    section("(b) 无泄漏检查：三层证据")

    paths, ds = cfg["paths"], cfg["datasets"]
    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    folds = fold_split["folds"]

    all_pass = True

    # ── 证据层 1：真实 fold 划分（来源：frozen/fold_split.json，M0 冻结）──────
    say("【证据层 1】真实 fold 划分   来源：frozen/fold_split.json（M0 冻结，只读）")
    say()
    for fn, fd in folds.items():
        tr_s, te_s = list(fd["train_stories"]), list(fd["test_stories"])
        say(f"  {fn}:  train {len(tr_s):>2} 故事   test {len(te_s):>2} 故事")
        say(f"    test  = {truncate_list(sorted(te_s), head=3, tail=2)}")
        inter = set(tr_s) & set(te_s)
        all_pass &= check(len(inter) == 0,
                          f"{fn}: train ∩ test = ∅（交集 {len(inter)} 个）")
    say()

    test_counts = [len(fd["test_stories"]) for fd in folds.values()]
    all_pass &= check(sorted(test_counts) == [27, 28, 28],
                      f"三折测试故事数 = {test_counts}（论文声明 28/28/27）")

    all_test = [s for fd in folds.values() for s in fd["test_stories"]]
    all_pass &= check(len(all_test) == len(set(all_test)),
                      f"每个故事只在一个折里做过测试（{len(all_test)} 次测试，"
                      f"{len(set(all_test))} 个不同故事）")

    # ── 证据层 2：运行时拦截 scaler/PCA/ridge 的 fit 调用 ────────────────────
    say()
    say("【证据层 2】运行时拦截   在真实 run_fold 上 monkeypatch sklearn 的 fit 方法，")
    say("           直接观测它们究竟看到了哪些行（缩小规模以保证秒级完成）")
    say()

    fold0 = folds[list(folds.keys())[0]]
    mini_train = sorted(fold0["train_stories"])[:3]
    mini_test = sorted(fold0["test_stories"])[:2]
    say(f"  缩小版 fold：train = {mini_train}")
    say(f"               test  = {mini_test}   （体素只取前 {MINI_VOXELS} 列）")
    say(f"  ⚠️ 说明：这是用**真实故事、真实特征、真实 run_fold 代码**跑的缩小版，")
    say(f"     目的是在秒级内运行时观测 fit 行为；完整 28/28/27 折的同一代码路径")
    say(f"     的审计结果见证据层 3。")
    say()

    voxel_mask = np.arange(MINI_VOXELS)
    story_data = assemble_all(
        mini_train + mini_test, DEMO_MODEL, DEMO_H, "main", SUBJECT,
        paths["cache_dir"], ds["data_dir"], ds["respdict"],
        str(Path(paths["frozen_dir"]) / "word_index.parquet"),
        voxel_mask=voxel_mask,
    )
    rows_by_story = {s: sd.X.shape[0] for s, sd in story_data.items()}
    train_rows = sum(rows_by_story[s] for s in mini_train)
    test_rows = sum(rows_by_story[s] for s in mini_test)
    say(f"  各故事 TR 行数：{rows_by_story}")
    say(f"  → train 合计 {train_rows} 行，test 合计 {test_rows} 行，"
        f"全体 {train_rows + test_rows} 行")
    say()

    # monkeypatch：记录每次 fit 看到的行数
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    fit_log: list[tuple[str, tuple]] = []
    orig_scaler_fit, orig_pca_fit = StandardScaler.fit, PCA.fit

    def spy_scaler_fit(self, X, y=None, **kw):
        fit_log.append(("StandardScaler.fit", np.asarray(X).shape))
        return orig_scaler_fit(self, X, y, **kw)

    def spy_pca_fit(self, X, y=None, **kw):
        fit_log.append(("PCA.fit", np.asarray(X).shape))
        return orig_pca_fit(self, X, y, **kw)

    solver_log: list[tuple] = []

    def spy_solver(Xtr, Ytr, Xte, lambda_grid, inner_folds, seed):
        solver_log.append(("ridge_solver", Xtr.shape, Xte.shape, len(lambda_grid), inner_folds))
        return numpy_ridgecv_solver(Xtr, Ytr, Xte, lambda_grid, inner_folds, seed)

    StandardScaler.fit, PCA.fit = spy_scaler_fit, spy_pca_fit
    try:
        run_fold(story_data, mini_train, mini_test, spy_solver,
                 roi_columns=None, pca_k=min(PCA_K, train_rows - 1),
                 verbose=False, tag="/demo")
    finally:
        StandardScaler.fit, PCA.fit = orig_scaler_fit, orig_pca_fit

    say(f"  拦截到的 fit 调用共 {len(fit_log)} 次：")
    for what, shape in fit_log:
        say(f"    {what:<22} 输入 shape = {shape}")
    say()

    scaler_calls = [s for w, s in fit_log if w == "StandardScaler.fit"]
    pca_calls = [s for w, s in fit_log if w == "PCA.fit"]

    all_pass &= check(len(scaler_calls) == 1 and scaler_calls[0][0] == train_rows,
                      f"StandardScaler.fit 只调用 1 次，且行数 {scaler_calls[0][0]} "
                      f"== train 行数 {train_rows}（未含 test 的 {test_rows} 行）")
    all_pass &= check(len(pca_calls) == 1 and pca_calls[0][0] == train_rows,
                      f"PCA.fit 只调用 1 次，且行数 {pca_calls[0][0]} "
                      f"== train 行数 {train_rows}（未含 test 的 {test_rows} 行）")
    all_pass &= check(all(s[0] != train_rows + test_rows for s in scaler_calls + pca_calls),
                      f"没有任何 fit 调用看到全体 {train_rows + test_rows} 行")

    if solver_log:
        _w, xtr_shape, xte_shape, n_lam, n_inner = solver_log[0]
        say()
        say(f"  ridge solver 收到：Xtr={xtr_shape}  Xte={xte_shape}  "
            f"λ 候选 {n_lam} 个  inner folds={n_inner}")
        all_pass &= check(xtr_shape[0] <= train_rows,
                          f"λ 的 inner validation 只在 train 行内切分"
                          f"（Xtr {xtr_shape[0]} 行 ≤ train {train_rows} 行）")
        all_pass &= check(xte_shape[0] == test_rows,
                          f"test 行仅用于最终预测，未进入任何 fit"
                          f"（Xte {xte_shape[0]} == test {test_rows} 行）")

    # ── 证据层 3：全量运行时落盘的审计记录 ───────────────────────────────────
    say()
    say("【证据层 3】全量运行的审计记录   来源：results/m4_full_matrix/*/cells/*.json")
    say("           （M4 完整 28/28/27 折真实运行时写下的字段，非事后重建）")
    say()

    audited, flags = 0, {"leakage_audit_pass": 0, "common_mask_verified": 0,
                         "scoring_mask_bit_identical": 0}
    by_subj = {}
    for subj in ("UTS01", "UTS02", "UTS03"):
        cdir = Path(paths["results_dir"]) / "m4_full_matrix" / subj / "cells"
        n_s, n_bit = 0, 0
        for p in sorted(cdir.glob("main_*.json")):
            c = json.load(open(p))
            audited += 1
            n_s += 1
            if c.get("scoring_mask_bit_identical") is True:
                n_bit += 1
            for k in flags:
                if c.get(k) is True:
                    flags[k] += 1
        by_subj[subj] = (n_s, n_bit)

    say(f"  扫描 {audited} 个 M4 主层 cell（三被试 × 4 模型 × 3H × 3 折）：")
    for k, v in flags.items():
        say(f"    {k:<32} True 的 cell 数 = {v} / {audited}")
    all_pass &= check(flags["leakage_audit_pass"] == audited,
                      f"全部 {audited} 个 cell 的 leakage_audit_pass 均为 True")
    all_pass &= check(flags["common_mask_verified"] == audited,
                      f"全部 {audited} 个 cell 的 common_mask_verified 均为 True")

    # scoring_mask_bit_identical 逐被试看：UTS03 是 pilot 期先跑的，当时 m4_driver 还没
    # 加这个「逐元素相同」的强字段（只有较弱的 n_eff 计数相等检查），因此字段缺失。
    # 该缺口由独立审计脚本 scripts/verify_scoring_mask_identity.py 事后补齐，
    # 产物在 results/mask_identity_audit/ —— 下面把这处证据一并读出，避免把
    # 「字段缺失」误报成「掩码不一致」。
    say()
    say("  scoring_mask_bit_identical 逐被试明细：")
    for subj, (n_s, n_bit) in by_subj.items():
        note = "" if n_bit == n_s else "  ← pilot 期运行，该字段当时尚未加入"
        say(f"    {subj}: {n_bit} / {n_s}{note}")

    audit_dir = Path(paths["results_dir"]) / "mask_identity_audit"
    say()
    say(f"  补充证据：results/mask_identity_audit/（独立核验脚本的产物，覆盖三被试）")
    audit_ok, audit_rows = True, []
    for subj in ("UTS01", "UTS02", "UTS03"):
        ap = audit_dir / subj / "mask_identity.json"
        if not ap.exists():
            audit_ok = False
            say(f"    {subj}: 未找到 {ap.name}")
            continue
        a = json.load(open(ap))
        n_st = a["step2_n_stories_checked"]
        bit = a["step2_all_masks_bit_identical"]
        audit_rows.append((subj, n_st, bit))
        say(f"    {subj}: 检查 {n_st} 个故事，逐元素相同 = {bit}")
        audit_ok &= bool(bit)

    all_pass &= check(audit_ok and len(audit_rows) == 3,
                      "三被试的 normal/shift 评分掩码均经独立核验为逐元素相同"
                      "（覆盖 m4 cells 字段缺失的 UTS03）")

    return all_pass


def main() -> int:
    header("D2 — 时间对齐的形状变换 + 无泄漏的可执行检查",
           "数据从词到 TR 是怎样被塑形的？“没有 held-out 泄漏”这句声明能被验证吗？")

    cfg = load_config()
    part_a(cfg)
    ok = part_b(cfg)

    say()
    rule("=")
    if ok:
        say("[全部通过] 三层证据一致：held-out 故事从未进入特征变换、超参数选择或模型拟合。")
        say("           证据层 1 = 冻结的 fold 划分；层 2 = 运行时拦截真实 fit 调用；")
        say("           层 3 = 全量运行时落盘的审计字段。")
    else:
        say("[存在未通过项] 见上方 [FAIL] 行")
    rule("=")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
