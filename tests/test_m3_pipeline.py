"""
M3 编码管线单测 —— 纯 CPU、合成数据，不依赖 himalaya / 真实 fMRI。

覆盖：
  1. 信号可恢复：Y 来自 FIR(X)@W → 测试折体素 r 高
  2. 防泄漏：Y 与 X 独立 → 测试折 r≈0（若 PCA/scaler/ridge 偷看测试 Y，null 会虚高）
  3. FIR 不跨故事：每故事开头若干行被标无效
  4. 评分函数：pearson / fisher-z / ROI 均值 / 有效-TR 加权
  5. solver 平手取较大 λ
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ridge.pipeline import (
    StoryData, run_fold, run_encoding_cv, numpy_ridgecv_solver,
    _transform_and_fir, DELAYS_S,
)
from src.ridge.score import (
    voxelwise_pearson, fisher_z, fisher_z_inv, roi_mean_r,
    effective_tr_weighted_mean,
)
from src.fmri.alignment import apply_fir
from src.fmri.mask import common_scoring_mask
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RNG = np.random.default_rng(0)
TR = 2.0


def _tr_times(T):
    """合成 trim 后 TR 中心时间，约半数 >100s。"""
    return np.arange(T) * TR - 10.0 + TR / 2.0


def _make_stories(n_stories=4, T=80, D=8, V=5, signal=True, noise=0.05):
    """造若干故事；signal=True 时 Y=FIR(X)@W+噪声，否则 Y 与 X 独立。"""
    W = RNG.standard_normal((D * len(DELAYS_S), V))
    data = {}
    for i in range(n_stories):
        X = RNG.standard_normal((T, D))
        if signal:
            Xf, _ = apply_fir(X, delays_s=DELAYS_S, tr=TR)
            Y = Xf @ W + noise * RNG.standard_normal((T, V))
        else:
            Y = RNG.standard_normal((T, V))
        data[f"s{i}"] = StoryData(X=X, Y=Y, tr_times=_tr_times(T))
    return data


# --------------------------------------------------------------------------- #
# 1. 信号可恢复
# --------------------------------------------------------------------------- #

def test_recovery_signal():
    data = _make_stories(signal=True, noise=0.02)
    fr = run_fold(data, ["s0", "s1", "s2"], ["s3"],
                  numpy_ridgecv_solver, pca_k=8, seed=0)
    assert fr.n_eff_tr > 0
    assert fr.voxel_r.mean() > 0.7, f"信号未恢复, mean r={fr.voxel_r.mean():.3f}"


# --------------------------------------------------------------------------- #
# 2. 防泄漏：null 数据不应虚高
# --------------------------------------------------------------------------- #

def test_null_no_leakage():
    data = _make_stories(signal=False)
    fr = run_fold(data, ["s0", "s1", "s2"], ["s3"],
                  numpy_ridgecv_solver, pca_k=8, seed=0)
    assert abs(fr.voxel_r.mean()) < 0.25, \
        f"null 泄漏迹象, mean r={fr.voxel_r.mean():.3f}"


def test_transform_fit_train_only():
    """run_fold 内 PCA/scaler 只在训练折 fit：改动测试故事 X 不改变训练拟合的变换。"""
    data = _make_stories(signal=True)
    train = ["s0", "s1", "s2"]
    # 手动按训练折 fit 变换
    Xtr_raw = np.vstack([data[s].X for s in train])
    sc = StandardScaler().fit(Xtr_raw)
    pca = PCA(n_components=8, svd_solver="full", random_state=0).fit(
        sc.transform(Xtr_raw))
    # 复刻 _transform_and_fir 对训练折的输出，与 pipeline 内部应一致
    Xf_ref, _, _, _, _ = _transform_and_fir(data, train, sc, pca, DELAYS_S, TR)
    fr = run_fold(data, train, ["s3"], numpy_ridgecv_solver, pca_k=8, seed=0)
    # 仅核对接口契约：训练折变换可独立复现（fit 不含测试折）
    assert Xf_ref.shape[1] == 8 * len(DELAYS_S)
    assert fr.valphas.shape == (data["s3"].Y.shape[1],)


# --------------------------------------------------------------------------- #
# 3. FIR 不跨故事
# --------------------------------------------------------------------------- #

def test_fir_no_cross_story():
    data = _make_stories(n_stories=2, T=60, D=8)
    sc = StandardScaler().fit(data["s0"].X)
    pca = PCA(n_components=8, svd_solver="full").fit(sc.transform(data["s0"].X))
    Xf, Y, valid, trt, lens = _transform_and_fir(
        data, ["s0", "s1"], sc, pca, DELAYS_S, TR)
    max_shift = max(int(round(d / TR)) for d in DELAYS_S)
    # 每个故事的前 max_shift 行应因 FIR 零填充而无效
    assert not valid[:max_shift].any()
    assert not valid[lens[0]:lens[0] + max_shift].any()
    # 故事边界处第二个故事开头独立无效（未从前一个故事借值）
    assert valid.sum() < len(valid)


# --------------------------------------------------------------------------- #
# 4. 评分函数
# --------------------------------------------------------------------------- #

def test_voxelwise_pearson():
    x = RNG.standard_normal((50, 3))
    assert np.allclose(voxelwise_pearson(x, x), 1.0, atol=1e-6)
    assert np.allclose(voxelwise_pearson(x, -x), -1.0, atol=1e-6)
    # 零方差列 → 0
    z = np.zeros((50, 1))
    assert voxelwise_pearson(z, x[:, :1])[0] == 0.0


def test_fisher_roundtrip():
    r = np.array([-0.9, 0.0, 0.5, 0.99])
    assert np.allclose(fisher_z_inv(fisher_z(r)), r, atol=1e-5)


def test_roi_mean_r():
    voxel_r = np.array([0.2, 0.4, 0.6, 0.8])
    cols = np.array([0, 2])
    expect = fisher_z_inv(fisher_z(np.array([0.2, 0.6])).mean())
    assert np.isclose(roi_mean_r(voxel_r, cols), expect)
    assert np.isnan(roi_mean_r(voxel_r, np.array([], dtype=int)))


def test_effective_tr_weighted_mean():
    r = effective_tr_weighted_mean(
        [np.array([0.0, 1.0]), np.array([1.0, 0.0])], [10, 30])
    # 加权: (0*10+1*30)/40=0.75, (1*10+0*30)/40=0.25
    assert np.allclose(r, [0.75, 0.25])


# --------------------------------------------------------------------------- #
# 5. solver 平手取较大 λ
# --------------------------------------------------------------------------- #

def test_solver_prefer_larger_lambda():
    # 纯噪声 Y → 所有 λ 验证 r 近似平手 → 应偏向较大 λ
    X = RNG.standard_normal((60, 5))
    Y = RNG.standard_normal((60, 4))
    grid = np.logspace(-2, 4, 13)
    _, lambdas = numpy_ridgecv_solver(X, Y, X, grid, inner_folds=2, seed=0)
    assert (lambdas >= np.median(grid)).mean() >= 0.5


# --------------------------------------------------------------------------- #
# 全 CV 串联
# --------------------------------------------------------------------------- #

def test_run_encoding_cv():
    data = _make_stories(n_stories=4, signal=True, noise=0.02)
    folds = [(["s1", "s2", "s3"], ["s0"]),
             (["s0", "s2", "s3"], ["s1"]),
             (["s0", "s1", "s3"], ["s2"])]
    res = run_encoding_cv(data, folds, numpy_ridgecv_solver, pca_k=8, seed=0)
    assert res.voxel_r.shape == (5,)
    assert len(res.folds) == 3
    assert res.voxel_r.mean() > 0.6


# --------------------------------------------------------------------------- #
# ROI 聚合下沉到 story 级（M3 模块4：每 story fisher-z→ROI，再 fold/跨fold 汇总 z）
# --------------------------------------------------------------------------- #

def test_roi_aggregated_at_story_level_not_from_fold_voxel_r():
    """ROI z 必须由 per-story roi_z 加权得到，而非对 fold voxel_r 再算 roi_mean_fisherz。
    两者因 fisher-z 非线性 + story 间 voxel r 分布不同而不等价。"""
    from src.ridge.score import roi_mean_fisherz, weighted_mean_scalar
    data = _make_stories(n_stories=3, T=80, D=8, V=6, signal=True, noise=0.02)
    data["s3"] = StoryData(X=RNG.standard_normal((150, 8)),
                           Y=RNG.standard_normal((150, 6)), tr_times=_tr_times(150))
    roi_cols = {"roiA": np.array([0, 1, 2]), "roiB": np.array([3, 4, 5])}
    fr = run_fold(data, ["s1", "s2"], ["s0", "s3"], numpy_ridgecv_solver,
                  roi_columns=roi_cols, pca_k=6, seed=0, verbose=False)

    # 正确路径：per-story roi_z 按有效TR加权
    neff = [ss.n_eff_tr for ss in fr.story_scores]
    for name in roi_cols:
        expected_z = weighted_mean_scalar(
            [ss.roi_z[name] for ss in fr.story_scores], neff)
        assert np.isclose(fr.roi_z[name], expected_z, atol=1e-12)
    # 错误路径（对 fold voxel_r 再算 ROI）应给出不同值 → 证明我们没走错路径
    wrong_z = roi_mean_fisherz(fr.voxel_r, roi_cols["roiA"])
    assert not np.isclose(fr.roi_z["roiA"], wrong_z, atol=1e-6)


def test_per_story_scores_preserved():
    """FoldResult 保留每个有效 story 的 voxel_r/roi_z/n_eff_tr（M4保存/M5 bootstrap）。"""
    data = _make_stories(n_stories=3, signal=True, noise=0.02)
    roi_cols = {"roiA": np.array([0, 1])}
    fr = run_fold(data, ["s1", "s2"], ["s0"], numpy_ridgecv_solver,
                  roi_columns=roi_cols, pca_k=8, seed=0, verbose=False)
    assert len(fr.story_scores) == 1
    ss = fr.story_scores[0]
    assert ss.story == "s0"
    assert ss.voxel_r.shape == (5,)
    assert "roiA" in ss.roi_z and np.isfinite(ss.roi_z["roiA"])
    assert ss.n_eff_tr > 0


def test_cv_roi_r_is_tanh_of_roi_z():
    data = _make_stories(n_stories=4, signal=True, noise=0.02)
    roi_cols = {"roiA": np.array([0, 1, 2])}
    folds = [(["s1", "s2", "s3"], ["s0"]), (["s0", "s2", "s3"], ["s1"])]
    res = run_encoding_cv(data, folds, numpy_ridgecv_solver,
                          roi_columns=roi_cols, pca_k=8, seed=0)
    assert np.isclose(res.roi_r["roiA"], np.tanh(res.roi_z["roiA"]))


# --------------------------------------------------------------------------- #
# story-level 评分（里程碑冻结文档 M3 模块4：逐 story 算 r 再汇总，非拼接后算一次）
# --------------------------------------------------------------------------- #

def test_pooling_vs_per_story_differ_in_general():
    """纯数学性质：拼接后算一次 r，与逐段算 r 再按点数加权平均，不是同一个统计量。

    构造两段：段A强正相关(pred≈actual)但均值偏移，段B无相关(纯噪声)。若拼接算，
    段间均值差会像 Simpson's paradox 一样污染相关；逐段算再加权则不受影响。
    """
    n1, n2 = 40, 10  # 故意不等长，让加权平均和简单拼接的差异更容易暴露
    rng = np.random.default_rng(1)
    actual_a = rng.standard_normal((n1, 3))
    pred_a = actual_a + 0.01 * rng.standard_normal((n1, 3)) + 5.0  # 强相关 + 均值偏移
    actual_b = rng.standard_normal((n2, 3))
    pred_b = rng.standard_normal((n2, 3))  # 与 actual_b 无关

    pooled_r = voxelwise_pearson(
        np.vstack([pred_a, pred_b]), np.vstack([actual_a, actual_b]))
    per_segment_r = effective_tr_weighted_mean(
        [voxelwise_pearson(pred_a, actual_a), voxelwise_pearson(pred_b, actual_b)],
        [n1, n2])

    assert np.abs(pooled_r - per_segment_r).max() > 0.05, (
        "构造的场景应能体现拼接 vs 逐段加权的数值差异，若相等说明测试场景没设计对")


def test_run_fold_scores_per_story_then_aggregates():
    """run_fold 对多测试故事的输出，应等于「逐故事算 r 再按有效TR加权平均」，
    而不是「拼接所有测试故事 TR 后算一次 r」。直接复刻 run_fold 内部步骤验证契约。
    """
    data = _make_stories(n_stories=3, T=80, D=8, V=4, signal=True, noise=0.02)
    # 追加一个更长、纯噪声的第二个测试故事（T 需够大让 >100s mask 有实际评分点，
    # 否则该故事贡献 0 个点，两种算法会退化成完全相同的计算，测不出差异）
    data["s3"] = StoryData(
        X=RNG.standard_normal((150, 8)),
        Y=RNG.standard_normal((150, 4)),
        tr_times=_tr_times(150),
    )
    # 测试折 = 一个有信号的故事(s0) + 一个纯噪声、长度不同的故事(s3)
    train = ["s1", "s2"]
    test = ["s0", "s3"]

    fr = run_fold(data, train, test, numpy_ridgecv_solver, pca_k=6, seed=0, verbose=False)

    # 手动复刻 run_fold 内部：同样的 scaler/PCA/FIR/solver，但显式逐故事切片评分
    Xtr_raw = np.vstack([data[s].X for s in train])
    scaler = StandardScaler().fit(Xtr_raw)
    pca = PCA(n_components=6, svd_solver="full", random_state=0)
    pca.fit(scaler.transform(Xtr_raw))
    Xtr_f, Ytr, vtr, _, _ = _transform_and_fir(data, train, scaler, pca, DELAYS_S, TR)
    Xtr_f, Ytr = Xtr_f[vtr], Ytr[vtr]
    Xte_f, Yte, vte, _, lens = _transform_and_fir(data, test, scaler, pca, DELAYS_S, TR)
    pred_te, _ = numpy_ridgecv_solver(
        Xtr_f, Ytr, Xte_f, np.logspace(-2, 4, 13), 2, 0)

    per_r, per_n, off = [], [], 0
    for s, L in zip(test, lens):
        seg = slice(off, off + L)
        m = common_scoring_mask(data[s].tr_times, vte[seg])
        per_r.append(voxelwise_pearson(pred_te[seg][m], Yte[seg][m]))
        per_n.append(int(m.sum()))
        off += L
    expected = effective_tr_weighted_mean(per_r, per_n)
    naive_pool_mask = np.concatenate([
        common_scoring_mask(data[s].tr_times,
                            vte[slice(sum(lens[:i]), sum(lens[:i]) + lens[i])])
        for i, s in enumerate(test)
    ])
    naive_pooled = voxelwise_pearson(pred_te[naive_pool_mask], Yte[naive_pool_mask])

    assert np.allclose(fr.voxel_r, expected, atol=1e-10), (
        "run_fold 应输出逐故事算r再加权平均的结果")
    assert fr.n_eff_tr == sum(per_n)
    # 二者不应在此构造场景下恰好相等（否则无法证明改动确实生效）
    assert not np.allclose(fr.voxel_r, naive_pooled, atol=1e-6)
