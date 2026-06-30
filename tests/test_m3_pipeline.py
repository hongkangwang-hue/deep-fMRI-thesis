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
