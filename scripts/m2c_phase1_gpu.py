"""
M2-C Phase 1 —— GPU 加速版（完全独立脚本，不调用原生 ridge.py）。

用 PyTorch 在 GPU 上重实现 LeBel bootstrap ridge，数学逻辑逐行对应
encoding/ridge_utils/ridge.py，忠实复现以下细节：
  - bootstrap 抽样：chunk-based，与 LeBel bootstrap_ridge 协议相同
  - bootstrap 内评分：use_corr=False → R² with 平滑方差 (1+var)/2（LeBel 特有）
  - 最终 corrs：Pearson r（对应 native return_wt=True → corrcoef 路径）
  - 奇异值截断：singcutoff=1e-10

性能优化：训练矩阵一次性加载到 GPU，bootstrap 内用 GPU 索引，避免重复传输。

输出格式与 m2c_phase1_native.py 完全相同，可直接用 scripts/m2c_compare.py 对照。
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import sys
import time
from os.path import join

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, join(PROJECT_ROOT, "encoding"))

from encoding_utils import apply_zscore_and_hrf, get_response          # noqa: E402
from feature_spaces import get_feature_space, _FEATURE_CONFIG          # noqa: E402
from config import EM_DATA_DIR                                         # noqa: E402

# 与 native 版完全相同的超参数（冻结见 frozen/m2c_reference_validation.yaml）
ALPHAS     = np.logspace(1, 3, 10).astype(np.float32)
NBOOTS     = 50
CHUNKLEN   = 40
NCHUNKS    = 125
TRIM       = 5
NDELAYS    = 4
SINGCUTOFF = 1e-10
USE_CORR   = False   # bootstrap 内评分用 R²（与 native 一致）


# ---------------------------------------------------------------------------
# GPU 核心函数
# ---------------------------------------------------------------------------

def _zs(x: torch.Tensor) -> torch.Tensor:
    """列 z-score（对应 LeBel `zs = lambda v: (v-v.mean(0))/v.std(0)`，ddof=0）。"""
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def _ridge_corr_gpu(
    RRstim: torch.Tensor, PRstim: torch.Tensor,
    RRresp: torch.Tensor, PRresp: torch.Tensor,
    alphas: np.ndarray, singcutoff: float, use_corr: bool,
) -> np.ndarray:
    """
    单次 bootstrap 的 ridge_corr（GPU tensors 版）。

    对应 ridge.py:ridge_corr，忠实复现：
      - SVD + 奇异值截断
      - UR = U.T @ Rresp，PVh = Pstim @ Vh.T
      - use_corr=False：平滑方差 Prespvar = (1 + actual_var) / 2  ← LeBel 特有
      - use_corr=True：z-score 相关

    Returns: (nalphas, nvox) float32 numpy array
    """
    U, S, Vh = torch.linalg.svd(RRstim, full_matrices=False)
    keep = S > singcutoff
    U, S, Vh = U[:, keep], S[keep], Vh[keep, :]

    UR  = U.T @ RRresp   # (k, nvox)
    PVh = PRstim @ Vh.T  # (nval, k)

    if use_corr:
        zPRresp = _zs(PRresp)
    else:
        # LeBel 平滑方差：避免方差极小的体素主导 alpha 选择
        PRresp_var = PRresp.var(0)
        smooth_var = (torch.ones_like(PRresp_var) + PRresp_var) / 2.0

    nalphas = len(alphas)
    nvox = RRresp.shape[1]
    Rcorrs = torch.zeros(nalphas, nvox, dtype=torch.float32, device=RRstim.device)

    for ai, alpha in enumerate(alphas):
        D    = S / (S ** 2 + float(alpha) ** 2)   # (k,)
        pred = (PVh * D) @ UR                      # (nval, nvox)

        if use_corr:
            zpred = _zs(pred)
            Rcorrs[ai] = (zPRresp * zpred).mean(0)
        else:
            resvar = (PRresp - pred).var(0)
            Rsq    = 1.0 - resvar / smooth_var
            Rcorrs[ai] = torch.sqrt(torch.abs(Rsq)) * torch.sign(Rsq)

    return Rcorrs.cpu().numpy()


def _final_corrs_gpu(
    Rstim_g: torch.Tensor, Pstim_g: torch.Tensor,
    Rresp_g: torch.Tensor, Presp_g: torch.Tensor,
    valphas_np: np.ndarray, singcutoff: float,
) -> np.ndarray:
    """
    全量训练数据计算最终 weights，返回 Pearson r。

    对应 native return_wt=True 路径：
      ridge.py:ridge(Rstim, Rresp, valphas) → wt
      pred = Pstim @ wt
      corrs = corrcoef(Presp[:,i], pred[:,i])  for each voxel
    """
    device = Rstim_g.device

    U, S, Vh = torch.linalg.svd(Rstim_g, full_matrices=False)
    keep = S > singcutoff
    U, S, Vh = U[:, keep], S[keep], Vh[keep, :]

    UR   = U.T @ Rresp_g       # (k, nvox)
    nfeat = Rstim_g.shape[1]
    nvox  = Rresp_g.shape[1]
    wt    = torch.zeros(nfeat, nvox, dtype=torch.float32, device=device)

    va_g = torch.tensor(valphas_np, dtype=torch.float32, device=device)
    for ua in torch.unique(va_g):
        selvox = (va_g == ua).nonzero(as_tuple=True)[0]
        D   = S / (S ** 2 + ua ** 2)
        awt = Vh.T @ (D.unsqueeze(1) * UR[:, selvox])   # (nfeat, |selvox|)
        wt[:, selvox] = awt

    pred = Pstim_g @ wt   # (ntest, nvox)

    # Pearson r（与 native np.corrcoef 路径等价）
    Pr_c   = Presp_g - Presp_g.mean(0)
    pred_c = pred    - pred.mean(0)
    num    = (Pr_c * pred_c).sum(0)
    denom  = Pr_c.norm(dim=0) * pred_c.norm(dim=0) + 1e-12
    corrs  = (num / denom).cpu().numpy()
    return np.nan_to_num(corrs)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def bootstrap_ridge_gpu(
    Rstim_np: np.ndarray, Rresp_np: np.ndarray,
    Pstim_np: np.ndarray, Presp_np: np.ndarray,
    alphas: np.ndarray,
    nboots: int, chunklen: int, nchunks: int,
    singcutoff: float = 1e-10, use_corr: bool = False,
    device: torch.device = None,
):
    """
    GPU bootstrap ridge，协议与 LeBel bootstrap_ridge 完全相同。

    优化：全量训练矩阵一次加载到 GPU，bootstrap 内用 GPU 索引，
    避免每轮 host→device 传输 1.7GB 响应矩阵。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[gpu_phase1] 预加载训练数据到 {device} ...", flush=True)
    Rstim_g = torch.tensor(Rstim_np, dtype=torch.float32, device=device)
    Rresp_g = torch.tensor(Rresp_np, dtype=torch.float32, device=device)
    Pstim_g = torch.tensor(Pstim_np, dtype=torch.float32, device=device)
    Presp_g = torch.tensor(Presp_np, dtype=torch.float32, device=device)
    if device.type == "cuda":
        used = torch.cuda.memory_allocated(device) / 1e9
        print(f"[gpu_phase1] GPU 显存已用: {used:.2f}GB", flush=True)

    nresp   = Rstim_np.shape[0]
    nvox    = Rresp_np.shape[1]
    nalphas = len(alphas)

    allRcorrs_list: list[np.ndarray] = []
    valinds: list[list[int]] = []
    t_start = time.time()

    for bi in range(nboots):
        t_boot = time.time()

        # 抽样（CPU，与 LeBel 完全相同协议）
        allinds   = list(range(nresp))
        indchunks = list(zip(*[iter(allinds)] * chunklen))
        random.shuffle(indchunks)
        heldinds    = list(itertools.chain(*indchunks[:nchunks]))
        notheldinds = sorted(set(allinds) - set(heldinds))
        valinds.append(heldinds)

        # GPU 索引（无 host-device 传输）
        ni_t = torch.tensor(notheldinds, device=device)
        hi_t = torch.tensor(heldinds,    device=device)

        RRstim = Rstim_g[ni_t]   # (nresp-nchunks*chunklen, nfeat)
        PRstim = Rstim_g[hi_t]   # (nchunks*chunklen, nfeat)
        RRresp = Rresp_g[ni_t]   # (nresp-nchunks*chunklen, nvox)
        PRresp = Rresp_g[hi_t]   # (nchunks*chunklen, nvox)

        Rcmat = _ridge_corr_gpu(RRstim, PRstim, RRresp, PRresp,
                                alphas, singcutoff, use_corr)
        allRcorrs_list.append(Rcmat)

        elapsed_boot  = time.time() - t_boot
        elapsed_total = time.time() - t_start
        remaining     = elapsed_boot * (nboots - bi - 1)
        print(
            f"[gpu_phase1] boot {bi+1:2d}/{nboots}  "
            f"{elapsed_boot:.1f}s  "
            f"已过 {elapsed_total/60:.1f}min  "
            f"剩余≈{remaining/60:.1f}min",
            flush=True,
        )

    # (nalphas, nvox, nboots)
    allRcorrs     = np.stack(allRcorrs_list, axis=2)
    meanbootcorrs = allRcorrs.mean(2)          # (nalphas, nvox)
    bestalphainds = np.argmax(meanbootcorrs, 0)
    valphas       = alphas[bestalphainds]

    print("[gpu_phase1] 所有 bootstrap 完成，计算最终预测 ...", flush=True)
    corrs = _final_corrs_gpu(Rstim_g, Pstim_g, Rresp_g, Presp_g,
                             valphas, singcutoff)

    return corrs, valphas, allRcorrs, valinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject",  default="UTS03")
    ap.add_argument("--feature",  default="eng1000", choices=list(_FEATURE_CONFIG))
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed",     type=int, default=20260629)
    ap.add_argument("--out-name", default="eng1000_gpu_rerun",
                    help="输出到 results/<out-name>/<subject>/")
    ap.add_argument("--device",   default="cuda",
                    help="torch device（cuda / cpu）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        stream=sys.stdout)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[gpu_phase1] 使用设备: {device}", flush=True)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[gpu_phase1] GPU: {props.name}  "
              f"显存: {props.total_memory/1e9:.1f}GB", flush=True)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 读取故事划分
    with open(join(EM_DATA_DIR, "sess_to_story.json")) as f:
        sess_to_story = json.load(f)
    train_stories, test_stories = [], []
    for sess in map(str, args.sessions):
        stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
        train_stories.extend(stories)
        if tstory not in test_stories:
            test_stories.append(tstory)
    assert not (set(train_stories) & set(test_stories)), "Train-Test overlap!"
    allstories = list(set(train_stories) | set(test_stories))

    save_location = join(PROJECT_ROOT, "results", args.out_name, args.subject)
    os.makedirs(save_location, exist_ok=True)
    print(f"[gpu_phase1] 输出: {save_location}  种子={args.seed}", flush=True)
    print(f"[gpu_phase1] train={len(train_stories)} 故事  test={test_stories}", flush=True)

    # 特征与响应（与 native 完全相同调用）
    feat     = get_feature_space(args.feature, allstories)
    delRstim = apply_zscore_and_hrf(train_stories, feat, TRIM, NDELAYS)
    delPstim = apply_zscore_and_hrf(test_stories,  feat, TRIM, NDELAYS)
    zRresp   = get_response(train_stories, args.subject)
    zPresp   = get_response(test_stories,  args.subject)
    print(f"[gpu_phase1] delRstim{delRstim.shape} delPstim{delPstim.shape} "
          f"zRresp{zRresp.shape} zPresp{zPresp.shape}", flush=True)

    print(f"[gpu_phase1] 开始 GPU bootstrap ridge  nboots={NBOOTS} ...", flush=True)
    t0 = time.time()

    corrs, valphas, bscorrs, valinds = bootstrap_ridge_gpu(
        delRstim.astype(np.float32), zRresp.astype(np.float32),
        delPstim.astype(np.float32), zPresp.astype(np.float32),
        ALPHAS, NBOOTS, CHUNKLEN, NCHUNKS,
        singcutoff=SINGCUTOFF, use_corr=USE_CORR, device=device,
    )

    elapsed = time.time() - t0
    print(f"[gpu_phase1] 总耗时 {elapsed/60:.1f} 分钟", flush=True)

    # 保存（格式与 native 版相同）
    np.savez(join(save_location, "corrs"),   corrs)
    np.savez(join(save_location, "valphas"), valphas)
    np.savez(join(save_location, "bscorrs"), bscorrs)
    np.savez(join(save_location, "valinds"), np.array(valinds))
    manifest = {
        "subject":      args.subject,
        "feature":      args.feature,
        "seed":         args.seed,
        "sessions":     args.sessions,
        "test_stories": test_stories,
        "device":       str(device),
        "dtype":        "float32",
        "alphas":       "logspace(1,3,10)",
        "nboots":       NBOOTS,
        "chunklen":     CHUNKLEN,
        "nchunks":      NCHUNKS,
        "trim":         TRIM,
        "ndelays":      NDELAYS,
        "use_corr":     USE_CORR,
        "corrs_shape":  list(corrs.shape),
        "note":         "GPU float32 版；bootstrap 评分用 LeBel 平滑方差；最终 corrs 为 Pearson r",
    }
    with open(join(save_location, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[gpu_phase1] corrs{corrs.shape} mean={np.nanmean(corrs):.4f} "
          f"max={np.nanmax(corrs):.4f} → 已保存", flush=True)
    print(f"[gpu_phase1] 下一步:\n"
          f"  python scripts/m2c_compare.py \\\n"
          f"      --ours {save_location}/corrs.npz \\\n"
          f"      --ours-valphas {save_location}/valphas.npz",
          flush=True)


if __name__ == "__main__":
    main()