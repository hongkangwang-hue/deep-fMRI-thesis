"""
M2-C Phase 1 —— GPU 加速版（完全独立脚本，不调用原生 ridge.py）。

用 PyTorch float64 在 GPU 上重实现 LeBel bootstrap ridge，数学逻辑逐行对应
encoding/ridge_utils/ridge.py，忠实复现以下细节：
  - bootstrap 抽样：chunk-based，与 LeBel bootstrap_ridge 协议相同
  - bootstrap 内评分：use_corr=False → R² with 平滑方差 (1+var)/2（LeBel 特有）
  - 最终 corrs：sqrt(|R²|)·sign(R²)（对应 native return_wt=True, use_corr=False 路径）
  - 奇异值截断：singcutoff=1e-10
  - 方差一律用 ddof=0（与 numpy .var(0) 一致；torch 默认 ddof=1，故手算）

内存管理：float64 的响应矩阵达 7.2GB，超过 24GB 显存上限（含中间变量）。
策略：只预加载小矩阵（Rstim/Pstim/Presp，合计 ~530MB）；
Rresp 按 bootstrap 按需传输（PCIe 传输约 0.23s/次，可忽略）；
alpha 循环按体素分块（默认 8000 列/块），避免实例化全量 pred 矩阵。

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
ALPHAS     = np.logspace(1, 3, 10)          # float64
NBOOTS     = 50
CHUNKLEN   = 40
NCHUNKS    = 125
TRIM       = 5
NDELAYS    = 4
SINGCUTOFF = 1e-10
USE_CORR   = False   # bootstrap 内评分用 R²（与 native 一致）
VOX_CHUNK  = 8000    # alpha 循环体素分块大小，控制峰值显存


# ---------------------------------------------------------------------------
# GPU 核心函数
# ---------------------------------------------------------------------------

def _var0(x: torch.Tensor) -> torch.Tensor:
    """ddof=0 方差，与 numpy `.var(0)` 一致。

    torch `.var(0)` 默认 correction=1（ddof=1），而 LeBel 全程用 numpy 默认 ddof=0。
    手算 ((x-mean)**2).mean(0) 既匹配 numpy 又不依赖 torch 版本（correction= kwarg 旧版不支持）。
    """
    return ((x - x.mean(0)) ** 2).mean(0)


def _zs(x: torch.Tensor) -> torch.Tensor:
    """列 z-score（对应 LeBel `zs = lambda v: (v-v.mean(0))/v.std(0)`，ddof=0）。"""
    return (x - x.mean(0)) / (torch.sqrt(_var0(x)) + 1e-12)


def _ridge_corr_gpu(
    RRstim_np: np.ndarray, PRstim_np: np.ndarray,
    RRresp_np: np.ndarray, PRresp_np: np.ndarray,
    alphas: np.ndarray, singcutoff: float, use_corr: bool,
    device: torch.device, vox_chunk: int = VOX_CHUNK,
) -> np.ndarray:
    """
    单次 bootstrap 的 ridge_corr，GPU float64 版，含显存管理。

    忠实复现 LeBel ridge.py:ridge_corr：
      - SVD + 奇异值截断
      - UR = U.T @ Rresp，PVh = Pstim @ Vh.T
      - use_corr=False：平滑方差 (1 + actual_var) / 2  ← LeBel 特有
    显存优化：
      - del U/RRresp/RRstim/Vh/PRstim 在不再需要后立刻释放
      - alpha 循环按 vox_chunk 列分块，避免实例化 (nval, nvox) 全量 pred

    Returns: (nalphas, nvox) float64 numpy array
    """
    # 传输到 GPU
    RRstim = torch.tensor(RRstim_np, dtype=torch.float64, device=device)
    PRstim = torch.tensor(PRstim_np, dtype=torch.float64, device=device)
    RRresp = torch.tensor(RRresp_np, dtype=torch.float64, device=device)
    PRresp = torch.tensor(PRresp_np, dtype=torch.float64, device=device)

    # SVD
    U, S, Vh = torch.linalg.svd(RRstim, full_matrices=False)
    del RRstim
    keep = S > singcutoff
    U, S, Vh = U[:, keep], S[keep], Vh[keep, :]

    UR = U.T @ RRresp    # (k, nvox)
    del U, RRresp

    PVh = PRstim @ Vh.T  # (nval, k)
    del PRstim, Vh

    nvox    = UR.shape[1]
    nalphas = len(alphas)
    Rcorrs  = np.zeros((nalphas, nvox), dtype=np.float64)

    # 预计算（use_corr=False）
    if not use_corr:
        PRresp_var = _var0(PRresp)               # ddof=0，匹配 numpy
        smooth_var = (1.0 + PRresp_var) / 2.0
        del PRresp_var
    else:
        zPRresp = _zs(PRresp)

    for ai, alpha in enumerate(alphas):
        D    = S / (S ** 2 + float(alpha) ** 2)   # (k,)
        PVhD = PVh * D                             # (nval, k)

        for vs in range(0, nvox, vox_chunk):
            ve = min(vs + vox_chunk, nvox)
            pred_chunk = PVhD @ UR[:, vs:ve].contiguous()   # (nval, chunk)

            if use_corr:
                zpred = _zs(pred_chunk)
                Rcorrs[ai, vs:ve] = (zPRresp[:, vs:ve] * zpred).mean(0).cpu().numpy()
                del zpred
            else:
                diff   = PRresp[:, vs:ve] - pred_chunk
                resvar = _var0(diff)             # ddof=0，匹配 numpy
                del diff
                Rsq = 1.0 - resvar / smooth_var[vs:ve]
                del resvar
                Rcorrs[ai, vs:ve] = (torch.sqrt(torch.abs(Rsq)) * torch.sign(Rsq)).cpu().numpy()
                del Rsq
            del pred_chunk

        del PVhD

    # 释放剩余 GPU 资源
    del UR, PVh, S
    if not use_corr:
        del smooth_var, PRresp
    else:
        del zPRresp, PRresp

    return Rcorrs


def _final_corrs_gpu(
    Rstim_g: torch.Tensor, Pstim_g: torch.Tensor,
    Rresp_np: np.ndarray, Presp_g: torch.Tensor,
    valphas_np: np.ndarray, singcutoff: float,
    device: torch.device,
) -> np.ndarray:
    """
    全量训练数据计算最终 weights，返回 sqrt(|R²|)·sign(R²)。

    对应 native return_wt=True, use_corr=False 路径：
      ridge(Rstim, Rresp, valphas) → wt；pred = Pstim @ wt
      resvar=(Presp-pred).var(0); Rsq=1-resvar/Presp.var(0); corrs=sqrt(|Rsq|)·sign(Rsq)
    Rresp 在此处按需加载，用完即释放。
    """
    # 临时加载 Rresp（函数返回后由调用方 del）
    Rresp_g = torch.tensor(Rresp_np, dtype=torch.float64, device=device)

    U, S, Vh = torch.linalg.svd(Rstim_g, full_matrices=False)
    keep = S > singcutoff
    U, S, Vh = U[:, keep], S[keep], Vh[keep, :]

    UR   = U.T @ Rresp_g    # (k, nvox)
    del U, Rresp_g

    nfeat = Rstim_g.shape[1]
    nvox  = UR.shape[1]
    wt    = torch.zeros(nfeat, nvox, dtype=torch.float64, device=device)

    va_g = torch.tensor(valphas_np, dtype=torch.float64, device=device)
    for ua in torch.unique(va_g):
        selvox = (va_g == ua).nonzero(as_tuple=True)[0]
        D   = S / (S ** 2 + ua ** 2)
        awt = Vh.T @ (D.unsqueeze(1) * UR[:, selvox])
        wt[:, selvox] = awt
        del awt
    del UR, S, Vh, va_g

    pred = Pstim_g @ wt   # (ntest, nvox) — small: 291×95556×8 = 0.22GB
    del wt

    # 与 native use_corr=False, return_wt=True 路径完全一致：
    #   resvar = (Presp - pred).var(0)        [numpy ddof=0]
    #   Rsqs   = 1 - resvar / Presp.var(0)
    #   corrs  = sqrt(|Rsqs|) * sign(Rsqs)
    # 注意：numpy .var(0) 默认 ddof=0；torch .var() 默认 correction=1，
    # 须显式指定 correction=0 以匹配 numpy 行为。
    pred_np   = pred.cpu().numpy()
    del pred
    Presp_np  = Presp_g.cpu().numpy()
    resvar    = (Presp_np - pred_np).var(0)   # ddof=0，与 numpy 一致
    Presp_var = Presp_np.var(0)               # ddof=0
    Rsqs      = 1.0 - resvar / Presp_var
    corrs     = np.nan_to_num(np.sqrt(np.abs(Rsqs)) * np.sign(Rsqs))
    return corrs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def bootstrap_ridge_gpu(
    Rstim_np: np.ndarray, Rresp_np: np.ndarray,
    Pstim_np: np.ndarray, Presp_np: np.ndarray,
    alphas: np.ndarray,
    nboots: int, chunklen: int, nchunks: int,
    singcutoff: float = 1e-10, use_corr: bool = False,
    device: torch.device = None, vox_chunk: int = VOX_CHUNK,
):
    """
    GPU bootstrap ridge（float64），显存安全版。

    内存策略：
      - 预加载小矩阵（Rstim/Pstim/Presp，~530MB）
      - Rresp 按 bootstrap 从 CPU 按需传输（~0.23s/次，可忽略）
      - alpha 内循环按 vox_chunk 体素分块（峰值 <12GB）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 只预加载小矩阵
    print(f"[gpu_phase1] 预加载 Rstim/Pstim/Presp 到 {device} ...", flush=True)
    Rstim_g = torch.tensor(Rstim_np, dtype=torch.float64, device=device)
    Pstim_g = torch.tensor(Pstim_np, dtype=torch.float64, device=device)
    Presp_g = torch.tensor(Presp_np, dtype=torch.float64, device=device)
    if device.type == "cuda":
        used = torch.cuda.memory_allocated(device) / 1e9
        print(f"[gpu_phase1] 静态显存: {used:.2f}GB（Rresp 按需传输）", flush=True)

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

        # Rstim 用 GPU 索引（小，无传输开销）
        ni_t = torch.tensor(notheldinds, device=device)
        hi_t = torch.tensor(heldinds,    device=device)
        RRstim_np = Rstim_np[notheldinds, :]
        PRstim_np = Rstim_np[heldinds,    :]
        # Rresp 从 CPU 按 bootstrap 子集传输
        RRresp_np = Rresp_np[notheldinds, :]
        PRresp_np = Rresp_np[heldinds,    :]
        del ni_t, hi_t

        Rcmat = _ridge_corr_gpu(
            RRstim_np, PRstim_np, RRresp_np, PRresp_np,
            alphas, singcutoff, use_corr, device, vox_chunk,
        )
        del RRstim_np, PRstim_np, RRresp_np, PRresp_np
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
    meanbootcorrs = allRcorrs.mean(2)
    bestalphainds = np.argmax(meanbootcorrs, 0)
    valphas       = alphas[bestalphainds]

    print("[gpu_phase1] 所有 bootstrap 完成，计算最终预测 ...", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    corrs = _final_corrs_gpu(
        Rstim_g, Pstim_g, Rresp_np, Presp_g, valphas, singcutoff, device,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    ap.add_argument("--vox-chunk", type=int, default=VOX_CHUNK,
                    help="alpha 循环体素分块大小（减小可降低显存峰值）")
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
    delRstim = apply_zscore_and_hrf(train_stories, feat, TRIM, NDELAYS).astype(np.float64)
    delPstim = apply_zscore_and_hrf(test_stories,  feat, TRIM, NDELAYS).astype(np.float64)
    zRresp   = get_response(train_stories, args.subject).astype(np.float64)
    zPresp   = get_response(test_stories,  args.subject).astype(np.float64)
    print(f"[gpu_phase1] delRstim{delRstim.shape} delPstim{delPstim.shape} "
          f"zRresp{zRresp.shape} zPresp{zPresp.shape}", flush=True)

    print(f"[gpu_phase1] 开始 GPU bootstrap ridge  nboots={NBOOTS} ...", flush=True)
    t0 = time.time()

    corrs, valphas, bscorrs, valinds = bootstrap_ridge_gpu(
        delRstim, zRresp, delPstim, zPresp,
        ALPHAS, NBOOTS, CHUNKLEN, NCHUNKS,
        singcutoff=SINGCUTOFF, use_corr=USE_CORR,
        device=device, vox_chunk=args.vox_chunk,
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
        "dtype":        "float64",
        "alphas":       "logspace(1,3,10)",
        "nboots":       NBOOTS,
        "chunklen":     CHUNKLEN,
        "nchunks":      NCHUNKS,
        "trim":         TRIM,
        "ndelays":      NDELAYS,
        "use_corr":     USE_CORR,
        "vox_chunk":    args.vox_chunk,
        "corrs_shape":  list(corrs.shape),
        "note":         "GPU float64；Rresp 按需传输；alpha 循环分块；最终 corrs=sqrt(|R²|)·sign(R²)；方差 ddof=0",
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