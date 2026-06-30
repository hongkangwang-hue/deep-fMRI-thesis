# M2-C English1000 参照验证 —— 服务器执行计划

**纪律**：指标/种子/容差已在 `frozen/m2c_reference_validation.yaml` 预注册冻结。
执行重计算前需经用户确认（commit hash + 命令 + 资源 + 耗时）。Phase 1 不通过先
定位差异来源，不放宽容差。

参考产物：`results/eng1000/UTS03/{corrs,valphas}.npz`（corrs (95556,)，mean 0.041,
max 0.70）。**该目录是历史参考，任何重跑都不得写入此目录。**

---

## Phase 1：复现 LeBel native 路径，对齐 corrs.npz（硬闸门）

**目的**：证明数据加载 + 时间对齐 + 评分忠实于参考。用 encoding/ 原生函数与原生
设置（非冻结 spec），固定种子，输出到独立目录。

**脚本**：`scripts/m2c_phase1_native.py`（已写，import/argparse 本地校验通过）
**对照**：`scripts/m2c_compare.py`（已写，自对照 sanity 通过）

**命令（服务器）**：
```bash
# 1) 重跑（输出 results/eng1000_native_rerun/UTS03/，不碰参考）
python scripts/m2c_phase1_native.py --subject UTS03 --feature eng1000 --seed 20260629
# 2) 对照冻结指标判 PASS/FAIL
python scripts/m2c_compare.py \
    --ours results/eng1000_native_rerun/UTS03/corrs.npz \
    --ours-valphas results/eng1000_native_rerun/UTS03/valphas.npz
```

**验收（硬闸门，全 PASS 才算过）**：
| 指标 | 判据 |
|------|------|
| voxel_r_vector_pearson（我方 vs 参考 corrs，95556 体素相关） | ≥ 0.995 |
| median_abs_delta_r | ≤ 0.010 |
| p95_abs_delta_r | ≤ 0.050 |
| valpha_within_1_grid_step_frac | ≥ 0.85 |

差异来源（若 FAIL，按序排查，不放宽容差）：故事划分 → trim(5) → FIR 延迟(1..4)
→ eng1000 词向量 lowercase/bad_words → 响应 z-score 状态。

**资源/耗时（估）**：纯 CPU（bootstrap_ridge 为 numpy）。内存峰值受训练响应叠加
影响（~25 训练故事 × ~300TR × 95556 × 8B，叠加 + 延迟矩阵）约 6–12 GB；建议 ≥16GB
内存。不保存 wt（GB 级，避免 OOM）。预计 15–40 分钟。

---

## Phase 2：运行冻结 M2 spec，与 native 比较差异（描述性）

**目的**：用当前冻结 spec（PCA-100 + himalaya + 3 折故事级 CV + >100s/FIR mask +
fisher-z ROI）跑 eng1000，记录与 native 的差异来源，确认无泄漏、维度正确、ROI 量级合理。

**前置（尚未就绪，需先建）**：冻结-spec 编码管线（PCA→FIR→himalaya 嵌套 CV→评分→
ROI 汇总）。该管线与 M3 竖切基础设施重叠，将作为下一步代码任务实现；它复用已冻结的
`src/fmri/{alignment,mask}.py`、`frozen/voxel_mask_UTS03.npy`、`frozen/roi_columns_UTS03.npz`。

**前置检查（服务器）**：himalaya 是否已装（torch 后端，服务器有 torch 2.3.0）。

**比较（描述性，非 bit 对齐——CV 划分/PCA/solver/测试集均不同）**：
| 项 | 期望 |
|----|------|
| spatial_pattern_pearson_vs_native（体素 r 空间相关） | ≥ 0.80（低于则排查，非硬闸门） |
| roi_mean_r_sign（IFG/PT 的 fisher-z 平均 r） | 为正且量级合理 |
| leakage_checks（PCA/scaler 仅训练折拟合、测试故事不入训练、FIR 不跨故事） | 必须通过（硬性） |

**资源/耗时（估）**：himalaya 可用 4090 GPU；PCA 降到 100×4=400 维，3 折 × 13 alpha ×
2 内折。预计 5–15 分钟。

---

## 执行顺序与门控

1. **现在**：提交 M2-C 计划与脚本（Phase 1 可跑、Phase 2 待建管线）。
2. **确认后**：服务器 pull → 跑 Phase 1 → compare 判定。PASS 则进 Phase 2 建管线；
   FAIL 则定位差异、不放宽容差。
3. Phase 2 管线建好后再跑（GPU），输出描述性比较报告。

**重计算门控**：Phase 1（CPU bootstrap_ridge）与 Phase 2（GPU himalaya）均为重计算，
启动前须经用户确认。本文档与脚本不触发任何 ridge。
