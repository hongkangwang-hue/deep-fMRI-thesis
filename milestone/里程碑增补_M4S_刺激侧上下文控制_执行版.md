# 完成老师任务（刺激侧上下文控制 C1）需要修改的地方

> 设计与统计口径**沿用《里程碑增补老师修改建议上下文控制.md》(V6.4.2)**，本文件不重复。
> 这里只列：为把这个实验做出来、并落进论文，**实际要改/加哪些东西**。

---

## A. 先修 V6.4.2 规范里三处与真实情况不符（不改会执行错）

| # | 规范版写的 | 真实情况（已核对） | 改成 |
|---|---|---|---|
| 1 | 论文用 Figure 1–14、Table 1–3；新图=Fig 12、新表=Table 4、旧图顺延 | 论文实际 **图 1–16、表 1–8**（`…V2全图补充版.docx` 实测）；time-shift 是 **§4.6** | 新表 **Table 9**、新图 **Figure 17**（追加不顺延）；§4.6 见 C |
| 2 | 现有条件叫 `shift40` | 代码真名 `normal` / `shift`（[m4_driver.py:183](../src/ridge/m4_driver.py#L183)、[estimands.py:96](../src/stats/estimands.py#L96)） | 新条件命名 `ctx1`，现有按 `shift` 引用 |
| 3 | eligible_h128 = 146,135 及其 hash | 146,135 是 83 CV 故事打分数；冻结集合**及 hash**是 84 故事 = 147,846 | 打分 146,135 ⊂ 冻结 147,846 |

> 其余数字（§7.3E 的 0.0006–0.0069 / +0.0014~+0.0044、§2.4 预期量级表、28/28/27、主层 9-9-17-2）
> 均已核对**与真实数据吻合**，不用动。

---

## B. 代码改动

**一个前提**：现有 `shift` 是对已提取 X 做内存位移（[m4_driver.py:131](../src/ridge/m4_driver.py#L131)，便宜）；
**C1 打乱的是喂给模型的词，必须重新前向**（贵）。所以 C1 走"重提取特征 → 独立拟合"，不能套 shift 的省钱路径。

| 文件 | 加/改 | 内容 | 本地可测 | 需算力 |
|---|---|---|---|---|
| `src/models/context_perturb.py` | **加** | `story_permutation(story_id, master_seed, n)`（sha256 稳定派生，禁 Python `hash()`）；`build_perturbed_window(words, perm, i, H)` = 乱序上下文 `s[i-H:i]` + 真实目标 `w_i` | ✅ | ❌ |
| `tests/test_context_perturb.py` | **加** | 置换可复现、相邻窗 H−1 重叠、目标词/末元素未改 | ✅ | ❌ |
| `scripts/m4s_extract_perturbed_features.py` | **加** | 复制 `m1_extract_features.py`，仅把 [base.py:155](../src/models/base.py#L155) 的 `build_window` 换成 `build_perturbed_window`，写独立缓存 `cache/features_ctx1/`（`meta` 记 permutation_sha256，**不碰 normal 缓存**） | 断言可本地 | ✅ GPU |
| `scripts/m4s_assert_impl.py` | **加** | V6.4.2 §2.1 A–F 硬断言 + §2.4 origin/form 诊断（只依赖置换索引+词表）——**跑特征前必须先过** | ✅ | ❌ |
| `scripts/m4s_ridge_perturbed.py` | **加** | assemble 指向 `cache/features_ctx1/` → 走 `run_fold` **normal 分支**（非 do_shift）→ `ctx1_*.json`；对每 story 断言 `mask_ctx1 == mask_normal`（复用 [m4_driver.py:176](../src/ridge/m4_driver.py#L176)） | — | ✅ Ridge |
| `src/stats/estimands.py` | **改** | 仿现有 `shift` 块加 `ctx1` estimands、`D_m = Δr_total^normal − Δr_total^ctx1`、`I_MP = D_Mamba − D_Pythia` | ✅ | ❌ |
| 现有 M5 bootstrap 入口 | **改** | condition 列表加 `ctx1`（normal/ctx1 共享 bootstrap indices），输出 `D_m`/`I_MP` + CI | — | ✅ 轻 |
| 现有 M6 出图脚本 | **改** | 出新控制图/表（见 C） | ✅ | ❌ |

> 缓存不改 `feature_cache.py`：`cache_dir` 与 `meta` 都是入参，换目录 + 补 meta 即可。
> ridge 不需新拟合逻辑：`run_fold` 本就每折独立 fit scaler/PCA/λ/Ridge，指向 ctx1 缓存即自动"独立拟合"。

---

## C. 论文改动（按真实 8 表 / 16 图）

- **结果章节**：现有 §4.6=时间平移、§4.7=最终层、§4.8=小结。改为 §4.6 拆「控制分析」下设
  **4.6.1 时间平移 / 4.6.2 刺激侧上下文控制**，§4.7/§4.8 编号不变（改动最小）。
- **新增控制表 = Table 9**（追加在 Table 8 后）；**新增控制图 = Figure 17**（追加在 Figure 16 后）。
  正文用 in-text 指针引用，避免重排 16 图。
- **星号一致性**：论文现有诊断性负控制/时移分面板图注已用星号 → 二选一并冻结：(A) 回溯统一去星号；
  (B) 保留并统一加注"星号仅表示 95% CI 不含零，非校正后显著"。**不得新旧两套并存无说明。**
- Methods/Results/Discussion/Limitations/Future Work 按 V6.4.2 §8.5–§8.10 增补；全文用
  "result-blind diagnostic amendment"，不用"预注册"。

---

## D. 执行分两批

1. **本地先做（零算力，是跑特征前的硬闸门）**：`context_perturb.py` + 单测 + `m4s_assert_impl.py`
   （§2.1 A–F 全过 + §2.4 与预期量级表对照），全过才允许进特征提取。
2. **服务器需你确认后跑（真实算力）**：重提取 ctx1 特征 → 54 单元 Ridge → M5 补 `D_m`/`I_MP` → M6 出图。
   算力紧则按 V6.4.2 §9 的 **L2**（仅 Mamba+Pythia、H{8,128}、三被试）——已足以回答主问题。
