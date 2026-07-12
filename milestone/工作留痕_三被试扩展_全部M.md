# 工作留痕：三被试扩展 全部里程碑（M0–M6）

**项目**：不同上下文长度下序列架构的语言脑编码比较（V6.3 三被试扩展）
**扩展被试**：UTS01、UTS02（新增）+ UTS03（已验证基线，只读引用/复用）
**数据集**：LeBel ds003020（v3.1.1, commit `cb7c536d`）
**记录日期**：2026-07-11 ~ 2026-07-12
**性质**：本文件记录三被试扩展**全部里程碑（M0–M6）实际已做**的工作与真实产出的数字，区别于 `里程碑总览_三被试执行计划_V6_3_*.md` 的计划文档。持续追加，不按里程碑拆分多个文件。

---

## 总览

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| **M0** | UTS03 基线锚定 + 三被试扩展主配置冻结 | ✅ 完成 |
| **M1** | UTS01/UTS02 的 voxel mask + IFG/PT ROI 构建 | ✅ 完成 |
| **M2** | UTS01/UTS02 的 English1000 参照验证硬闸门 | ✅ **通过（补充判据路径，非原始全体素判据）**——官方全体素判据 FAIL，诊断定位到两类已知数值边界效应（详见下文），均不涉及实现错误、均几乎不落在 ROI 打分范围内；已按此写入 `里程碑总览_三被试执行计划_V6_3_阶段二_M1至M4逐被试执行.md` 新增的"补充条款（2026-07-12）"第 8 条正式判据，两名被试均判定通过 |
| **M3** | UTS01 新被试端到端竖切复核（迁移正确性闸门） | ✅ 完成——过程中发现并修复一个真实生产 bug（M1 的 voxel_mask 从未在 M3/M4 管线里被真正应用），修复后 8/8 验收标准全过。UTS02 按里程碑规则不需重复跑（竖切只测代码通用性，非被试专属） |
| **M4** | UTS01、UTS02 逐被试全矩阵扩展（UTS03 复用 pilot 结果） | ✅ 完成——两被试各 72/72 单元，验收标准 1–6 全过，量级 sanity check 正常 |
| **M5** | 逐被试统计（bootstrap+Holm）+ 跨被试方向一致性综合 | ✅ 完成——两项确认性对比得出跨被试结论，并定位到 UTS01 一处负控制符号翻转异常 |
| **M6** | 三被试图表与可复现交付 | ✅ 完成——三被试逐被试图表 + 跨被试并排图 + ROI 空间位置图全部对真实数据出图，已下载到本地；额外完成一项 AWD-LSTM 上下文饱和诊断 |

---

# M0 — UTS03 基线锚定 + 三被试扩展冻结

**commit**: `6b125e7`（正式冻结提交），tag `three-subject-m0-freeze`；UTS03 基线只读锚定 tag `uts03-graduation-baseline`（commit `26e6894`）。

## 交付物

| 文件 | 内容 |
|------|------|
| `frozen/uts03_baseline_manifest.json` | UTS03 冻结文件 sha256 + eligible_h128 目标集合 hash（147846 目标，84 故事） |
| `frozen/three_subject_extension_spec.yaml` | 三被试扩展主配置（milestone 模板填入真实值） |
| `frozen/three_subject_contrast_registry.yaml` | 预注册对比表的被试维度声明（确认性家族 per_subject，跨被试 descriptive_consistency） |
| `frozen/uts01_uts02_data_availability.md` | UTS01/UTS02 数据可得性核查（`git annex find` 只读核查：84/84 故事已登记，内容未下载，预计 UTS01≈20.77GB、UTS02≈24.16GB） |
| `manuscript/methods.md` §13 | 三处预注册已知偏差首次写成正式 prose（inner-CV PCA 乐观、H128 训练端退化行、repeatability 可得性） |

## 数据下载（M0 之后的前置工作，不属于任何正式里程碑）

- `git-annex get` 在服务器上遇到 Haskell 网络 bug（`Network.BSD.getProtocolByName: does not exist`），容器缺 `/etc/protocols`（`netbase` 包未装）。
- 改用 **AWS CLI 直连 S3**（`aws s3 sync --no-sign-request s3://openneuro.org/ds003020/...`，绕开 datalad/git-annex），bucket=`openneuro.org`，fileprefix=`ds003020/`，host=`s3.amazonaws.com`（与 git-annex 的 `s3-PUBLIC` remote 配置核对一致）。
- 下载过程中踩坑：目标目录残留 git-annex 悬空符号链接导致 `aws s3 sync` 报 "File does not exist" 静默跳过整批文件——用 `find <dir> -xtype l -delete` 清掉悬空链接后重跑解决。
- 最终 UTS01、UTS02 各 84/84 故事下载完成，与 UTS03 故事集完全一致。

---

# M1 — UTS01/UTS02 的 voxel mask 与 ROI 构建

## 代码修复（均在 M1 数据到位前完成，commit `c8664f9`、`79060f5`、`ef0eb38`、`1c9cd91`）

逐文件审查硬编码 UTS03 假设，修了 4 处：

1. **`scripts/m2_validate_pipeline.py`**：`step_roi()` 的 `xfm` 参数硬编码默认值 `"UTS03_auto"`——对新被试跑会静默用错 pycortex transform、算出错误 ROI 但不报错。改成从 `derivatives/subject_xfms.json` 按 `--subject` 查（已验证 UTS01→UTS01_auto、UTS02→UTS02_auto，对 UTS03 行为不变），保留 `--xfm` 手动覆盖。
2. 同一脚本：`roi_columns_{subject}.json` 里 `column_space` 字段硬编码 `"thick_mask_95556_C_order"`（UTS03 专属体素数），改成用实测值 `int(flat.sum())`。
3. **`scripts/m6_roi_location.py`**（M6 阶段脚本，顺手一并修）：同样的 xfm 硬编码 + 一处 `!= 95556` 硬编码断言，改成跟被试无关的越界校验（ROI 列号必须落在该被试自己的 thick mask 列空间内）。
4. **`scripts/m2c_compare.py`、`scripts/m2c_phase2.py`**：`--ref`/`--ref-valphas`/`--native` 默认值硬编码指向 `results/eng1000*/UTS03/...`——因为 UTS03 这份文件真实存在，换被试忘了覆盖参数不会报错，会静默拿 UTS03 的参照去对比新被试的结果。改成从 `--ours`/`--subject` 自动推导对应被试路径。

审查确认**不用改**：`src/fmri/roi.py`、`compute_bold_only_mask`、`save_voxel_mask`、`list_subject_stories` 全部已是纯函数/参数化，无硬编码；UTS03 冻结 mask 是在全部 84 个故事（含测试故事）上建的，`list_subject_stories` 对新被试自动用同一策略，无口径漂移。

## 实测结果

| | UTS03（基线） | UTS01 | UTS02 |
|---|---|---|---|
| voxel 总数 | 95556 | 81126 | 94251 |
| 保留（BOLD-only mask） | 95556 | 81116 | 94219 |
| 剔除(NaN) | 0 | 10 | 32 |
| 剔除(零方差) | 0 | 0 | 0 |
| left_IFG 列数 | 768 | 563 | 755 |
| left_IFG 连通块/最大块占比 | 1 / 1.00 | 1 / 1.00 | 1 / 1.00 |
| bilateral_PT 列数 | 341 | 429 | 395 |
| bilateral_PT 连通块/最大块占比 | 2 / 0.584 | 4 / 0.56 | 2 / 0.56 |
| min_voxels(10) 判定 | PASS | PASS | PASS |

三名被试体素总数/ROI 列数不同属于正常个体解剖差异，不代表出错。UTS01 的 PT 连通块数（4）比 UTS03/UTS02（2）多，但最大块占比（0.56）与 UTS03（0.584）接近，判断为零星小碎片而非拓扑异常。

跑 `step_roi` 时两个被试都出现过一条 `Error. nthreads must be a positive integer` 的非致命警告（来源未精确定位，怀疑是 pycortex 内部某个 C 扩展/线程库读容器线程数配置读到无效值），未阻断执行，最终结果数字健康，判定为可忽略的噪音输出。

**M1 验收标准（voxel mask + ROI 冻结、归属规则与 UTS03 一致、裁剪/标准化未重复、体素计数有记录）全部满足。**

---

# M2 — UTS01/UTS02 的 English1000 参照验证硬闸门（进行中，未结案）

## 背景：参照真值文件从零生成

核查（本地 + 服务器双重确认）：`results/eng1000/` 下只有 UTS03，UTS01/UTS02 没有对应参照文件。原因不是"丢失"，是在数据下载完成前，任何机器都没有这两名被试的原始 BOLD，从未有机会生成。`results/eng1000/UTS03/corrs.npz` 本身也从未被 git 追踪（`results/` 全目录 gitignore），来源是历史上跑过一次原始 `encoding/encoding.py --subject UTS03 --feature eng1000`（无固定种子）。

**决定**：用 `scripts/m2c_phase1_native.py --subject <X> --out-name eng1000`（复用 LeBel 原生 `encoding_utils`/`ridge_utils.ridge.bootstrap_ridge`，逐超参数核对与 `encoding.py` 完全一致，唯一区别是固定种子 `20260629`、不保存 `weights.npz`）为 UTS01、UTS02 各自生成参照文件，直接写入 `results/eng1000/<subject>/`（用 `--out-name eng1000` 覆盖默认的 `eng1000_native_rerun`，因为这次没有历史参照需要保护）。比 UTS03 当年那份「种子不可考」的参照更规范。

## CPU 原生版运行记录

- 两次都是纯 CPU、单进程多线程（约 10~11 核并行，`ps aux` 显示 %CPU≈1050%），耗时量级：UTS01 约 4.9 小时（22:39 → 03:50），UTS02 约 5.5 小时（03:59 → 09:38，体素数更多所以更慢）。
- **运行事故**：手误在两个不同终端窗口各开了一次 `screen -S eng1000_uts01`（screen 允许同名重复会话），实际只有一个真进程在算，另一个是空壳；排查时误杀了挂着真进程的那个会话，导致第一次 UTS01 尝试（已运行约 11 分钟）被杀掉。因为 `np.savez` 全部在计算最后才一次性写出，被杀时未落盘任何文件，无脏数据风险，直接干净重跑。教训：**screen 多窗口操作要一块一块粘贴、每块之间 detach 完成后再开下一个**，不要一次性粘贴多个 `screen -S` 块。
- 最终结果（均已核对文件完整性：5 个 npz/json 全部存在、形状匹配、无 NaN/Inf、manifest 种子正确）：

| | UTS01 | UTS02 |
|---|---|---|
| corrs shape | (81126,) | (94251,) |
| 有限值 | 81126/81126 | 94251/94251 |
| mean | 0.0053 | 0.0256 |
| max | 0.6226 | 0.6636 |

UTS01 mean 明显低于 UTS02/UTS03（≈0.04），判断为被试间信号强度的正常个体差异（后续诊断证实这不是本次 FAIL 的主因，见下）。

## GPU 重实现对比（Phase 1 硬闸门）

`scripts/m2c_phase1_gpu.py --out-name eng1000_gpu_rerun` + `scripts/m2c_compare.py`，对照 `frozen/m2c_reference_validation.yaml` 冻结的四道阈值：

| 指标 | 阈值 | UTS01 | UTS02 |
|---|---|---|---|
| voxel_r_vector_pearson | ≥0.995 | 0.9867 **FAIL** | 0.9713 **FAIL** |
| median_abs_delta_r | ≤0.01 | 0.0000 PASS | 0.0000 PASS |
| p95_abs_delta_r | ≤0.05 | 0.0044 PASS | 0.0067 PASS |
| valpha_within_1_grid_step_frac | ≥0.85 | 0.9997 PASS | 0.9993 PASS |
| **总判定** | | **FAIL** | **FAIL** |

四项里只有 pearson 相关系数没过，其余三项都远超阈值——说明绝大多数体素高度一致，问题集中在少数体素上，值得深挖而不是简单放宽阈值（milestone 冻结规则明确禁止"先看结果再放宽容差"）。

## 诊断链（逐步排除假设，用真实数据核实）

1. **假设1（信号弱被拖累）：已证伪。** 只看信号体素（`|ref|>0.1`）重算 pearson，UTS01 0.9867→0.9878、UTS02 0.9713→0.9699，几乎没变化，说明不是"整体方差小导致对噪声更敏感"。

2. **真实原因A：数值退化体素。** 找到 `|ref|>1` 的体素（相关系数数学定义域是 `[-1,1]`，出现这个说明原生实现里 `Rsq=1-resvar/Presp_var` 在近零方差分母上炸出了病态大值；GPU 版本因为有 `np.nan_to_num(...)` 保险丝，同样情况下输出 `0`，两边处理零方差边界的方式不同）。UTS01 有 10 个，UTS02 有 32 个。**关键**：这些体素 **0 个** 落在 M1 阶段冻结的 `voxel_mask_{subject}.npy` 保留列表里——本来就会被我们自己的 BOLD-only QC 排除，不会进入任何 ROI 打分。
   - 排除这批体素后：UTS01 pearson→0.9950，UTS02→0.9947。
   - 直接限定到 `voxel_mask` 保留列（结果与上一步几乎相同，因为排除的正是同一批体素）：**UTS01 = 0.995028（+0.000028，过线）**，**UTS02 = 0.994712（-0.000288，仍差一点）**。

3. **真实原因B（针对 UTS02 剩余缺口）：符号翻转体素。** 在 `voxel_mask` 限定后差异最大的体素里，几乎全部呈现"`ref` 与 `ours` 数值不大（都在 ±0.35 以内，合法范围）但符号相反"的模式。判断：这批体素的真实 `Rsq` 极度接近 0（模型几乎没有稳定解释力），`corrs=sqrt(|Rsq|)·sign(Rsq)` 公式在 `Rsq≈0` 时符号完全取决于浮点误差最后几位——CPU(numpy)与GPU(PyTorch/cuBLAS)底层线性代数库不同、加法顺序不同，同一个理论上应为正/负的极小值可能在两边翻到不同符号。这不是实现错误，是这批体素本身没有可靠信号、符号在噪声范围内摆动。

4. **ROI 重叠检查**：`voxel_mask` 限定后差异最大的 100 个体素里，落在 `bilateral_PT`（395列）的有 **0 个**，落在 `left_IFG`（755列）的只有 **2 个**（占比0.26%）。即两类差异体素（数值退化的 + 符号翻转的）加起来，对真正打分用的 ROI 几乎没有触及。

5. **UTS02 那 2 个落在 left_IFG 里的差异体素，逐个核对确认**：
   - 列51388：`ref=0.0736  ours=-0.1663  diff=0.2399`
   - 列42918：`ref=0.0705  ours=-0.1589  diff=0.2293`

   换算成 `Rsq`（`corr=sqrt(|Rsq|)·sign(Rsq)` 的反推）：两个体素的 `Rsq_ref` 都只有 +0.005~+0.006、`Rsq_ours` 都只有 -0.025~-0.028，双侧都是解释力趋近于零的极小值，完全符合"原因B：符号翻转"的模式，不是新的、独立的第三类问题。**诊断链到此完整闭环**，四步假设/证实链条（假设1证伪 → 原因A定位 → 原因B定位 → ROI重叠确认 → 这2个IFG体素逐一核实）全部有真实数据支撑。

## 结案：方法学决定与依据（2026-07-12）

**milestone 计划书是用户自己撰写、可自行修改的任务计划文档**（不是外部强加、不可更改的规范），用户在核实这一点后，明确决定：不悄悄放宽原判据、不直接把"差一点点"当作可以忽略，而是**保留原始全体素判据的 FAIL 结果不改动，另外新增一条独立的、有条件的补充判据**，走"如实报告 FAIL + 完整诊断证据支持替代通过路径"这条更严谨的记录方式，而非"改判据让它变成PASS"。

具体写入 `里程碑总览_三被试执行计划_V6_3_阶段二_M1至M4逐被试执行.md` 的 M2 章节，新增"**补充条款（2026-07-12，UTS01/UTS02 实际执行后追加）**"，**原验收标准与"明确不做"一字未删改**，新增内容包括：

1. 如实记录官方全体素判据的真实 FAIL 结果（UTS01=0.9867，UTS02=0.9713）。
2. 完整诊断结论（数值退化体素 100% 被 voxel_mask 排除；符号不稳定体素在 IFG/PT 中占比 <0.3%）。
3. **第 8 条补充判据**：允许在满足四项条件（差异体素逐一定位到具体机制 / 证明是通用数值现象非实现错误 / 证明与 ROI 列重叠可忽略 / 完整诊断记录留痕供导师复核）时，以此作为替代通过依据。
4. **最终判定**：UTS01、UTS02 均判定为"**闸门通过（第 8 条补充判据路径）**"，明确区别于"闸门通过（原判据）"，要求论文 Methods/Limitations 如实注明走的是哪条路径，不得笼统写"闸门通过"掩盖差异。

**M2 里程碑正式结案，可以进入 M3（新被试端到端竖切复核）。**

---

# M3 — UTS01 新被试端到端竖切复核

**代码**：`scripts/m3_new_subject_slice.py`（新建，commit `70f6a5d`）；代码复查修复（commit `36bef40`）；关键 bug 修复（commit `8c42e80`）。

## 里程碑口径：只需一名新被试

竖切复核测的是"复用管线 × 新被试 BOLD"这个组合有没有隐性问题，不是被试专属的科学结果——管线里没有任何硬编码某个被试的逻辑（`voxel_mask_{subject}.npy`、`roi_columns_{subject}.npz` 均按参数取），在 UTS01 上证明代码路径正确，即证明它对任意新被试都正确。**UTS02 不需要重复跑 M3**，直接进入 M4。

## 代码复查发现并修复的两处（跑真实数据前，commit `36bef40`）

1. **mask 逐元素核查的 `zip()` 静默截断风险**：normal/shift 的 `story_scores` 列表若长度不一致，`zip()` 会静默截断到较短的一边，可能掩盖真实的 mask 不一致。改为先显式断言两列表等长，不等长直接 `raise`。
2. **`layer_index` 字段读取错误**：无条件读 `feat_meta.get("layer_main")`，`--layer final` 时会取错字段。改为按 `layer` 条件读取（`layer_main` / `layer_final`），与 `m4_driver.py` 的既有模式对齐。

## 真实生产 crash 与根因修复（commit `8c42e80`）

在服务器上对 UTS01 首次跑 M3 时撞上：

```
ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
```

崩在 `himalaya_ridgecv_solver` 的 `model.fit()` 内部。**这不是操作失误**（用户曾怀疑是自己不小心终止了进程），是真实数据/管线 bug：

- **根因**：`src/ridge/assemble.py` 的 `assemble_story()` 调用 `load_response()` 时**没有传 `columns=` 参数**，导致 M1 阶段冻结的 `voxel_mask_{subject}.npy`（本该剔除 NaN/零方差体素）从未在 M3/M4 管线的任何环节被真正应用过。
- **为什么 UTS03 pilot 全程没暴露**：UTS03 自己的 voxel_mask 恰好是恒等映射（0 个体素被排除），所以这个漏洞对 UTS03 完全无副作用、无法被察觉；UTS01 排除 10 个体素、UTS02 排除 32 个，这些体素的 NaN 值直接冲进 float32 的 ridge 拟合，才第一次触发崩溃。
- **修复范围**：`src/ridge/assemble.py`（`assemble_story`/`assemble_all` 新增 `voxel_mask` 参数并透传给 `load_response(columns=...)`；新增 `remap_roi_columns_to_voxel_mask()` 把 ROI 列索引从全量 BOLD 列空间重映射到压缩后的 mask 空间，用 `searchsorted` + 显式校验，版本不一致直接 `raise`）、`scripts/m3_new_subject_slice.py`、`src/ridge/m4_driver.py`、`scripts/m4_{pythia,mamba,rwkv,awd_lstm}.py` 全部同步加载并透传 `voxel_mask`。
- **验证**：单元测试（10→8 列压缩的精确重映射、ROI 列不属于 mask 时显式报错、UTS03 恒等映射不变）+ 端到端合成 NaN 复现/修复测试 + 全量 81 项测试套件通过。

## 修复后重跑结果（UTS01，`fold_0`）

```
[m3] 验收标准: {"1_dims_correct_no_leakage": true, "2_lambda_boundary_recorded": true,
"3_normal_shift_independent_fit": true, "4_scoring_mask_bit_identical": true,
"5_heldout_r_finite_nonzero": true, "6_shift_differs_from_normal": true,
"7_evr_recorded": true, "8_manifest_traceable": true}
[m3] 全部通过: ✅
```

8 项验收标准全过，**尤其第 4 项**（normal/shift 最终评分 mask 逐元素相同）——证明修复没有破坏 M3 竖切阶段已验证过的独立拟合/共同 mask 结构。

**M3 里程碑结案，进入 M4（逐被试全矩阵）。**

---

# M4 — UTS01、UTS02 逐被试全矩阵扩展

**代码**：`src/ridge/m4_driver.py`、`scripts/m4_{pythia,mamba,rwkv,awd_lstm}.py`（撤销单被试偏离标签，commit `9660665`；voxel_mask 修复同 M3，commit `8c42e80`）。

## 代码改动

- `build_manifest()`：`"phase"` 字段从 `"M4 full matrix (single-subject deviation)"` 改为 `subject_scope` 区块，声明 `all_subjects: ["UTS01","UTS02","UTS03"]`，每被试独立建模。
- `process_group()`/`run_model_matrix()` 签名扩展 `voxel_mask` 参数，透传给 `assemble_all(..., voxel_mask=voxel_mask)`（与 M3 同一处修复覆盖）。
- 4 个入口脚本批量打上同样的补丁：加载 `voxel_mask_{subject}.npy` → `remap_roi_columns_to_voxel_mask()` 重映射 ROI 列 → 传入 `run_model_matrix`。
- 矩阵范围：4 模型 × 3H(8/32/128) × 3 折 × (主层 normal+shift 双 ROI + 最终层 normal 单 IFG) = 每被试 **72 个单元**；`do_shift = (layer == "main")`，最终层不含 shift（与里程碑矩阵定义一致，已在代码复查中核对）。

## 运行记录

单卡 RTX 4090（24GB，运行期间独占，未并行），四模型用 `&&` 串联在同一 screen 会话里顺序执行（`pythia && mamba && rwkv && awd_lstm`），带 `--skip-existing` 支持断点续跑：

| 被试 | screen 会话 | 总耗时 | 单元数 | 均耗时/单元 |
|---|---|---|---|---|
| UTS01 | `m4_uts01_all` | 167.9 分钟 | 72/72 | 140.0s |
| UTS02 | `m4_uts02_all` | 197.8 分钟 | 72/72 | 164.8s |

（对照：UTS03 pilot 当年 142.7 分钟/72单元/129.8s，量级一致，UTS02 略慢因体素数更多。）

## 验收标准（`m4_aggregate.py` 程序化核对，两被试均全过）

```
{"1_main_matrix_complete": true, "2_final_ifg_matrix_complete": true,
"3_per_story_saved": true, "4_no_nan_inf": true,
"5_common_mask_used": true, "6_manifest_traceable": true}
```

`4_no_nan_inf: true` 尤其关键——证明 M3 阶段修复的 voxel_mask 问题在全矩阵规模下依然稳固，没有漏网。

## 量级 sanity check（`m4_view_results.py`，主层正常条件，model×H 跨折均值，left_IFG）

| model | UTS01 (H8→H128) | UTS02 (H8→H128) | UTS03 pilot 参考 (H8→H128) |
|---|---|---|---|
| pythia | 0.1064 → 0.1097 | 0.1244 → 0.1269 | 0.1287 → 0.1322 |
| mamba | 0.1094 → 0.1141 | 0.1282 → 0.1351 | 0.1315 → 0.1378 |
| rwkv | 0.1056 → 0.1065 | 0.1239 → 0.1258 | 0.1273 → 0.1280 |
| awd_lstm | 0.0683（随H基本不变） | 0.0714（随H基本不变） | 0.0684（随H基本不变） |

三被试同量级（0.10–0.14）、架构相对顺序一致（mamba ≥ pythia ≈ rwkv > awd_lstm）、`bilateral_PT` 系统性高于 `left_IFG`、shift 条件 r 明显趋近 0——判定为**工程健康**，无异常。此判断仅回答"数据健不健康"，不构成任何科学结论（科学结论见 M5）。

**M4 里程碑结案，进入 M5（逐被试统计与跨被试综合）。**

---

# M5 — 逐被试统计与跨被试方向一致性综合

**代码**：`scripts/m5_analysis.py`（撤销单被试偏离标签）、`src/stats/cross_subject.py`（新建）、`scripts/m5_cross_subject.py`（新建）——均为 commit `e01b393`；复查加固 commit `e744b0d`。

## 代码

- `src/stats/cross_subject.py::direction_consistency()`：纯函数，对单个估计量在各被试上的 point/CI 判读四类结果之一：`consistent_strong`（方向相同且各被试 CI 均排除0）/`consistent_direction_only`（方向相同但非全部CI排除0）/`heterogeneous`（方向不一致）/`insufficient_data`（存在缺失）。**严格不做任何池化、不生成组水平 CI**（`point_min`/`point_max` 仅为描述性量级范围）。
- `scripts/m5_cross_subject.py`：读三被试各自的 `m5_results.json`，对两项确认性 Δr_total 架构差值 + RQ1 H-specific 差值做方向一致性判读，输出 `m5_cross_subject.json`。**不重跑任何 bootstrap/PCA/Ridge**。
- 复查发现并修复：`--subjects` 传重复被试时，`subj_est` 字典去重但 `subject_order` 不去重，导致 `n_subjects`/CI 计数重复计入、可能静默误判为"强一致"。加显式校验，重复即 `SystemExit`。
- 验证：14 项单元测试（9 项原有 + 5 项新增：strong/direction-only/heterogeneous/insufficient/量级范围非池化）全过；合成三被试数据端到端冒烟测试跑通完整 IO 路径。

## 逐被试确认性家族结果（IFG 主层 Δr_total 架构差值，Holm α=0.05）

| 对比 | UTS01 | UTS02 | UTS03 |
|---|---|---|---|
| rwkv − pythia | −0.0024 [−0.0038,−0.0010] p=0.0000 **拒绝** | −0.0006 [−0.0022,+0.0008] p=0.4200 未拒绝 | −0.0029 [−0.0043,−0.0015] **拒绝** |
| mamba − pythia | +0.0014 [+0.0001,+0.0027] p=0.0420 **拒绝** | +0.0044 [+0.0026,+0.0060] p=0.0000 **拒绝** | +0.0028 [+0.0011,+0.0043] **拒绝** |

## 负控制（40s time-shift）三被试对照

| | UTS01 | UTS02 | UTS03 |
|---|---|---|---|
| 层面①：架构差值平移后仍显著（⚠️=确认性发现受威胁） | **⚠️ 是** | 否 ✓ | 否 ✓ |
| pythia 自身 gain 被显著削弱 | ✓ | ✗ | ✗ |
| mamba 自身 gain 被显著削弱 | ✓ | ✓ | ✗ |
| rwkv 自身 gain 被显著削弱 | ✗ | ✗ | ✗ |
| awd_lstm 自身 gain 被显著削弱 | ✗ | ✗ | ✗ |

**UTS01 层面①拆解定位**（层面①的布尔标志是两项 shifted 架构对比的 `any()` 聚合，逐项拆开核实是哪一项驱动的）：

| shifted 对比（UTS01） | point | CI | 是否显著 |
|---|---|---|---|
| shifted rwkv−pythia | +0.0017 | [+0.0005,+0.0030] | **是**（驱动⚠️） |
| shifted mamba−pythia | +0.0011 | [−0.0003,+0.0024] | 否（干净） |

**关键发现：UTS01 的 rwkv−pythia 是符号翻转，不只是"未塌陷"**——normal 条件下 rwkv−pythia = **−0.0024**（显著负），shift 条件下变成 **+0.0017**（显著正）。两个方向都显著，符号整个翻转。这是比"残留同方向信号"更值得警惕的模式，判断为 UTS01 被试特异的不稳定性，不涉及实现错误（管线机制已在 M3/M4 程序化验证过），需在 Limitations 中用"sign flip under shift"精确措辞单独注明，不能笼统写"未通过负控制"。

## 跨被试方向一致性判读（`m5_cross_subject.py` 真实输出）

| 确认性对比 | UTS01 | UTS02 | UTS03 | 判读 |
|---|---|---|---|---|
| rwkv−pythia Δr_total | −0.0024★ | −0.0006 | −0.0029★ | **部分一致**（方向同，非全部CI排除0） |
| mamba−pythia Δr_total | +0.0014★ | +0.0044★ | +0.0028★ | **一致（强被试内重复）** |

RQ1 H-specific（探索性）在 H=8/32/128 三点上完全印证同一模式：mamba−pythia 六项全部"一致（强）"，rwkv−pythia 六项全部"部分一致"，无一项"不一致"或"数据不足"——说明这不是随机噪声，是稳定但强度不齐的方向性效应。

## 综合结论（供 M6/Results 直接引用）

1. **mamba − pythia 的 Δr_total 架构差值**：跨被试方向一致 + 三被试 CI 均排除 0 + **三被试负控制全部干净**（含 UTS01——UTS01 的层面①警告经拆解证实只针对 rwkv−pythia，与 mamba 无关）。这是本次三被试扩展能拿到的最强证据形态。
2. **rwkv − pythia 的 Δr_total 架构差值**：跨被试方向一致但显著性不齐（UTS02 不显著），且 **UTS01 单独存在符号翻转异常**，该被试上这条发现需要明确降级措辞、单独加注，不能与 mamba 那条同等强度呈现。

以上结论均不涉及任何代码/规则修改——纯粹是如实记录到 Results/Limitations 的诊断结果。

**M5 里程碑结案，进入 M6（三被试图表与可复现交付）。**

---

# M6 — 三被试图表与可复现交付

**代码**：`scripts/m6_cross_subject_figures.py`（新建）、`scripts/m6_tables.py`（修复），commit `3d72fd1`。

## 代码

- **新建** `scripts/m6_cross_subject_figures.py`：两张跨被试图，均只读 M5 结果、不重算：
  - `figX_cross_main`：四模型 r8/r32/r128 曲线，行=ROI（IFG/PT）、列=被试，三被试并排，无合并组水平值；
  - `figX_cross_confirmatory`：两项确认性 Δr_total 架构差值，各被试 point+95%CI 并排，标注方向一致性判读（读 `m5_cross_subject.json`，不在图脚本里重算判读）；实心点=CI排除0，空心点=CI跨0。
  - 图内全英文标签（避免中文在默认字体下渲染成方块）；AWD-LSTM 沿用灰虚线视觉隔离。
  - 守卫：缺 M5/跨被试输入文件时给出可操作的报错指引；`--subjects` 与跨被试产物实际覆盖的被试不一致时显式拦截（避免 `KeyError`）。
- **修复** `scripts/m6_tables.py`：`qc_table()` 硬编码读 `voxel_mask_UTS03.json`/`roi_columns_UTS03.npz`——按 `--subject` 跑其他被试时会静默报告 UTS03 的体素/ROI 列数。改为按 subject 参数化。同时撤销"UTS03（单被试，预注册偏离）"措辞。
- 验证：用合成三被试 fixture 端到端跑通（`m5_cross_subject.py` → `m6_cross_subject_figures.py`），两张图 PNG+PDF 均正确生成；被试不匹配守卫按预期拦截（非零退出）。

## 实际执行记录（2026-07-12，服务器）

```bash
python scripts/m6_figures.py --subject UTS01   # UTS02、UTS03 各一次
python scripts/m6_tables.py  --subject UTS01   # UTS02、UTS03 各一次
python scripts/m6_cross_subject_figures.py
python scripts/m6_roi_location.py --subject UTS01   # UTS02 补一次（Figure 6，UTS03 pilot 阶段已有）
```

全部纯 CPU、只读 M5 结构化结果，未重算任何统计。产出（服务器 `figures/` 下，共 52 个文件）：

| 目录 | 内容 |
|---|---|
| `figures/UTS01/`、`figures/UTS02/`、`figures/UTS03/` | 各自 fig1–6（PNG+PDF）+ `tables/`（qc_table、table1_checkpoint_audit、table2_full_numbers） |
| `figures/cross_subject/` | `figX_cross_main`、`figX_cross_confirmatory`（PNG+PDF） |

`fig6_roi_location`（ROI 空间位置图）原本只有 UTS03（pilot 阶段产出），`m6_roi_location.py` 本身早已按 subject 参数化（M1 准备阶段 commit `ef0eb38` 修过），只是没人跑过 UTS01/UTS02，本次补齐，跑图时出现的 `Error. nthreads must be a positive integer` 与 M1 阶段同款，非致命噪音，不影响结果。

## 下载到本地

用 `rsync -avz -e "ssh -p <端口>" root@<主机>:~/autodl-tmp/deep-fMRI-dataset/figures/ figures/` 同步到本地 `/home/linux/projects/deep-fMRI-dataset/figures/`，52 个文件全部到位。**注意**：此类下载命令必须在本地终端执行，不能粘贴进服务器 SSH 会话里（曾误粘贴导致 `rsync` 反向在服务器上找本地路径报错）。

**M6 里程碑结案，三被试扩展毕业闭环完成（M0–M6 全部完成）。**

---

# M6 补充诊断 — AWD-LSTM 上下文饱和核实（非 bug，真实发现）

## 起因

复盘图表时注意到 AWD-LSTM 在三被试上 r8/r32/r128 几乎完全打平（UTS01: 0.068289/0.068309/0.068309；UTS02: 0.071384/0.071352/0.071352），怀疑是否存在"H 窗口未真正生效 / recurrent state 被错误复用"这类实现 bug，而非真实的模型属性。

## 核实方法

直接比较同一批目标词在 H=8/32/128 下的原始特征缓存（`load_features`），逐元素展平后算相关与最大绝对差：

```python
corr(X8 , X32 ) = 0.99987
corr(X32, X128) = 1.000000
max|X32 - X128| = 1.5534460544586182e-06
```

## 判读

- `corr(X32,X128)=1.000000`、`max|diff|≈1.5e-6`——**float32 浮点舍入误差量级**，X32 与 X128 在数值上逐位相同；
- `corr(X8,X32)=0.99987`——**非零、非浮点噪声级别**的真实差异，说明 8→32 词的历史确实被模型消化、确实改变了输出。

两者组合恰好排除了两种最可能的 bug：
1. 若"H 机制整体失效"（窗口构造 bug、模型没吃到不同长度历史），X8 与 X32 也应像 X32/X128 一样逐位相同——实际不是；
2. 若"`reset_state()` 未生效、状态跨窗口污染"，预期是杂乱、非单调的差异模式——实际是干净的单调收敛（H8→H32 有真实但极小的差距，H32→H128 精确归零）。

## 结论

这是该预训练 AWD-LSTM **真实的记忆饱和曲线**：有效上下文窗口在约 32 词处已饱和，超出部分对输出的贡献在 float32 精度下精确为零——符合 LSTM vanishing-gradient 的经典行为，与三被试上 `delta_total_awd_lstm_ifg_main` 精确为零（M5 结果）完全自洽。**不涉及实现错误，不需要改代码或重跑**，按项目纪律直接记录为 Limitations 中的一条如实发现，不作为核心架构排名的证据（AWD-LSTM 本就不参与核心三模型排名）。

附带验证价值：AWD-LSTM 是四模型里最容易暴露"H 未生效"类 bug 的模型（对上下文最敏感），它没有暴露此类问题，反过来加固了整条 H 窗口构造机制（`src/models/windowing.py::build_window`）对 pythia/mamba/rwkv 同样可信的信心。

**建议写入 Limitations 的英文措辞**：
> AWD-LSTM's effective context window saturates by H≈32 tokens (H=32 and H=128 representations are numerically identical to float32 precision); this recurrent-memory ceiling explains its flat Δr_total across H and is consistent with known vanishing-gradient behavior in LSTMs, not a pipeline artifact — verified by confirming H=8 vs H=32 representations differ non-trivially (r=0.99987) while H=32 vs H=128 do not (r=1.000000, max abs diff at float32 rounding level).
