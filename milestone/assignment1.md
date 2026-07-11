# Assignment 1 — 从「UTS03 单被试」扩展到「UTS01+UTS02+UTS03 三被试」的完整修改分析

**编写日期**：2026-07-03
**性质**：纯分析文档（**尚未修改任何代码**）。逐项列出把当前 UTS03 单被试管线回归到里程碑原始冻结口径（3 被试）所需的全部改动、其原因、涉及的具体文件/行号，以及仍是空白、需要新写的部分。
**背景**：当前项目是 UTS03 单被试 pilot（`config/config.yaml` 版本号 `v4.9-uts03-pilot`）。里程碑冻结文档原始口径是 **UTS01+UTS02+UTS03 三被试**，改成单被试是「数据可得性硬约束」下的已确认预注册偏离（散见于 `m3b_dual_path.py`、`m4_driver.py` 的注释）。本文档分析「若把偏离撤销、回到三被试」需要动什么。

---

## 0. 先决条件（不是代码问题，但决定这件事能否做）

三被试扩展的**根本前提是数据**，不是代码：

| 需要的数据 | 当前状态 | 用途 |
|---|---|---|
| `data/ds003020/derivatives/preprocessed_data/UTS01/*.hf5`、`UTS02/*.hf5` | ❌ 只下载了 UTS03 | 每个被试的 BOLD 响应（voxel 时间序列） |
| `data/ds003020/derivatives/freesurfer_subjdir/UTS01/`、`UTS02/` | ❌ 只有 UTS03 | ROI 提取需要的 FreeSurfer annot（`?h.aparc.a2009s.annot`） |
| pycortex-db 中 UTS01/UTS02 的 transform（如 `UTS01_auto`） | ❌ 只有 `UTS03_auto` | voxel↔皮层映射，ROI 归属与 Fig6 都要用 |

**结论**：在这三类数据齐全之前，下面所有代码改动都无法真正跑通。这是第一步。数据经 git-annex 获取（`cd data/ds003020 && git annex get <path>`，远端 S3-PUBLIC=OpenNeuro 公开可达）。

---

## 1. 全局结论：哪些「不用改」，哪些「要改」

项目的核心计算代码在设计上大多是**被试无关**的——它们接收数组、路径、subject 字符串作为参数，本身不硬编码某个被试。真正要改的是「**入口层 / 配置层 / 被试专属产物**」以及「**目前完全不存在的跨被试聚合层**」。

### 1.1 完全不用改（被试无关，天然可复用）

| 模块 | 为什么不用改 |
|---|---|
| `scripts/m1_extract_features.py` + `cache/features/` | 特征是「语言模型对文本的激活」，同一个故事对所有被试是同一份特征。全脚本无 `subject`/`UTS0` 字样（已 grep 确认）。**12 组合特征缓存三被试共用，无需重提取。** |
| `src/ridge/pipeline.py`、`src/ridge/score.py` | ridge 拟合 / 评分核心，输入是 `StoryData`(X,Y,tr_times) 数组，不认识「被试」概念。 |
| `src/stats/bootstrap.py`、`src/stats/estimands.py` | 统计核心是纯 numpy，喂什么数组算什么，无 `subject` 字样（已确认）。**但见 §4——它只在单被试语境验证过，跨被试聚合逻辑不存在。** |
| `frozen/word_index.parquet`、`frozen/story_manifest.csv`（词/故事层） | 刺激（TextGrid、词序列）对所有被试相同。`m0_build_frozen.py` 的 word_index 直接读 TextGrids，与被试无关。 |
| `src/fmri/derivatives.py`、`src/fmri/mask.py`、`src/fmri/alignment.py` 的函数 | `load_response(data_dir, subject, story)`、`compute_bold_only_mask(data_dir, subject, stories)`、`list_subject_stories(...)` 都已把 `subject` 作为**参数**接收，函数本身已支持任意被试，无需改函数体，只需上层传对被试。 |

### 1.2 被试专属产物（不是「改代码」，是「每个被试各生成一份」）

这些文件名本身就带被试后缀，因为每个人脑解剖不同（voxel 数、ROI 体素位置都不同，UTS03 是 95556 voxel，UTS01/02 几乎一定是别的数）：

| 产物 | 当前 | 三被试需要 |
|---|---|---|
| `frozen/voxel_mask_{subject}.npy` / `.json` | 只有 `_UTS03` | 再生成 `_UTS01`、`_UTS02` |
| `frozen/roi_columns_{subject}.npz` / `.json` | 只有 `_UTS03` | 再生成 `_UTS01`、`_UTS02` |
| `results/*/{subject}/...` | 只有 `UTS03/` | 每被试各一套结果目录 |
| `figures/{subject}/...` | 只有 `UTS03/` | 每被试各一套图表 |

这部分不需要改代码逻辑，而是**把已有脚本对 UTS01、UTS02 各跑一遍**。计算量 ×3。

---

## 2. 需要修改的代码（逐文件、逐行）

### 2.1 `config/config.yaml` — 配置入口

| 位置 | 现状 | 改成 | 说明 |
|---|---|---|---|
| `version:` (行2) | `v4.9-uts03-pilot` | `v4.9`（去掉 pilot 标记） | 语义标记，非功能 |
| `datasets.subjects` (行16) | `[UTS03]` | `[UTS01, UTS02, UTS03]` | 核心开关。但注意：**只改这一行不会让脚本自动跑三被试**，因为下游多数脚本是「每次一个 `--subject`」的 CLI，见下。 |

### 2.2 `scripts/m0_build_frozen.py` — 冻结产物生成（**有硬编码，必须改**）

- **[m0_build_frozen.py:42](../scripts/m0_build_frozen.py#L42)**：`subject = cfg["datasets"]["subjects"][0]` —— **只取列表第一个**。即使 config 改成三被试，story_manifest 也只按 `subjects[0]` 的 BOLD 目录判断 `bold_available`。
- **[m0_build_frozen.py:36-57](../scripts/m0_build_frozen.py#L36) `build_story_manifest`**：`bold_available` 标志按单个被试的 `bold_dir` 存在性判断。**关键陷阱**：LeBel 数据集中三个被试听的故事集合**不完全相同**，某故事可能 UTS03 有、UTS01 没有。
- **[m0_build_frozen.py:121-161](../scripts/m0_build_frozen.py#L121) `build_fold_split`**：fold 划分基于 `manifest[bold_available]`。**若三被试故事集不同，fold_split 会因被试而异**。

**需要的设计决策（不只是改代码）**：
1. **共同故事集方案**：只用三被试都有 BOLD 的故事交集来建 fold_split，保证一份 fold 划分对三被试都成立（推荐，便于跨被试配对）。→ `build_story_manifest` 要改成对三被试各查一遍 `bold_available`，取交集。
2. **各自 fold 方案**：每被试各建 `fold_split_{subject}.json`。→ 下游所有读 `fold_split.json` 的地方都要改成读被试专属版本。

无论哪种，`m0` 都要从「取 `subjects[0]`」改成「遍历 `subjects`」。word_index / story_manifest 的词层部分不变（刺激相同）。

### 2.3 `scripts/m2_validate_pipeline.py` — voxel mask + ROI 生成

- CLI 已有 `--subject`（[:191](../scripts/m2_validate_pipeline.py#L191)），`step_voxel_mask`/`step_roi` 已参数化，函数体基本可复用。
- **[m2_validate_pipeline.py:100](../scripts/m2_validate_pipeline.py#L100)**：`def step_roi(cfg, subject, xfm="UTS03_auto")` —— **xfm 默认值硬编码 `UTS03_auto`**。每个被试在 pycortex 里的 transform 名不同（`UTS01_auto` 等）。要么改默认为 `f"{subject}_auto"`，要么在 CLI 暴露 `--xfm` 并每被试传对。
- 用法：对 UTS01/UTS02 各跑 `--step voxel_mask` 和 `--step roi`，产出各自的 `frozen/voxel_mask_{subject}.*` 和 `roi_columns_{subject}.*`。这是**每被试各一遍**的重复执行，逻辑不变。

### 2.4 `scripts/m4_*.py` + `src/ridge/m4_driver.py` — 编码矩阵主计算

- CLI 已有 `--subject`（默认 `UTS03`），`roi_columns_{subject}.npz`、`results/<out>/<subject>/` 都已按被试参数化（[m4_pythia.py](../scripts/m4_pythia.py) 读 `roi_columns_{args.subject}.npz`）。
- **不用改逻辑，但要跑三遍**：4 模型 × 3 被试 = 12 次进程启动。计算量相对单被试 **×3**（GPU 时间约 ×3）。
- **[m4_driver.py:10](../src/ridge/m4_driver.py#L10)** 及 **[:267-276](../src/ridge/m4_driver.py#L267)**：manifest 里写死的「预注册偏离：3 subjects → UTS03 单被试」`reason: dataset availability constraint` 字段。三被试跑完后，这段偏离声明要**撤销/改写**为「已回归原始冻结三被试口径」——否则 manifest 会自相矛盾（既声称三被试又声称偏离为单被试）。同样的注释也在各 `m4_*.py` 的 `--subject` help 文本里（如 [m4_pythia.py:38-40](../scripts/m4_pythia.py#L38)）。

### 2.5 `scripts/m5_analysis.py` — 统计分析

- CLI 已有 `--subject`（[:142](../scripts/m5_analysis.py#L142)），可对每被试各跑一遍，产出 `results/m5_stats/{subject}/`。
- **[m5_analysis.py:283](../scripts/m5_analysis.py#L283)**：`"phase": "M5 ...(single-subject deviation...)"` 同样要改偏离声明。
- **核心缺口**：`load_bootstrap_data` 只扫**单个被试**的 `cells/`，bootstrap 是 `paired_story_within_fold`——**单被试内**的配对。三被试的正确统计做法不是「跑三次各自出一个 CI」，而是要有一层**跨被试聚合**（见 §4，这是新代码，不是改现有）。

### 2.6 `scripts/m6_*.py` — 图表/表格（**有硬编码 bug**）

- **[m6_tables.py:104-105](../scripts/m6_tables.py#L104)**：
  ```python
  vmask = json.load(open(frozen / "voxel_mask_UTS03.json"))
  roi_cols = dict(np.load(frozen / "roi_columns_UTS03.npz"))
  ```
  **字面量 `UTS03` 写死，没用 `args.subject`**。即使传 `--subject UTS01` 也会去读 UTS03 的 mask —— 这是多被试下的真实 bug，必须改成 `f"..._{args.subject}..."`。
- **[m6_figures.py:319](../scripts/m6_figures.py#L319)**、**[m6_roi_location.py:74-75](../scripts/m6_roi_location.py#L74)**：`--subject` 默认 `UTS03`、`--xfm` 默认 `UTS03_auto` 硬编码，每被试要传对 xfm。
- 其余 M6 大部分已按 `args.subject` 参数化输出到 `figures/{subject}/`，跑三遍即可。

---

## 3. 修改清单速查表

| 文件 | 改动类型 | 关键点 |
|---|---|---|
| `config/config.yaml` | 改配置 | `subjects: [UTS01,UTS02,UTS03]`；version 去 pilot |
| `scripts/m0_build_frozen.py` | **改逻辑** | `[0]` 取首个 → 遍历；story 可得性取交集；fold_split 策略决策 |
| `scripts/m2_validate_pipeline.py` | 改默认值 + 跑3遍 | xfm 硬编码 `UTS03_auto` → `f"{subject}_auto"` |
| `src/ridge/m4_driver.py` | 改偏离声明 + 跑3遍 | manifest 的 single-subject deviation 字段撤销 |
| `scripts/m4_{pythia,mamba,rwkv,awd_lstm}.py` | 改 help 文本 + 跑3遍 | 计算量 ×3 |
| `scripts/m5_analysis.py` | 改偏离声明 + **补跨被试聚合** | 见 §4 |
| `scripts/m6_tables.py` | **改 bug** | `voxel_mask_UTS03` 硬编码没用 `args.subject` |
| `scripts/m6_figures.py`、`m6_roi_location.py` | 改默认 xfm + 跑3遍 | |
| **（新文件）跨被试聚合** | **新写** | §4，目前完全不存在 |
| `frozen/voxel_mask_{01,02}.*`、`roi_columns_{01,02}.*` | 重新生成 | 每被试各一份 |

---

## 4. 目前完全空白、必须新写的部分：跨被试统计聚合

这是三被试扩展中**最实质、也最容易被低估**的一块——它不是「改几行」，而是「新增一层统计设计+代码」，因为当前整套统计（`src/stats/`）从未处理过多被试。

**现状**：`src/stats/bootstrap.py` 的抽样单元是 `paired_story_within_fold`——在**一个被试内**、fold 内对 story 有放回抽样。聚合链是「fold 内按有效 TR 加权 Fisher-z → 跨 fold 加权 → tanh」。整条链里没有「被试」这一层。

**三被试需要决定的设计问题**（属于统计方法学，需与冻结文档核对，甚至可能要写进论文 Methods）：
1. **被试作为聚合层级**：是否在「fold 内 → 跨 fold」之上再加「跨被试」一层？三被试的 delta_total 如何合并——简单平均？被试内先算 CI 再跨被试固定/随机效应？
2. **配对结构**：确认性家族（`rwkv_minus_pythia_delta_total_ifg_main` 等）的配对 bootstrap，在三被试下配对单元是 (subject, story) 还是先被试内配对再跨被试合并？
3. **权重**：跨被试合并时被试是否等权，还是按各自有效 TR / voxel 数加权？

**结论**：`bootstrap.py` 的 `BootstrapData` 结构、`aggregate_to_r` 聚合函数都要扩展以容纳被试维度，`m5_analysis.py` 的 `load_bootstrap_data` 要能同时读三被试的 cells 并按上面选定的设计聚合。**这部分现在一行都没有**，是三被试相比单被试真正「新增」的工作量核心。

---

## 5. 执行顺序建议（若真要做）

1. **数据**：git-annex 拉取 UTS01/UTS02 的 preprocessed_data + freesurfer_subjdir + pycortex transforms（§0）。
2. **M0**：改 `m0_build_frozen.py` 支持遍历被试 + 故事交集/fold 策略，重生成冻结产物。
3. **M2**：对 UTS01/UTS02 各跑 voxel_mask + roi（改 xfm 默认），产出各自 frozen mask/ROI。
4. **M1**：**跳过**——特征被试无关，已有缓存直接复用。
5. **M4**：对 UTS01/UTS02 各跑全矩阵（UTS03 已有）。GPU 时间 ×2 增量。
6. **统计层**：先做 §4 的跨被试聚合设计与新代码，再跑 M5。
7. **M6**：修 `m6_tables.py` 硬编码 bug，对三被试出图表。
8. **文档**：撤销/改写所有 manifest 与注释里的「single-subject deviation」声明。

---

## 6. 一句话总结

代码结构本身是被试无关、可复用的，真正的工作量集中在四块：**①数据下载（先决条件）**、**②`m0` 单被试硬编码改批量 + fold 策略决策**、**③M2/M4/M6 对新被试各跑一遍（含 `m6_tables.py` 的 `UTS03` 硬编码 bug 与若干 xfm 默认值）**、**④从零新增跨被试统计聚合层（`src/stats/` 目前完全不支持多被试）**。其中 ④ 是相对单被试最本质的新增，也是最需要统计方法学决策的部分。