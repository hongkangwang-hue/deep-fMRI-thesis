# UTS01 / UTS02 数据可得性报告（M0 deliverable #4）

核查方式：只读 `git annex find` / `git annex info`（不下载任何内容）。核查时
commit：`26e68944a272f5713bf16c368f43dce72e1b5969`，2026-07-11。

## 结论

**BIDS 结构与 story manifest 均已在 git-annex 中登记（符号链接存在），但内容
（BOLD/FreeSurfer/pycortex 数据本体）均未下载（`present: false`）。** 这是
M1（新被试 BOLD 接入）无法立即启动的直接原因，需先 `git annex get` 才能进入
M1，与 milestone M0 文档"当前真实数据状态"一节的描述一致。

## 逐项核查

| 数据 | UTS01 | UTS02 | UTS03（对照） |
|---|---|---|---|
| `derivatives/preprocessed_data/{subj}/*.hf5`（84 故事） | 84/84 已登记，0/84 内容已下载 | 84/84 已登记，0/84 内容已下载 | 84/84 内容已下载（基线） |
| story 文件名集合 | 与 UTS03 逐一相同（`diff` 无差异） | 与 UTS03 逐一相同 | — |
| `derivatives/freesurfer_subjdir/{subj}/` | 564 个符号链接已登记，0 内容已下载 | （与 UTS01 合计 564） | 212 个（已下载） |
| `derivatives/pycortex-db/{subj}/` | 64 个符号链接已登记，0 内容已下载 | （与 UTS01 合计 64） | 25 个（已下载） |
| pycortex transform 名 | `UTS01_auto`（`derivatives/subject_xfms.json` 已含此键） | `UTS02_auto`（已含） | `UTS03_auto` |
| `derivatives/respdict.json`（故事→TR 数） | 与被试无关，84 条已覆盖全部故事 | 同左 | 同左 |
| 预计下载体积 | ≈20.77 GB（BOLD 20.12G + FreeSurfer 0.36G + pycortex 0.30G） | ≈24.16 GB（BOLD 23.37G + FreeSurfer 0.45G + pycortex 0.34G） | 已下载 |

## 判读

1. **不是"数据不存在"，是"数据未拉取"。** git-annex 的符号链接（指向
   `.git/annex/objects/...`）已经在仓库里，story 覆盖与 UTS03 完全一致（84
   个故事逐一对应，无缺口），说明上游 OpenNeuro（`s3-PUBLIC`）确实有这两名
   被试的完整数据、且已被本仓库登记过。只需要 `git annex get` 拉取内容。
2. **M1 前置动作**：`cd data/ds003020 && git annex get derivatives/preprocessed_data/UTS01 derivatives/preprocessed_data/UTS02 derivatives/freesurfer_subjdir/UTS01 derivatives/freesurfer_subjdir/UTS02 derivatives/pycortex-db/UTS01 derivatives/pycortex-db/UTS02`，合计约 45GB，需要在服务器（或有对应磁盘余量的机器）上执行，不在本次 M0 范围内（M0 明确不产生新被试 brain score）。
3. **`m2_validate_pipeline.py` 的 `xfm` 默认值** 目前硬编码 `UTS03_auto`
   （见 `milestone/assignment1.md` §2.3）；UTS01/UTS02 的正确值已确认为
   `UTS01_auto`/`UTS02_auto`，与被试名规律一致，M1/M2 执行时按被试传入
   即可，不需要额外查询。
4. **English1000 参照真值文件是否存在** 是与本报告独立的问题（关于
   `corrs.npz` 或等价物是否覆盖 UTS01/UTS02），milestone 文档已标注为
   "需在 M2 开始前单独核实"，不在本次 BOLD/FreeSurfer/pycortex 可得性核查
   范围内，留待 M2 前核实。
