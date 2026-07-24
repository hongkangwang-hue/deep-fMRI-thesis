# 实验补充——刺激侧上下文控制（回应导师建议）

## 这个文件夹是什么

导师建议：固定目标词，把它前面的上下文打乱/替换，检验 Context Gain 是否真的依赖
语言历史本身，而不只是"输入更长/特征更平滑/故事位置更靠后"。这个文件夹是为完成
这条建议而**新增**的代码，不修改主实验（`src/`、`scripts/`）的任何已有文件，唯一
例外是 `src/models/base.py` 的 `extract()`/`extract_batch()` 各加了一个可选参数
（`window_override`/`windows_override`，默认 `None`，不传时行为与之前逐位相同——
34 个既有单测 + 新增单测全部通过，见 `tests/test_extract_batch_override_wiring.py`）。

对应文档：
- 设计规范：`../milestone/里程碑增补老师修改建议上下文控制.md`（V6.4.2，导师建议原文的科学设计）
- 修改点清单：`../milestone/里程碑增补_M4S_刺激侧上下文控制_执行版.md`

## 与主实验的关系

只读、附加、不覆盖：新条件叫 `ctx1`，用独立缓存目录、独立统计键、独立图表编号；
主实验（normal/shift 两条件）的所有已产出结果不受影响。

## 进度

| 步骤 | 内容 | 算力 | 状态 |
|---|---|---|---|
| 0 | 冻结决策（seed、scope、condition 命名） | 无 | ✅ `config/freeze_manifest.json` |
| 1 | 上下文置换纯函数 + 单测 | 无 | ✅ `src/context_perturb.py` + `tests/`（15 单测） |
| 2 | 硬闸门断言 + 词形重复诊断（§2.4） | 无 | ✅ `scripts/m4s_assert_impl.py` → `results/`（对全部84故事跑通） |
| 3 | 打乱上下文特征重提取 | GPU | 🟡 代码已写好，本地已验证接线正确，**待你确认后上服务器跑** |
| 4 | ctx1 条件 Ridge 重拟合 | Ridge | ⏸ 待步骤 3 完成 |
| 5 | 统计量 `D_m`/`I_MP` + bootstrap CI | 轻量 | ⏸ 待步骤 4 完成 |
| 6 | 论文 Figure 17 / Table 9 | 无 | ⏸ 待步骤 5 完成 |

Step 0–2 全部本地零算力，已产出真实结果（见 `results/`）。

**Step 3 代码状态**：`scripts/m4s_extract_perturbed_features.py` 已写好，结构与
`../scripts/m1_extract_features.py` 逐行对应。本地已验证（无需 torch/GPU）：
- 模块可正常 import（惰性加载，不因缺 torch 报错）；
- `--from-fold-split` 精确解析出 83 个 CV 故事；
- 置换落盘 → 读回 → SHA-256 全部正确（真实跑过一次，非纸面设计）；
- `base.py` 的 `windows_override` 接线用带因果依赖的假适配器证明"确实生效、
  不会静默退回真实上下文"（`test_batch_windows_override_actually_differs_from_normal_batch`）。

**真正需要 GPU 的部分**（加载 Pythia/RWKV/Mamba 做前向）尚未执行，按项目规矩
需你确认后在服务器上跑：
```
python 实验补充/scripts/m4s_extract_perturbed_features.py \
    --models pythia rwkv mamba --H 8 128 --from-fold-split --device cuda
```
算力紧可先降到 L2（`--models pythia mamba`）。
