# demo/ — 答辩录屏演示脚本

三个面向屏幕录制的独立演示脚本,用于满足答辩评分标准中"展示报告中已完成但未详尽涵盖的
方面,对单纯通过文字或图表难以表达的内容提供进一步澄清"这一项(占 40%)。

**设计原则**:每个脚本把论文里一句**无法被读者验证的文字声明**变成屏幕上可见的数值事实。
不修改任何现有代码,只调用/读取现有模块与已有缓存结果。

---

## 运行环境

全部在 AutoDL 服务器上运行(项目数据与模型权重都在那里):

```bash
ssh -p <端口> root@connect.bjb2.seetacloud.com
cd /root/autodl-tmp/deep-fMRI-dataset
conda activate deepfmri          # 或 source activate deepfmri
```

| 脚本 | 需要 GPU | 实测耗时 | 依赖数据 |
|---|---|---|---|
| **D1(默认,读缓存)** | 否 | **0.08 s** | `demo/cached_results/d1_results.json` |
| D1 `--live`(现场真算) | 建议开卡 | GPU **7.5 s** / 无卡 CPU **120 s** | HF 本地权重 + `frozen/word_index.parquet` |
| D2 | 否(纯 CPU) | **2.9 s** | `cache/features/` + `frozen/` + `results/m4_full_matrix/` + BOLD |
| D3 | 否(纯 CPU) | **4.7 s** | `results/m4_full_matrix/` + `results/m5_stats/` |

**录屏时三个脚本合计约 8 秒,完全不需要等待。**

### D1 为什么默认读缓存

D1 是唯一需要真实模型前向的演示。开卡时只要 7.5 秒,但**无卡模式下 CPU 前向要 120 秒**
(实测),录屏会出现两分钟静默。因此 D1 默认从 `demo/cached_results/d1_results.json` 读取
并渲染,**0.08 秒**出结果。

该缓存不是手写数字,而是 `--live` 模式真实计算后自动落盘的,内含生成时间、设备、
git commit,输出顶部会明确标注:

```
⚠ 以下数值读取自缓存：demo/cached_results/d1_results.json
   该缓存由 --live 模式真实计算后落盘，非手写数字：
     生成时间 = 2026-08-17 21:39:34   设备 = cpu   代码版本 = 3ecfb87
   现场重算请加 --live（GPU 约 7 s；无卡 CPU 约 120 s）
```

两种模式的输出格式**完全一致**(同一套 `render()` 函数),所以答辩时若被要求现场重算,
直接 `bash demo/run.sh d1-live` 即可,画面内容不变。

> D2/D3 **保持现场真跑**:它们本来就只要 2.9 s / 4.7 s,而且真跑的说服力强得多——
> D2 是运行时拦截真实 `run_fold` 的 fit 调用,D3 是真的跑满 1000 次重采样。
> 这两处"现场算出来"正是演示的价值所在,不建议改成读缓存。

---

## 运行命令

**务必使用 `demo/run.sh` 启动器,不要直接 `python3 demo/dX.py`。**

```bash
bash demo/run.sh d1        # D1(读缓存,0.08s)
bash demo/run.sh d2        # D2(现场真跑,2.9s)
bash demo/run.sh d3        # D3(现场真跑,4.7s)
bash demo/run.sh all       # 依次跑三个(录屏建议分三段单独录)

bash demo/run.sh d1-live   # D1 现场真算并更新缓存(GPU~7s / 无卡CPU~120s)
bash demo/run.sh live      # ★ 三个全部现场真算(需开卡,合计约 15s)
```

---

## 两种演示模式:选哪个

**首先明确:D2 与 D3 在任何模式下都是现场真算的**,它们不读任何汇总结果:

- D2 现场做 Lanczos 重采样 + FIR 展开 + 掩码构建,并**运行时 monkeypatch 拦截真实
  `run_fold`** 的 `StandardScaler.fit` / `PCA.fit`,屏幕上的 `(754, 768)` 是当场观测到的;
- D3 现场跑满 **1000 次** paired-story bootstrap 重采样(只是不重跑特征提取与 ridge,
  那两步读 M4 已落盘的 story 级分数——这符合"能读缓存绝不重算"的要求)。

唯一有区别的是 D1:

| | 模式 A(默认) | 模式 B(全真现场) |
|---|---|---|
| 命令 | `bash demo/run.sh d1` | `bash demo/run.sh d1-live` |
| D1 行为 | 读缓存 JSON 渲染 | **三个模型真实加载 + 真实前向** |
| 耗时 | 0.08 s | **开卡 7.5 s** / 无卡 CPU 120 s |
| 需要开卡 | 否 | **建议开卡** |
| 三段合计 | 约 8 s | **约 15 s** |

**要做全真现场演示,请开有卡模式**,然后:

```bash
bash demo/check_ready.sh          # 先自检(应显示"全部就绪")
nvidia-smi                        # 确认 GPU 在线(录屏时这一行也值得留着)

clear; bash demo/run.sh d1-live   # 三个模型真实前向,约 7.5 s
clear; bash demo/run.sh d2        # 约 2.9 s
clear; bash demo/run.sh d3        # 约 4.7 s
```

`d1-live` 的输出顶部会显示 `● 现场真实计算（--live）  设备 = cuda`,与读缓存模式的
`⚠ 以下数值读取自缓存` 一眼可分。两种模式的其余输出格式完全一致(同一套 `render()`)。

> **为什么全真模式建议开卡**:无卡模式下 D1 的 CPU 前向实测 **120 秒**(三个模型各做
> 4 次 120–129 token 的前向),录屏会出现两分钟静默。开卡后同样的计算只要 7.5 秒。
> 开卡几分钟的费用换来完全真实的现场演示,通常是值得的。

### 为什么必须用 run.sh(三个实测踩到的坑)

1. **HF 离线模式** — 服务器无法访问 huggingface.co。`transformers` 加载模型前会发 HEAD
   请求检查更新,失败后按 1/2/4/8/8 秒重试 5 次 × 多个配置文件。**实测 RWKV 因此卡了
   数分钟毫无输出**;设 `HF_HUB_OFFLINE=1` 后加载只需 0.4 秒(权重本就在本地缓存)。
2. **`OMP_NUM_THREADS`** — 服务器上该变量为非法值,libgomp 会在**任何 Python 代码运行前**
   往 stderr 打印两行警告,污染录屏第一屏。必须在 shell 层设置,脚本内 `os.environ` 来不及。
3. **`python3 -u`** — Python 经 SSH 管道输出是块缓冲的,不加 `-u` 录屏时会长时间空白然后
   所有文字一次性刷出,观感极差。

---

## D1 — 窗口边界与 H 的严格上界

**证明的声明**(论文 2.2 节):"Each target window was processed in a cold-start forward
call so that H remained a strict upper bound",以及 H 计的是**原始词**而非 subtoken。

**PPT 播放位置**:Slide 4 之后 | **建议时长**:35–45 s

### 预期输出摘要

(a) 同一个真实目标词(取自 `eligible_h128` 集合,非编造)在三个 H 下的窗口:

```
   H      窗口词数   subtoken数      目标词subtoken数   窗口原文（长窗截断中间）
   8         9           9                 1   STOP AND IT WAS LIKE  [...1 words...]  OUT IN FRONT
  32        33          33                 1   MY PARENTS AND UH I  [...25 words...]  OUT IN FRONT
 128       129         132                 1   MEET YOUR PARENTS AND I  [...121 words...]  OUT IN FRONT
```

**这是最关键的一屏**:窗口词数恒为 H+1(8→9、32→33、128→129),而 H=128 时 subtoken 数
是 **132 > 129** —— 直接证明 H 数的是原始词,BPE 把部分词切成了多个 subtoken。

(b) 三个 checkpoint 的严格上界检验:

```
模型       checkpoint                     cold-start  allclose         状态继承         拼接前文
pythia   EleutherAI/pythia-160m           0.00e+00      True     6.38e+00     6.38e+00
rwkv     RWKV/rwkv-4-169m-pile            0.00e+00      True     2.87e+01     2.87e+01
mamba    state-spaces/mamba-130m-hf       0.00e+00      True          n/a     3.29e+00
```

- **cold-start 全部精确为 0.00e+00** → 先用 120 个无关词做一次前向,再提取同一目标词,
  结果逐位相同,窗口外文本不留任何痕迹。
- **对照列量级达 10⁰~10¹** → 一旦让窗口外的词进入同一次前向,目标词向量剧变,
  远非浮点误差。H 边界是实质约束,不是形式声明。

> Mamba 的"状态继承"列为 n/a:其 HF 实现不接受"多 token 序列 + 预置 cache"(抛 ValueError),
> 故其泄漏用通用的"拼接前文"列呈现,结论一致。这一点在输出中如实标注。

---

## D2 — 时间对齐的形状变换 + 无泄漏的可执行检查

**证明的声明**:PPT Slide 6 横幅 "No held-out story contributes to feature transformation,
hyperparameter selection or model fitting";论文 2.3 节对 leakage 的文字辩护。

**PPT 播放位置**:Slide 6 之后 | **建议时长**:40–45 s

### 预期输出摘要

(a) 单个真实故事的形状变换链,每步一行:

```
1. 词级特征（缓存）        (1528, 768)   1528 个合格目标词 × 768 维
2. 词中点时间戳            (1528,)       范围 39.1s ~ 503.1s（不等间隔）
3. TR 时间轴（等间隔 2s）   (256,)        respdict=261，减 pad 5 帧
4. Lanczos 重采样 词→TR    (256, 768)    不等间隔词 → 等间隔 TR 网格
5. 裁边 [10:-5]           (241, 768)    去头 10 帧(30s)、去尾 5 帧(10s)
6. [折内] scaler + PCA     (T, 100)      降到 100 维，仅用训练折拟合
7. FIR 四延迟展开          (241, 400)    维度 ×4 = 100×4
8. 评分掩码 ∩ >100s        196 / 241     最终参与评分的 TR 数
总结：1528 个词级向量 → 241 个 TR 行 → 400 列设计矩阵 → 196 个 TR 参与打分
```

(b) 无泄漏检查,**三层证据**,每条带 `[PASS]`:

- **证据层 1 — 冻结的 fold 划分**:打印三折 train/test 数量(55/28、55/28、56/27),
  断言 train ∩ test = ∅,断言测试故事数为 28/28/27,断言 83 个故事各只被测一次。
- **证据层 2 — 运行时拦截**(最有说服力):monkeypatch `StandardScaler.fit` / `PCA.fit`,
  再调用**真实的 `run_fold`**,直接打印它们看到的输入 shape:
  ```
    StandardScaler.fit     输入 shape = (754, 768)
    PCA.fit                输入 shape = (754, 768)
  [PASS] 行数 754 == train 行数 754（未含 test 的 703 行）
  [PASS] 没有任何 fit 调用看到全体 1457 行
  [PASS] test 行仅用于最终预测，未进入任何 fit（Xte 703 == test 703 行）
  ```
  为保证秒级完成,这一层用 3 训练 + 2 测试故事、150 个体素的**缩小版** fold,
  但跑的是真实故事、真实特征、未修改的 `run_fold`。输出中明确标注了这一点。
- **证据层 3 — 全量运行的落盘审计**:扫描 108 个 M4 主层 cell,
  `leakage_audit_pass` 与 `common_mask_verified` 均为 108/108。

> **一处如实披露**:`scoring_mask_bit_identical` 只有 72/108,因为 UTS03 是 pilot 期先跑的,
> 当时 `m4_driver` 还没加这个"逐元素相同"的强字段。该缺口由独立脚本
> `scripts/verify_scoring_mask_identity.py` 事后补齐(产物在 `results/mask_identity_audit/`,
> 三被试各 83 故事全部 True)。脚本把两处证据都读出来并说明,不把"字段缺失"误报成
> "掩码不一致",也不掩盖它。

---

## D3 — Paired-story bootstrap 的配对性

**证明的声明**(论文 2.4 节):"the same sampled indices are used for every checkpoint,
H condition, and layer",以及表 4 的三行确认性结果。

**PPT 播放位置**:Slide 9 之后 | **建议时长**:25–30 s

### 预期输出摘要

(1) 配对机制 —— 打印第 1 次重采样的实际索引,并证明两个模型用的是同一组:

```
  fold_2: 从 27 个故事中抽 27 个 → 索引 [25, 9, 7, 8, 26, 18, ... (18 more) ..., 16, 2, 20]
  [PASS] 两个模型的抽样索引数组逐元素相等 = True
  [PASS] 实际上是**同一个索引对象**被传给所有 key（不是巧合相同）
```

(2) 跑满 1000 次,**精确复现论文表 4**:

```
被试           本次重跑 point                     95% CI         p       论文表4 point     一致?
UTS01         +0.001375     [+0.000123, +0.002723]     0.042          +0.0014       ✓
UTS02         +0.004395     [+0.002601, +0.006029]    <0.001          +0.0044       ✓
UTS03         +0.002837     [+0.001142, +0.004317]     0.004          +0.0028       ✓
```

差值明细全部为 `Δ=+0.0000`,并与 M5 原始产物**逐位一致**(`|Δ|=0.00e+00`),
证明同种子下完全可复现。

(3) Holm 校正判定 —— 6 行(3 被试 × 2 对比),含 UTS02 的 RWKV−Pythia 未拒绝(p=0.420)。

(4) **对照实验:如果不配对会怎样**(不属于论文任何结果,仅说明配对的价值):

```
被试              配对 95% CI 宽度          非配对 95% CI 宽度       倍数
UTS01               0.002600               0.003956     1.5×
UTS02               0.003428               0.005380     1.6×
UTS03               0.003175               0.004445     1.4×
```

区间宽 1.4–1.6 倍 —— 故事难度差异不再被消掉,全变成噪声。这是"为什么坚持同一组索引"
最直观的答案,也是纯文字最难讲清的部分。

---

## 建议录屏顺序与时长

| 顺序 | 脚本 | 时长 | 接在哪张 PPT 之后 | 录屏要点 |
|---|---|---|---|---|
| 1 | D1 | 35–45 s | Slide 4 | 在 (a) 的三行表格上停顿,口播"窗口 129 词但 132 个 subtoken";在汇总表 cold-start 全 0 那一列停顿 |
| 2 | D2 | 40–45 s | Slide 6 | 形状链逐行滚动即可;在证据层 2 的两个 fit shape 上停顿,强调 754 vs 1457 |
| 3 | D3 | 25–30 s | Slide 9 | 在"同一个索引对象"两个 PASS 停顿;在表 4 三行 ✓ 停顿;末尾对照表停顿 |

**总计约 100–120 秒**。三段建议**分别单独录制**(而非 `run.sh all` 一次录完),便于在
每段前后插入 PPT 讲解、单独重录某一段。

录屏前建议先跑一遍预热(HF 缓存与页缓存都会变热,D1 的模型加载会更快)。

---

## 数据来源声明

三个脚本的所有数值均为**现场真实计算或读取真实产物**,无任何预存/硬编码的演示数字:

| 演示 | 计算/读取 |
|---|---|
| D1(默认) | 读 `demo/cached_results/d1_results.json` —— 由 `--live` 真实前向后落盘,输出顶部标注来源与生成时间 |
| D1 `--live` | 三个模型的真实前向(现场计算);目标词取自 `frozen/word_index.parquet` |
| D2(a) | 读 `cache/features/`(M1 已提取)+ 现场做 Lanczos/FIR/掩码 |
| D2(b) | 现场调用真实 `run_fold` + 读 M4/audit 落盘记录 |
| D3 | 读 M4 story 级分数 + 现场跑 1000 次重采样(不重跑特征提取/ridge) |

论文表 4 的期望值以常量形式写在 `d3_paired_bootstrap.py` 的 `PAPER_TABLE4` 中,
仅用于**对照打印**,不参与任何计算。
