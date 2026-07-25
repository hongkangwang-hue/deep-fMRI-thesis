# Figure 17 与 Figure 18 的结构、标题和坐标轴修改建议

## 总体修改原则

这两个图都属于 **stimulus-side context control 的诊断性结果**，不是确认性主结果。因此图的标题、坐标轴和图注都要避免给人一种“已经证明机制”的感觉。

建议整体遵循三点：

1. **标题要中性、概括性强**  
   不建议用过于结论化的词，例如 “advantage”“collapse”“degradation” 作为主标题。  
   这些词可以在正文解释中使用，但图标题最好更中性。

2. **坐标轴要直接说明统计量是什么**  
   尽量写清楚 `Δr_total`、`D_m`、`I_MP` 的定义，不要只写 “interaction” 或 “gain lost”。

3. **图内文字不要过度解释结果**  
   例如 “opposite of expectation”“all CIs exclude zero”“why Dm and IMP cannot be read as evidence...” 这类判断性文字，建议从图内移到正文或图注中。

---

# 一、Figure 17 修改建议

## 1. 当前图的主要问题

Figure 17 当前标题是：

> Stimulus-side context control (C1: same-story shuffled context)

这个标题基本可以，但还不够概括。如果以后不止 C1，或者你想让标题更适合所有情况，可以改成更中性的说法。

当前四个 panel 是：

- A. Context Gain `Δr_total`: normal vs shuffled-context
- B. `D_m = Δr_total^normal - Δr_total^ctx1`
- C. Difference-in-differences interaction
- D. Mamba–Pythia advantage: normal vs shuffled context

问题主要有三个：

1. **Panel C 和 Panel D 的顺序建议互换**  
   读者更容易按这个逻辑阅读：  
   原始 gain → 模型内变化 → 架构差异 → 最终交互量。

2. **“advantage” 这个词不够中性**  
   如果未来某个条件下 Mamba−Pythia 是负值，叫 advantage 就不合适。建议改成 “contrast”。

3. **“gain lost when context shuffled” 有一定结论导向**  
   如果 `D_m` 为负，说明 shuffled 条件下 gain 反而更高，就不能叫 “lost”。建议改成中性标题。

---

## 2. 推荐的总标题

### 最推荐版本

```text
Figure 17. Effects of Stimulus-Side Context Perturbation on Total Context Gain
```

中文含义：

> 刺激侧上下文扰动对总 Context Gain 的影响。

这个标题比较适合所有情况，不管结果是 gain 下降、上升，还是没有变化，都能用。

### 如果想强调 C1

```text
Figure 17. Stimulus-Side Context Perturbation: Total Context Gain under Normal and Shuffled Contexts
```

### 不太建议继续使用的标题

```text
Figure 17. Stimulus-side context control (C1: same-story shuffled context)
```

这个标题不是错，但它偏方法名，不够概括图中展示的核心内容。

---

## 3. 推荐的四面板结构

建议将 Figure 17 调整为以下结构：

| Panel | 推荐标题 | 作用 |
|---|---|---|
| A | Total Context Gain by condition | 展示 normal 和 C1 下各模型的 `Δr_total` |
| B | Model-wise context-perturbation effect | 展示每个模型的 `D_m` |
| C | Architecture contrasts by condition | 展示 normal 和 C1 下的 Mamba−Pythia / RWKV−Pythia |
| D | Architecture-by-condition interaction | 展示最终的 `I_MP` / `I_RP` |

也就是说，建议把当前的 **Panel C 和 Panel D 互换位置**。

新的阅读逻辑是：

1. A：先看 normal 和 C1 下各模型的总 gain；
2. B：再看每个模型被打乱上下文影响多少；
3. C：再看 normal 和 C1 下架构差异分别是多少；
4. D：最后看 difference-in-differences，也就是最关键的诊断性交互量。

---

## 4. Figure 17 各 panel 推荐标题与坐标轴

### Panel A

#### 推荐标题

```text
A. Total Context Gain under normal and shuffled contexts
```

#### X 轴

```text
Participant
```

#### Y 轴

```text
Total Context Gain, Δr_total = r_128 − r_8 (95% CI)
```

#### 图例

建议使用：

```text
Pythia normal
Pythia C1
Mamba normal
Mamba C1
RWKV normal
RWKV C1
```

不建议在图例中只写 `ctx1`，因为论文读者不一定知道代码变量名。  
第一次出现时可以写：

```text
C1 shuffled context
```

---

### Panel B

#### 推荐标题

```text
B. Model-wise context-perturbation effect
```

#### X 轴

```text
Participant
```

#### Y 轴

```text
D_m = Δr_total^normal − Δr_total^C1 (95% CI)
```

#### 建议加一条小注释

```text
Positive values indicate lower total Context Gain under C1.
```

中文意思：

> 正值表示 C1 打乱上下文后 total Context Gain 下降。

这样比 “gain lost when context shuffled” 更中性。  
因为如果以后 `D_m` 为负，也不会出现标题和结果矛盾的问题。

---

### Panel C

这个 panel 建议放现在 Figure 17 的 D 图内容。

#### 推荐标题

```text
C. Architecture contrasts under normal and shuffled contexts
```

#### X 轴

```text
Participant
```

#### Y 轴

如果只展示 Mamba−Pythia：

```text
A_MP = Δr_total(Mamba) − Δr_total(Pythia) (95% CI)
```

如果同时展示 RWKV−Pythia：

```text
Architecture contrast in total Context Gain (95% CI)
```

#### 图例

如果只展示 Mamba−Pythia：

```text
A_MP normal
A_MP C1
```

如果同时展示两组：

```text
Mamba−Pythia normal
Mamba−Pythia C1
RWKV−Pythia normal
RWKV−Pythia C1
```

#### 建议

不建议标题写：

```text
Mamba–Pythia advantage
```

建议改成：

```text
Mamba–Pythia contrast
```

因为 “contrast” 更中性。

---

### Panel D

这个 panel 建议放现在 Figure 17 的 C 图内容，即 `I_MP` 和 `I_RP`。

#### 推荐标题

```text
D. Architecture-by-condition interaction
```

#### X 轴

```text
Participant
```

#### Y 轴

```text
I = A^normal − A^C1 (95% CI)
```

或者更具体：

```text
Difference-in-differences interaction, I (95% CI)
```

#### 图例

```text
I_MP: Mamba−Pythia
I_RP: RWKV−Pythia
```

#### 建议加一条小注释

```text
Positive values indicate that the architecture contrast is reduced under C1.
```

中文意思：

> 正值表示打乱上下文后架构差异变小。

这对解释 `I_MP` 很重要。因为你现在的结果是负数，说明 Mamba−Pythia 没有减弱，反而扩大。

---

## 5. Figure 17 推荐图注

可以直接用下面这个版本：

```text
Figure 17. Effects of stimulus-side context perturbation on total Context Gain. 
C1 denotes the same-story shuffled-context condition. 
Panel A shows total Context Gain under the normal and C1 conditions. 
Panel B shows the model-wise perturbation effect, D_m = Δr_total^normal − Δr_total^C1. 
Panel C shows architecture contrasts under normal and C1 contexts. 
Panel D shows the architecture-by-condition interaction, I = A^normal − A^C1. 
All intervals are 95% story-paired bootstrap confidence intervals. 
These analyses are diagnostic, uncorrected, and not included in the confirmatory family.
```

---

# 二、Figure 18 修改建议

## 1. 当前图的主要问题

Figure 18 当前标题是：

> Representational degradation under shuffled context

这个标题能表达你的结果，但不够中性。  
因为 “degradation” 已经带有结果判断，最好不要作为主标题。

当前副标题是：

> diagnostic · uncorrected · why Dm and IMP cannot be read as evidence for dependence on linguistic structure

这个句子太长，而且太像结论。建议移到正文或图注，不要放在图内标题区。

当前四个 panel 是：

- A. Effective dimensionality
- B. Temporal smoothness
- C. H-dependent degradation
- D. Rank-order agreement

问题主要有四个：

1. **主标题过于结论化**  
   建议用 “representation-statistic diagnostics” 这类中性表达。

2. **Panel C 的标题过于绝对**  
   “all three CIs exclude zero” 如果以后换数据就不一定成立，不适合作为通用标题。

3. **Panel D 的标题过于强**  
   “the model whose representation collapses most also loses the most Context Gain” 容易被理解成因果关系。实际上这个 panel 只有 3 个模型，是描述性 rank-order，不是统计检验。

4. **坐标轴里负数不太直观**  
   如果 x 轴用 `PR DiD`，越负表示 collapse 越强。小白读者会比较难理解。可以考虑改成正向指标：`representational collapse = −PR DiD`。

---

## 2. 推荐的总标题

### 最推荐版本

```text
Figure 18. Representation-Statistic Diagnostics under Stimulus-Side Context Perturbation
```

中文含义：

> 刺激侧上下文扰动下的表示统计诊断。

这个标题比较中性，适合所有情况。

### 如果想更短

```text
Figure 18. Representation Statistics under Normal and Shuffled Contexts
```

### 不太建议继续使用

```text
Figure 18. Representational degradation under shuffled context
```

原因是 “degradation” 太结论化。

---

## 3. 推荐的四面板结构

Figure 18 当前结构基本可以保留，但标题需要修改。

建议结构如下：

| Panel | 推荐标题 | 作用 |
|---|---|---|
| A | Effective dimensionality | 展示 participation ratio 随 H 的变化 |
| B | Temporal smoothness | 展示 lag-1 随 H 的变化 |
| C | H-dependent shift in effective dimensionality | 展示 C1 相比 normal 的 H 依赖变化 |
| D | Descriptive link between representation shift and gain reduction | 描述表示塌缩和 Context Gain 下降之间的关系 |

---

## 4. Figure 18 各 panel 推荐标题与坐标轴

### Panel A

#### 推荐标题

```text
A. Effective dimensionality
```

#### 副标题

```text
Higher participation ratio indicates more distributed representations.
```

#### X 轴

```text
Context Length H (raw words)
```

#### Y 轴

```text
Participation ratio (median over stories)
```

#### 图例

建议统一写：

```text
Pythia normal
Pythia C1
Mamba normal
Mamba C1
RWKV normal
RWKV C1
```

或者颜色表示模型、线型表示条件：

```text
Model: Pythia / Mamba / RWKV
Condition: normal / C1
```

---

### Panel B

#### 推荐标题

```text
B. Temporal smoothness
```

#### 副标题

```text
Higher lag-1 cosine indicates slower-changing features.
```

#### X 轴

```text
Context Length H (raw words)
```

#### Y 轴

```text
Lag-1 cosine similarity between adjacent TRs
```

或者更简短：

```text
Lag-1 cosine similarity
```

#### 不建议继续使用的副标题

```text
shuffling makes representations MORE inert (opposite of expectation)
```

原因是这个说法太口语化，也太结论化。可以在正文里解释，不建议放在图内。

---

### Panel C

#### 推荐标题

```text
C. H-dependent shift in effective dimensionality
```

#### X 轴

```text
Model
```

#### Y 轴

推荐使用中性版本：

```text
PR interaction: ΔPR_C1 − ΔPR_normal (95% CI)
```

其中：

```text
ΔPR = PR_H128 − PR_H8
```

如果想让小白读者更容易理解，也可以改成正向版本：

```text
Dimensionality-collapse index, −(ΔPR_C1 − ΔPR_normal) (95% CI)
```

这样数值越大，就表示 collapse 越强，更直观。

#### 标题不要写

```text
all three CIs exclude zero
```

这个应该放进正文结果描述，而不是图标题。

---

### Panel D

#### 推荐标题

```text
D. Descriptive relation between dimensionality shift and gain reduction
```

#### X 轴

如果继续使用负向 PR DiD：

```text
PR interaction, ΔPR_C1 − ΔPR_normal
```

但是为了更直观，建议改成正向指标：

```text
Dimensionality-collapse index, −(ΔPR_C1 − ΔPR_normal)
```

#### Y 轴

```text
Mean context-gain reduction, D_m (mean ± SD over participants)
```

#### 图内注释

建议保留但稍微改得更正式：

```text
n = 3 models; descriptive comparison only; no statistical test
```

#### 不建议继续使用的标题

```text
Rank-order agreement: the model whose representation collapses most also loses the most Context Gain
```

原因是这个表述太像结论或因果解释。更安全的表达是：

```text
Descriptive relation between dimensionality shift and gain reduction
```

---

## 5. Figure 18 推荐图注

可以直接用下面这个版本：

```text
Figure 18. Representation-statistic diagnostics under stimulus-side context perturbation. 
Panel A shows the effective dimensionality of model representations, measured by participation ratio. 
Panel B shows temporal smoothness, measured by the lag-1 cosine similarity of adjacent TR-level features. 
Panel C summarises the H-dependent shift in participation ratio between the normal and C1 conditions. 
Panel D descriptively compares the dimensionality shift with the average model-wise reduction in total Context Gain. 
These diagnostics show whether the shuffled-context condition changes representation statistics in addition to disrupting linguistic structure. 
All analyses are diagnostic and uncorrected.
```

如果你的正文已经明确说结果显示 OOD 退化，可以在图注最后加一句：

```text
Because C1 also changes representation statistics, D_m and I_MP should not be interpreted as pure measures of dependence on intact linguistic structure.
```

---

# 三、两个图之间的逻辑关系

这两个图建议在论文中这样安排：

## Figure 17：先展示结果本身

Figure 17 回答：

> 打乱上下文后，Context Gain 和模型差异发生了什么？

它展示的是结果层面：

- normal 和 C1 的 `Δr_total`；
- 每个模型损失多少 `D_m`；
- Mamba−Pythia 差异是否变化；
- `I_MP` 是否支持 Mamba 优势依赖完整语言历史。

## Figure 18：再解释为什么要谨慎

Figure 18 回答：

> 为什么 Figure 17 不能被简单解释为“谁更依赖真实语言结构”？

它展示的是诊断层面：

- C1 改变了有效维度；
- C1 改变了时间平滑度；
- 这些变化随 H 增强；
- 因此 C1 不只是破坏语言意义，也改变了表示统计。

所以两张图的章节顺序建议是：

```text
4.7.2 Stimulus-Side Context Control

Paragraph 1: 说明 C1 设计目的
Figure 17: 展示 Context Gain 和 architecture contrasts 的变化
Paragraph 2: 解释 Figure 17 的核心结果
Figure 18: 展示 lag-1 和 OOD 诊断
Paragraph 3: 解释为什么该控制只能作为 diagnostic evidence
```

---

# 四、最终推荐标题汇总

## Figure 17

### 推荐标题

```text
Effects of Stimulus-Side Context Perturbation on Total Context Gain
```

### 更简洁版本

```text
Stimulus-Side Context Perturbation and Total Context Gain
```

### 更方法化版本

```text
Stimulus-Side Context Control under Normal and Shuffled Contexts
```

---

## Figure 18

### 推荐标题

```text
Representation-Statistic Diagnostics under Stimulus-Side Context Perturbation
```

### 更简洁版本

```text
Representation Statistics under Normal and Shuffled Contexts
```

### 更强调诊断作用的版本

```text
Diagnostic Representation Statistics for the Shuffled-Context Control
```

---

# 五、最推荐的最终命名方案

如果你想让标题风格统一，建议使用：

```text
Figure 17. Effects of Stimulus-Side Context Perturbation on Total Context Gain
```

```text
Figure 18. Representation-Statistic Diagnostics under Stimulus-Side Context Perturbation
```

这两个标题的优点是：

1. 都使用 `Stimulus-Side Context Perturbation`，风格统一；
2. Figure 17 强调结果量 `Total Context Gain`；
3. Figure 18 强调诊断量 `Representation Statistics`；
4. 不会提前暗示结果一定是 gain loss、advantage 或 degradation；
5. 适合你现在的结果，也适合未来如果结果方向变化的情况。

---

# 六、绘图代码修改清单

## Figure 17

- [ ] 总标题改为 `Effects of Stimulus-Side Context Perturbation on Total Context Gain`
- [ ] Panel C 和 Panel D 互换
- [ ] `ctx1` 在图例中改为 `C1` 或 `C1 shuffled`
- [ ] “advantage” 改为 “contrast”
- [ ] “gain lost” 改为 “context-perturbation effect”
- [ ] Y 轴统一写 `95% CI`
- [ ] Panel A Y 轴写 `Total Context Gain, Δr_total = r_128 − r_8 (95% CI)`
- [ ] Panel B Y 轴写 `D_m = Δr_total^normal − Δr_total^C1 (95% CI)`
- [ ] Panel D Y 轴写 `I = A^normal − A^C1 (95% CI)`
- [ ] 图注标明 `diagnostic, uncorrected, not included in the confirmatory family`

## Figure 18

- [ ] 总标题改为 `Representation-Statistic Diagnostics under Stimulus-Side Context Perturbation`
- [ ] “degradation” 从主标题中删除
- [ ] Panel A 标题保留 `Effective dimensionality`
- [ ] Panel A X 轴改为 `Context Length H (raw words)`
- [ ] Panel A Y 轴保留 `Participation ratio (median over stories)`
- [ ] Panel B 标题保留 `Temporal smoothness`
- [ ] Panel B X 轴改为 `Context Length H (raw words)`
- [ ] Panel B Y 轴改为 `Lag-1 cosine similarity between adjacent TRs`
- [ ] Panel C 标题改为 `H-dependent shift in effective dimensionality`
- [ ] Panel C 不要写 `all three CIs exclude zero`
- [ ] Panel D 标题改为 `Descriptive relation between dimensionality shift and gain reduction`
- [ ] Panel D 图内保留 `n = 3 models; descriptive comparison only; no statistical test`
- [ ] 不要在图标题中写过强结论，例如 `why D_m and I_MP cannot be read as evidence...`
