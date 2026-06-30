# results/ — 实验结果目录约定

本目录存放所有编码实验的输出。**内容不进 git**（大文件，可由代码+冻结 spec 重新生成），
仅本说明文件随仓库走。`config/config.yaml: paths.results_dir = results` 是唯一真相源，
所有脚本输出都归到这里。

## 目录命名约定

```
results/<run-name>/<subject>/
```

| run-name | 来源脚本 | 说明 |
|---|---|---|
| `eng1000/` | LeBel 原始（外部参考） | **参考产物，勿改**。Phase 1 对照基准 |
| `eng1000_gpu_rerun/` | `scripts/m2c_phase1_gpu.py` | M2-C Phase 1 GPU 重算（已 PASS） |
| `eng1000_native_rerun/` | `scripts/m2c_phase1_native.py` | M2-C Phase 1 CPU 重算（备用） |
| `m3_<model>_H<H>_<layer>/` | `scripts/m3_vertical_slice.py` | M3 垂直切片，如 `m3_pythia_H32_main` |

## 每个结果目录的标准文件

| 文件 | 内容 |
|---|---|
| `corrs.npz` / `voxel_r.npz` | `arr_0` = 每体素相关 `(95556,)` |
| `valphas.npz` / `fold_valphas.npz` | 每体素选定 alpha/lambda |
| `run_manifest.json` | **检索钩子**：模型/H/层/种子/solver/指标/spec 版本/对齐方案 |

## 日后怎么找实验

每个目录都有 `run_manifest.json`，按字段 grep 即可定位，例如：

```bash
# 列出所有 run 的关键指标
for m in results/*/*/run_manifest.json; do
  echo "== $m =="; python -c "import json,sys;d=json.load(open('$m'));print({k:d.get(k) for k in ('model','H','layer','seed','solver','voxel_r_mean','roi_mean_r')})"
done

# 找某个模型/H 的结果
grep -l '"H": 32' results/*/*/run_manifest.json
```

manifest 里 `spec` 字段指向所用的冻结 spec（`frozen/analysis_spec.yaml` 或
`frozen/m2c_reference_validation.yaml`），`alignment` 字段记录对齐方案（如 `plan_A_target_words_only`）。
