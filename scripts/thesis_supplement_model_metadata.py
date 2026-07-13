"""
硕士论文核查清单 —— 第3节：model_metadata.csv。

数据来源分三类：
  1. 已由真实 M4 运行记录在 figures/<subject>/tables/table1_checkpoint_audit.csv 里的
     "代码实际用到的" model_id / revision / 层索引（三名被试完全一致，任取一份）——
     这是运行时真值，不是 config.yaml 里写的计划值（两者对 rwkv/mamba 并不相同，见下）。
  2. 本机对 HuggingFace 的只读网络请求（config.json/tokenizer.json/API，不下载权重，
     不需要 torch/transformers）：hidden_size、层数、tokenizer 真实词表大小、当前
     commit sha、以及 pythia/mamba 两个模型可从 safetensors 元数据精确读到的参数量。
  3. RWKV 与 AWD-LSTM 的精确参数量、AWD-LSTM 的真实 embedding 词表大小与 xxunk
     比例——这三项没有 safetensors 清单可读，需要实际把模型权重载入一次求和/
     过词表，本机没有 torch/fastai，需要在服务器上跑一次（已实现，见
     scripts/thesis_supplement_server_model_audit.py），此处先留空并如实标注。

用法：python3 scripts/thesis_supplement_model_metadata.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config   # noqa: E402

OUT_DIR = PROJECT_ROOT / "thesis_supplement"

# 只对确实是 HF Hub 模型的三个核心模型发只读请求；AWD-LSTM 不在 HF 上，跳过
HF_MODELS = {
    "pythia": "EleutherAI/pythia-160m",
    "mamba": "state-spaces/mamba-130m-hf",
    "rwkv": "RWKV/rwkv-4-169m-pile",
}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def fetch_hf_metadata(model_id: str) -> dict:
    """只读 config.json + tokenizer.json + Hub API，不下载权重。"""
    out = {}
    cfg = _get_json(f"https://huggingface.co/{model_id}/raw/main/config.json")
    out["hidden_size"] = cfg.get("hidden_size") or cfg.get("d_model")
    out["n_layers"] = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    out["embedding_vocab_size"] = cfg.get("vocab_size")  # padded，非真实 tokenizer 词表数
    try:
        tok = _get_json(f"https://huggingface.co/{model_id}/raw/main/tokenizer.json")
        out["tokenizer_real_vocab_size"] = len(tok.get("model", {}).get("vocab", {}))
    except Exception as e:
        out["tokenizer_real_vocab_size"] = None
        print(f"  [warn] {model_id} tokenizer.json 读取失败：{e}")
    try:
        info = _get_json(f"https://huggingface.co/api/models/{model_id}")
        out["current_main_sha"] = info.get("sha")
        st = info.get("safetensors")
        if st and len(st.get("parameters", {})) == 1:
            # 只有一种权重格式（如 mamba 只有 F32）时，total 就是精确参数量；
            # pythia 仓库同时放了 F16 主权重与一个不相关的 U8 量化文件，total 会
            # 把两者相加造成假象，需要单独取 F16 那一项。
            out["exact_param_count"] = st.get("total")
        elif st and "F16" in st.get("parameters", {}):
            out["exact_param_count"] = st["parameters"]["F16"]
        else:
            out["exact_param_count"] = "NEEDS_SERVER"
    except Exception as e:
        out["current_main_sha"] = None
        out["exact_param_count"] = None
        print(f"  [warn] {model_id} Hub API 读取失败：{e}")
    return out


def load_table1(cfg: dict) -> pd.DataFrame:
    """三名被试的 table1_checkpoint_audit.csv 理应完全一致（层索引与被试无关），
    这里读 UTS03 那份并做一次跨被试一致性核对，避免误用某个被试的特例。"""
    figs = Path(cfg["paths"]["figures_dir"])
    dfs = {}
    for subj in ("UTS01", "UTS02", "UTS03"):
        p = figs / subj / "tables" / "table1_checkpoint_audit.csv"
        if p.exists():
            dfs[subj] = pd.read_csv(p).set_index("model")
    if not dfs:
        raise SystemExit("找不到任何 figures/<subject>/tables/table1_checkpoint_audit.csv，"
                         "请先确认已从服务器同步 figures/ 到本地。")
    subjects = list(dfs)
    base = dfs[subjects[0]]
    for s in subjects[1:]:
        cols = ["model_id", "revision", "primary_layer_idx", "robustness_layer_idx",
               "layer_idx_in_main_cell", "layer_idx_in_final_cell"]
        if not base[cols].equals(dfs[s][cols]):
            print(f"  [警告] {subjects[0]} 与 {s} 的 table1 不完全一致，"
                 "说明层索引/model_id 并非真正跨被试恒定，需要人工核查！")
    print(f"  跨被试一致性核对：使用 {subjects} 中的 {subjects[0]}（"
         f"{'一致' if len(subjects) > 1 else '只有一份可核对'}）")
    return base


def main():
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/2] 读取真实运行记录 table1_checkpoint_audit.csv ...")
    t1 = load_table1(cfg)

    print("[2/2] 只读网络请求 HF config.json / tokenizer.json / API（不下载权重）...")
    rows = []
    for short_name in ("pythia", "rwkv", "mamba", "awd_lstm"):
        row = {"model": short_name}
        actual_id = t1.loc[short_name, "model_id"]
        row["model_id_actually_used"] = actual_id
        row["model_id_in_config_yaml"] = {
            "pythia": cfg["models"]["transformer"],
            "rwkv": cfg["models"]["rwkv"],
            "mamba": cfg["models"]["mamba"],
            "awd_lstm": cfg["models"]["lstm"],
        }[short_name]
        row["config_yaml_matches_actual"] = (row["model_id_actually_used"]
                                             == row["model_id_in_config_yaml"])
        row["primary_layer_idx"] = int(t1.loc[short_name, "primary_layer_idx"])
        row["robustness_layer_idx"] = int(t1.loc[short_name, "robustness_layer_idx"])
        row["layer_idx_in_main_cell"] = int(t1.loc[short_name, "layer_idx_in_main_cell"])
        row["layer_idx_in_final_cell"] = int(t1.loc[short_name, "layer_idx_in_final_cell"])
        row["state_reset_per_window"] = cfg["models"]["state_reset_per_window"]

        if short_name in HF_MODELS:
            print(f"  拉取 {HF_MODELS[short_name]} ...")
            meta = fetch_hf_metadata(HF_MODELS[short_name])
            row.update({
                "hidden_size": meta["hidden_size"],
                "n_layers_total": meta["n_layers"],
                "embedding_vocab_size_padded": meta["embedding_vocab_size"],
                "tokenizer_real_vocab_size": meta["tokenizer_real_vocab_size"],
                "tokenizer_name": "GPTNeoXTokenizer (shared BPE, 50254 real tokens, "
                                  "20B-tokenizer)",
                "revision_current_main_sha": meta["current_main_sha"],
                "exact_param_count": meta["exact_param_count"],
                "param_count_source": ("HF safetensors metadata（精确，来自权重文件"
                                       "张量形状，非估算）"
                                       if isinstance(meta["exact_param_count"], int)
                                       else "NEEDS_SERVER：该仓库无 safetensors 清单，"
                                       "需要在服务器上实际载入权重求和"),
                "xxunk_count": "N/A（仅 AWD-LSTM 词级模型概念适用）",
                "xxunk_rate": "N/A",
            })
        else:  # awd_lstm —— 不在 HF 上，config.json 概念不适用
            row.update({
                "hidden_size": "1152（主层，src/models/awd_lstm_adapter.py 架构常量 "
                               "n_hid）/ 400（最终层 rnns[2] 输出宽度）/ 400（emb_sz）",
                "n_layers_total": 3,
                "embedding_vocab_size_padded": "NEEDS_SERVER（WT103 itos.pkl 实际词表长度，"
                                               "本机无 fastai 无法本地读取）",
                "tokenizer_real_vocab_size": "NEEDS_SERVER（同上，词级模型自建词表，"
                                             "非 HF tokenizer）",
                "tokenizer_name": "fastai WT103 word-level vocab（OOV→xxunk），非 BPE",
                "revision_current_main_sha": "fastai_URLs.WT103_FWD（非 HF repo，无 git sha；"
                                             "fastai 用固定 URL 分发单一版本权重，无 revision "
                                             "概念）",
                "exact_param_count": "NEEDS_SERVER",
                "param_count_source": "NEEDS_SERVER：无 safetensors 清单，需服务器实际载入 "
                                       "AWD_LSTM(vocab_sz, emb_sz=400, n_hid=1152, n_layers=3) "
                                       "求 sum(p.numel())",
                "xxunk_count": "NEEDS_SERVER",
                "xxunk_rate": "NEEDS_SERVER",
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "model_metadata.csv", index=False)
    print(f"\n已写 {OUT_DIR / 'model_metadata.csv'}")

    mismatches = df[~df["config_yaml_matches_actual"].fillna(True)]
    if len(mismatches):
        print("\n[重要] 以下模型 config.yaml 里声明的 model_id 与代码实际运行时用的不同"
             "（真值以 table1_checkpoint_audit.csv / 实际 M4 cell 元数据为准）：")
        for _, r in mismatches.iterrows():
            print(f"  {r['model']}: config.yaml写 {r['model_id_in_config_yaml']!r}，"
                 f"实际用 {r['model_id_actually_used']!r}")
    needs_server = df[df["exact_param_count"].astype(str).str.contains("NEEDS_SERVER")]
    if len(needs_server):
        print(f"\n[待补] {list(needs_server['model'])} 的精确参数量/AWD-LSTM 词表/xxunk 比例"
             "本机无法算，需服务器跑 scripts/thesis_supplement_server_model_audit.py。")


if __name__ == "__main__":
    main()
