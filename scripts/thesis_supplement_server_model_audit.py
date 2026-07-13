"""
硕士论文核查清单 —— 第3节剩余缺口：RWKV/AWD-LSTM 精确参数量 + AWD-LSTM 真实
embedding 词表大小 + xxunk 数量/比例。

**为什么本机做不了、必须在服务器跑**：pythia/mamba 两个模型的精确参数量已经从
HuggingFace safetensors 元数据直接精确读到（见 thesis_supplement_model_metadata.py，
纯 API 调用，不下载权重），但 RWKV 仓库没有 safetensors 清单，AWD-LSTM 根本不在
HF 上——这两个的精确参数量唯一可靠的来源是**真的把权重载入一次、`sum(p.numel()
for p in model.parameters())`**。AWD-LSTM 的词表大小/xxunk 比例同理，需要真的解开
WT103 的 itos.pkl 词表文件。本机没有 torch/fastai，服务器已经装好且模型已缓存
（M1 特征提取时下载过），这里只是**读取已加载对象的元数据，不做任何新的语言模型
推理或 fMRI 相关计算**——不碰 GPU、不碰 Ridge/bootstrap、不产生任何新的 brain
score，CPU 上运行预计数十秒到几分钟（主要是 AWD-LSTM 权重从磁盘反序列化的时间）。

按项目规矩（新计算一律先写脚本、交给用户确认后在服务器上跑，我不擅自执行），
本脚本只是"写好、等你确认"，不是我已经跑过的产物。

用法（在服务器项目根目录，deepfmri 或装了 torch+transformers+fastai 的环境下）：
  python3 scripts/thesis_supplement_server_model_audit.py
输出：results/thesis_supplement/model_param_audit.json（结构化，供本机
thesis_supplement_model_metadata.py 合并进最终 model_metadata.csv）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config   # noqa: E402


def count_rwkv_params() -> dict:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-4-169m-pile")
    n_params = sum(p.numel() for p in model.parameters())
    return {"model_id": "RWKV/rwkv-4-169m-pile", "exact_param_count": int(n_params),
           "source": "sum(p.numel() for p in model.parameters()) after "
                     "AutoModelForCausalLM.from_pretrained（真实载入权重求和）"}


def count_awd_lstm_params_and_vocab(word_index_path: Path) -> dict:
    import pickle
    import torch
    import pandas as pd
    from fastai.text.all import AWD_LSTM, URLs, untar_data

    path = untar_data(URLs.WT103_FWD)
    wgt_path = next(path.glob("*.pth"))
    itos_path = next(path.glob("*.pkl"))
    with open(itos_path, "rb") as f:
        vocab = pickle.load(f)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    unk_id = stoi.get("xxunk", 0)

    vocab_sz = len(vocab)
    encoder = AWD_LSTM(vocab_sz, emb_sz=400, n_hid=1152, n_layers=3)
    wgts = torch.load(wgt_path, map_location="cpu")
    state = wgts.get("model", wgts)
    enc_state = {k.replace("0.encoder.", "").replace("encoder.", ""): v
                for k, v in state.items() if "decoder" not in k}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    n_params = sum(p.numel() for p in encoder.parameters())

    # xxunk 比例：对 word_index.parquet 里全部原始词（跟 M1 特征提取时用的是
    # 同一份词表/同一套小写+查表规则，直接复用 awd_lstm_adapter.py 的逻辑）
    word_index = pd.read_parquet(word_index_path)
    words = word_index["word"].tolist()
    is_unk = [stoi.get(w.lower(), unk_id) == unk_id for w in words]
    xxunk_count = int(sum(is_unk))
    xxunk_rate = xxunk_count / len(words) if words else 0.0

    return {
        "model_id": "fastai_AWD_LSTM_WT103_FWD",
        "exact_param_count": int(n_params),
        "embedding_vocab_size": vocab_sz,
        "xxunk_count": xxunk_count,
        "xxunk_rate": xxunk_rate,
        "n_words_checked": len(words),
        "encoder_load_missing_keys": list(missing),
        "encoder_load_unexpected_keys": list(unexpected),
        "source": "sum(p.numel() for p in encoder.parameters()) + itos.pkl 词表长度 + "
                  "对 frozen/word_index.parquet 全部原始词（158598个）逐词小写查表统计",
    }


def main():
    cfg = load_config()
    out_dir = Path(cfg["paths"]["results_dir"]) / "thesis_supplement"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] RWKV 精确参数量（transformers.AutoModelForCausalLM，只读结构，无推理）...")
    rwkv = count_rwkv_params()
    print(f"  {rwkv}")

    print("\n[2/2] AWD-LSTM 精确参数量 + 词表 + xxunk 比例...")
    word_index_path = Path(cfg["paths"]["frozen_dir"]) / "word_index.parquet"
    awd = count_awd_lstm_params_and_vocab(word_index_path)
    print(f"  参数量={awd['exact_param_count']}, 词表={awd['embedding_vocab_size']}, "
         f"xxunk={awd['xxunk_count']}/{awd['n_words_checked']} "
         f"({awd['xxunk_rate']:.4%})")

    result = {"rwkv": rwkv, "awd_lstm": awd}
    out_path = out_dir / "model_param_audit.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n已写 {out_path}")


if __name__ == "__main__":
    main()
