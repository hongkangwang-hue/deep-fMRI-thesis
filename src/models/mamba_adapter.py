"""
Mamba-130M 适配器（状态空间模型 SSM，via transformers MambaForCausalLM）。

依赖 transformers + torch（+ 可选 mamba-ssm/causal-conv1d 加速核，仅 AutoDL）。
Mamba 在一次 forward 中按序扫过整个窗口，SSM latent 缺省从零状态开始，不跨窗
复用 → 「每窗冷启动」为默认；reset_state 为 no-op。状态重置的正反测试在测试
代码中通过传入/不传入 cache_params 构造。

不分析 SSM latent 本身（明确不做项），只取各层 hidden_states。

层号约定（AutoDL 核验）：output_hidden_states 返回 n_layers+1 个张量，
[0]=embedding，[b+1]=第 b 个 block（0-based）之后。Mamba-130m 共 24 层，
hidden 768。
    主层 mamba=16 → hidden_states[17]；最终层 mamba=23 → hidden_states[24]
"""

from __future__ import annotations

import numpy as np

from .base import LayerSpec, ModelAdapter


class MambaAdapter(ModelAdapter):
    is_recurrent = True

    def __init__(self, model_id: str = "state-spaces/mamba-130m-hf",
                 revision: str = "main", device: str = "cuda"):
        # transformers 兼容权重为 state-spaces/mamba-130m-hf。
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.model = None
        self.tokenizer = None
        self.hidden_width = None
        self.n_layers = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision,
            output_hidden_states=True, torch_dtype=torch.float32,
        ).to(self.device).eval()

        cfg = self.model.config
        self.hidden_width = cfg.hidden_size
        self.n_layers = cfg.num_hidden_layers
        self._assert_no_chat_tuning()
        self._assert_layer_convention()

    def _assert_no_chat_tuning(self) -> None:
        name = self.model_id.lower()
        assert not any(t in name for t in ("instruct", "chat", "sft", "rlhf")), \
            f"疑似微调模型: {self.model_id}"

    def _assert_layer_convention(self) -> None:
        import torch
        with torch.no_grad():
            ids = self.tokenizer("hello world", return_tensors="pt").to(self.device)
            out = self.model(**ids)
        n_hs = len(out.hidden_states)
        assert n_hs == self.n_layers + 1, (
            f"Mamba hidden_states 长度 {n_hs} != n_layers+1 "
            f"({self.n_layers + 1})，层号约定需核对"
        )

    def reset_state(self) -> None:
        # 正式提取路径不传入 cache_params → 冷启动；no-op。
        pass

    def tokenize_with_spans(self, words):
        # 词一律小写以匹配 eng1000 词流（make_word_ds→DataSequence 为小写）并
        # 避免全大写词被 BPE 切碎；见 pythia_adapter 说明。
        token_ids, spans, is_unk = [], [], []
        for k, w in enumerate(words):
            wl = w.lower()
            text = wl if k == 0 else " " + wl
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 0:
                ids = [self.tokenizer.unk_token_id or 0]
            start = len(token_ids)
            token_ids.extend(ids)
            spans.append((start, len(token_ids)))
            is_unk.append(False)
        return token_ids, spans, is_unk

    def forward_hidden(self, token_ids, layers: LayerSpec):
        import torch
        idx = torch.tensor([token_ids], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=idx, use_cache=False)
        hs = out.hidden_states
        result = {}
        for layer in {layers.main, layers.final}:
            result[layer] = hs[layer + 1][0].float().cpu().numpy()
        return result
