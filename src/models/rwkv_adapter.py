"""
RWKV-4-Pile-169M 适配器（RNN/线性注意力，via transformers RwkvModel）。

依赖 transformers + torch（仅 AutoDL）。RWKV 是循环模型：transformers 实现
在一次 forward 中按序处理整个窗口，state 缺省从零开始（不传入上一窗 state），
因此「每窗冷启动」是默认行为——reset_state 为 no-op，但状态重置测试仍须验证
「故意传入上一窗 state 会得到不同输出」（在测试代码里构造，不在正式提取路径）。

层号约定（AutoDL 核验）：与 transformers 通用约定一致——output_hidden_states
返回 n_layers+1 个张量，[0]=embedding，[b+1]=第 b 个 block（0-based）之后。
RWKV-4-169m 共 12 层，hidden 768。
    主层 rwkv=8 → hidden_states[9]；最终层 rwkv=11 → hidden_states[12]
"""

from __future__ import annotations

import numpy as np

from .base import LayerSpec, ModelAdapter


class RWKVAdapter(ModelAdapter):
    is_recurrent = True

    def __init__(self, model_id: str = "RWKV/rwkv-4-169m-pile",
                 revision: str = "main", device: str = "cuda"):
        # 注意：HF Hub 上的官方转换权重为 RWKV/rwkv-4-169m-pile。
        # 若使用 BlinkDL 原始权重需另行转换；revision 在 AutoDL 上锁定。
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
        self.hidden_width = getattr(cfg, "hidden_size", None) or cfg.attention_hidden_size
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
            f"RWKV hidden_states 长度 {n_hs} != n_layers+1 "
            f"({self.n_layers + 1})，层号约定需核对"
        )

    def reset_state(self) -> None:
        # 正式提取路径不传入上一窗 state，天然冷启动；no-op。
        pass

    def tokenize_with_spans(self, words):
        token_ids, spans, is_unk = [], [], []
        for k, w in enumerate(words):
            text = w if k == 0 else " " + w
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
            # 不传 state → 冷启动；use_cache=False 不保留跨步缓存
            out = self.model(input_ids=idx, use_cache=False)
        hs = out.hidden_states
        result = {}
        for layer in {layers.main, layers.final}:
            result[layer] = hs[layer + 1][0].float().cpu().numpy()
        return result
