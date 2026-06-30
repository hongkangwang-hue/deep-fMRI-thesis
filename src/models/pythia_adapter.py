"""
Pythia-160M 适配器（Transformer / GPTNeoX）。

依赖 transformers + torch（仅 AutoDL）。本文件本地可 import 结构，但 load()/
forward 需 GPU 环境。

层号约定（必须在 AutoDL 上核验，off-by-one 风险点）：
  transformers 的 output_hidden_states 返回长度 n_layers+1 的 tuple：
    hidden_states[0]   = embedding 层输出
    hidden_states[k]   = 第 k 个 transformer block 之后的输出（k=1..n_layers）
  配置中的层号为 0-based block 索引（pythia 共 12 个 block，索引 0..11）。
  因此 0-based block b 的输出 = hidden_states[b + 1]。
    主层  pythia=8  → hidden_states[9]
    最终层 pythia=11 → hidden_states[12]（最后一层）
  load() 后用 _assert_layer_convention() 对真实模型核验。

不复用窗口外 KV cache：每个窗口都是独立 forward（use_cache=False），天然无
跨窗状态。
"""

from __future__ import annotations

import numpy as np

from .base import LayerSpec, ModelAdapter


class PythiaAdapter(ModelAdapter):
    is_recurrent = False

    def __init__(self, model_id: str = "EleutherAI/pythia-160m",
                 revision: str = "main", device: str = "cuda"):
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
            f"疑似 instruction/chat 微调模型: {self.model_id}"

    def _assert_layer_convention(self) -> None:
        """核验 hidden_states 长度 == n_layers + 1，确认层号约定成立。"""
        import torch
        with torch.no_grad():
            ids = self.tokenizer("hello world", return_tensors="pt").to(self.device)
            out = self.model(**ids)
        n_hs = len(out.hidden_states)
        assert n_hs == self.n_layers + 1, (
            f"hidden_states 长度 {n_hs} != n_layers+1 ({self.n_layers + 1})，"
            f"层号约定需重新核对"
        )

    def reset_state(self) -> None:
        # Transformer 无跨窗状态；每次 forward 独立，no-op。
        pass

    def tokenize_with_spans(self, words):
        """逐词编码并记录每个词的 subtoken 跨度。

        采用「逐词分别编码再拼接」以获得明确的词→subtoken 边界。第一个词正常
        编码，后续词加前导空格以匹配 GPT-NeoX BPE 的空格语义。

        词一律小写：原始 TextGrid 标注为全大写，而原始 eng1000 流程的词流
        （make_word_ds → DataSequence）为小写。已验证 word.lower() 与该词流
        逐词精确相等。喂大写会让 BPE 把每个词切成大量碎 subtoken，产生退化
        表示，且与 eng1000 基线、AWD-LSTM 词流不一致。
        """
        token_ids: list[int] = []
        spans: list[tuple[int, int]] = []
        is_unk: list[bool] = []
        for k, w in enumerate(words):
            wl = w.lower()
            text = wl if k == 0 else " " + wl
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 0:  # 极端情况：空编码，退回 unk 占位
                ids = [self.tokenizer.unk_token_id or self.tokenizer.eos_token_id]
            start = len(token_ids)
            token_ids.extend(ids)
            spans.append((start, len(token_ids)))
            is_unk.append(False)  # BPE 无真正 OOV
        return token_ids, spans, is_unk

    def forward_hidden(self, token_ids, layers: LayerSpec):
        import torch
        idx = torch.tensor([token_ids], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=idx, use_cache=False)
        hs = out.hidden_states  # tuple, [0]=emb, [b+1]=block b 输出
        result = {}
        for layer in {layers.main, layers.final}:
            result[layer] = hs[layer + 1][0].float().cpu().numpy()
        return result

    def forward_hidden_batch(self, token_id_lists, layers: LayerSpec):
        from .base import hf_forward_hidden_batch
        return hf_forward_hidden_batch(
            self.model, self.device, token_id_lists, layers
        )
