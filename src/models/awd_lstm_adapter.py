"""
AWD-LSTM 适配器（fastai WikiText-103 Forward，约 44M 参数，历史参照）。

⚠️ 高风险模块：设 3 个工作日风险预算。若在 AutoDL 上无法通过加载/状态重置/
H=128 稳定运行，须提交 scope amendment（修改题目/RQ/图表/结论边界），不得
口头降级为 Future Work（见冻结文档 awd_lstm_policy）。

依赖 fastai + torch（仅 AutoDL）。本文件本地可 import 结构，load()/forward 需库。

模型结构（AWD_LSTM 默认）：emb_sz=400, n_hid=1152, n_layers=3。
  rnns[0]: 400→1152   rnns[1]: 1152→1152   rnns[2]: 1152→400
配置层号（0-based recurrent layer）：
  主层 awd_lstm=1 → 第 2 个 LSTM 层 raw hidden，宽度 1152
  最终层 awd_lstm=2 → 第 3 个 LSTM 层 raw hidden，宽度 400
取 raw hidden（不取 decoder logits）。

词→token 映射：AWD-LSTM 为词级模型。每个原始词小写后在 WT103 词表查找，
OOV → xxunk（仅记录、不删除目标词，见冻结策略）。故每词恰好 1 个 token，
span 平凡为 (k, k+1)。该词级映射策略须在 AutoDL 上对照 fastai 分词核验。

状态重置：每个窗口 forward 前调用 model.reset()，清空各层 hidden/cell。
状态重置正反测试：冷启动两次应一致；故意继承上一窗 hidden 应不同。
"""

from __future__ import annotations

import numpy as np

from .base import LayerSpec, ModelAdapter


class AWDLSTMAdapter(ModelAdapter):
    is_recurrent = True

    def __init__(self, device: str = "cuda",
                 main_layer: int = 1, final_layer: int = 2):
        self.model_id = "fastai_AWD_LSTM_WT103_FWD"
        self.revision = "fastai_URLs.WT103_FWD"
        self.device = device
        self.encoder = None       # AWD_LSTM 模块
        self.vocab = None         # itos 列表
        self.stoi = None          # {token: id}
        self.unk_id = None
        self.hidden_width = 1152  # 主层宽度
        self.n_layers = 3
        self._main_layer = main_layer
        self._final_layer = final_layer
        self._captured = {}       # 前向 hook 捕获的逐层输出

    def load(self) -> None:
        import pickle
        import torch
        from fastai.text.all import AWD_LSTM, URLs, untar_data

        path = untar_data(URLs.WT103_FWD)
        # 目录内含权重 .pth 与 itos .pkl（文件名随 fastai 版本，AutoDL 上确认）
        wgt_path = next(path.glob("*.pth"))
        itos_path = next(path.glob("*.pkl"))
        with open(itos_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.unk_id = self.stoi.get("xxunk", 0)

        vocab_sz = len(self.vocab)
        self.encoder = AWD_LSTM(vocab_sz, emb_sz=400, n_hid=1152, n_layers=3)
        wgts = torch.load(wgt_path, map_location="cpu")
        state = wgts.get("model", wgts)
        # 只取 encoder 权重（去掉 decoder/0. 前缀差异在 AutoDL 上对齐）
        enc_state = {k.replace("0.encoder.", "").replace("encoder.", ""): v
                     for k, v in state.items() if "decoder" not in k}
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
        # 在 AutoDL 上断言 missing/unexpected 仅为 decoder 相关，否则报错
        self.encoder = self.encoder.to(self.device).eval()
        self._register_hooks()

    def _register_hooks(self):
        """在每个 LSTM 层注册前向 hook，捕获其 raw hidden 输出。"""
        self._captured = {}

        def make_hook(idx):
            def hook(_module, _inp, out):
                # fastai RNN 层输出 (output, hidden)；取 output
                t = out[0] if isinstance(out, tuple) else out
                self._captured[idx] = t.detach()
            return hook

        for k, rnn in enumerate(self.encoder.rnns):
            rnn.register_forward_hook(make_hook(k))

    def reset_state(self) -> None:
        if self.encoder is not None:
            self.encoder.reset()

    def tokenize_with_spans(self, words):
        """词级映射：每词小写查表，OOV→xxunk；每词恰好 1 token。"""
        token_ids, spans, is_unk = [], [], []
        for k, w in enumerate(words):
            tok = w.lower()
            tid = self.stoi.get(tok, self.unk_id)
            token_ids.append(tid)
            spans.append((k, k + 1))
            is_unk.append(tid == self.unk_id)
        return token_ids, spans, is_unk

    def forward_hidden(self, token_ids, layers: LayerSpec):
        import torch
        self._captured = {}
        idx = torch.tensor([token_ids], device=self.device)
        with torch.no_grad():
            _ = self.encoder(idx)
        result = {}
        for layer in {layers.main, layers.final}:
            t = self._captured[layer]          # (1, seq, width)
            result[layer] = t[0].float().cpu().numpy()
        return result

    def forward_hidden_batch(self, token_id_lists, layers: LayerSpec):
        """右侧 padding 批量前向。LSTM 严格左→右，pad 在目标之后不影响其
        hidden。pad_id=0；reset_state() 已由 extract_batch 在前向前调用，故每个
        batch 元素从零状态独立开始（batch 维彼此独立）。

        ⚠️ AutoDL 上需核验：fastai AWD_LSTM 跨调用缓存 hidden，要求 reset() 后
        按当前 batch 大小重新初始化；末批 batch 较小时务必确认 reset 生效。
        """
        import torch
        n = len(token_id_lists)
        lengths = [len(t) for t in token_id_lists]
        max_len = max(lengths)
        idx = torch.zeros((n, max_len), dtype=torch.long)
        for b, t in enumerate(token_id_lists):
            idx[b, : lengths[b]] = torch.tensor(t, dtype=torch.long)
        idx = idx.to(self.device)
        self._captured = {}
        with torch.no_grad():
            _ = self.encoder(idx)
        result = {}
        for layer in {layers.main, layers.final}:
            t = self._captured[layer]          # (batch, max_len, width)
            result[layer] = t.float().cpu().numpy()
        return result

    def xxunk_rate(self, words) -> float:
        """该词表下的 xxunk 比例（用于 AWD-LSTM 专项报告）。"""
        _, _, is_unk = self.tokenize_with_spans(words)
        return float(np.mean(is_unk)) if is_unk else 0.0
