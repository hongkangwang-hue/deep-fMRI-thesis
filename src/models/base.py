"""
M1 — 模型适配器抽象接口（四模型共用）

四个模型（Pythia / RWKV / Mamba / AWD-LSTM）实现同一接口，使得开窗、状态
重置、目标词 pooling 和缓存逻辑完全一致，差异只封装在各自的 load /
tokenize / forward 内。

关键约定：
  - 状态重置：每个窗口在 forward 前调用 reset_state()；Transformer 不复用
    窗口外 KV cache（每窗独立 forward 即可，天然无跨窗 cache）。
  - 目标词表示：取目标词（窗口最后一个词）的「最后一个 subtoken」的 hidden
    （last-subtoken pooling）。
  - 层号 → hidden 索引：见各适配器 forward_hidden 的实现说明。配置里的层号
    在 AutoDL 上必须对着真实模型核验后才能定稿（off-by-one 风险点）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .windowing import build_window


@dataclass(frozen=True)
class LayerSpec:
    """要提取的两层：主层与最终（稳健）层，按配置中的 0-based 层号。"""
    main: int
    final: int


@dataclass
class WindowRepr:
    """单个目标词在某 H 下的双层表示与诊断信息。"""
    main: np.ndarray            # shape (hidden_main,)
    final: np.ndarray           # shape (hidden_final,)；宽度可与 main 不同
    n_tokens: int               # 该窗口 tokenize 后的 token 数
    target_token_index: int     # 目标词最后一个 subtoken 在序列中的位置
    n_target_subtokens: int = 1 # 目标词被切成几个 subtoken（BPE 可 >1）
    is_unk: bool = False        # 目标词是否被映射为 unk（AWD-LSTM xxunk）


class ModelAdapter(ABC):
    """四模型统一前向接口。子类只实现 load / reset_state / tokenize /
    forward_hidden；extract() 在基类中统一，保证开窗与 pooling 完全一致。"""

    model_id: str
    revision: str
    hidden_width: int
    n_layers: int
    is_recurrent: bool  # RWKV/Mamba/AWD-LSTM=True；Transformer=False

    # ── 子类必须实现 ──────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """加载 checkpoint，固定 revision，核验非 instruction/chat 微调。"""

    @abstractmethod
    def reset_state(self) -> None:
        """清空所有跨步状态（RWKV/Mamba 的 recurrent state、LSTM 的
        hidden/cell）。Transformer 为 no-op。"""

    @abstractmethod
    def tokenize_with_spans(
        self, words: list[str]
    ) -> tuple[list[int], list[tuple[int, int]], list[bool]]:
        """把 H+1 个原始词编码为 token 序列。

        Returns:
            token_ids: 该窗口的 token id 列表。
            spans: 长度等于词数；spans[k] = (start, end) 表示第 k 个词对应
                token_ids[start:end]。
            is_unk: 长度等于词数；该词是否被映射为 unk。
        """

    @abstractmethod
    def forward_hidden(
        self, token_ids: list[int], layers: LayerSpec
    ) -> dict[int, np.ndarray]:
        """对 token 序列做一次无外部 cache 的前向，返回所请求层的 hidden。

        Returns:
            {layer_number: ndarray(shape=(n_tokens, hidden))}，key 为
            layers.main 与 layers.final 的配置层号。
        """

    # ── 基类统一逻辑 ──────────────────────────────────────────────────────

    def extract(
        self, words: list[str], i: int, H: int, layers: LayerSpec
    ) -> WindowRepr:
        """提取故事内位置 i、上下文长度 H 的目标词双层表示。"""
        window = build_window(words, i, H)
        self.reset_state()
        token_ids, spans, is_unk = self.tokenize_with_spans(window)

        # 目标词 = 窗口最后一个词；取其最后一个 subtoken
        target_start, target_end = spans[-1]
        if target_end <= target_start:
            raise ValueError(f"目标词未产生任何 token（位置 i={i}, H={H}）")
        target_tok = target_end - 1

        hidden = self.forward_hidden(token_ids, layers)
        return WindowRepr(
            main=np.asarray(hidden[layers.main][target_tok], dtype=np.float32),
            final=np.asarray(hidden[layers.final][target_tok], dtype=np.float32),
            n_tokens=len(token_ids),
            target_token_index=target_tok,
            n_target_subtokens=target_end - target_start,
            is_unk=bool(is_unk[-1]),
        )

    def extract_inheriting_state(
        self, words: list[str], i: int, H: int, layers: LayerSpec
    ) -> WindowRepr:
        """【状态重置反向测试用】故意把前一窗口的循环状态带入目标窗口。

        正式提取路径绝不调用本方法；它只为单元测试证明「若不重置状态，输出会
        不同」——即状态重置确实有意义。循环模型（RWKV/Mamba/AWD-LSTM）必须在
        AutoDL 上用各自的 state/cache_params API 重写本方法；Transformer 无跨窗
        状态，反向测试不适用。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 尚未实现状态继承路径；"
            f"需在 AutoDL 上用该模型的 state/cache API 重写（里程碑要求的"
            f"状态重置反向测试）。"
        )

    def audit_row(self) -> dict:
        """模型审计表的一行：ID/revision/参数与层信息。"""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "hidden_width": self.hidden_width,
            "n_layers": self.n_layers,
            "is_recurrent": self.is_recurrent,
        }
