"""三个演示脚本共用的环境设置与排版工具（面向屏幕录制）。

⚠️ 必须在 import torch / transformers **之前** 调用 setup_env()。

为什么需要这个模块（都是实测踩到的坑，不是预防性代码）：
  1. HF 离线模式：本项目服务器无法访问 huggingface.co。transformers 加载模型前会发
     HEAD 请求检查更新，失败后按 1/2/4/8/8 秒重试 5 次 × 多个配置文件 → 累计数分钟
     白等（实测 RWKV 因此卡住数分钟；设为离线后加载只需 0.5 秒）。权重已在本地
     ~/.cache/huggingface，离线模式直接读盘。
  2. 无缓冲输出：Python 经 SSH 管道输出是块缓冲的，录屏时会长时间空白后一次性刷出
     全部文字。脚本请用 `python3 -u` 运行；本模块的 say() 也强制 flush。
  3. OMP_NUM_THREADS：服务器该变量为非法值，libgomp 会在第一行打印警告污染录屏。
"""

from __future__ import annotations

import os
import sys

WIDTH = 96  # 所有输出行宽上限，适配录屏画面


def setup_env() -> None:
    """设置离线模式与线程数，抑制无关日志。必须在 import torch 之前调用。"""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if not os.environ.get("OMP_NUM_THREADS", "").strip().isdigit():
        os.environ["OMP_NUM_THREADS"] = "4"

    import warnings
    warnings.filterwarnings("ignore")
    import logging
    for name in ("transformers", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)


def project_root() -> str:
    """仓库根目录（本文件位于 <root>/demo/）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def add_project_to_path() -> str:
    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


# ── 录屏排版工具 ──────────────────────────────────────────────────────────────

def say(*args, **kwargs) -> None:
    """强制 flush 的 print，保证录屏时输出逐行实时出现。"""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def rule(char: str = "=") -> None:
    say(char * WIDTH)


def header(title: str, question: str) -> None:
    """脚本开头的标题块：明确这个演示在回答什么问题。"""
    rule("=")
    say(title)
    rule("=")
    say(f"本演示回答的问题：{question}")
    rule("=")
    say()


def section(label: str) -> None:
    say()
    rule("-")
    say(label)
    rule("-")


def check(ok: bool, msg: str) -> bool:
    """打印 [PASS]/[FAIL] 断言行，返回原布尔值以便累计。"""
    say(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    return ok


def truncate_list(items, head: int = 3, tail: int = 2) -> str:
    """长列表截断显示：前 head 个 ... 后 tail 个（总数始终标出）。"""
    items = list(items)
    if len(items) <= head + tail:
        return ", ".join(str(x) for x in items)
    front = ", ".join(str(x) for x in items[:head])
    back = ", ".join(str(x) for x in items[-tail:])
    return f"{front}, ... ({len(items) - head - tail} more) ..., {back}"


def truncate_words(words, head: int = 6, tail: int = 4) -> str:
    """长词序列截断显示，让观众直观看到窗口在变长。"""
    words = list(words)
    if len(words) <= head + tail:
        return " ".join(words)
    return (" ".join(words[:head]) + f"  [...{len(words) - head - tail} words...]  "
            + " ".join(words[-tail:]))
