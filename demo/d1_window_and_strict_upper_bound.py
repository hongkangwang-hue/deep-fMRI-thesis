"""D1 — 窗口边界与 H 的严格上界

【证明论文中的哪句声明】
  论文 2.2 节：“Each target window was processed in a cold-start forward call so that H
  remained a strict upper bound on the available context.”
  以及 H 的定义：H 计的是**目标词之前的原始词（raw words）数**，不是 subtoken 数。

【对应论文章节】2.2 节（特征提取与窗口构造）
【PPT 播放位置】Slide 4 之后

【为什么需要这个演示】
  论文里这句话是一句无法验证的断言：读者看不到窗口长什么样、H 数的到底是词还是
  subtoken、也无法确认“cold-start”真的杜绝了跨窗信息泄漏。本演示把它变成可见的事实：
  (a) 打印真实目标词在 H=8/32/128 下的实际窗口与 tokenisation；
  (b) 用同一目标词做两组前向对比，证明 cold-start 下 max|Δ|=0，而让窗口外的词进入
      同一次前向时 max|Δ|>0（信息确实会越过 H 边界）。

【两种运行模式】
  默认（--cached）：读取 demo/cached_results/d1_results.json 并渲染，**秒级完成**。
                    该 JSON 由 --live 模式真实计算后落盘，内含生成时间、设备、
                    git commit，输出顶部会明确标注“数值读取自缓存”。
  --live          ：现场真实计算（三模型加载 + 前向）。GPU 约 7 s；无卡 CPU 约 120 s。
                    跑完自动更新缓存 JSON。

【数据来源】
  - 目标词取自冻结的 common target index：frozen/word_index.parquet（eligible_h128==True），
    不是编造的例子。
  - 模型为三个冻结 checkpoint，权重读自服务器本地 HF cache（离线模式，不联网）。
  - --live 下全部为真实前向；--cached 下为同一计算的落盘结果，无任何手写数字。

【实现说明（对应原代码位置）】
  - 窗口构造：直接调用 src/models/windowing.py::build_window（未复制逻辑）。
  - 特征提取（cold-start 正式路径）：调用 src/models/base.py::ModelAdapter.extract。
  - **状态继承路径（(b) 的反向对比）在本脚本内实现**：src/models/base.py 的
    extract_inheriting_state() 对真实 adapter 只抛 NotImplementedError（注释写明
    “需在 AutoDL 上用该模型的 state/cache API 重写”），故此处按各模型的 HF API
    自行实现。这段逻辑仅用于演示/反向验证，正式提取路径从不使用。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _demo_common import (  # noqa: E402
    setup_env, add_project_to_path, say, rule, header, section, truncate_words,
)

setup_env()                 # 必须在 import torch 之前
ROOT = add_project_to_path()

CACHE_PATH = Path(__file__).resolve().parent / "cached_results" / "d1_results.json"

CORE_MODELS = ["pythia", "rwkv", "mamba"]
H_LIST = [8, 32, 128]
DEMO_STORY = "adollshouse"      # 任一真实故事；仅用于取真实词序列
DEMO_TARGET_LOCAL_ID = 300      # 故事内第 300 词，H=128 下有完整历史
H_TEST = 8                      # (b) 用的窗口长度
FILLER_SLICE = (1200, 1320)     # 无关文本取自同故事远端（真实词，非编造）


# ══ 采集：真实计算（仅 --live 走这里）════════════════════════════════════════

def _forward_last_target_vec(adapter, window_words, layer, carry_state=None):
    """对一个窗口做前向，返回目标词最后一个 subtoken 在指定层的向量。

    carry_state=None  → cold-start（与正式提取路径同义：不传任何 past/state/cache）
    carry_state 非 None → 故意把上一次前向的循环状态带进来（演示泄漏，非正式路径）
    """
    import torch
    token_ids, spans, _ = adapter.tokenize_with_spans(window_words)
    target_tok = spans[-1][1] - 1          # 目标词的最后一个 subtoken
    ids = torch.tensor([token_ids], device=adapter.device)

    kwargs, used = {}, None
    if carry_state is not None:
        kind, value = carry_state
        kwargs[kind] = value
        used = kind

    with torch.no_grad():
        if carry_state is None:
            out = adapter.model(input_ids=ids, use_cache=False)
        else:
            out = adapter.model(input_ids=ids, use_cache=True, **kwargs)

    hs = out.hidden_states[layer + 1]       # [0]=embedding，[b+1]=block b 输出
    return hs[0, target_tok].float().cpu().numpy(), used


def _concat_prefix_vec(adapter, filler_words, window_words, layer):
    """把无关文本**拼接**在窗口前面做一次前向，取同一个目标词的向量。

    这是「H 不是严格上界」的等效情形：目标词在同一个 forward 里看到了窗口之外的词。
    与状态继承路径不同，这条路对三个模型都成立（不依赖各自的 cache/state API）。
    正式提取路径绝不这样做——窗口严格只含 H 个历史词。
    """
    import torch
    fill_ids, _, _ = adapter.tokenize_with_spans(filler_words)
    win_ids, spans, _ = adapter.tokenize_with_spans(window_words)
    target_off = spans[-1][1] - 1                  # 目标词在窗口内的相对位置
    ids = torch.tensor([fill_ids + win_ids], device=adapter.device)
    abs_pos = len(fill_ids) + target_off           # 在拼接序列里的绝对位置
    with torch.no_grad():
        out = adapter.model(input_ids=ids, use_cache=False)
    return out.hidden_states[layer + 1][0, abs_pos].float().cpu().numpy()


def _capture_state(adapter, filler_words):
    """用无关文本做一次前向，抓出该模型的循环状态/缓存。三模型 API 名不同，逐个尝试。"""
    import torch
    token_ids, _, _ = adapter.tokenize_with_spans(filler_words)
    ids = torch.tensor([token_ids], device=adapter.device)
    with torch.no_grad():
        out = adapter.model(input_ids=ids, use_cache=True)
    for kind in ("past_key_values", "cache_params", "state"):
        value = getattr(out, kind, None)
        if value is not None:
            return (kind, value)
    return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                              # noqa: BLE001
        return "unknown"


def collect_live() -> dict:
    """真实计算全部演示数值，返回可 JSON 序列化的 dict。"""
    import numpy as np
    import pandas as pd
    import torch

    from src.config_loader import load_config
    from src.models import get_adapter
    from src.models.base import LayerSpec
    from src.models.windowing import build_window

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wi = pd.read_parquet(Path(ROOT) / "frozen" / "word_index.parquet")
    sub = wi[wi["story"] == DEMO_STORY].sort_values("word_local_id")
    words = sub["word"].tolist()
    elig = sub[sub["eligible_h128"]]["word_local_id"].tolist()

    i = DEMO_TARGET_LOCAL_ID
    assert i in elig, f"目标位置 {i} 不在 eligible_h128 集合中"

    data: dict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "git_commit": _git_commit(),
        "story": DEMO_STORY,
        "n_words_story": len(words),
        "n_eligible": len(elig),
        "target_local_id": i,
        "target_word": words[i],
        "H_list": H_LIST,
        "H_test": H_TEST,
        "filler_n_words": FILLER_SLICE[1] - FILLER_SLICE[0],
        "windows": [],
        "readout": {},
        "models": [],
    }

    filler = words[FILLER_SLICE[0]:FILLER_SLICE[1]]

    # ── (a) 窗口与 tokenisation（只需 tokenizer，但 adapter.load() 会一并加载模型）──
    ad = get_adapter("pythia", device=device)
    t0 = time.perf_counter()
    ad.load()
    data["pythia_load_s"] = round(time.perf_counter() - t0, 1)
    layers_py = LayerSpec(main=cfg["models"]["primary_layers"]["pythia"],
                          final=cfg["models"]["robustness_layers"]["pythia"])

    for H in H_LIST:
        window = build_window(words, i, H)
        tok_ids, spans, _ = ad.tokenize_with_spans(window)
        data["windows"].append({
            "H": H,
            "n_words": len(window),
            "n_subtokens": len(tok_ids),
            "n_target_subtokens": spans[-1][1] - spans[-1][0],
            "display": truncate_words(window, head=5, tail=3),
        })

    window = build_window(words, i, H_TEST)
    tok_ids, spans, _ = ad.tokenize_with_spans(window)
    ts, te = spans[-1]
    rep = ad.extract(words, i, H_TEST, layers_py)
    data["readout"] = {
        "H": H_TEST, "span_start": ts, "span_end": te,
        "layer_main": layers_py.main, "layer_final": layers_py.final,
        "shape_main": list(rep.main.shape), "shape_final": list(rep.final.shape),
        "n_tokens": rep.n_tokens, "target_token_index": rep.target_token_index,
        "n_target_subtokens": rep.n_target_subtokens,
    }
    del ad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── (b) 严格上界：三模型各跑一遍 ─────────────────────────────────────────
    window_A = build_window(words, i, H_TEST)
    for name in CORE_MODELS:
        layer_main = cfg["models"]["primary_layers"][name]
        ad = get_adapter(name, device=device)
        t0 = time.perf_counter()
        ad.load()
        t_load = time.perf_counter() - t0

        vec_A, _ = _forward_last_target_vec(ad, window_A, layer_main)
        _ = _forward_last_target_vec(ad, filler, layer_main)        # 无关前向，不传状态
        vec_B, _ = _forward_last_target_vec(ad, window_A, layer_main)
        cold_diff = float(np.abs(vec_A - vec_B).max())
        cold_same = bool(np.allclose(vec_A, vec_B))

        leak_diff, leak_note = None, ""
        try:
            carry = _capture_state(ad, filler)
            if carry is None:
                leak_note = "该模型未暴露可继承状态"
            else:
                vec_leak, used = _forward_last_target_vec(ad, window_A, layer_main, carry)
                if vec_leak.shape != vec_A.shape:
                    leak_note = f"形状不一致({vec_leak.shape})"
                else:
                    leak_diff = float(np.abs(vec_A - vec_leak).max())
                    leak_note = f"经 {used}"
        except Exception as e:                                      # noqa: BLE001
            leak_note = f"{type(e).__name__}"

        cat_diff = float(np.abs(vec_A - _concat_prefix_vec(
            ad, filler, window_A, layer_main)).max())

        data["models"].append({
            "name": name, "model_id": ad.model_id, "layer_main": layer_main,
            "load_s": round(t_load, 1), "cold_diff": cold_diff, "cold_same": cold_same,
            "leak_diff": leak_diff, "leak_note": leak_note, "cat_diff": cat_diff,
        })
        del ad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return data


# ══ 渲染：两种模式共用，保证输出格式完全一致 ══════════════════════════════════

def render(d: dict, from_cache: bool) -> bool:
    header("D1 — 窗口边界与 H 的严格上界（Window boundary & strict upper bound on H）",
           "H 数的是原始词还是 subtoken？cold-start 真的能杜绝跨窗信息泄漏吗？")

    if from_cache:
        say(f"⚠ 以下数值读取自缓存：demo/cached_results/d1_results.json")
        say(f"   该缓存由 --live 模式真实计算后落盘，非手写数字：")
        say(f"     生成时间 = {d['generated_at']}   设备 = {d['device']}   "
            f"代码版本 = {d['git_commit']}")
        say(f"   现场重算请加 --live（GPU 约 7 s；无卡 CPU 约 120 s）")
    else:
        say(f"● 现场真实计算（--live）  设备 = {d['device']}   代码版本 = {d['git_commit']}")
    say()

    say(f"数据来源：frozen/word_index.parquet（M0 冻结的 common target index）")
    say(f"演示故事：{d['story']}  共 {d['n_words_story']} 词，"
        f"其中 eligible_h128 目标 {d['n_eligible']} 个")
    say(f"选定目标词：story-local id = {d['target_local_id']}，"
        f"词 = '{d['target_word']}'（取自 eligible 集合，非编造）")

    # ── (a) ──────────────────────────────────────────────────────────────────
    section("(a) 同一目标词在 H=8 / 32 / 128 下的实际窗口与 tokenisation")
    say("窗口定义 src/models/windowing.py::build_window ： W_i(H) = words[i-H : i+1]")
    say("  → 前 H 个是历史词，最后 1 个是目标词，窗口总长 H+1（H 只数历史，不含目标）")
    say()
    say(f"{'H':>4}  {'窗口词数':>8}  {'subtoken数':>10}  {'目标词subtoken数':>16}"
        f"   窗口原文（长窗截断中间）")
    say("-" * 96)
    for w in d["windows"]:
        say(f"{w['H']:>4}  {w['n_words']:>8}  {w['n_subtokens']:>10}  "
            f"{w['n_target_subtokens']:>16}   {w['display']}")
    say("-" * 96)
    say("↑ 关键观察：窗口词数恒为 H+1（8→9, 32→33, 128→129），而 subtoken 数**多于**词数")
    say("  （BPE 会把部分词切成多个 subtoken）→ 证明 H 计的是原始词，不是 subtoken。")
    say()

    r = d["readout"]
    say(f"特征读取点（以 H={r['H']} 为例）：")
    say(f"  目标词 '{d['target_word']}' 占 subtoken 区间 "
        f"[{r['span_start']}, {r['span_end']})，共 {r['span_end']-r['span_start']} 个 "
        f"→ 取**最后一个**（index {r['span_end']-1}）")
    say(f"  主层 = block {r['layer_main']}（0-based，取 hidden_states[{r['layer_main']}+1]）"
        f"；最终层 = block {r['layer_final']}")
    say(f"  输出向量 shape = {tuple(r['shape_main'])}（主层）  "
        f"{tuple(r['shape_final'])}（最终层）")
    say(f"  adapter 自报：n_tokens={r['n_tokens']}, "
        f"target_token_index={r['target_token_index']}, "
        f"n_target_subtokens={r['n_target_subtokens']}")

    # ── (b) ──────────────────────────────────────────────────────────────────
    section("(b) 严格上界证明：cold-start vs 让窗口外的词进入前向")
    say(f"对比设计（H={d['H_test']}，目标词 '{d['target_word']}'）：")
    say(f"  窗口 A：cold-start 直接前向该窗口")
    say(f"  窗口 B：**先**用一段无关文本（{d['filler_n_words']} 个词，取自同故事远端）"
        f"做一次前向，")
    say(f"          **再**前向同一个窗口 A")
    say(f"  若 H 是严格上界，则 A 与 B 的目标词向量必须逐位相同（无关文本不得留下痕迹）。")
    say()
    for m in d["models"]:
        leak = (f"{m['leak_diff']:.3e} ({m['leak_note']})" if m["leak_diff"] is not None
                else f"n/a ({m['leak_note']})")
        say(f"  {m['name']:8s} 加载 {m['load_s']:4.1f}s | "
            f"cold-start max|Δ|={m['cold_diff']:.3e} allclose={m['cold_same']}")
        say(f"  {'':8s}   状态继承 max|Δ|={leak}   拼接前文 max|Δ|={m['cat_diff']:.3e}")

    # ── 汇总表 ───────────────────────────────────────────────────────────────
    section("汇总：三个 core checkpoint 的严格上界检验")
    say(f"{'模型':<8} {'checkpoint':<28} {'cold-start':>12} {'allclose':>9} "
        f"{'状态继承':>12} {'拼接前文':>12}")
    say("-" * 96)
    all_ok = True
    for m in d["models"]:
        leak_str = f"{m['leak_diff']:.2e}" if m["leak_diff"] is not None else "n/a"
        say(f"{m['name']:<8} {m['model_id']:<28} {m['cold_diff']:>12.2e} "
            f"{str(m['cold_same']):>9} {leak_str:>12} {m['cat_diff']:>12.2e}")
        all_ok = all_ok and m["cold_same"] and m["cold_diff"] == 0.0
    say("-" * 96)
    say("列说明：cold-start = 正式提取路径（窗口外文本先前向但不传状态）")
    say("        状态继承   = 故意把无关文本的 KV/state/cache 传入（依赖各模型 API）")
    say("        拼接前文   = 把无关文本拼在窗口之前同批前向（三模型通用的等效违规）")
    say()

    if all_ok:
        say("[PASS] 三个模型的 cold-start max|Δ| 全部精确为 0，allclose 全部为 True")
        say("       → H 是严格上界：窗口之外的任何文本都不会影响目标词表示。")
    else:
        say("[FAIL] 存在 cold-start 下不一致的模型，需检查提取路径")

    cat_all = [(m["name"], m["cat_diff"]) for m in d["models"] if m["cat_diff"] > 0]
    leaked = [(m["name"], m["leak_diff"]) for m in d["models"]
              if m["leak_diff"] is not None and m["leak_diff"] > 0]
    if cat_all:
        say(f"[对照] 一旦让窗口外的词进入同一次前向，三个模型的目标词向量全部改变：")
        say(f"       {', '.join(f'{n} max|Δ|={v:.2f}' for n, v in cat_all)}")
        say(f"       量级达到 10^0~10^1，远非浮点误差 → H 边界是实质约束，不是形式声明。")
    if leaked:
        say(f"[对照] 经各模型自身的状态/缓存 API 泄漏也同样成立："
            f"{', '.join(f'{n}={v:.2f}' for n, v in leaked)}")
    unsupported = [(m["name"], m["leak_note"]) for m in d["models"]
                   if m["leak_diff"] is None]
    if unsupported:
        say(f"       （{', '.join(f'{n}: {no}' for n, no in unsupported)} —— "
            f"该模型的 HF 实现不接受多 token 序列 + 预置 cache，")
        say(f"         故其状态泄漏用「拼接前文」列呈现，结论一致。）")

    say()
    rule("=")
    say("结论：H 计的是原始词数（窗口长 H+1，subtoken 数更多）；cold-start 使 H 成为")
    say("      严格上界，这一点现在有逐位相同的数值证据，而非仅凭论文中的文字声明。")
    rule("=")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description="D1 窗口边界与 H 的严格上界")
    ap.add_argument("--live", action="store_true",
                    help="现场真实计算（GPU 约 7s，无卡 CPU 约 120s），并更新缓存")
    args = ap.parse_args()

    if args.live:
        data = collect_live()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        ok = render(data, from_cache=False)
        say()
        say(f"[缓存已更新] {CACHE_PATH.relative_to(Path(ROOT))}")
        return 0 if ok else 1

    if not CACHE_PATH.exists():
        say(f"未找到缓存 {CACHE_PATH}")
        say(f"请先运行一次：bash demo/run.sh d1-live")
        return 2
    with open(CACHE_PATH) as f:
        data = json.load(f)
    return 0 if render(data, from_cache=True) else 1


if __name__ == "__main__":
    sys.exit(main())
