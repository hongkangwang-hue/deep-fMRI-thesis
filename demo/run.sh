#!/usr/bin/env bash
# 演示脚本统一启动器（面向屏幕录制）
#
# 为什么需要它，而不是直接 python3 demo/dX.py：
#   1. OMP_NUM_THREADS 必须在 **Python 解释器启动前** 就是合法值。服务器上该变量为
#      非法值，libgomp 会在任何 Python 代码运行之前就往 stderr 打印两行警告，污染
#      录屏第一屏。脚本内部 os.environ 设置为时已晚。
#   2. HF 离线模式同理越早越好：服务器无法访问 huggingface.co，transformers 会重试
#      5 次 × 多个配置文件（实测 RWKV 因此卡数分钟）。权重已在本地 HF cache。
#   3. python3 -u 关闭输出缓冲，否则经 SSH 管道时输出会攒着一次性刷出，录屏观感差。
#
# 用法：
#   bash demo/run.sh d1        # 只跑 D1
#   bash demo/run.sh d2
#   bash demo/run.sh d3
#   bash demo/run.sh all       # 依次跑三个（录屏建议分三段单独录）

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONWARNINGS=ignore

cd "$(dirname "$0")/.."

D1=demo/d1_window_and_strict_upper_bound.py
D2=demo/d2_alignment_and_no_leakage.py
D3=demo/d3_paired_bootstrap.py

# T series: English, compact, data-first (second version)
T1=demo/t_d1_window_bound.py
T2=demo/t_d2_alignment_leakage.py
T3=demo/t_d3_paired_bootstrap.py

case "${1:-all}" in
  d1)       python3 -u "$D1" ;;            # 读缓存，秒级（录屏用这个）
  d1-live)  python3 -u "$D1" --live ;;     # 现场真算：GPU~7s / 无卡CPU~120s，并更新缓存
  d2)       python3 -u "$D2" ;;
  d3)       python3 -u "$D3" ;;
  all)
    python3 -u "$D1"
    echo; echo; sleep 1
    python3 -u "$D2"
    echo; echo; sleep 1
    python3 -u "$D3"
    ;;
  live)
    # 三个演示全部现场真算（D1 用 --live；D2/D3 本来就是现场计算）。
    # 建议开有卡模式：D1 在 GPU 上约 7.5s，无卡 CPU 约 120s。
    python3 -u "$D1" --live
    echo; echo; sleep 1
    python3 -u "$D2"
    echo; echo; sleep 1
    python3 -u "$D3"
    ;;
  t1)       python3 -u "$T1" ;;            # English, cached
  t1-live)  python3 -u "$T1" --live ;;     # English, live forward passes
  t2)       python3 -u "$T2" ;;
  t3)       python3 -u "$T3" ;;
  t-all)
    python3 -u "$T1"
    echo; echo; sleep 1
    python3 -u "$T2"
    echo; echo; sleep 1
    python3 -u "$T3"
    ;;
  t-live)
    python3 -u "$T1" --live
    echo; echo; sleep 1
    python3 -u "$T2"
    echo; echo; sleep 1
    python3 -u "$T3"
    ;;
  *)
    echo "用法: bash demo/run.sh <target>" >&2
    echo "  中文版（第一版）:" >&2
    echo "    d1        读缓存（0.08s）" >&2
    echo "    d1-live   现场真算（开卡~7.5s / 无卡CPU~120s）并更新缓存" >&2
    echo "    d2 / d3   现场真算（2.9s / 4.7s）" >&2
    echo "    all       三个依次跑，D1 用缓存" >&2
    echo "    live      三个依次跑，D1 现场真算" >&2
    echo "  英文精简版（第二版，T 系列）:" >&2
    echo "    t1        读缓存" >&2
    echo "    t1-live   现场真算并更新缓存" >&2
    echo "    t2 / t3   现场真算" >&2
    echo "    t-all     三个依次跑，T1 用缓存" >&2
    echo "    t-live    三个依次跑，T1 现场真算" >&2
    exit 2
    ;;
esac
