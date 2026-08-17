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

case "${1:-all}" in
  d1)  python3 -u "$D1" ;;
  d2)  python3 -u "$D2" ;;
  d3)  python3 -u "$D3" ;;
  all)
    python3 -u "$D1"
    echo; echo; sleep 1
    python3 -u "$D2"
    echo; echo; sleep 1
    python3 -u "$D3"
    ;;
  *)
    echo "用法: bash demo/run.sh [d1|d2|d3|all]" >&2
    exit 2
    ;;
esac
