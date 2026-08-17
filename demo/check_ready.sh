#!/usr/bin/env bash
# 演示前自检：确认三个演示所需的数据是否齐全。
#
# 用途：服务器关机重启后、或换了新实例后，先跑这个，10 秒内知道能不能直接开录。
#   bash demo/check_ready.sh
#
# 背景（AutoDL 存储行为）：
#   /root/autodl-tmp  = 数据盘，**关机重启保留**，释放实例才删除
#   /                 = 系统盘，同样关机重启保留（HF 模型权重在这里）
#   本项目整个位于 /root/autodl-tmp/deep-fMRI-dataset，因此正常关机重启后
#   三个演示都能直接跑，无需任何重新配置。

cd "$(dirname "$0")/.."

# 颜色只在真正的终端里启用；经管道/重定向时禁用，避免录屏或日志里出现 [0;32m 乱码
if [ -t 1 ]; then
  GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
else
  GREEN=''; RED=''; YELLOW=''; NC=''
fi
ok=0; miss=0

# 中文字符显示宽度是 2 但 printf 按字符数补齐，强行对齐反而更乱。
# 因此：就绪时只打描述；缺失时才补上路径，方便排查。
chk() {   # chk <描述> <路径>
  if [ -e "$2" ]; then
    printf "  ${GREEN}[有]${NC} %s\n" "$1"
    ok=$((ok+1))
  else
    printf "  ${RED}[缺]${NC} %s  ← %s\n" "$1" "$2"
    miss=$((miss+1))
  fi
}

echo "================================================================================"
echo "演示前自检 — demo/check_ready.sh"
echo "================================================================================"
echo "当前目录: $(pwd)"
echo "数据盘状态:"
df -h . | tail -1 | sed 's/^/  /'
echo

echo "── D1（默认读缓存，0.08s，最健壮：只依赖一个 2KB 的 JSON）────────────────────"
chk "D1 缓存结果" "demo/cached_results/d1_results.json"
d1_ready=$([ -e demo/cached_results/d1_results.json ] && echo yes || echo no)
echo

echo "── D2（现场真跑，2.9s，依赖最重：需要 BOLD 与特征缓存）──────────────────────"
chk "词级特征缓存(单故事即可)" "cache/features/pythia/adollshouse_H8.npz"
chk "冻结文件 word_index" "frozen/word_index.parquet"
chk "冻结文件 fold_split" "frozen/fold_split.json"
chk "BOLD 数据(UTS03 单故事)" "data/ds003020/derivatives/preprocessed_data/UTS03/adollshouse.hf5"
chk "M4 story级结果(UTS01)" "results/m4_full_matrix/UTS01/cells"
chk "掩码独立审计(UTS01)" "results/mask_identity_audit/UTS01/mask_identity.json"
echo

echo "── D3（现场真跑，4.7s，只依赖统计结果，不需要 BOLD/特征）────────────────────"
for s in UTS01 UTS02 UTS03; do
  chk "M4 cells ($s)" "results/m4_full_matrix/$s/cells"
  chk "M5 统计结果 ($s)" "results/m5_stats/$s/m5_results.json"
done
echo

echo "── 仅 d1-live 现场真算才需要（默认 d1 不需要）───────────────────────────────"
chk "Pythia 权重" "$HOME/.cache/huggingface/hub/models--EleutherAI--pythia-160m"
chk "RWKV 权重"   "$HOME/.cache/huggingface/hub/models--RWKV--rwkv-4-169m-pile"
chk "Mamba 权重"  "$HOME/.cache/huggingface/hub/models--state-spaces--mamba-130m-hf"
echo

echo "================================================================================"
if [ "$miss" -eq 0 ]; then
  printf "${GREEN}全部就绪（%d 项）——三段演示可以直接开录${NC}\n" "$ok"
  echo
  echo "  clear; bash demo/run.sh d1"
  echo "  clear; bash demo/run.sh d2"
  echo "  clear; bash demo/run.sh d3"
else
  printf "${YELLOW}就绪 %d 项，缺失 %d 项${NC}\n" "$ok" "$miss"
  echo
  if [ "$d1_ready" = yes ]; then
    printf "  ${GREEN}D1 仍可正常演示${NC}（它只依赖已提交进 git 的缓存 JSON）\n"
  fi
  echo "  缺失项的恢复办法："
  echo "    - demo/ 下的文件      → git pull mine master（缓存 JSON 已随仓库提交）"
  echo "    - frozen/            → git pull（已随仓库提交）"
  echo "    - results/           → 未入版本库，需从本机 scp 上传，或重跑 M4/M5"
  echo "    - cache/features/    → 需重跑 M1 特征提取（GPU 约 1 小时）"
  echo "    - data/ds003020/     → datalad get 重新下载（单被试约 23G）"
  echo "    - HF 权重            → 联网后首次 d1-live 会自动下载；无网则只能用默认 d1"
fi
echo "================================================================================"
exit 0
