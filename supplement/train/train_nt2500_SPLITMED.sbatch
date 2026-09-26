#!/bin/bash
#SBATCH --job-name=nt2500SPLITMED
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16 --mem=32G --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/path/to/work/nt2500SPLITMED_%j.log
# noText @2500 (2247 base + 249 backfill), stage 2 ONLY -- from the raw base, no stage 1.
# Decay is inlined: a separate decay job sat in the queue while all 8 GPUs idled.
# Only the 249 backfill rows are decayed. The base 2247 already went through decay when it was
# built (0 rows over the 70k budget); running build_decay over the whole file re-resolves EVERY
# row image path by basename across dirs and dropped 978 rows on the first attempt.
# MAX_PIXELS is load-bearing in TRAINING (swift/Qwen-VL processor) -- dropping it OOMs the H200.
# max_length/padding_free/liger = the config the 82 paper runs used; 262144/no-pack OOM at step 2.
set -e
S=/path/to/work
NODE=$(hostname -s)
if [ "$NODE" = NODE ]; then DT=/path/to/work/rw30_sft
else DT=/path/to/work/rw30_sft; fi
echo "=== NODE=$NODE  DT=$DT ==="
ENV=$DT/env
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
export HF_HOME=$DT/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 MAX_PIXELS=602112
export NPROC_PER_NODE=2 VLDR_ROOT=$S/VL_DR
export TRITON_CACHE_DIR=$DT/triton TORCHINDUCTOR_CACHE_DIR=$DT/inductor XDG_CACHE_HOME=$DT/xdg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/usr/local/cuda-12.9

DATA_OUT=$S/sft_train_noText2500_dec_thid.jsonl
if [ -s "$DATA_OUT" ]; then
  echo "=== 数据已存在,跳过重建: $DATA_OUT ($(wc -l < $DATA_OUT) 行) ==="
else
python3 -c "
S=\"$S/\"
base=set(open(S+\"sft_train_rpmix_noText_net.jsonl\"))
keep=[];fill=[]
for l in open(S+\"sft_train_noText2500.jsonl\"):
    (keep if l in base else fill).append(l)
open(S+\"_nt_base.jsonl\",\"w\").writelines(keep)
open(S+\"_nt_fill.jsonl\",\"w\").writelines(fill)
print(\"split: base %d + fill %d\" % (len(keep), len(fill)))
"
/path/to/work/vllmenv/bin/python $S/build_decay_multi.py \
  $S/_nt_fill.jsonl $S/_nt_fill_dec.jsonl $S/cfw_images_fill250
IN=$(wc -l < $S/_nt_fill.jsonl); OUT=$(wc -l < $S/_nt_fill_dec.jsonl)
[ "$IN" = "$OUT" ] || { echo "FATAL: decay dropped $((IN-OUT)) fill rows"; exit 1; }
cat $S/_nt_base.jsonl $S/_nt_fill_dec.jsonl > $S/sft_train_noText2500_dec.jsonl
TOT=$(wc -l < $S/sft_train_noText2500_dec.jsonl)
[ "$TOT" = "2496" ] || { echo "FATAL: merged $TOT != 2496"; exit 1; }
sed "s|/net/NODE/data|/data|g" $S/sft_train_noText2500_dec.jsonl > $S/sft_train_noText2500_dec_thid.jsonl
echo "decay ok: $TOT rows"
fi

if [ "$NODE" = NODE ]; then DATA=$S/sft_train_nt2500_SPLITMED_thid.jsonl
else DATA=$S/sft_train_nt2500_SPLITMED.jsonl; fi
ROWS=$(wc -l < $DATA); STEPS=$(( (ROWS + 15) / 16 ))
OUT=$DT/out_noText2500_nos1_SPLITMED
for p in $DT/model $ENV/bin/python $DATA; do [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }; done
$ENV/bin/python $S/register_cfwv3.py $ENV/lib/python3.12/site-packages/swift/loss_scale


# ---- PREFLIGHT: 图片可达性(换节点后最容易静默失败的地方)----
# _thid.jsonl 的 15592 张图在 /path/to/work(仅 NODE 本地可见);非 NODE 节点必须
# 用 _dec.jsonl(那 15592 张写作 /net/NODE/data)。路径不可达不会崩,会喂进空白图。
$ENV/bin/python - "$DATA" <<'IMGCHK' || { echo "=== PREFLIGHT FAILED: 图片路径不可达,拒绝启动 ==="; exit 1; }
import json, os, sys, random, collections
allp=[]
for line in open(sys.argv[1]):
    line=line.strip()
    if line: allp += (json.loads(line).get("images") or [])
pref=collections.Counter("/".join(p.split("/")[:4]) for p in allp)
print("PREFLIGHT 图片前缀:", dict(pref))
random.seed(0); samp=random.sample(allp, min(200, len(allp)))
miss=[p for p in samp if not os.path.exists(p)]
assert not miss, "抽样 200 张缺失 %d 张,例: %s" % (len(miss), miss[0])
print("PREFLIGHT OK: %d 张图,抽样 200 张全部可达" % len(allp))
IMGCHK

# ---- PREFLIGHT: 证明 per-token 反事实权重真的到达训练器 ----
# 2026-09-13: 所有历史训练用的是 `cfwv3+ignore_empty_think`。`+` 会构造 ConcatLossScale,
# 它只调用子对象的 get_loss_scale(context)(拿不到 messages),而 cfwv3 的逻辑全在
# __call__(context_list, context_types, messages) 里 -> 永不被调用 -> 所有响应 token 权重=1.0。
# 结果:全部 cfwV3rank 模型其实是普通 SFT。register_cfwv3.py 自带的 smoke-test 测的是
# `cfwv3` 单独使用(好的那条路),所以从未报警。这道闸门直接检查编码后的权重种类。
DT=$DT $ENV/bin/python - "$DATA" <<'PREFLIGHT' || { echo "=== PREFLIGHT FAILED: 反事实权重未进入训练器,拒绝启动 ==="; exit 1; }
import json, sys, os, collections
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
M = os.environ["DT"] + "/model"
from swift.model import get_model_info_meta
from swift.template import get_template
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
mi, mm = get_model_info_meta(M); tok.model_info, tok.model_meta = mi, mm
tmpl = get_template(tok, max_length=131072, loss_scale="stepw"); tmpl.set_mode("train")
n = 0; vals = collections.Counter()
for line in open(sys.argv[1]):
    d = json.loads(line); n += 1
    if n > 5: break
    enc = tmpl.encode({"messages": [dict(m) for m in d["messages"]]})
    vals.update(round(float(x), 4) for x in enc["loss_scale"])
nz = {k: v for k, v in vals.items() if k != 0}
print("PREFLIGHT: 前5行编码后非零权重取值 =", sorted(nz.items(), key=lambda x: -x[1])[:8])
assert len(nz) > 2, "只有 %d 种非零权重 -> 逐步权重没进来" % len(nz)
wm = sum(k * v for k, v in nz.items()) / sum(nz.values())
assert 1.20 < wm < 1.80, "加权均值 %.4f 不在预期 1.47 附近 -> 数据或 loss_scale 用错" % wm
print("PREFLIGHT OK: %d 种非零权重,加权均值 %.4f(预期约 1.4745)" % (len(nz), wm))
PREFLIGHT

echo "EFFECTIVE BATCH = $NPROC_PER_NODE x 1 x 8 = $((NPROC_PER_NODE*8))  (基线为 16)"
echo "NODE=$(hostname) noText2500 NO-STAGE1 rows=$ROWS -> 1 epoch = $STEPS steps -> $OUT"
$ENV/bin/python -m swift.cli.main sft \
  --model $DT/model --dataset $DATA \
  --loss_scale stepw --disable_ignore_empty_think true \
  --tuner_type lora --target_modules all-linear \
  --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
  --learning_rate 1e-4 --num_train_epochs 1 --max_steps $STEPS \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --data_seed 42 --save_steps 10 --save_total_limit 99 --logging_steps 10 \
  --max_length 131072 --truncation_strategy delete --padding_free true --use_liger_kernel true \
  --output_dir $OUT --torch_dtype bfloat16 --attn_impl flash_attn
echo "=== DONE noText2500 nos1 CFWFIX $(date) ==="
