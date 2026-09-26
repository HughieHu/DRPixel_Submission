#!/bin/bash
#SBATCH --job-name=evalrp
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/work/beval_%x_%j.log
set -u
# lorax 有不申 gres 的裸抢者: 预检分到的 GPU, 被占>2GB 则退
# 分级守卫: 小残留(<20GB)放行并压低 sglang 显存占比; 真占卡(>=20GB)才退出
SQUAT_MAX=0
for u in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$u" -gt "$SQUAT_MAX" ] && SQUAT_MAX=$u
done
if [ "$SQUAT_MAX" -ge 8000 ]; then
  echo "SQUATTED HARD: GPU has ${SQUAT_MAX}MiB in use, aborting"; exit 1
elif [ "$SQUAT_MAX" -gt 2000 ]; then
  echo "SQUATTED SOFT: ${SQUAT_MAX}MiB in use -> continuing with reduced mem-fraction"
  export MEMFRAC=0.60
fi
# A node cannot automount its OWN /net export: on NODE every /net/NODE/... path
# resolves to nothing, so the job dies in 2s with "No such file or directory" on hf_cache.
# Prefer the node-local /data when it is really NODE storage; fall back to /net elsewhere.
if [ -d /path/to/work/hf_cache ] && [ -x /path/to/work/vllmenv/bin/python ] \
   && [ -x /path/to/work/milesenv/bin/python ] && [ -d /path/to/work/VL_DR/experiment ]; then
  D=/path/to/work
else
  D=/path/to/work
fi
S=/path/to/work
echo "=== ROOT D=$D on $(hostname -s) ==="
for _p in "$D/hf_cache" "$D/vllmenv/bin/python" "$D/milesenv/bin/python" "$D/VL_DR/experiment/run.py"; do
  [ -e "$_p" ] || { echo "=== FATAL: missing $_p on $(hostname -s) -- refusing to start ==="; exit 1; }
done
export TMPDIR=/tmp/tsh_ev_${SLURM_JOB_ID}; mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor TRITON_CACHE_DIR=$TMPDIR/triton XDG_CACHE_HOME=$TMPDIR/xdg RAY_TMPDIR=$TMPDIR/ray
trap "rm -rf $TMPDIR" EXIT
SGENV=$D/milesenv; EENV=$D/vllmenv; E=/path/to/vldr/experiment
MODE=${MODE:-base}; CONFIG=${CONFIG:-page_q8}; PORT=${PORT:-$((17000 + SLURM_JOB_ID % 2000))}
export MAX_PIXELS=602112 HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=$SGENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# doc_map 修复: /net/.../doc_map.json 里存的是 /data 路径(NODE本地视角), 非NODE节点读它会
# 全部 No-such-file(实测 horton 上 42-99% 检索失败). 用 /net 视角的副本覆盖到 job 私有 E 目录.
mkdir -p $TMPDIR/expdir && cp -f $S/doc_map_net.json $TMPDIR/expdir/doc_map.json 2>/dev/null && export VLDR_DOCMAP=$TMPDIR/expdir/doc_map.json
BASE=$D/rw30_sft/model; [ -f $BASE/config.json ] || BASE=$(find $D/hf_cache/hub/models--Qwen--Qwen3.6-27B/snapshots -maxdepth 1 -mindepth 1 -type d | head -1)

if [ "$MODE" = "v3rank" ]; then
  OUTDIR=${OUTDIR:-$D/rw30_sft/out_cfwV3}; SWENV=$D/rw30_sft/env; TAG=cfwV3rank${RTAG:-}
  if [ -n "${MODEL_DIR:-}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    echo "=== SKIP MERGE: serving ${MODEL_DIR} ==="; SERVE_MODEL="${MODEL_DIR}"
  else
  # CKPT_STEP=<n> pins one checkpoint. Without it the newest is taken -- which
  # silently made a "sweep" evaluate the same final weights N times.
  if [ -n "${CKPT_STEP:-}" ]; then
    CKPT=$(ls -d $OUTDIR/v*/checkpoint-${CKPT_STEP} 2>/dev/null | tail -1)
    [ -z "$CKPT" ] && { echo "NO checkpoint-${CKPT_STEP} in $OUTDIR"; exit 1; }
  else
  CKPT=$(for c in $(ls -d $OUTDIR/v*/checkpoint-* 2>/dev/null | sort -V); do [ -f "$c/adapter_config.json" ] && [ -f "$c/adapter_model.safetensors" ] && echo "$c"; done | tail -1)
  fi
  echo "=== CKPTCHK using $CKPT ==="
  [ -z "$CKPT" ] && { echo "NO CKPT in $OUTDIR"; exit 1; }
  MERGED=$D/merged_cfwV3rank${RTAG:-}; rm -rf $MERGED
  echo "=== merge $CKPT -> $MERGED ==="
  $SWENV/bin/python -m swift.cli.main export ${MERGE_BASE:+--model $MERGE_BASE} --adapters $CKPT --merge_lora true --output_dir $MERGED --torch_dtype bfloat16 2>&1
  [ -f "$MERGED/config.json" ] || { echo "MERGE FAILED"; ls -la $MERGED; exit 1; }
  SERVE_MODEL=$MERGED
  fi
else
  SERVE_MODEL=$BASE; TAG=qwen27b_base${RTAG:-}
fi
CT=$SERVE_MODEL/chat_template.jinja; [ -f "$CT" ] || CT=$BASE/chat_template.jinja
echo "=== MODE=$MODE TAG=$TAG SERVE_MODEL=$SERVE_MODEL PORT=$PORT ==="

SLOG=$S/beval_serve_${SLURM_JOB_ID}.log
if ss -tln 2>/dev/null | grep -q ":$PORT "; then echo "=== PORT $PORT ALREADY IN USE -- refusing to start (port-collision guard) ==="; exit 1; fi
CUDA_VISIBLE_DEVICES=0,1,2,3 $SGENV/bin/python -m sglang.launch_server --model-path $SERVE_MODEL --served-model-name qwen3-27b \
  --host 0.0.0.0 --port $PORT --tp 4 --attention-backend fa3 --enable-multimodal --context-length 131072 \
  --mem-fraction-static ${MEMFRAC:-0.78} --chat-template $CT --enable-mm-global-cache --mamba-radix-cache-strategy auto --trust-remote-code > $SLOG 2>&1 & SPID=$!
for i in $(seq 1 120); do
  curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && { echo "=== SERVE READY ~$((i*10))s ==="; break; }
  kill -0 $SPID 2>/dev/null || { echo "=== SERVE DIED ==="; tail -30 $SLOG; exit 1; }
  sleep 10
done

SERVED_PATH=$(curl -sf http://localhost:$PORT/get_model_info 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('model_path',''))" 2>/dev/null)
echo "=== served model_path = $SERVED_PATH (expected $SERVE_MODEL) ==="
if [ -n "$SERVED_PATH" ] && [ "$SERVED_PATH" != "$SERVE_MODEL" ]; then
  echo "=== PORT COLLISION: port $PORT serves someone else -- aborting ==="; kill $SPID 2>/dev/null; exit 1
fi

export VLDR_ROOT=/path/to/vldr PYTHONPATH=$D/VL_DR/embedding_build:${PYTHONPATH:-} TOKENIZERS_PARALLELISM=false
export IDX_PACKED=${IDX_PACKED:-$D/pgq8_packed_bench_all_n2} VLDR_BENCH=1
echo "=== IDXCHK IDX_PACKED=$IDX_PACKED exists=$([ -f ${IDX_PACKED}.X.npy ] && echo YES || echo NO) ==="
export AGENT_MODEL=qwen3-27b-local QWEN_SERVED=qwen3-27b QWEN_PORT=$PORT QWEN_HOST=localhost HARNESS=chat CONCURRENCY=${CONCURRENCY:-12}
export IMG_MAX_PIXELS=${RES_PX:-0} MAX_IMAGES=24   # RES_PX=602112 -> 低分辨率; unset/0 = 全分辨率(默认)
# seed must be explicit: agent_model.py defaults to 0, which silently made past "r3" runs a
# repeat of seed 0 instead of a new seed. Override with SEED=<n> at sbatch time.
export QWEN_SEED=${SEED:-0}
echo "=== SEEDCHK QWEN_SEED=$QWEN_SEED ==="
cd $E; mkdir -p $E/results_chat $S/VL_DR/experiment/results_chat
for BENCH in ${BENCHES:-bench_irp_100 bench_m3_100}; do
  cp -f $S/${BENCH}.json $E/${BENCH}.json 2>/dev/null  # __CP_BENCH__
  RES=$E/results_chat/${BENCH}__${TAG}__${CONFIG}.json
  [ -n "${FORCE_FRESH:-}" ] && rm -f "$RES" "$S/VL_DR/experiment/results_chat/$(basename $RES)"
  echo "=== EVAL $BENCH -> $(basename $RES) $(date) ==="
  CUDA_VISIBLE_DEVICES=1 $EENV/bin/python run.py --config $CONFIG --bench $E/${BENCH}.json --out $RES 2>&1
  echo "=== rc=$? ==="
  cp -f $RES $S/VL_DR/experiment/results_chat/ 2>/dev/null && echo "copied $(basename $RES) to scratch"
done
kill $SPID 2>/dev/null; echo "=== ALL DONE $(date) ==="
