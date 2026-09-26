#!/bin/bash
#SBATCH --job-name=mergeSPLITMED
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=NODE
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=250G --time=3:00:00
#SBATCH --output=/path/to/work/mergeSPLITMED_%j.log
set -u
D=/path/to/work
CKPT=$(ls -d $D/rw30_sft/out_noText2500_nos1_SPLITMED/v*/checkpoint-156 | tail -1)
OUT=$D/SPLITMED_merged
echo "merging $CKPT -> $OUT"
rm -rf $OUT
$D/rw30_sft/env/bin/python -m swift.cli.main export --adapters $CKPT --merge_lora true --output_dir $OUT --torch_dtype bfloat16 2>&1 | tail -5
[ -f "$OUT/config.json" ] && echo MERGE_OK || { echo MERGE_FAILED; exit 1; }
