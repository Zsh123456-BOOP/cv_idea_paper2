#!/usr/bin/env bash
# Pilot (user 2026-11-08): can a taught Qwen3-VL-8B read the struck text straight from the scan, better than reading after erasing?
#  1. data.struck_pack train: every region with confirmed, readable struck text of the 201 regions' export, training batch 1 and the
#     test page's export of 10-22, no writer of the evaluation strikes; data.struck_pack eval: the 122 strikes (41 validation, 81
#     development) of the writing gate's deciding batch, each with the scan and RevSep s364's erased crop.
#  2. train_reader --pack --condition raw_struck: a LoRA on the 8B, the words asking only for what is struck out.
#  3. one engine (recognize_lines --backend hf) reads the pack: the untaught 8B on the scan asked for the struck text (raw_struck), on
#     the scan plainly (raw_neutral) and on RevSep's erased crop (erased_neutral, the "erase, then read" route); the taught 8B on the scan
#     asked for the struck text.
#  4. eval.struck_reader, the reference the untaught 8B after erasing; the rule written before the reading is in its docstring.
# Unattended; a step whose marker is in $R is skipped.  Every job goes to whichever card has room (_common.sh open_card).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_struck_reader_pilot20261108
E=${N}_revsep_gate_versions_final; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
TRAIN=${N}_struck_reader_train_pack; EVAL=${N}_struck_reader_eval_pack; LORA=${N}_struck_reader_8b
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR

if ! done_ packs; then
    [ -e "$TRAIN/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack train --export "$O/paper2_pixel_round_struck20261012_inputs/training_annotations.json" \
        --export "$O/paper2_training_batch1_returned/export.json" --export "$O/paper2_test_export20261022/export.json" --pages "$O/paper2_full_review20261010/images" \
        --same-student "$O/paper2_splits/same_student_pairs.json" --exclude-pack "$E/inputs" --output-dir "$TRAIN" > "$R/packs.log" 2>&1
    [ -e "$EVAL/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack eval --evaluation "$E" --restorer RevSep-s364 --output-dir "$EVAL" >> "$R/packs.log" 2>&1
    mark packs; fi
job() {   # CARD NUMA OUT -- COMMAND...: one GPU job
    local card="$1" numa="$2" out="$3"; shift 4
    bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$out" -- "$@" > "$out.log" 2>&1
}
if ! done_ train; then
    place_job 22000 job "$LORA" -- "$PY" -m revrender.train_reader --pack "$TRAIN" --condition raw_struck --model "$MODEL" --steps 120 --batch-size 2 \
        --accumulate 8 --warmup 10 --eval-every 40 --checkpointing --workers 2 --output-dir "$LORA"
    wait; test -e "$LORA/complete.json"; mark train; fi
if ! done_ read; then
    [ -e "$R/read_untaught/complete.json" ] || place_job 22000 job "$R/read_untaught" -- "$PY" -m revrender.recognize_lines --pack "$EVAL" --model "$MODEL" --backend hf \
        --batch-size 8 --conditions raw_struck raw_neutral erased_neutral --output-dir "$R/read_untaught"
    [ -e "$R/read_taught/complete.json" ] || place_job 22000 job "$R/read_taught" -- "$PY" -m revrender.recognize_lines --pack "$EVAL" --model "$MODEL" --backend hf \
        --adapter "$LORA/adapter" --batch-size 8 --conditions raw_struck --output-dir "$R/read_taught"
    wait; test -e "$R/read_untaught/complete.json"; test -e "$R/read_taught/complete.json"; mark read; fi
done_ score || { "$PY" -m revrender.eval.struck_reader --pack "$EVAL" --readings "$R/read_untaught" "$R/read_taught" \
    --reference "Qwen3-VL-8B-Instruct|erased_neutral" --output "$R/scores.json" > "$R/score.log" 2>&1; mark score; }
touch "$R/COMPLETE"; stamp complete
