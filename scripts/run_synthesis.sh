#!/usr/bin/env bash
# Synthetic strikes on clean lines as a controlled trial for the prefill model (plans and rules written before any reading:
# work/synthesis_plan20261116/合成数据_方案与判定规则_2026-11-16.md, its sections three to five replaced by
# 合成数据_第二版_按一笔来_方案与判定规则_2026-11-17.md; user 2026-11-17 through the planning session).
# PART=materials (the check before any pasting or training): the three sources of strikes as single strokes -- H the annotator's erased
#   pixels of the round-2 training regions cut into lines, groups and blobs, lines straightened and repaired; P paper 1's bank lines
#   regrown on their page and repaired; D drawn ones where an H unit lay -- of writers other than the held-out, fixed-validation and
#   paper 1's test writers, 200 of each put once on a host's own writing in the same place (data.synthesis materials): checks 1, 2,
#   3 and 5 and the page to look at; and check 4: the units alone on white read by the untaught Qwen3-VL-8B (raw_revision, loops
#   stopped), the share with a letter, digit or character read against the drawn lines (synthesis read-check).
# Card 2 only, waiting while it is taken.  Outputs paper2_clean_synth_materials_r$ROUND_ (from 3); marker $O/paper2_synth20261116.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_synth20261116
CARD=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; NUMA=1; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
CATALOGUE=/home/zsh/hme/code_358_spatial_v1/gold/line_edit_all5002_policy_decisions_20260814_v1/canonical/exact_crops_46902.jsonl
ROUND_=${ROUND_:?set ROUND_ (3 or more)}; MAT=${N}_synth_materials_r$ROUND_
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR

if [ "${PART:-materials}" = materials ]; then
    if ! done_ "materials_r$ROUND_"; then
        "$PY" -m revrender.data.synthesis materials --set ${N}_prefill_set_r2 --bank ${N}_strike_bank_strokes \
            --dataset ${N}_struck_dataset20261116/regions.json --catalogue "$CATALOGUE" --mapping $O/paper2_splits/index_manifest.json \
            --same-student $O/paper2_splits/same_student_pairs.json --templates ${N}_page_templates/templates.npz \
            --pages $O/paper2_pages_clean_unlabeled3976 --output-dir "$MAT" > "$R/materials_r$ROUND_.log" 2>&1
        mark "materials_r$ROUND_"; fi
    if ! done_ "read_r$ROUND_"; then
        bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "${MAT}_read" -- "$PY" -m revrender.recognize_lines --pack "$MAT/reader" \
            --model "$MODEL" --backend hf --batch-size 8 --conditions raw_revision --loop-limit 4 --output-dir "${MAT}_read" > "$R/read_r$ROUND_.log" 2>&1
        "$PY" -m revrender.data.synthesis read-check --pack "$MAT/reader" --predictions "${MAT}_read/predictions.jsonl" \
            --output "$R/read_check_r$ROUND_.json" >> "$R/read_r$ROUND_.log" 2>&1
        mark "read_r$ROUND_"; fi
    stamp "materials round $ROUND_ complete"; touch "$R/MATERIALS_R${ROUND_}_COMPLETE"
fi
