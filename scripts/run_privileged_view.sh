#!/usr/bin/env bash
# The method stage, fourth round (the planning session's plan, rules written before any reading:
# work/method_candidates20261120/第四轮_特权视图三样试验_方案与判定规则_2026-11-20.md): the clean image a region has at training
# time and not at test time, used to teach the reader.  On the second round's five folds (paper2_clean_nafnet_folds20261119: its
# folds, shared readers, NAFNet's held-out erased crops; its H, E-NAFNet and R0 readings are the controls, not read again):
#   X  a context_revision reader per fold taught on NAFNet's held-out erased crops of the other folds (train_reader --all-rest, step
#      100, the shared reader's recipe), reading its fold's E-NAFNet input;
#   V  the reader's vision tower taught to see the scan as clean (train_clean_view): V-真 on the real pairs, V-真+合 with the
#      synthetic pairs as well (data.synth_views); read by the fold's shared reader, the scan through the frozen tower, the second image
#      (the scan again) and the line through the student; a wiring check first (fold 0 read through the same path, the adapter off,
#      against H's readings: at most 2% differ, else the round stops); the feature distances;
#   A  a raw_context_revision reader taught 200 steps on synthetic regions the fold's shared reader read (the clean host as the second
#      image), then 100 steps on the real regions (R0's recipe); its control R0-长, R0's recipe for 300 steps.
# eval.transcription against E-NAFNet, R0, H, V-真 and R0-长; eval.privileged the plan's rules.  Card 1 only (the user's word,
# 2026-11-19: 「只用 1号卡」); cards 0, 2 and 3 are not touched, ALLOW_GPU3 is not set.  PART=smoke: a few steps, eight regions of
# fold 0 read, 40 synthetic regions a fold; PART=main: the round.  Unattended; a step whose marker is in $R is skipped.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; B=${N}_privileged_view20261120; PART=${PART:-main}
# V_STEPS: the vision adapter's step cap, from the smoke test's pace (about 0.7 s a step, two trainings on the card) to keep a training
# within about 30 minutes (plan, 4.2): 2,000 steps, about 25 minutes and the validations
case "$PART" in main) TAG=; STEPS=100; LONG=300; FIRST=200; V_STEPS=2000; V_EVAL=50; TAKE=3000; TEACH=1500; LIMIT=(); TLIMIT=(); READ=(0 1 2 3 4) ;;
    smoke) TAG=_smoke; STEPS=2; LONG=3; FIRST=2; V_STEPS=20; V_EVAL=10; TAKE=40; TEACH=16; LIMIT=(--limit 8); TLIMIT=(--limit 16); READ=(0) ;;
    *) echo "PART is main or smoke"; exit 2 ;; esac
R=$B$TAG; mkdir -p "$R"
P2=${N}_nafnet_folds20261119; FOLDS_PACK=${N}_revision_folds_human_masked20261117b; V=${FOLDS_PACK}_views
EV=${P2}_erased_views; RAWV=${P2}_raw_views; SYN=${B}${TAG}_synth
MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
test -e "$P2/COMPLETE" || { echo 'the second round is not complete'; exit 2; }
CARD=${ALL_CARDS[1]}; NUMA=$(numa_of "$CARD")
place() { one_card_place "$CARD" "$@"; }   # MiB WHOLE FUNCTION ARGS...
job() {   # OUT -- COMMAND...: one GPU job on card 1
    local out="$1"; shift 2
    bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1
}
teach() {   # OUT CONDITION PACK FOLD STEPS [MORE...]: a reader taught on every region of the other folds (the shared reader's recipe)
    local out="$1" condition="$2" pack="$3" fold="$4" steps="$5"; shift 5
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.train_reader --pack "$pack" --condition "$condition" --fold "$fold" --all-rest --model "$MODEL" \
        --steps "$steps" --batch-size 2 --accumulate 8 --warmup 10 --eval-every 10 --checkpointing --workers 2 --answer latex "$@" --output-dir "$out"
}
read_with() {   # OUT CONDITION PACK ADAPTER FOLD [MORE...]: the fold read, loops stopped
    local out="$1" condition="$2" pack="$3" adapter="$4" fold="$5"; shift 5
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.recognize_lines --pack "$pack" --model "$MODEL" --backend hf --adapter "$adapter" \
        --batch-size 8 --conditions "$condition" --fold "$fold" --loop-limit 4 "${LIMIT[@]}" "$@" --output-dir "$out"
}
dirs() { local k s=; for k in "${READ[@]}"; do s+="${s:+,}$1_f$k"; done; echo "$s"; }   # PREFIX -> the folds' folders
SH=${P2}_reader_shared_f

# The synthetic regions each fold may learn from (CPU).
if ! done_ data; then
    [ -e "$SYN/summary.json" ] || "$PY" -m revrender.data.synth_views build --synth "$O/paper2_clean_synth_regions_ink" --pack "$V" \
        --set "$O/paper2_clean_prefill_set_r2" --bank "$O/paper2_clean_strike_bank_strokes" --same-student "$O/paper2_splits/same_student_pairs.json" \
        --dataset "$O/paper2_clean_line_dataset20261117/regions.json" --test-halves "$ROOT/configs/test_halves.json" \
        --pages "$O/paper2_pages_clean_unlabeled3976" --take "$TAKE" --output-dir "$SYN" > "$SYN.log" 2>&1
    # the plan's section five: a fold with fewer than 1,500 regions stops the round until the planning session decides
    [ "$PART" != main ] || "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); bad=[k for k,v in s["folds"].items() if v["kept"]<3000]; sys.exit("folds short of 3,000: "+",".join(bad) if bad else 0)' "$SYN/summary.json"
    mark data; fi

# X: readers taught on NAFNet's held-out erased crops.
if ! done_ X; then
    for k in "${READ[@]}"; do place 22000 1 teach "${B}${TAG}_reader_X_f$k" context_revision "$EV" "$k" "$STEPS"; wait; done
    for k in "${READ[@]}"; do place 22000 1 read_with "$R/read_X_f$k" context_revision "$EV" "${B}${TAG}_reader_X_f$k/adapter" "$k"; wait; done
    for k in "${READ[@]}"; do test -e "$R/read_X_f$k/complete.json"; done
    mark X; fi

# V: the vision adapters, the wiring check, the distances, the readings.
student() {   # OUT FOLD [--synthetic DIR]
    local out="$1" fold="$2"; shift 2
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.train_clean_view train --model "$MODEL" --pack "$V" --raw-pack "$RAWV" --fold "$fold" \
        --steps "$V_STEPS" --eval-every "$V_EVAL" "$@" --output-dir "$out"
}
distance() {   # OUT STUDENT FOLD
    [ -e "$1/distances.jsonl" ] || job "$1" -- "$PY" -m revrender.train_clean_view distance --model "$MODEL" --student "$2" --pack "$V" --raw-pack "$RAWV" \
        --erased-pack "$EV" --fold "$3" --output-dir "$1"
}
if ! done_ V_train; then
    for k in "${READ[@]}"; do place 12000 0 student "${B}${TAG}_view_real_f$k" "$k"; place 12000 0 student "${B}${TAG}_view_synth_f$k" "$k" --synthetic "$SYN"; done
    wait; for k in "${READ[@]}"; do test -e "${B}${TAG}_view_real_f$k/student/adapter_config.json"; test -e "${B}${TAG}_view_synth_f$k/student/adapter_config.json"; done
    mark V_train; fi
if ! done_ V_wiring; then
    # the teacher's features of the annotator's erased crop and line, through the student's path with its adapter off: as H read
    place 22000 1 read_with "$R/wiring_f0" context_revision "$V" "${SH}0/adapter" 0 --view-student "${B}${TAG}_view_synth_f0/student" --student-images 2 3 --student-off; wait
    "$PY" - "$R/wiring_f0" "$P2/read_H_f0" "$R/wiring.json" <<'PY'
import json, sys
from pathlib import Path
read = lambda d: {g['id']: g['text'] for g in map(json.loads, (Path(d)/'predictions.jsonl').read_text().splitlines())}
mine, h = read(sys.argv[1]), read(sys.argv[2]); ids = sorted(mine)
differ = [k for k in ids if mine[k] != h[k]]; share = len(differ)/len(ids)
Path(sys.argv[3]).write_text(json.dumps({'regions': len(ids), 'differ': len(differ), 'share': share, 'ids': differ, 'passed': share <= .02}, indent=2))
print(json.dumps({'regions': len(ids), 'differ': len(differ), 'share': share}))
sys.exit(0 if share <= .02 else 'wiring check failed: more than 2% of the readings differ from H')
PY
    mark V_wiring; fi
if ! done_ V_read; then
    for k in "${READ[@]}"; do for arm in real synth; do place 6000 0 distance "$R/distance_${arm}_f$k" "${B}${TAG}_view_${arm}_f$k/student" "$k"; done; done; wait
    for k in "${READ[@]}"; do for arm in real synth; do
        place 22000 1 read_with "$R/read_V-${arm}_f$k" context_revision "$RAWV" "${SH}$k/adapter" "$k" --view-student "${B}${TAG}_view_${arm}_f$k/student" --student-images 2 3; wait
    done; done
    for k in "${READ[@]}"; do for arm in real synth; do test -e "$R/read_V-${arm}_f$k/complete.json"; test -e "$R/distance_${arm}_f$k/distances.jsonl"; done; done
    mark V_read; fi

# A: the teacher's answers, the two-stage readers, R0-长.
teacher() {   # FOLD: the fold's shared reader reads the first regions of its synthetic clean pack
    [ -e "$R/teacher_f$1/complete.json" ] || job "$R/teacher_f$1" -- "$PY" -m revrender.recognize_lines --pack "$SYN/fold_$1_clean" --model "$MODEL" --backend hf \
        --adapter "${SH}$1/adapter" --batch-size 8 --conditions context_revision --loop-limit 4 --limit "$TEACH" --output-dir "$R/teacher_f$1"
}
first_stage() {   # FOLD
    [ -e "${B}${TAG}_reader_A1_f$1/complete.json" ] || job "${B}${TAG}_reader_A1_f$1" -- "$PY" -m revrender.train_reader --pack "$R/answered_f$1" \
        --condition raw_context_revision --model "$MODEL" --steps "$FIRST" --batch-size 2 --accumulate 8 --warmup 10 --eval-every 10 --checkpointing --workers 2 \
        --answer latex --output-dir "${B}${TAG}_reader_A1_f$1"
}
if ! done_ A; then
    for k in "${READ[@]}"; do place 22000 1 teacher "$k"; wait
        [ -e "$R/answered_f$k/pack.json" ] || "$PY" -m revrender.data.synth_views answered --pack "$SYN/fold_${k}_raw" --readings "$R/teacher_f$k" --take "$TEACH" \
            --output-dir "$R/answered_f$k" > "$R/answered_f$k.log" 2>&1
        place 22000 1 first_stage "$k"; wait
        place 22000 1 teach "${B}${TAG}_reader_A_f$k" raw_context_revision "$RAWV" "$k" "$STEPS" --init-adapter "${B}${TAG}_reader_A1_f$k/adapter"; wait
        place 22000 1 teach "${B}${TAG}_reader_R0long_f$k" raw_context_revision "$RAWV" "$k" "$LONG"; wait
        place 22000 1 read_with "$R/read_A_f$k" raw_context_revision "$RAWV" "${B}${TAG}_reader_A_f$k/adapter" "$k"; wait
        place 22000 1 read_with "$R/read_R0long_f$k" raw_context_revision "$RAWV" "${B}${TAG}_reader_R0long_f$k/adapter" "$k"; wait
    done
    for k in "${READ[@]}"; do test -e "$R/read_A_f$k/complete.json"; test -e "$R/read_R0long_f$k/complete.json"; done
    mark A; fi

# The scores and the rules.
ARMS_=(--arm "X=$(dirs "$R/read_X")" --arm "V-真=$(dirs "$R/read_V-real")" --arm "V-真+合=$(dirs "$R/read_V-synth")" --arm "A=$(dirs "$R/read_A")"
       --arm "R0-长=$(dirs "$R/read_R0long")" --arm "E-NAFNet=$(dirs "$P2/read_E-NAFNet")" --arm "R0=$(dirs "$P2/read_R0")" --arm "H=$(dirs "$P2/read_H")")
if ! done_ score; then
    for ref in E-NAFNet R0 H V-真 R0-长; do [ -e "$R/scores_vs_$ref.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference "$ref" \
        $([ "$ref" != E-NAFNet ] || echo --readings "$R/readings.jsonl") --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
    [ -e "$R/analysis/analysis.json" ] || "$PY" -m revrender.eval.privileged analyse --pack "$V" "${ARMS_[@]}" \
        $(for ref in E-NAFNet R0 H V-真 R0-长; do echo --scores "$ref=$R/scores_vs_$ref.json"; done) \
        --distances "V-真=$(dirs "$R/distance_real")" --distances "V-真+合=$(dirs "$R/distance_synth")" \
        $(for k in "${READ[@]}"; do echo --teacher "$R/answered_f$k/run_manifest.json"; done) --output-dir "$R/analysis" > "$R/analysis.log" 2>&1
    mark score; fi
touch "$R/COMPLETE"; stamp "$PART complete"
