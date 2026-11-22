#!/usr/bin/env bash
# The plug-in's structure (plan of 2026-11-22, 插件结构优化, as revised in its section eight), on Qwen3-VL-8B, the 610 regions of the
# second round, five folds, each fold read by its shared H reader with images 2 and 3 through the fold's adapter (V's reading):
#   RUN=maa    S1 MAA's three-branch adapters on every block's input (models.maa), V-修's loss and settings otherwise;
#              S2 the same adapters on MAA's own: mean squared error + 0.1 x its token discriminator (1e-6) on the merged tokens,
#              5e-5, transformers' cosine with 100 warm-up steps to 0, kept by its validation mean squared error.
#   RUN=judge  B V-修 with the fold's reader as judge: the alignment + 0.1 x the reader's answer loss of the step's real regions,
#              kept by the validation alignment loss (reading_signal).  Its check 5 first (reading_signal gap, five folds): a mean
#              difference under 0.05 means the reader gives no signal -- B is not run (B_NOT_RUN), as the plan says.
# PART=smoke: the plan's checks -- 1-3 (models.maa check: a new set gives the frozen tower's levels bit for bit, the grids and the patch
# order bit for bit, only adapter parameters with gradients) or 3 for B (reading_signal grads), 4 five steps of each arm (finite
# losses, peak memory, seconds a step, the estimate), 5 (B), 6 (tools.plugin_leakage), and eight regions of fold 0 read through each.
# PART=main: the five folds, the readings, the spread (MMD squared; S1 and S2), eval.transcription, eval.privileged with rule S / B
# (--judged, 97.5 %) and the reported S2 pairs.  At most two of our jobs on a card;
# the cards: _common.sh place_on_cards (the user's word of 2026-11-23: any of 0-3, at most two of them ours at a time, the least used first).
# Unattended; a step whose marker is in $R is skipped.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; PART=${PART:-main}; RUN=${RUN:-}
case "$RUN" in maa) B=${N}_plugin_structure20261122 ;; judge) B=${N}_plugin_judge20261122 ;; *) echo 'RUN is maa or judge'; exit 2 ;; esac
case "$PART" in main) R=$B; P_STEPS=2000; P_EVAL=50; STOP=(); LIMIT=(); READ=(0 1 2 3 4) ;;
    smoke) R=${B}_smoke; P_STEPS=2000; P_EVAL=5; STOP=(--stop-after 5); LIMIT=(--limit 8); READ=(0) ;;
    *) echo "PART is main or smoke"; exit 2 ;; esac
P2=${N}_nafnet_folds20261119; V=${N}_revision_folds_human_masked20261117b_views; EV=${P2}_erased_views; RAWV=${P2}_raw_views
P4=${N}_privileged_view20261120; SYN=${P4}_synth; FIX=${N}_covered_view20261121; MAA6=${N}_maa_view20261121; SH=${P2}_reader_shared_f
SHARES=$FIX/checks/strike_shares.json; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
mkdir -p "$R/checks"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
test -e "$P2/COMPLETE" && test -e "$P4/COMPLETE" && test -e "$FIX/COMPLETE" && test -e "$SHARES" || { echo 'the second, fourth or sixth round missing'; exit 2; }
place() { place_on_cards "$@"; }   # MiB WHOLE FUNCTION ARGS...: on the least used of our two cards it fits on (_common.sh)
job() {   # OUT -- COMMAND...: one GPU job on $CARD
    local out="$1"; shift 2
    bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1
}
dirs() { local k s=; for k in "${READ[@]}"; do s+="${s:+,}$1_f$k"; done; echo "$s"; }   # PREFIX -> the folds' folders
mine() { echo "${R}_view_$1_f$2"; }
plugin() {   # OUT FOLD [MORE...]: a tower adapter on V-修's data (real and synthetic pairs, half each)
    local out="$1" fold="$2"; shift 2
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.train_clean_view train --model "$MODEL" --pack "$V" --scan-pack "$RAWV" --fold "$fold" \
        --synthetic "$SYN" --steps "$P_STEPS" --eval-every "$P_EVAL" "${STOP[@]}" "$@" --output-dir "$out"
}
read_with() {   # OUT ADAPTER FOLD: the fold read under context_revision by its shared H reader, images 2 and 3 through ADAPTER
    [ -e "$1/complete.json" ] || job "$1" -- "$PY" -m revrender.recognize_lines --pack "$RAWV" --model "$MODEL" --backend hf --adapter "${SH}$3/adapter" \
        --batch-size 8 --conditions context_revision --fold "$3" --loop-limit 4 "${LIMIT[@]}" --view-student "$2" --student-images 2 3 --output-dir "$1"
}
steps_check() {   # OUT ARM...: check 4 from the five-step runs of fold 0
    "$PY" - "$1" "$R" "$(mine x 0)" "${@:2}" <<'PY'
import json, math, sys
from pathlib import Path
out, R, pattern, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
r = {}
for arm in arms:
    d = Path(pattern.replace('_view_x_', f'_view_{arm}_')); log = [json.loads(l) for l in open(d/'train_log.jsonl')]; done = json.load(open(d/'complete.json'))
    steps = [x for x in log if x['step'] > 0]; secs = [b['seconds']-a['seconds'] for a, b in zip(steps, steps[1:]) if 'val_loss' not in b]
    m = json.load(open(d/'run_manifest.json'))
    r[arm] = {'losses': [x['loss'] for x in steps], 'finite': all(math.isfinite(x['loss']) for x in steps), 'step0_val': log[0]['val_loss'],
              'seconds_a_step': secs, 'peak_allocated_gib': done.get('peak_allocated_gib'), 'trainable_parameters': m['trainable_parameters'],
              'saved_files': sorted(p.name for p in (d/'student').iterdir()), 'extra_step_keys': sorted({k for x in steps for k in x} - {'step', 'loss', 'lr', 'synthetic', 'covered', 'seconds'})}
    per = sorted(secs)[len(secs)//2] if secs else None
    r[arm]['estimate_hours_a_fold'] = round(per*2000/3600, 2) if per else None
r['passed'] = all(v['finite'] for v in r.values() if isinstance(v, dict))
Path(out).write_text(json.dumps(r, indent=2)); print(json.dumps({a: {k: v[k] for k in ('finite', 'peak_allocated_gib', 'estimate_hours_a_fold')} for a, v in r.items() if isinstance(v, dict)}))
sys.exit(0 if r['passed'] else 'a loss is not finite')
PY
}
score() {   # ARMS_ REFS JUDGED PAIRS are set by the caller
    for ref in "${REFS[@]}"; do [ -e "$R/scores_vs_$ref.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference "$ref" \
        $([ "$ref" != V-修 ] || echo --readings "$R/readings.jsonl") --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
    [ -e "$R/analysis/analysis.json" ] || "$PY" -m revrender.eval.privileged analyse --pack "$V" "${ARMS_[@]}" --shares "$SHARES" \
        $(for ref in "${REFS[@]}"; do echo --scores "$ref=$R/scores_vs_$ref.json"; done) "${JUDGED[@]}" "${PAIRS[@]}" --output-dir "$R/analysis" > "$R/analysis.log" 2>&1
}

if [ "$PART" = smoke ] && ! done_ leakage; then
    "$PY" -m revrender.tools.plugin_leakage --pack "$V" --scan-pack "$RAWV" --synthetic "$SYN" --output "$R/checks/check6_leakage.json"
    mark leakage; fi

if [ "$RUN" = maa ]; then
    declare -A MORE=([s1]="--structure maa" [s2]="--structure maa --loss mse --discriminator --lr 5e-5 --schedule maa --warmup 100")
    if [ "$PART" = smoke ] && ! done_ adapter_checks; then
        [ -e "$R/checks/check1_3.json" ] || place 12000 0 job "$R/checks/check1_3.json" -- "$PY" -m revrender.models.maa check --model "$MODEL" --pack "$V" \
            --scan-pack "$RAWV" --regions 20 --output "$R/checks/check1_3.json"; wait
        "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({**r["passed"], "tower_parameters_trainable": r["check3"]["tower_parameters_trainable"]})); sys.exit(0 if all(r["passed"].values()) else "checks 1-3 failed")' "$R/checks/check1_3.json"
        mark adapter_checks; fi
    if ! done_ plugins; then
        for arm in s1 s2; do for k in "${READ[@]}"; do read -ra more <<< "${MORE[$arm]}"; place 12000 0 plugin "$(mine $arm "$k")" "$k" "${more[@]}"; done; done; wait
        for arm in s1 s2; do for k in "${READ[@]}"; do test -e "$(mine $arm "$k")/student/maa_config.json"; done; done
        [ "$PART" != smoke ] || steps_check "$R/checks/check4_steps.json" s1 s2
        mark plugins; fi
    if ! done_ reads; then
        for arm in s1 s2; do for k in "${READ[@]}"; do place 22000 1 read_with "$R/read_${arm}_f$k" "$(mine $arm "$k")/student" "$k"; done; done; wait
        for arm in s1 s2; do for k in "${READ[@]}"; do test -e "$R/read_${arm}_f$k/complete.json"; done; done
        mark reads; fi
    if [ "$PART" = main ]; then
        if ! done_ measures; then
            [ -e "$R/spread.json" ] || place 12000 1 job "$R/spread.json" -- "$PY" -m revrender.train_clean_view spread --model "$MODEL" --pack "$V" \
                --student "S1=${R}_view_s1_f{fold}/student" --student "S2=${R}_view_s2_f{fold}/student" --folds "${READ[@]}" --output "$R/spread.json"; wait
            test -e "$R/spread.json"; mark measures; fi
        if ! done_ score; then
            ARMS_=(--arm "S1=$(dirs "$R/read_s1")" --arm "S2=$(dirs "$R/read_s2")" --arm "V-修=$(dirs "$FIX/read_fix")" --arm "V-判别=$(dirs "$MAA6/read_disc")"
                   --arm "E-NAFNet=$(dirs "$P2/read_E-NAFNet")" --arm "R0=$(dirs "$P2/read_R0")" --arm "H=$(dirs "$P2/read_H")")
            REFS=(V-修 S2 H R0 E-NAFNet); JUDGED=(--judged S1); PAIRS=(--pair V-修:S2 --pair S1:S2)
            score; mark score; fi
    fi
    touch "$R/COMPLETE"; stamp "$PART maa complete"; exit 0
fi

# RUN=judge: check 5 (run once, in the smoke part; the main part takes its answer) decides whether B runs at all
GAP=${B}_smoke/checks/check5_gap.json
if [ "$PART" = smoke ] && ! done_ gap; then
    for k in 0 1 2 3 4; do [ -e "$R/checks/gap_f$k.json" ] || place 22000 1 job "$R/checks/gap_f$k.json" -- "$PY" -m revrender.reading_signal gap \
        --model "$MODEL" --reader-adapter "${SH}$k/adapter" --pack "$V" --scan-pack "$RAWV" --fold "$k" --output "$R/checks/gap_f$k.json"; done; wait
    "$PY" - "$R/checks" "$GAP" <<'PY'
import json, sys
from pathlib import Path
here, out = Path(sys.argv[1]), Path(sys.argv[2])
per = {k: {f: json.loads((here/f'gap_f{k}.json').read_text())[f] for f in ('regions', 'mean_scan', 'mean_clean', 'difference')} for k in range(5)}
mean = sum(v['difference'] for v in per.values())/5
r = {'folds': per, 'mean_difference': mean, 'rule': "B runs only when the five folds' mean difference is at least 0.05", 'b_runs': mean >= .05}
out.write_text(json.dumps(r, indent=2)); print(json.dumps({'mean_difference': mean, 'b_runs': r['b_runs']}))
PY
    mark gap; fi
test -e "$GAP" || { echo 'check 5 (the smoke part) has not run'; exit 2; }
if ! "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["b_runs"] else 1)' "$GAP"; then
    touch "$R/B_NOT_RUN" "$R/COMPLETE"; stamp "$PART judge: the reader gives no signal (check 5), B not run"; exit 0
fi
if [ "$PART" = smoke ] && ! done_ grads; then
    [ -e "$R/checks/check3_grads.json" ] || place 22000 1 job "$R/checks/check3_grads.json" -- "$PY" -m revrender.reading_signal grads --model "$MODEL" \
        --reader-adapter "${SH}0/adapter" --pack "$V" --scan-pack "$RAWV" --fold 0 --output "$R/checks/check3_grads.json"; wait
    "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({k: r[k] for k in ("passed", "parameters_with_gradients", "tower_parameters_trainable", "reader_parameters_trainable")})); sys.exit(0 if r["passed"] else "check 3 failed")' "$R/checks/check3_grads.json"
    mark grads; fi
if ! done_ plugins; then
    for k in "${READ[@]}"; do place 22000 1 plugin "$(mine b "$k")" "$k" --reader-adapter "${SH}$k/adapter"; done; wait
    for k in "${READ[@]}"; do test -e "$(mine b "$k")/student/adapter_config.json"; done
    [ "$PART" != smoke ] || steps_check "$R/checks/check4_steps.json" b
    mark plugins; fi
if ! done_ reads; then
    for k in "${READ[@]}"; do place 22000 1 read_with "$R/read_b_f$k" "$(mine b "$k")/student" "$k"; done; wait
    for k in "${READ[@]}"; do test -e "$R/read_b_f$k/complete.json"; done
    mark reads; fi
if [ "$PART" = main ] && ! done_ score; then
    ARMS_=(--arm "B=$(dirs "$R/read_b")" --arm "V-修=$(dirs "$FIX/read_fix")" --arm "E-NAFNet=$(dirs "$P2/read_E-NAFNet")" --arm "R0=$(dirs "$P2/read_R0")" --arm "H=$(dirs "$P2/read_H")")
    REFS=(V-修 H R0 E-NAFNet); JUDGED=(--judged B); PAIRS=()
    score; mark score; fi
touch "$R/COMPLETE"; stamp "$PART judge complete"; exit 0
