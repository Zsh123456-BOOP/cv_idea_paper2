#!/usr/bin/env bash
# The method stage's fifth round (plan of 2026-11-20, 第五轮_X与V结合): the fourth round's X and V together, six new readings of the
# 610 regions of the second round's folds:
#   VR-真+合, VR-真  the fourth round's vision adapters (V) with a matched reader: a reader taught on what the adapters make of its
#                    images, each training region through its own fold's adapter (train_reader --view-students; section 4.1);
#   NV, NVR          new adapters (N) that look at NAFNet's held-out erased crop and line instead of the scan (real pairs only;
#                    section 4.2), read by the shared reader / by a matched one;
#   MV, MVR          new adapters (M) that look at the scan with NAFNet's erasure in red (data.struck_pack tinted; section 4.3),
#                    read by the shared reader / by a matched one;
# the feature distances of N and M, eval.transcription against V-真+合, V-真, VR-真, NVR, H, R0, E-NAFNet and X, eval.privileged
# the plan's rules.  PART=smoke (fold 0, a few steps, eight regions read) also runs the plan's section-six checks: 1 the matched
# path's trajectory with the adapters off against the shared reader of fold 0, 2 the training and reading features
# (train_clean_view agree), 3-4 the adapters each matched reader used and its trainable parameters (after every matched reader, in
# both parts), 5 the adapter training with the raw views pack against V-真's step-0 validation loss, 6 NAFNet's pack through the
# adapter path with the adapter off against E-NAFNet, 7 the tinted pack read back (in the builder).  Card 1 only (the user's word,
# 2026-11-19: 「只用 1号卡」); cards 0, 2 and 3 are not touched, ALLOW_GPU3 is not set.  Unattended; a step whose marker is in $R is
# skipped.  HELPER=1 (PART=main, beside a main run under way; the user's word, 2026-11-21: 「可以在多用1张空卡跑」): card 2 takes
# work from the end of the main run's later stages -- the distances, then NV / MV readings and NVR / MVR of folds 4, 3, 2 --
# each only while the main run, going up from fold 0, cannot reach it before it is done (``ahead``); the main run then finds those
# outputs complete and skips them.  It writes no stage marker; its lock and failure marker are its own.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; B=${N}_matched_view20261120; PART=${PART:-main}
case "$PART" in main) TAG=; STEPS=100; P_STEPS=2000; P_EVAL=50; LIMIT=(); READ=(0 1 2 3 4) ;;
    smoke) TAG=_smoke; STEPS=2; P_STEPS=10; P_EVAL=10; LIMIT=(--limit 8); READ=(0) ;;
    *) echo "PART is main or smoke"; exit 2 ;; esac
R=$B$TAG; mkdir -p "$R/checks"
P2=${N}_nafnet_folds20261119; V=${N}_revision_folds_human_masked20261117b_views; EV=${P2}_erased_views; RAWV=${P2}_raw_views
P4=${N}_privileged_view20261120; TINT=${B}_tinted; SH=${P2}_reader_shared_f
MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
exec 8>"$R/lock${HELPER:+_card2}"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/${HELPER:+card2_}FAILED"' ERR
test -e "$P2/COMPLETE" && test -e "$P4/COMPLETE" || { echo 'the second or fourth round is not complete'; exit 2; }
CARD=${ALL_CARDS[$([ -n "${HELPER:-}" ] && echo 2 || echo 1)]}; NUMA=$(numa_of "$CARD")
place() { one_card_place "$CARD" "$@"; }   # MiB WHOLE FUNCTION ARGS...
job() {   # OUT -- COMMAND...: one GPU job on card 1
    local out="$1"; shift 2
    bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1
}
teach() {   # OUT PACK FOLD [MORE...]: a context_revision reader taught on every region of the other folds (the shared reader's recipe)
    local out="$1" pack="$2" fold="$3"; shift 3
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.train_reader --pack "$pack" --condition context_revision --fold "$fold" --all-rest --model "$MODEL" \
        --steps "$STEPS" --batch-size 2 --accumulate 8 --warmup 10 --eval-every 10 --checkpointing --workers 2 --answer latex "$@" --output-dir "$out"
}
read_with() {   # OUT PACK ADAPTER FOLD [MORE...]: the fold read under context_revision, loops stopped
    local out="$1" pack="$2" adapter="$3" fold="$4"; shift 4
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.recognize_lines --pack "$pack" --model "$MODEL" --backend hf --adapter "$adapter" \
        --batch-size 8 --conditions context_revision --fold "$fold" --loop-limit 4 "$@" --output-dir "$out"
}
plugin() {   # OUT SCAN_PACK FOLD: a vision adapter taught on real pairs, looking at the scan pack's window and line (V-真's recipe)
    [ -e "$1/complete.json" ] || job "$1" -- "$PY" -m revrender.train_clean_view train --model "$MODEL" --pack "$V" --scan-pack "$2" --fold "$3" \
        --steps "$P_STEPS" --eval-every "$P_EVAL" --output-dir "$1"
}
distance() {   # OUT STUDENT SCAN_PACK FOLD
    [ -e "$1/distances.jsonl" ] || job "$1" -- "$PY" -m revrender.train_clean_view distance --model "$MODEL" --student "$2" --pack "$V" --raw-pack "$RAWV" \
        --erased-pack "$EV" --scan-pack "$3" --fold "$4" --output-dir "$1"
}
verify_reader() {   # READER FOLD PACK: checks 3 and 4 (the adapters a matched reader used, its trainable parameters, its saved adapter)
    "$PY" - "$1" "$2" "$3" "$R/checks/reader_$(basename "$1").json" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
from safetensors import safe_open
run, fold, pack, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
m = json.loads((run/'run_manifest.json').read_text()); c = json.loads((run/'complete.json').read_text())
rows = Counter(r['fold'] for r in json.loads((pack/'pack.json').read_text())['rows'] if r['fold'] != fold)
with safe_open(str(run/'adapter'/'adapter_model.safetensors'), 'pt') as f: keys = list(f.keys())
result = {'students_used': m['students_used'], 'pack_rows': {f'fold{j}': n for j, n in sorted(rows.items())}, 'students': m['students'],
          'students_encoded': c['students_encoded'], 'trainable_parameters': m['trainable_parameters'], 'adapter_tensors': len(keys),
          'adapter_visual_tensors': sum('visual' in k for k in keys), 'adapter_language_tensors': sum('language_model' in k for k in keys)}
checks = {'fold_not_used': f'fold{fold}' not in m['students_used'] and f'fold{fold}' not in m['students'] and f'fold{fold}' not in c['students_encoded'],
          'own_fold_adapters': all(f'_f{k[4:]}/' in p+'/' for k, p in m['students'].items()),
          'counts_match_the_pack': m['students_used'] == result['pack_rows'],
          'trainable_43646976': m['trainable_parameters'] == 43646976,
          'adapter_language_only': result['adapter_visual_tensors'] == 0 and result['adapter_language_tensors'] == len(keys)}
result['checks'] = checks; result['passed'] = all(checks.values())
out.write_text(json.dumps(result, indent=2)); print(json.dumps({'reader': run.name, **checks}))
sys.exit(0 if result['passed'] else f'checks 3-4 failed for {run}')
PY
}
matched() {   # NAME PACK STUDENT_PATTERN FOLD: a matched reader, its checks, its reading of the fold through the fold's own adapter
    local name="$1" pack="$2" pattern="$3" k="$4" out="${B}${TAG}_reader_$1_f$4"
    place 22000 1 teach "$out" "$pack" "$k" --view-students "$pattern" --student-images 2 3; wait
    verify_reader "$out" "$k" "$pack"
    place 22000 1 read_with "$R/read_${name}_f$k" "$pack" "$out/adapter" "$k" "${LIMIT[@]}" --view-student "${pattern//\{fold\}/$k}" --student-images 2 3; wait
}
compare_reads() {   # MINE THEIRS OUT LABEL: the share of readings that differ (at most 2%)
    "$PY" - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
from pathlib import Path
read = lambda d: {g['id']: g['text'] for g in map(json.loads, (Path(d)/'predictions.jsonl').read_text().splitlines())}
mine, theirs = read(sys.argv[1]), read(sys.argv[2]); ids = sorted(mine)
differ = [k for k in ids if mine[k] != theirs[k]]; share = len(differ)/len(ids)
Path(sys.argv[3]).write_text(json.dumps({'regions': len(ids), 'differ': len(differ), 'share': share, 'ids': differ, 'passed': share <= .02}, indent=2))
print(json.dumps({'check': sys.argv[4], 'regions': len(ids), 'differ': len(differ), 'share': share}))
sys.exit(0 if share <= .02 else f'{sys.argv[4]} failed: more than 2% of the readings differ')
PY
}
dirs() { local k s=; for k in "${READ[@]}"; do s+="${s:+,}$1_f$k"; done; echo "$s"; }   # PREFIX -> the folds' folders
SYNTH_V="${P4}_view_synth_f{fold}/student"; REAL_V="${P4}_view_real_f{fold}/student"
NPLUG="${B}${TAG}_view_N_f{fold}/student"; MPLUG="${B}${TAG}_view_M_f{fold}/student"

if [ -n "${HELPER:-}" ]; then
    [ "$PART" = main ] && done_ NM_train || { echo 'the helper goes beside a main run past the adapters'; exit 2; }
    # a fold k is taken only while the main run has not begun its fold k-2 of that stage: it has two folds of work (about twice the
    # helper's) before it reaches fold k
    ahead() { [ ! -e "$1$(($2-2))" ]; }   # MAIN_PREFIX K
    # the distances: only while the main run has a whole fold of NVR / MVR still to begin
    if [ ! -e "${B}_reader_NVR_f1" ]; then
        for k in 0 1 2 3 4; do
            place 6000 0 distance "$R/distance_N_f$k" "${NPLUG//\{fold\}/$k}" "$EV" "$k"; place 6000 0 distance "$R/distance_M_f$k" "${MPLUG//\{fold\}/$k}" "$TINT" "$k"
        done; wait
        for k in 0 1 2 3 4; do test -e "$R/distance_N_f$k/distances.jsonl"; test -e "$R/distance_M_f$k/distances.jsonl"; done; stamp "card 2: distances"
    fi
    for k in 4 3 2; do
        ahead "$R/read_NV_f" "$k" || break
        place 22000 1 read_with "$R/read_NV_f$k" "$EV" "${SH}$k/adapter" "$k" --view-student "${NPLUG//\{fold\}/$k}" --student-images 2 3; wait
        place 22000 1 read_with "$R/read_MV_f$k" "$TINT" "${SH}$k/adapter" "$k" --view-student "${MPLUG//\{fold\}/$k}" --student-images 2 3; wait
        test -e "$R/read_NV_f$k/complete.json"; test -e "$R/read_MV_f$k/complete.json"; stamp "card 2: NV, MV fold $k"
    done
    for k in 4 3 2; do
        ahead "${B}_reader_NVR_f" "$k" || break
        matched NVR "$EV" "$NPLUG" "$k"; matched MVR "$TINT" "$MPLUG" "$k"
        test -e "$R/read_NVR_f$k/complete.json"; test -e "$R/read_MVR_f$k/complete.json"; stamp "card 2: NVR, MVR fold $k"
    done
    stamp "card 2 helper complete"; exit 0
fi

# The tinted pack (CPU; check 7 inside), shared by both parts.
if ! done_ data; then
    [ -e "$TINT/summary.json" ] || "$PY" -m revrender.data.struck_pack tinted --raw "$RAWV" --erased "$EV" --output-dir "$TINT" > "$TINT.log" 2>&1
    cp "$TINT/summary.json" "$R/checks/check7_tinted.json"
    mark data; fi

# The checks of the plan's section six that need a model (smoke only: the main run starts after the planning session has seen them).
if [ "$PART" = smoke ] && ! done_ checks; then
    # 5: the adapter training, its scans now from a scan pack: the raw views pack gives V-真's step-0 validation loss of fold 0
    if [ ! -e "$R/checks/check5_replay.json" ]; then
        place 12000 0 job "$R/check5_replay" -- "$PY" -m revrender.train_clean_view train --model "$MODEL" --pack "$V" --scan-pack "$RAWV" --fold 0 \
            --steps 1 --eval-every 1 --output-dir "$R/check5_replay"; wait
        "$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["initial_val_loss"]; b=json.load(open(sys.argv[2]))["initial_val_loss"]; r={"replay":a,"fourth_round":b,"difference":a-b,"passed":abs(a-b)<=.001}; json.dump(r,open(sys.argv[3],"w"),indent=2); print(json.dumps(r)); sys.exit(0 if r["passed"] else "check 5 failed")' \
            "$R/check5_replay/complete.json" "${P4}_view_real_f0/complete.json" "$R/checks/check5_replay.json"; fi
    # 1: the matched path with the adapters off, the annotator's pack, against the shared reader of fold 0's first four steps.  The
    # bar as revised before any reading (plan, section six, 2026-11-20 evening; the first bar, 0.01, was tighter than the training's
    # own jitter): step 1 within 0.001 of the record, steps 2-4 each within the largest difference of the recipe rerun as it was
    if [ ! -e "$R/checks/check1_trajectory_revised.json" ]; then
        place 22000 1 teach "$R/check1_trajectory" "$V" 0 --steps 100 --stop-after 4 --view-students "$SYNTH_V" --student-images 2 3 --student-off; wait
        place 22000 1 teach "$R/diag_plain_rerun" "$V" 0 --steps 100 --stop-after 4; wait
        "$PY" - "$R/check1_trajectory/train_log.jsonl" "${SH}0/train_log.jsonl" "$R/diag_plain_rerun/train_log.jsonl" "$R/diag_matched_off_rerun/train_log.jsonl" \
            "$R/checks/check1_trajectory_revised.json" <<'PY'
import json, sys
from pathlib import Path
read = lambda p: [json.loads(l)['loss'] for l in open(p)][:4] if Path(p).exists() else None
mine, record, plain, again = map(read, sys.argv[1:5])
bar = max(abs(x-y) for x, y in zip(plain[1:], record[1:]))
d = [x-y for x, y in zip(mine, record)]
r = {'record_shared_f0': record, 'recipe_rerun': plain, 'matched_off': mine, 'matched_off_again': again, 'differences': d, 'bar_steps_2_4': bar,
     'first_bar_0.01_passed': max(map(abs, d)) <= .01, 'passed': len(mine) == 4 and abs(d[0]) <= .001 and max(map(abs, d[1:])) <= bar}
Path(sys.argv[5]).write_text(json.dumps(r, indent=2)); print(json.dumps(r))
sys.exit(0 if r['passed'] else 'check 1 failed')
PY
    fi
    # 2: the training and reading features of eight regions of folds 1-4, the fourth round's real+synthetic adapters on
    # (as revised, plan section six, second revision: 2a bit for bit against adapters switched by hand, 2b eval, 2c the training
    # model against the reading Reader; the first version's result stays in check2_agree.json)
    [ -e "$R/checks/check2_revised.json" ] || place 22000 1 job "$R/checks/check2_revised.json" -- "$PY" -m revrender.train_clean_view agree --model "$MODEL" \
        --pack "$RAWV" --students "$SYNTH_V" --fold 0 --regions 8 --reader-adapter "${SH}0/adapter" --output "$R/checks/check2_revised.json"; wait
    "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else "check 2 failed")' "$R/checks/check2_revised.json"
    # 6: NAFNet's pack through the adapter path with the adapter off, the whole of fold 0: as E-NAFNet read
    place 22000 1 read_with "$R/check6_wiring" "$EV" "${SH}0/adapter" 0 --view-student "${SYNTH_V//\{fold\}/0}" --student-images 2 3 --student-off; wait
    compare_reads "$R/check6_wiring" "$P2/read_E-NAFNet_f0" "$R/checks/check6_wiring.json" 'check 6'
    mark checks; fi

# Rule 1 first: VR-真+合, the fourth round's real+synthetic adapters with matched readers.
if ! done_ VR_synth; then
    for k in "${READ[@]}"; do matched VR-synth "$RAWV" "$SYNTH_V" "$k"; done
    for k in "${READ[@]}"; do test -e "$R/read_VR-synth_f$k/complete.json"; done
    mark VR_synth; fi

# The N and M adapters, two at a time (a smoke run teaches all five folds: a matched reader of fold 0 needs folds 1-4).
if ! done_ NM_train; then
    for k in 0 1 2 3 4; do place 12000 0 plugin "${B}${TAG}_view_N_f$k" "$EV" "$k"; place 12000 0 plugin "${B}${TAG}_view_M_f$k" "$TINT" "$k"; done; wait
    for k in 0 1 2 3 4; do test -e "${B}${TAG}_view_N_f$k/student/adapter_config.json"; test -e "${B}${TAG}_view_M_f$k/student/adapter_config.json"; done
    mark NM_train; fi

# VR-真: the fourth round's real-pair adapters with matched readers.
if ! done_ VR_real; then
    for k in "${READ[@]}"; do matched VR-real "$RAWV" "$REAL_V" "$k"; done
    for k in "${READ[@]}"; do test -e "$R/read_VR-real_f$k/complete.json"; done
    mark VR_real; fi

# NV and MV: the shared readers through the N and M adapters.
if ! done_ shared_reads; then
    for k in "${READ[@]}"; do
        place 22000 1 read_with "$R/read_NV_f$k" "$EV" "${SH}$k/adapter" "$k" "${LIMIT[@]}" --view-student "${NPLUG//\{fold\}/$k}" --student-images 2 3; wait
        place 22000 1 read_with "$R/read_MV_f$k" "$TINT" "${SH}$k/adapter" "$k" "${LIMIT[@]}" --view-student "${MPLUG//\{fold\}/$k}" --student-images 2 3; wait
    done
    for k in "${READ[@]}"; do test -e "$R/read_NV_f$k/complete.json"; test -e "$R/read_MV_f$k/complete.json"; done
    mark shared_reads; fi

# NVR and MVR: matched readers on the N and M adapters.
if ! done_ NM_matched; then
    for k in "${READ[@]}"; do matched NVR "$EV" "$NPLUG" "$k"; matched MVR "$TINT" "$MPLUG" "$k"; done
    for k in "${READ[@]}"; do test -e "$R/read_NVR_f$k/complete.json"; test -e "$R/read_MVR_f$k/complete.json"; done
    mark NM_matched; fi

# The feature distances of N and M (the fourth round's touched tokens), two at a time.
if ! done_ distances; then
    for k in "${READ[@]}"; do
        place 6000 0 distance "$R/distance_N_f$k" "${NPLUG//\{fold\}/$k}" "$EV" "$k"; place 6000 0 distance "$R/distance_M_f$k" "${MPLUG//\{fold\}/$k}" "$TINT" "$k"
    done; wait
    for k in "${READ[@]}"; do test -e "$R/distance_N_f$k/distances.jsonl"; test -e "$R/distance_M_f$k/distances.jsonl"; done
    mark distances; fi

# The scores and the rules.
ARMS_=(--arm "VR-真+合=$(dirs "$R/read_VR-synth")" --arm "VR-真=$(dirs "$R/read_VR-real")" --arm "NV=$(dirs "$R/read_NV")" --arm "NVR=$(dirs "$R/read_NVR")"
       --arm "MV=$(dirs "$R/read_MV")" --arm "MVR=$(dirs "$R/read_MVR")" --arm "V-真+合=$(dirs "$P4/read_V-synth")" --arm "V-真=$(dirs "$P4/read_V-real")"
       --arm "X=$(dirs "$P4/read_X")" --arm "E-NAFNet=$(dirs "$P2/read_E-NAFNet")" --arm "R0=$(dirs "$P2/read_R0")" --arm "H=$(dirs "$P2/read_H")")
REFS=(V-真+合 V-真 VR-真 NVR H R0 E-NAFNet X)
if ! done_ score; then
    for ref in "${REFS[@]}"; do [ -e "$R/scores_vs_$ref.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference "$ref" \
        $([ "$ref" != V-真+合 ] || echo --readings "$R/readings.jsonl") --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
    [ -e "$R/analysis/analysis.json" ] || "$PY" -m revrender.eval.privileged analyse --pack "$V" "${ARMS_[@]}" \
        $(for ref in "${REFS[@]}"; do echo --scores "$ref=$R/scores_vs_$ref.json"; done) \
        --distances "N=$(dirs "$R/distance_N")" --distances "M=$(dirs "$R/distance_M")" --output-dir "$R/analysis" > "$R/analysis.log" 2>&1
    mark score; fi
touch "$R/COMPLETE"; stamp "$PART complete"
