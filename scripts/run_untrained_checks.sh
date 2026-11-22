#!/usr/bin/env bash
# Three checks with nothing taught (plan of 2026-11-22, 三个不训练的检验, revision one), on Qwen3-VL-8B, the 610 regions of the second
# round, five folds, each fold read by its shared H reader (context_revision, --loop-limit 4, batches of 8, as V-修's readings):
#   T1  --student-images 2 (V-只2) and 3 (V-只3) beside V-修 (2 3) read again;
#   T2  images 2 and 3 take features on the chord from the frozen tower's on the scan to its on the human erased images
#       (recognize_lines --mix-pack, models.feature_mix): alpha 0, 0.25, 0.5, 0.75, 1, and M / M' (per token as close as V-修's
#       features; s farther than x -> x, or s);
#   T3  eval.feature_scale on the server CPU (float32, 8 threads, nice; no card): norms, variances, the chord's closeness.
# PART=smoke: check 4 (feature_mix grids, all 610), T3 (its check 6 inside), check 3 (feature_mix routes, 20 regions of fold 0), and fold
# 0 read as V-修, alpha 0 and 1, the --student-off path, M and M' for checks 1, 2 and 5; check 7 the seconds.  PART=main: the ten arms
# on the five folds, eval.untrained_checks (rules T1, T2 and the reported items), eval.feature_scale summary (rule T3).
# One reading on a card at a time (run on cards 1 and 2);
# the cards: _common.sh place_on_cards (the user's word of 2026-11-23: any of 0-3, at most two of them ours at a time, the least used first).  Unattended; a step whose marker is in $R is skipped.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; PART=${PART:-main}; B=${N}_untrained_checks20261122
case "$PART" in main) R=$B; READ=(0 1 2 3 4) ;; smoke) R=${B}_smoke; READ=(0) ;; *) echo "PART is main or smoke"; exit 2 ;; esac
P2=${N}_nafnet_folds20261119; V=${N}_revision_folds_human_masked20261117b_views; RAWV=${P2}_raw_views; SH=${P2}_reader_shared_f
FIX=${N}_covered_view20261121; SHARES=$FIX/checks/strike_shares.json; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
STUDENT="${FIX}_view_fix_f{fold}/student"; T3=${B}_t3
mkdir -p "$R/checks"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
test -e "$P2/COMPLETE" && test -e "$FIX/COMPLETE" && test -e "$SHARES" || { echo 'the second or sixth round missing'; exit 2; }
place() { place_on_cards "$@"; }   # MiB WHOLE FUNCTION ARGS...: on the least used of our two cards it fits on (_common.sh)
job() {   # OUT -- COMMAND...: one GPU job on $CARD
    local out="$1" t0; shift 2; t0=$(date +%s)
    bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1
    echo $(( $(date +%s)-t0 )) > "$out.wall"
}
dirs() { local k s=; for k in "${READ[@]}"; do s+="${s:+,}$1_f$k"; done; echo "$s"; }   # PREFIX -> the folds' folders
student() { echo "${STUDENT//\{fold\}/$1}"; }
read_arm() {   # OUT FOLD ARGS...: the fold read under context_revision by its shared H reader from the scan pack
    local out="$1" fold="$2"; shift 2
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.recognize_lines --pack "$RAWV" --model "$MODEL" --backend hf --adapter "${SH}$fold/adapter" \
        --batch-size 8 --conditions context_revision --fold "$fold" --loop-limit 4 "$@" --output-dir "$out"
}
arm() {   # NAME FOLD: one of the ten arms
    local name="$1" k="$2" out="$R/read_$1_f$2"
    case "$name" in
        V-修) read_arm "$out" "$k" --view-student "$(student "$k")" --student-images 2 3 ;;
        V-只2) read_arm "$out" "$k" --view-student "$(student "$k")" --student-images 2 ;;
        V-只3) read_arm "$out" "$k" --view-student "$(student "$k")" --student-images 3 ;;
        off) read_arm "$out" "$k" --view-student "$(student "$k")" --student-images 2 3 --student-off ;;
        a*) read_arm "$out" "$k" --mix-pack "$V" --mix-alpha "${name#a}" --student-images 2 3 ;;
        M) read_arm "$out" "$k" --mix-pack "$V" --mix-matched x --view-student "$(student "$k")" --student-images 2 3 ;;
        Mp) read_arm "$out" "$k" --mix-pack "$V" --mix-matched s --view-student "$(student "$k")" --student-images 2 3 ;;
    esac
}

# T3 on the server CPU (no card), once: its check 6 stops it if the closeness is off the records
t3() {
    [ -e "$T3/complete.json" ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 nice "$PY" -m revrender.eval.feature_scale measure --model "$MODEL" --pack "$V" --raw-pack "$RAWV" \
        --students "$STUDENT" --records "$FIX/distance_fix_f{fold}" --threads 8 --output-dir "$T3" > "$T3.log" 2>&1
}

if [ "$PART" = smoke ]; then
    done_ check4 || { CUDA_VISIBLE_DEVICES= nice "$PY" -m revrender.models.feature_mix grids --model "$MODEL" --pack "$RAWV" --mix-pack "$V" --places 2 3 --output-dir "$R/check4_grids"; mark check4; }
    [ -e "$T3/complete.json" ] || { t3 & T3PID=$!; }
    done_ check3 || { place 22000 1 job "$R/check3_routes" -- "$PY" -m revrender.models.feature_mix routes --model "$MODEL" --pack "$RAWV" --student "$(student 0)" \
        --fold 0 --regions 20 --output-dir "$R/check3_routes"; wait; mark check3; }
    if ! done_ readings; then
        for name in V-修 a0 a1 off M Mp; do place 22000 1 arm "$name" 0; done; wait; mark readings
    fi
    "$PY" - "$R" "$FIX/read_fix_f0" "$P2/read_H_f0" > "$R/checks/checks_1_2_5_7.log" <<'PY'
import json, sys
from pathlib import Path
R, fix, H = (Path(a) for a in sys.argv[1:])
load = lambda d: {g['id']: g for g in map(json.loads, (d/'predictions.jsonl').read_text().splitlines())}
def same(a, b):
    A, B = load(a), load(b); differ = [{'id': k, 'a': A[k]['raw'], 'b': B[k]['raw']} for k in sorted(A) if A[k]['raw'] != B.get(k, {}).get('raw')]
    return {'regions': len(A), 'same': len(A)-len(differ), 'differ': differ, 'passed': not differ and set(A) == set(B)}
r = {'check1_V-修_again_vs_record': same(R/'read_V-修_f0', fix), 'check2_alpha1_vs_H': same(R/'read_a1_f0', H), 'check2_alpha0_vs_student_off': same(R/'read_a0_f0', R/'read_off_f0')}
for arm in ('M', 'Mp'):
    stats = [json.loads(l) for l in (R/f'read_{arm}_f0'/'mix_stats.jsonl').read_text().splitlines()]
    levels = [lv for g in stats for img in g['images'].values() for lv in img.values()]
    gap = max(lv['largest_cos_gap'] for lv in levels); back = sum(lv['taken_back'] for lv in levels); tokens = sum(lv['tokens'] for lv in levels)
    e = {'regions': len(stats), 'largest_cos_gap': gap, 'within_1e-4': gap <= 1e-4, 'taken_back_share': back/tokens,
         'taken_back_share_by_level': {n: sum(g['images'][i][n]['taken_back'] for g in stats for i in g['images'])/sum(g['images'][i][n]['tokens'] for g in stats for i in g['images'])
                                       for n in ('merged', 'deepstack1', 'deepstack2', 'deepstack3')},
         'taken_back_share_window_merged': sum(g['images']['2']['merged']['taken_back'] for g in stats)/sum(g['images']['2']['merged']['tokens'] for g in stats)}
    if arm == 'Mp':
        e['kept_s_bit_for_bit'] = all(lv['kept_s_bit_for_bit'] for lv in levels)
    e['passed'] = e['within_1e-4'] and e.get('kept_s_bit_for_bit', True)
    r[f'check5_{arm}'] = e
r['check7_seconds'] = {d.name: {**json.loads((d/'complete.json').read_text()), 'wall_seconds': int((Path(str(d)+'.wall')).read_text())} for d in sorted(R.glob('read_*_f0'))}
r['passed'] = all(v['passed'] for k, v in r.items() if isinstance(v, dict) and 'passed' in v)
(R/'checks'/'checks_1_2_5_7.json').write_text(json.dumps(r, ensure_ascii=False, indent=1))
print(json.dumps({k: v.get('passed') for k, v in r.items() if isinstance(v, dict)}, ensure_ascii=False))
PY
    [ -z "${T3PID:-}" ] || wait "$T3PID"
    stamp "smoke done"; touch "$R/SMOKE_COMPLETE"; exit 0
fi

# PART=main
ARMS=(V-修 V-只2 V-只3 a0 a0.25 a0.5 a0.75 a1 M Mp)
if ! done_ readings; then
    for k in "${READ[@]}"; do for name in "${ARMS[@]}"; do place 22000 1 arm "$name" "$k"; done; done; wait; mark readings
fi
done_ analysis || { "$PY" -m revrender.eval.untrained_checks --pack "$V" --shares "$SHARES" \
    --arm "V-修=$(dirs "$R/read_V-修")" --arm "V-只2=$(dirs "$R/read_V-只2")" --arm "V-只3=$(dirs "$R/read_V-只3")" \
    --arm "α0=$(dirs "$R/read_a0")" --arm "α0.25=$(dirs "$R/read_a0.25")" --arm "α0.5=$(dirs "$R/read_a0.5")" --arm "α0.75=$(dirs "$R/read_a0.75")" \
    --arm "α1=$(dirs "$R/read_a1")" --arm "M=$(dirs "$R/read_M")" --arm "M'=$(dirs "$R/read_Mp")" --mix-stats "M=$(dirs "$R/read_M")" \
    --output-dir "$R/analysis" > "$R/analysis.log" 2>&1; mark analysis; }
t3
done_ t3_summary || { "$PY" -m revrender.eval.feature_scale summary --measured "$T3" --shares "$SHARES" --output-dir "$R/t3_summary" > "$R/t3_summary.log" 2>&1; mark t3_summary; }
stamp "all done"; touch "$R/COMPLETE"
