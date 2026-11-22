#!/usr/bin/env bash
# The premise check of an early, local repair (plan of 2026-11-22, 早期局部修_前提检验, revision one): on Qwen3-VL-8B, the 610 regions of
# the second round, five folds, each fold read by its shared H reader (context_revision, --loop-limit 4, batches of 8, as V-修's
# readings), images 2 and 3 transplanted (recognize_lines --transplant-pack, models.transplant): after K = 3, 6, 9, 13, 17 blocks of the
# frozen tower, the human erased image's tokens put in at the struck patches, or the struck and their neighbours, by the human strike;
# one arm after 9 blocks, struck and next by NAFNet's fold-out erased crops (the removal metrics' rule: brighter by 0.2 on ink).
# A diagnosis with the held-out folds' human erased crops and masks, not a method's score.
# PART=cpu: eval.transplant_closeness on the server CPU (float32, 8 threads, nice; no card): the eleven arms' four outputs beside the
# erased image's, its checks inside, then its summary.  PART=smoke: checks 1 and 2 on the features (models.transplant, 20 regions of
# fold 0, bit for bit), fold 0 read under the eleven arms and under every patch after 9 blocks (against H's readings) and none (against
# T2's --student-off reading), the patches taken against the CPU run's, and the seconds.  PART=main: the eleven arms on the five folds.
# Cards 1 and 2, one reading on a card at a time; cards 0 and 3 not touched.  Unattended; a step whose marker is in $R is skipped.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; PART=${PART:-main}; B=${N}_transplant20261122
case "$PART" in main) R=$B; READ=(0 1 2 3 4) ;; smoke) R=${B}_smoke; READ=(0) ;; cpu) R=${B}_cpu_steps; READ=() ;; *) echo "PART is main, smoke or cpu"; exit 2 ;; esac
P2=${N}_nafnet_folds20261119; V=${N}_revision_folds_human_masked20261117b_views; RAWV=${P2}_raw_views; EV=${P2}_erased_views; SH=${P2}_reader_shared_f
SPREAD=${N}_strike_spread20261122; OFF=${N}_untrained_checks20261122_smoke/read_off_f0; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
CPU=${B}_cpu
mkdir -p "$R/checks"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
test -e "$P2/COMPLETE" && test -e "$EV/pack.json" && test -e "$SPREAD/complete.json" || { echo 'the second round or the spread run missing'; exit 2; }

if [ "$PART" = cpu ]; then
    [ -e "$CPU/complete.json" ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 nice "$PY" -m revrender.eval.transplant_closeness measure --model "$MODEL" --pack "$RAWV" \
        --transplant-pack "$V" --nafnet-pack "$EV" --spread "$SPREAD" --threads 8 --output-dir "$CPU" > "$CPU.log" 2>&1
    done_ summary || { "$PY" -m revrender.eval.transplant_closeness summary --measured "$CPU" --pack "$V" --output-dir "${CPU}_summary" > "${CPU}_summary.log" 2>&1; mark summary; }
    stamp "cpu done"; touch "$R/COMPLETE"; exit 0
fi

CARDS=("${ALL_CARDS[1]}" "${ALL_CARDS[2]}")
place() { place_on_cards "$@"; }   # MiB WHOLE FUNCTION ARGS...: on card 1 or 2, whichever opens first (_common.sh)
job() {   # OUT -- COMMAND...: one GPU job on $CARD
    local out="$1" t0; shift 2; t0=$(date +%s)
    bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1
    echo $(( $(date +%s)-t0 )) > "$out.wall"
}
read_arm() {   # OUT FOLD ARGS...: the fold read under context_revision by its shared H reader from the scan pack, images 2 and 3 transplanted
    local out="$1" fold="$2"; shift 2
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.recognize_lines --pack "$RAWV" --model "$MODEL" --backend hf --adapter "${SH}$fold/adapter" \
        --batch-size 8 --conditions context_revision --fold "$fold" --loop-limit 4 --transplant-pack "$V" --student-images 2 3 "$@" --output-dir "$out"
}
arm() {   # NAME FOLD: k{K}_struck, k{K}_next, k9_next_NAFNet, or the checks' all9 / none
    local name="$1" k="$2" out="$R/read_$1_f$2"
    case "$name" in
        k9_next_NAFNet) read_arm "$out" "$k" --transplant-after 9 --transplant-select next --transplant-places "$EV" --transplant-rule brighter ;;
        all9) read_arm "$out" "$k" --transplant-after 9 --transplant-select all ;;
        none) read_arm "$out" "$k" --transplant-after 9 --transplant-select none ;;
        k*_*) local after="${name%%_*}"; read_arm "$out" "$k" --transplant-after "${after#k}" --transplant-select "${name#*_}" ;;
    esac
}
ARMS=(k3_struck k3_next k6_struck k6_next k9_struck k9_next k13_struck k13_next k17_struck k17_next k9_next_NAFNet)

if [ "$PART" = smoke ]; then
    # check 3 needs only the CPU run's patches taken per region (regions.jsonl, all 610), not its float32 checks
    [ "$(wc -l < "$CPU/regions.jsonl")" = 610 ] && test -e "$OFF/complete.json" || { echo 'the CPU run'"'"'s 610 regions or T2'"'"'s --student-off reading missing'; exit 2; }
    done_ features || { place 22000 1 job "$R/check12_features" -- "$PY" -m revrender.models.transplant --model "$MODEL" --pack "$RAWV" --transplant-pack "$V" \
        --fold 0 --regions 20 --output-dir "$R/check12_features"; wait; test -e "$R/check12_features/result.json"; mark features; }
    if ! done_ readings; then
        for name in all9 none "${ARMS[@]}"; do place 22000 1 arm "$name" 0; done; wait; mark readings
    fi
    "$PY" - "$R" "$P2/read_H_f0" "$OFF" "$CPU" "$SPREAD" > "$R/checks/checks.log" <<'PY'
import json, sys
from pathlib import Path
R, H, off, cpu, spread_dir = (Path(a) for a in sys.argv[1:])
load = lambda d: {g['id']: g for g in map(json.loads, (d/'predictions.jsonl').read_text().splitlines())}
def same(a, b):
    A, B = load(a), load(b); differ = [{'id': k, 'a': A[k]['raw'], 'b': B.get(k, {}).get('raw')} for k in sorted(A) if A[k]['raw'] != B.get(k, {}).get('raw')]
    return {'regions': len(A), 'same': len(A)-len(differ), 'differ': differ, 'passed': not differ and set(A) == set(B)}
features = json.loads((R/'check12_features'/'result.json').read_text())
r = {'check1_2_features': {'passed': features['passed'], **{k: {'images': v['images'], 'as_wanted': v['as_wanted']} for k, v in features['checks'].items()}},
     'check1_all_after_9_vs_H': same(R/'read_all9_f0', H), 'check2_none_vs_student_off': same(R/'read_none_f0', off)}
taken = {g['id']: g for g in map(json.loads, (cpu/'regions.jsonl').read_text().splitlines())}
views = {'2': 'window', '3': 'line'}; differ = []; arms = [d.name[len('read_'):-len('_f0')] for d in sorted(R.glob('read_k*_f0'))]
for arm in arms:
    for g in map(json.loads, (R/f'read_{arm}_f0'/'transplant_stats.jsonl').read_text().splitlines()):
        for image, st in g['images'].items():
            want = taken[g['id']][views[image]]['taken'][arm]
            if st['replaced'] != want:
                differ.append({'arm': arm, 'id': g['id'], 'image': image, 'read': st['replaced'], 'cpu': want})
r['check3_patches_taken_vs_cpu'] = {'arms': arms, 'differ': differ[:50], 'differing': len(differ), 'passed': not differ}
spread = {g['id']: g for g in map(json.loads, (spread_dir/'regions.jsonl').read_text().splitlines())}; differ, images = [], 0
for arm in [a for a in arms if not a.endswith('NAFNet')]:
    for g in map(json.loads, (R/f'read_{arm}_f0'/'transplant_stats.jsonl').read_text().splitlines()):
        for image, st in g['images'].items():
            rec = spread[g['id']][views[image]]['class_shares']['patches']; want = rec['struck']+(rec['next'] if arm.endswith('_next') else 0); images += 1
            if st['replaced'] != want:
                differ.append({'arm': arm, 'id': g['id'], 'image': image, 'read': st['replaced'], 'spread': want})
r['check3_patches_taken_vs_spread_records'] = {'images': images, 'differ': differ[:50], 'differing': len(differ), 'passed': not differ and images > 0}
r['check5_seconds'] = {d.name: {**json.loads((d/'complete.json').read_text()), 'wall_seconds': int((Path(str(d)+'.wall')).read_text())} for d in sorted(R.glob('read_*_f0'))}
r['passed'] = all(v['passed'] for v in r.values() if isinstance(v, dict) and 'passed' in v)
(R/'checks'/'checks.json').write_text(json.dumps(r, ensure_ascii=False, indent=1))
print(json.dumps({k: v.get('passed') for k, v in r.items() if isinstance(v, dict)}, ensure_ascii=False))
PY
    stamp "smoke done"; touch "$R/SMOKE_COMPLETE"; exit 0
fi

# PART=main
if ! done_ readings; then
    for k in "${READ[@]}"; do for name in "${ARMS[@]}"; do place 22000 1 arm "$name" "$k"; done; done; wait; mark readings
fi
stamp "all done"; touch "$R/COMPLETE"
