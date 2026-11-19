#!/usr/bin/env bash
# The method stage, second round (the planning session's plan, rules written before any reading:
# work/method_stage20261118/NAFNet五折与差距分析_方案与判定规则_2026-11-19.md): NAFNet trained once per fold of the ceiling's pack
# (paper2_clean_revision_folds_human_masked20261117b_views, its own five folds by writer), on all the other folds, stopped by the fold
# it leaves out, the recipe, windows and stopping rule of the first round's NAFNet (configs/restoration.toml [method_round1]), the
# weights of its chosen validation and the two after kept.  Per fold a shared reader (context_revision) and an R0 reader
# (raw_context_revision), taught on every region of the other folds (train_reader --all-rest), the adapter of step 100.  Eight
# readings of every region by readers that never saw its writer: H, E-NAFNet, R0, the three F (NAFNet's erased crop with its strike
# left, its writing erased, or both put right from the annotator's pixels: data.struck_pack fixed) and E-later1/2 (the two validations
# after the chosen one).  eval.transcription against R0, E-NAFNet, H and F-residual; eval.erase_gap the analyses A-F, the plan's
# verdicts and the sheets of G.  Card 1 only (the user's word, 2026-11-19: 「只用 1号卡」); cards 0, 2 and 3 are not touched, ALLOW_GPU3
# is not set.  At most two jobs of ours on it, by its free memory.  PART=smoke: two passes, two reader steps, eight regions of fold 0
# read; PART=main: the round.  Unattended; a step whose marker is in $R is skipped.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; B=${N}_nafnet_folds20261119; PART=${PART:-main}
case "$PART" in main) TAG=; BUDGET=method_round1; STEPS=100; LIMIT=(); READ=(0 1 2 3 4) ;;
    smoke) TAG=_smoke; BUDGET=method_smoke; STEPS=2; LIMIT=(--limit 8); READ=(0) ;;   # the smoke reads fold 0 only
    *) echo "PART is main or smoke"; exit 2 ;; esac
R=$B$TAG; mkdir -p "$R"
DATASET=$O/paper2_clean_line_dataset20261117/regions.json; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
PAGES=(--pages "$O/paper2_struck_dataset20261113/pages" --pages "$O/paper2_full_review20261010/images"); HOLD=(--hold-out test-page)
FOLDS_PACK=${N}_revision_folds_human_masked20261117b; V=${FOLDS_PACK}_views
UP=$O/paper2_restoration_sources20261008; SETS=${B}_sets; RAW=${B}_raw; RAWV=${B}_raw_views
source "$ROOT/scripts/_restorers.sh"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
CARD=${ALL_CARDS[1]}; NUMA=$(numa_of "$CARD")
place() { one_card_place "$CARD" "$@"; }   # MiB WHOLE FUNCTION ARGS...
job() {   # OUT -- COMMAND...: one GPU job on card 1
    local out="$1"; shift 2
    bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1
}
views() {   # PACK OUT: the pack with each region's line (CPU)
    [ -e "$2/pack.json" ] || "$PY" -m revrender.data.struck_pack views --pack "$1" --dataset "$DATASET" "${PAGES[@]}" "${HOLD[@]}" --output-dir "$2" > "$2.log" 2>&1
}

# The data: the readers' windows of the pack's five folds, each fold's training tiles, the scan as its own erased crop.
if ! done_ data; then
    [ -e "$SETS/fold_4/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack fold-sets --dataset "$DATASET" "${PAGES[@]}" "${HOLD[@]}" --folds-from "$FOLDS_PACK" \
        --windowed --output-dir "$SETS" > "$SETS.log" 2>&1
    for k in 0 1 2 3 4; do [ -e "${B}_tiles_f$k/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "$SETS/fold_$k" --no-replay \
        --output-dir "${B}_tiles_f$k" > "${B}_tiles_f$k.log" 2>&1; done
    [ -e "$RAW/pack.json" ] || "$PY" -m revrender.data.struck_pack separated --pack "$FOLDS_PACK" --sets "$SETS" $(for k in 0 1 2 3 4; do echo --separator none; done) \
        --upstream "$UP" --output-dir "$RAW" > "$RAW.log" 2>&1
    views "$RAW" "$RAWV"
    "$PY" - "$V" "$SETS" "$B" <<'PY'
import json, sys
from pathlib import Path
rows = json.loads((Path(sys.argv[1])/'pack.json').read_text())['rows']; sets = Path(sys.argv[2])
assert len(rows) == 610 and sum(r['readable'] for r in rows) == 2*257 and len({r['writer'] for r in rows}) == 77, 'not the plan\'s 610 regions'
for k in range(5):
    held = json.loads((sets/f'fold_{k}'/'heldout_native.json').read_text()); train = json.loads((sets/f'fold_{k}'/'training_native.json').read_text())
    assert {n['id'] for n in held} == {r['id'] for r in rows if r['fold'] == k} and {n['id'] for n in train} == {r['id'] for r in rows if r['fold'] != k}
    assert not {n['group_uuid'] for n in held} & {n['group_uuid'] for n in train}, f'fold {k}: a writer on both sides'
    tiles = json.loads((Path(sys.argv[3]+f'_tiles_f{k}')/'records.json').read_text())
    s = [r for r in rows if r['fold'] == k]
    print(json.dumps({'fold': k, 'writers': len({r['writer'] for r in s}), 'regions': len(s), 'readable': sum(r['readable'] for r in s), 'training_tiles': len(tiles)}))
PY
    mark data; fi

# Per fold: the two readers (each fills the card), then NAFNet; its erased crops come after all five.
teach() {   # OUT CONDITION PACK FOLD: a reader taught on every region of the other folds, the adapter of its last step
    [ -e "$1/complete.json" ] || job "$1" -- "$PY" -m revrender.train_reader --pack "$3" --condition "$2" --fold "$4" --all-rest --model "$MODEL" --steps "$STEPS" \
        --batch-size 2 --accumulate 8 --warmup 10 --eval-every 10 --checkpointing --workers 2 --answer latex --output-dir "$1"
}
read_with() {   # OUT CONDITION PACK READER FOLD: the fold read, loops stopped
    [ -e "$1/complete.json" ] || job "$1" -- "$PY" -m revrender.recognize_lines --pack "$3" --model "$MODEL" --backend hf --adapter "$4/adapter" \
        --batch-size 8 --conditions "$2" --fold "$5" --loop-limit 4 "${LIMIT[@]}" --output-dir "$1"
}
nafnet() {   # FOLD
    VAL=$SETS/fold_$1 RESTORER_MEMORY=6000 RESTORER_ARGS=--keep-all finetune_restorer NAFNet "$CARD" "$NUMA" - "${B}_tiles_f$1" "${B}${TAG}_NAFNet_f$1" "$BUDGET"
    test -e "${B}${TAG}_NAFNet_f$1/selection.json"
}
if ! done_ readers; then
    for k in "${READ[@]}"; do place 22000 1 teach "${B}${TAG}_reader_shared_f$k" context_revision "$V" "$k"; wait
        place 22000 1 teach "${B}${TAG}_reader_R0_f$k" raw_context_revision "$RAWV" "$k"; wait; done
    for k in "${READ[@]}"; do test -e "${B}${TAG}_reader_shared_f$k/complete.json"; test -e "${B}${TAG}_reader_R0_f$k/complete.json"; done
    mark readers; fi
if ! done_ train; then
    for k in 0 1 2 3 4; do place 6000 0 nafnet "$k"; done
    wait; for k in 0 1 2 3 4; do test -e "${B}${TAG}_NAFNet_f$k/selection.json"; done
    mark train; fi

# The packs read: E-NAFNet, E-later1/2 (each fold's two validations after its chosen one), the three F; each with its lines.
RUNS=($(for k in 0 1 2 3 4; do echo "--separator ${B}${TAG}_NAFNet_f$k"; done))
later() {   # N -> --checkpoint per fold: the N-th validation after the chosen one
    "$PY" - "$1" "${B}${TAG}_NAFNet_f" "$PART" <<'PY'
import json, sys
from pathlib import Path
n, prefix, part = int(sys.argv[1]), sys.argv[2], sys.argv[3]
for k in range(5):
    run = Path(f'{prefix}{k}'); chosen = json.loads((run/'selection.json').read_text())['step']
    steps = sorted(int(p.name[5:]) for p in run.glob('step_*') if p.name[5:].isdigit()); after = [s for s in steps if s > chosen]
    if part == 'smoke' and len(after) < n: after = [steps[-1]]*n      # two validations in the smoke: the last stands in
    assert len(after) >= n, f'fold {k}: no {n}th validation after the chosen one'
    print(f'--checkpoint step_{after[n-1]}')
PY
}
E=${B}${TAG}_erased
erase() {   # OUT [--checkpoint ...]
    local out="$1"; shift
    [ -e "$out/pack.json" ] || job "$out" -- "$PY" -m revrender.data.struck_pack separated --pack "$FOLDS_PACK" --sets "$SETS" "${RUNS[@]}" --upstream "$UP" "$@" --output-dir "$out"
}
if ! done_ packs; then
    place 6000 0 erase "$E"; wait; test -e "$E/pack.json"
    for n in 1 2; do place 6000 0 erase "${E}_later$n" $(later "$n"); done; wait
    for f in residual damage both; do [ -e "${E}_fix_$f/pack.json" ] || "$PY" -m revrender.data.struck_pack fixed --pack "$E" --sets "$SETS" --fix "$f" \
        --output-dir "${E}_fix_$f" > "${E}_fix_$f.log" 2>&1; done
    for p in "$E" "${E}_later1" "${E}_later2" "${E}_fix_residual" "${E}_fix_damage" "${E}_fix_both"; do views "$p" "${p}_views"; done
    mark packs; fi

# The readings: eight per fold.
SH=${B}${TAG}_reader_shared_f; R0R=${B}${TAG}_reader_R0_f
SPECS=("H context_revision $V $SH" "E-NAFNet context_revision ${E}_views $SH" "R0 raw_context_revision $RAWV $R0R"
       "F-residual context_revision ${E}_fix_residual_views $SH" "F-damage context_revision ${E}_fix_damage_views $SH" "F-both context_revision ${E}_fix_both_views $SH"
       "E-later1 context_revision ${E}_later1_views $SH" "E-later2 context_revision ${E}_later2_views $SH")
if ! done_ read; then
    for spec in "${SPECS[@]}"; do set -- $spec; for k in "${READ[@]}"; do place 22000 1 read_with "$R/read_$1_f$k" "$2" "$3" "$4$k" "$k"; done; done
    wait; for spec in "${SPECS[@]}"; do set -- $spec; for k in "${READ[@]}"; do test -e "$R/read_$1_f$k/complete.json"; done; done
    mark read; fi

# The scores and the analyses.
arm() { local k s=; for k in "${READ[@]}"; do s+="${s:+,}$R/read_$1_f$k"; done; echo "$1=$s"; }
ARMS_=(); for spec in "${SPECS[@]}"; do set -- $spec; ARMS_+=(--arm "$(arm "$1")"); done
if ! done_ score; then
    for ref in R0 E-NAFNet H F-residual; do [ -e "$R/scores_vs_$ref.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference "$ref" \
        $([ "$ref" != E-NAFNet ] || echo --readings "$R/readings.jsonl") --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
    [ -e "$R/gap/analysis.json" ] || "$PY" -m revrender.eval.erase_gap analyse --pack "$V" --sets "$SETS" --erased "$E" "${ARMS_[@]}" \
        $(for ref in R0 E-NAFNet H F-residual; do echo --scores "$ref=$R/scores_vs_$ref.json"; done) --output-dir "$R/gap" > "$R/gap.log" 2>&1
    [ "$PART" != main ] || "$PY" -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["regions"]==610 and a["readable"]==2*257, "not every region read"' "$R/gap/analysis.json"
    [ -e "$R/sheets/index.json" ] || "$PY" -m revrender.eval.erase_gap sheets --table "$R/gap/regions.jsonl" --human "$V" --erased "$E" --output-dir "$R/sheets" > "$R/sheets.log" 2>&1
    mark score; fi
touch "$R/COMPLETE"; stamp "$PART complete"
