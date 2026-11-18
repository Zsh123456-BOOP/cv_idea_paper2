#!/usr/bin/env bash
# The method stage, first round (the planning session's plan, rules written before any reading:
# work/method_stage20261118/基线统一训练第一轮_方案与判定规则_2026-11-18.md).  Ten restorers, each trained once from its own random start
# on the training writers of configs/method_split.json (data.struck_pack split, 9:1 by writer), each to its own convergence by one
# stopping rule (configs/restoration.toml [method_round1]: a validation every 10 passes, removal F1 inside the validation writers'
# regions); training pairs, validation and the restorers' inputs are the readers' windows (the region's polygon, white outside).
# A reader shared by every restorer, taught once on the training writers with the annotator's erased crops (context_revision, the
# ceiling's recipe, the adapter of step 100), reads the validation writers with each restorer's erased crop (E-<model>), with the
# annotator's (H) and with the scan in its place (E-none); R0 is a reader taught on the scan and its line alone (raw_context_revision),
# P-NAFNet one taught on NAFNet's erased crops of the training writers; the shared reader also reads NAFNet's two validations after
# its chosen one.  rules.json holds the plan's section four, readings_raw.jsonl every reading and its raw output.
# Cards 1 and 2 only (the user's word, 2026-11-18): CARDS takes no other; cards 0 and 3 are never touched, ALLOW_GPU3 never set.
# PART=smoke: two passes of each restorer ([method_smoke]), two steps of each reader, eight regions read; PART=main: the round.
# Unattended; a step whose marker is in $R is skipped.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; B=${N}_baselines20261118; PART=${PART:-main}
case "$PART" in main) TAG=; BUDGET=method_round1; STEPS=100; LIMIT=() ;; smoke) TAG=_smoke; BUDGET=method_smoke; STEPS=2; LIMIT=(--limit 8) ;;
    *) echo "PART is main or smoke"; exit 2 ;; esac
R=$B$TAG; mkdir -p "$R"
DATASET=$O/paper2_clean_line_dataset20261117/regions.json; SPLIT=$ROOT/configs/method_split.json; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
PAGES=(--pages "$O/paper2_struck_dataset20261113/pages" --pages "$O/paper2_full_review20261010/images"); HOLD=(--hold-out test-page)
CEILING_PACK=${N}_revision_folds_human_masked20261117b_views
UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream; NEW_UP=$O/paper2_clean_baselines20261118_sources
PACK=${B}_human; HV=${B}_human_views; SETS=${B}_sets; TILES=${B}_tiles; RAW=${B}_raw; RAWV=${B}_raw_views; VAL=$SETS/fold_0
MODELS=(StainRestorer Restormer DocDiff DocRes VCGAN Strikethrough-CycleGAN-A Strikethrough-CycleGAN NAFNet DE-GAN COR-Generator)   # the longest first
source "$ROOT/scripts/_restorers.sh"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR

PICK=(); for i in ${CARDS:-1 2}; do case "$i" in 1|2) PICK+=("${ALL_CARDS[$i]}") ;; *) echo "card $i is not ours this round"; exit 2 ;; esac; done
ALL_CARDS=("${PICK[@]}")
mib_of() {   # MODEL -> the free MiB its card must have when it starts (its peak in the smoke run and a margin)
    case "$1" in COR-Generator) echo 3000 ;; NAFNet|DE-GAN|Strikethrough-CycleGAN|VCGAN) echo 6000 ;; DocDiff) echo 8000 ;;
        Strikethrough-CycleGAN-A|DocRes) echo 12000 ;; *) echo 15000 ;; esac
}
roomy() {   # the cards with room for an 8B reader (22,000 MiB free) and no job of ours, the most room first
    local c free
    for c in "${ALL_CARDS[@]}"; do free=$(nvidia-smi -i "$c" --query-gpu=memory.free --format=csv,noheader,nounits)
        [ "$(ours_on "$c")" -eq 0 ] && [ "$free" -ge 22000 ] && echo "$free $c"; done | sort -rn | awk '{print $2}'
}
steady() { local first; first=$(roomy | sed -n 1p); sleep 10; [ -n "$first" ] && [ "$(roomy | sed -n 1p)" = "$first" ] && echo "$first"; }
alone() {   # FUNCTION ARGS...: FUNCTION CARD NUMA ARGS... in the background on a card with room for the 8B and no other job of ours
    local card; until card=$(steady); do sleep 20; done
    "$1" "$card" "$(numa_of "$card")" "${@:2}" & PLACED_PIDS+=($!); PLACED_CARDS+=("$card"); sleep 20
}
job() {   # CARD NUMA OUT -- COMMAND...: one GPU job
    local card="$1" numa="$2" out="$3"; shift 4
    bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$out" -- "$@" > "$out.log" 2>&1
}
views() {   # PACK OUT: the pack with each region's line (CPU)
    [ -e "$2/pack.json" ] || "$PY" -m revrender.data.struck_pack views --pack "$1" --dataset "$DATASET" "${PAGES[@]}" "${HOLD[@]}" --output-dir "$2" > "$2.log" 2>&1
}

# The data: the regions with the split's two folds (0 validation, 1 training), the windowed sets, the training tiles, the packs to read.
if ! done_ data; then
    [ -e "$PACK/pack.json" ] || "$PY" -m revrender.data.struck_pack folds --dataset "$DATASET" "${PAGES[@]}" "${HOLD[@]}" --split "$SPLIT" --output-dir "$PACK" > "$PACK.log" 2>&1
    [ -e "$SETS/fold_1/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack fold-sets --dataset "$DATASET" "${PAGES[@]}" "${HOLD[@]}" --split "$SPLIT" --output-dir "$SETS" > "$SETS.log" 2>&1
    [ -e "$TILES/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "$VAL" --no-replay --output-dir "$TILES" > "$TILES.log" 2>&1
    views "$PACK" "$HV"
    [ -e "$RAW/pack.json" ] || "$PY" -m revrender.data.struck_pack separated --pack "$PACK" --sets "$SETS" --separator none --separator none --upstream "$UP" --output-dir "$RAW" > "$RAW.log" 2>&1
    views "$RAW" "$RAWV"
    "$PY" - "$PACK" "$CEILING_PACK" "$SPLIT" "$TILES" <<'PY'
import hashlib, json, sys
from collections import Counter
from pathlib import Path
pack, ceiling, split, tiles = map(Path, sys.argv[1:])
rows = json.loads((pack/'pack.json').read_text())['rows']; old = {r['id']: r for r in json.loads((ceiling/'pack.json').read_text())['rows']}
digest = lambda root, rel: hashlib.sha256((root/rel).read_bytes()).hexdigest()
assert len(rows) == 610 and sum(r['readable'] for r in rows) == 2*257 and len({r['writer'] for r in rows}) == 77, 'not the 610 regions of the plan'
assert {r['id'] for r in rows} == set(old) and all(r['target'] == old[r['id']]['target'] for r in rows), 'not the regions and answers of the ceiling pack'
assert all(digest(pack, r['images'][k]) == digest(ceiling, old[r['id']]['images'][k]) for r in rows for k in ('raw', 'erased', 'map')), 'crops differ'
s = json.loads(split.read_text()); val = [r for r in rows if r['fold'] == 0]
assert {r['writer'] for r in val} == set(s['validation']) and len(val) == 65 and sum(r['readable'] for r in val) == 50
records = json.loads((tiles/'records.json').read_text())
assert not {r['group_uuid'] for r in records} & set(s['validation']), 'a validation writer among the training tiles'
print(json.dumps({'regions': len(rows), 'validation': len(val), 'training_tiles': len(records), 'tiles_by_mode': Counter(r.get('mode') for r in records)}))
PY
    mark data; fi

# A restorer: trained, then its erased windows (the validation writers; the training writers too for NAFNet, whose reader P-NAFNet
# learns on them), their lines, their pixel scores.
erase() {   # CARD NUMA MODEL RUN OUT TRAINING_SEPARATOR [CHECKPOINT]
    local card="$1" numa="$2" model="$3" run="$4" out="$5" second="$6" step="${7:-}"
    [ -e "$out/pack.json" ] || ( restorer_env "$model"
        job "$card" "$numa" "$out" -- "$RUNTIME" -m revrender.data.struck_pack separated --pack "$PACK" --sets "$SETS" --separator "$run" --separator "$second" \
            --upstream "$(upstream_of "$model")" ${step:+--checkpoint "$step"} --output-dir "$out" )
    views "$out" "${out}_views"
    [ -e "$out.pixels.json" ] || "$PY" -m revrender.data.struck_pack scores --pack "$out" --sets "$SETS" --fold 0 --output "$out.pixels.json" > "$out.pixels.log" 2>&1
}
model_job() {   # CARD NUMA MODEL
    local card="$1" numa="$2" model="$3" run=${B}${TAG}_$3
    RESTORER_MEMORY=$(mib_of "$model") RESTORER_ARGS="$([ "$model" != NAFNet ] || echo --keep-all)" finetune_restorer "$model" "$card" "$numa" - "$TILES" "$run" "$BUDGET"
    test -e "$run/selection.json"
    erase "$card" "$numa" "$model" "$run" "${B}${TAG}_erased_$model" "$([ "$model" = NAFNet ] && echo "$run" || echo none)"
    stamp "restorer $model done"
}
teach() {   # CARD NUMA OUT CONDITION PACK: a reader taught on the training writers (fold 1), the adapter of its last step
    [ -e "$3/complete.json" ] || job "$1" "$2" "$3" -- "$PY" -m revrender.train_reader --pack "$5" --condition "$4" --fold 0 --model "$MODEL" --steps "$STEPS" \
        --batch-size 2 --accumulate 8 --warmup 10 --eval-every 10 --checkpointing --workers 2 --answer latex --output-dir "$3"
}
read_with() {   # CARD NUMA OUT CONDITION PACK READER: the validation writers (fold 0) read, loops stopped
    [ -e "$3/complete.json" ] || job "$1" "$2" "$3" -- "$PY" -m revrender.recognize_lines --pack "$5" --model "$MODEL" --backend hf --adapter "$6/adapter" \
        --batch-size 8 --conditions "$4" --fold 0 --loop-limit 4 "${LIMIT[@]}" --output-dir "$3"
}
SHARED=${B}${TAG}_reader_shared; R0R=${B}${TAG}_reader_R0; PR=${B}${TAG}_reader_P-NAFNet
# The two readers that need no restorer first (each fills a card; the restorers would race them for it), then the restorers, two a card.
if ! done_ readers; then
    alone teach "$SHARED" context_revision "$HV"; alone teach "$R0R" raw_context_revision "$RAWV"
    wait; test -e "$SHARED/complete.json"; test -e "$R0R/complete.json"; mark readers; fi
if ! done_ train; then
    for m in "${MODELS[@]}"; do done_ "restorer_$m" || place_job "$(mib_of "$m")" model_job "$m"; done
    wait; for m in "${MODELS[@]}"; do test -e "${B}${TAG}_erased_${m}_views/pack.json"; touch "$R/restorer_$m.done"; done
    mark train; fi

# The readings of the validation writers.
NAF=${B}${TAG}_NAFNet
LATER=($("$PY" - "$NAF" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); chosen = json.loads((run/'selection.json').read_text())['step']
steps = sorted(int(p.name[5:]) for p in run.glob('step_*') if p.name[5:].isdigit())
print(' '.join(f'step_{s}' for s in [s for s in steps if s > chosen][:2]))
PY
))
if ! done_ read; then
    for k in "${!LATER[@]}"; do alone erase NAFNet "$NAF" "${B}${TAG}_erased_NAFNet_later$((k+1))" none "${LATER[$k]}"; done
    alone teach "$PR" context_revision "${B}${TAG}_erased_NAFNet_views"
    wait
    readings=("H context_revision $HV $SHARED" "E-none context_revision $RAWV $SHARED" "R0 raw_context_revision $RAWV $R0R"
              "P-NAFNet context_revision ${B}${TAG}_erased_NAFNet_views $PR")
    for m in "${MODELS[@]}"; do readings+=("E-$m context_revision ${B}${TAG}_erased_${m}_views $SHARED"); done
    for k in "${!LATER[@]}"; do readings+=("E-NAFNet-later$((k+1)) context_revision ${B}${TAG}_erased_NAFNet_later$((k+1))_views $SHARED"); done
    for spec in "${readings[@]}"; do set -- $spec; alone read_with "$R/read_$1" "$2" "$3" "$4"; done
    wait; for spec in "${readings[@]}"; do set -- $spec; test -e "$R/read_$1/complete.json"; done
    mark read; fi

# The scores (eval.transcription, paired by writer, 1,000 draws, seed 42) against E-NAFNet, R0 and H, and the rules of section four.
if ! done_ score; then
    arms=(); for d in "$R"/read_*; do [ -d "$d" ] && arms+=(--arm "${d##*/read_}=$d"); done
    for ref in E-NAFNet R0 H; do [ -e "$R/scores_vs_$ref.json" ] || "$PY" -m revrender.eval.transcription --pack "$HV" "${arms[@]}" --reference "$ref" \
        $([ "$ref" != E-NAFNet ] || echo --readings "$R/readings.jsonl") --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
    [ -e "$R/rules.json" ] || "$PY" - "$R" "$B$TAG" "$TILES" "${MODELS[@]}" <<'PY'
import json, sys
from pathlib import Path
R, prefix, tiles, models = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
train_tiles = sum(r['split'] == 'train' for r in json.loads((tiles/'records.json').read_text()))
scores = {ref: json.loads((R/f'scores_vs_{ref}.json').read_text())['arms'] for ref in ('E-NAFNet', 'R0', 'H')}
def restorer(m):
    run = Path(f'{prefix}_{m}'); sel = json.loads((run/'selection.json').read_text())
    config = json.loads((run/'run_manifest.json').read_text())['config']; batch = config['batch_size']
    curve = sorted((json.loads(p.read_text()) for p in run.glob('evaluation_*.json')), key=lambda e: (e['step'], e.get('variant', '')))
    chosen = json.loads((run/sel['evaluation']).read_text())['human_removal']; events = [json.loads(l) for l in open(run/'events.jsonl')]
    pixels = json.loads(Path(f'{prefix}_erased_{m}.pixels.json').read_text())
    return {'selected_step': sel['step'], 'selected_pass': round(sel['step']*batch/train_tiles, 1), 'actual_steps': sel['actual_steps'],
            'actual_passes': round(sel['actual_steps']*batch/train_tiles, 1), 'stop_reason': sel['stop_reason'],
            'not_levelled_at_the_limit': sel['stop_reason'] == 'declared_budget' and sel['selected_at_last_evaluation'],
            'seconds': events[-1]['elapsed_seconds'], 'batch': batch, 'validation': chosen,
            'trained': chosen['recall'] >= .10, 'psnr': pixels['psnr'], 'ssim': pixels['ssim'],
            'f1_curve': [(e['step'], e.get('variant', ''), e['human_removal']['f1']) for e in curve]}
def against(ref, arm):
    a = scores[ref][arm]['against_reference']; return {'ar': a['ar'], 'exact': a['exact'], 'ned': a['ned']}
def verdict(arm):   # section four, rule 1: against E-NAFNet
    a = scores['E-NAFNet'][arm]['against_reference']; lo, hi = a['exact']['interval']
    return ('clearly_better' if lo > 0 else 'clearly_worse' if hi < 0 else 'cannot_tell') if (lo > 0 or hi < 0) and abs(a['ar']['difference']) >= .02 else 'cannot_tell'
out = {'restorers': {m: restorer(m) for m in models},
       'arms': {arm: {k: v for k, v in s.items() if k != 'against_reference'} for arm, s in scores['E-NAFNet'].items()},
       'rule1_vs_E-NAFNet': {f'E-{m}': {**against('E-NAFNet', f'E-{m}'), 'verdict': verdict(f'E-{m}')} for m in models if m != 'NAFNet'},
       'rule3_vs_R0': {f'E-{m}': against('R0', f'E-{m}') for m in models}, 'rule3_vs_H': {f'E-{m}': against('H', f'E-{m}') for m in models},
       'reported_only': {arm: against('E-NAFNet', arm) for arm in scores['E-NAFNet'] if arm in ('E-none', 'P-NAFNet') or arm.startswith('E-NAFNet-later')}}
(R/'rules.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
read = {d.name[5:]: {g['id']: g for g in map(json.loads, open(d/'predictions.jsonl', encoding='utf-8'))} for d in R.glob('read_*') if d.is_dir()}
with (R/'readings_raw.jsonl').open('x', encoding='utf-8') as stream:
    for row in map(json.loads, open(R/'readings.jsonl', encoding='utf-8')):
        extra = {f'{a}_{k}': read[a][row['region']][k] for a in read if row['region'] in read[a] for k in ('raw', 'finish_reason')}
        stream.write(json.dumps({**row, **extra}, ensure_ascii=False)+'\n')
print(json.dumps({arm: v['verdict'] for arm, v in out['rule1_vs_E-NAFNet'].items()}))
PY
    mark score; fi
touch "$R/COMPLETE"; stamp "$PART complete"
