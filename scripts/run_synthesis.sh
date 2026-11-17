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
# PART=train (stage two, after round 4 passed; planning session 2026-11-17): data.synthesis paste puts H's 364 units up to 12 times
#   each, P and D at their shares (8,736 regions), cleaned (S) and as cut (S0) by the same maps, the strike's ink the host writing's
#   plus a real difference; check 6 (the strike-less-writing ink against the real, beside the set pasted before) must pass; R (its version
#   scored before) on 1,000 of them beside its held-out numbers (reported, not judged); then NAFNet from scratch, seed 42, one at a
#   time: R+S and R+S0 on R's data and budget, M+S on M's, each sample synthetic with half a chance (baselines.restoration --extra),
#   the fixed 16 as their logged validation (the final step is the one used).
# PART=score (the last step, each model once on the 163 held-out regions): R, M at their final steps, R+S, R+S0, M+S, and the
#   library's model (round 1's main, step 8372); the rules of the 11-16 plan's section seven (switch, tiers, the unseen 10 writers).
# Card 2 only, waiting while it is taken.  Outputs paper2_clean_synth_materials_r$ROUND_ (from 3), paper2_clean_synth_*;
# marker $O/paper2_synth20261116.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_synth20261116
CARD=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; NUMA=1; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
CATALOGUE=/home/zsh/hme/code_358_spatial_v1/gold/line_edit_all5002_policy_decisions_20260814_v1/canonical/exact_crops_46902.jsonl
ROUND_=${ROUND_:-}; MAT=${N}_synth_materials_r$ROUND_
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR

if [ "${PART:-materials}" = materials ]; then
    [ -n "$ROUND_" ] || { echo "set ROUND_"; exit 1; }
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

SET2=${N}_prefill_set_r2; FIXED=$O/paper2_fixed_validation20261007; UP=$O/paper2_restoration_sources20261008
PASTE=${N}_synth_regions_ink; RUN_R=${N}_prefill_model_r2_real; RUN_M=${N}_prefill_model_r2_main; LIB=${N}_prefill_model_main:step_8372
final() { echo "$1:step_$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["steps"])' "$1/run_manifest.json")"; }
if [ "${PART:-}" = train ]; then
    source "$ROOT/scripts/_restorers.sh"; VAL=$FIXED
    if ! done_ paste_ink; then
        "$PY" -m revrender.data.synthesis paste --set $SET2 --bank ${N}_strike_bank_strokes --dataset ${N}_struck_dataset20261116/regions.json \
            --catalogue "$CATALOGUE" --mapping $O/paper2_splits/index_manifest.json --same-student $O/paper2_splits/same_student_pairs.json \
            --templates ${N}_page_templates/templates.npz --pages $O/paper2_pages_clean_unlabeled3976 --output-dir "$PASTE" > "$R/paste_ink.log" 2>&1
        mark paste_ink; fi
    if ! done_ ink_check; then            # check 6 (plan of 2026-11-17, 墨色对齐): the strike's ink against the writing's, before training
        "$PY" -m revrender.data.synthesis ink-check --set $SET2 --paste "$PASTE" --before ${N}_synth_regions_unmatched_ink \
            --pages $O/paper2_pages_clean_unlabeled3976 --output-dir "$R/ink_check" > "$R/ink_check.log" 2>&1
        mark ink_check; fi
    "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$R/ink_check/ink_check.json" || { stamp "check 6 failed"; touch "$R/INK_FAILED"; exit 1; }
    if ! done_ synthetic_table; then
        bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$R/synthetic_scores" -- "$PY" -m revrender.eval.prefill score --model "R=$RUN_R" \
            --set "synthetic=$PASTE/scored.json" --upstream "$UP" --output-dir "$R/synthetic_scores" > "$R/synthetic_scores.log" 2>&1
        "$PY" - "$R/synthetic_scores/scores.json" "$O/paper2_prefill_model20261116/scores/scores.json" "$R/synthetic_table.json" <<'PY'
import json, sys
syn = json.load(open(sys.argv[1]))['R']; real = json.load(open(sys.argv[2]))['r2-real']
def row(regions, s):
    return {'regions': s['regions'], 'blocks': s['blocks'], 'blocks_per_1000_strike': 1000*sum(r['blocks'] for r in regions)/max(1, sum(r['strike'] for r in regions)),
            'missed_share': s['missed_share'], 'writing_erased': s['writing_erased'], 'no_edit_share': s['no_edit_share']}
table = {'model': 'R (real only), the version scored on the held-out writers before', 'checkpoint': syn['checkpoint'], 'checkpoint_before': real['checkpoint'],
         'synthetic': row(syn['sets']['synthetic']['regions'], syn['sets']['synthetic']['summary']),
         'held_out_before': row(real['sets']['held_out']['regions'], real['sets']['held_out']['summary'])}
json.dump(table, open(sys.argv[3], 'x'), indent=1); print(json.dumps(table))
PY
        mark synthetic_table; fi
    arm() {   # NAME DATA BUDGET EXTRA
        RESTORER_MEMORY=10000 RESTORER_ARGS="--select blocks --extra $4" finetune_restorer NAFNet "$CARD" "$NUMA" - "$2" "${N}_synth_model_$1" "$3" > "$R/train_$1.log" 2>&1
        test -e "${N}_synth_model_$1/selection.json"
    }
    if ! done_ train; then
        arm rs ${N}_prefill_tiles_r2_real real_only "$PASTE/S"
        arm rs0 ${N}_prefill_tiles_r2_real real_only "$PASTE/S0"
        arm ms ${N}_prefill_tiles_r2_main prefill "$PASTE/S"
        mark train; fi
    stamp "stage two trained"; touch "$R/TRAIN_COMPLETE"
fi
if [ "${PART:-}" = score ]; then
    test -e "$R/TRAIN_COMPLETE"
    if ! done_ score; then
        bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$R/held_out" -- "$PY" -m revrender.eval.prefill score --model "R=$(final $RUN_R)" \
            --model "RS=$(final ${N}_synth_model_rs)" --model "RS0=$(final ${N}_synth_model_rs0)" --model "M=$(final $RUN_M)" \
            --model "MS=$(final ${N}_synth_model_ms)" --model "library=$LIB" --set "held_out=$SET2/validation_native.json" --upstream "$UP" \
            --output-dir "$R/held_out" > "$R/held_out.log" 2>&1
        mark score; fi
    S=$R/held_out/scores.json
    for pair in R:RS M:MS RS0:RS library:RS library:MS; do
        [ -e "$R/switch_${pair/:/_}.json" ] || "$PY" -m revrender.eval.prefill switch --scores "$S" --set held_out --current "${pair%%:*}" --candidate "${pair#*:}" --output "$R/switch_${pair/:/_}.json"
        [ -e "$R/unseen_${pair/:/_}.json" ] || "$PY" -m revrender.eval.prefill switch --scores "$S" --set held_out --current "${pair%%:*}" --candidate "${pair#*:}" \
            --without-writers-of "${N}_prefill_pseudo_main/records.json" --output "$R/unseen_${pair/:/_}.json"
    done
    [ -e "$R/tiers.json" ] || "$PY" -m revrender.eval.prefill tiers --scores "$S" --set held_out --output "$R/tiers.json" > "$R/tiers.log" 2>&1
    [ -e "$R/rules.json" ] || "$PY" - "$R" "${N}_prefill_pseudo_main/records.json" <<'PY'
import json, sys
from pathlib import Path
R = Path(sys.argv[1]); load = lambda n: json.loads((R/n).read_text()); seen = {r['group_uuid'] for r in json.load(open(sys.argv[2]))}
def useful(n):
    s = load(n); hi = s['interval'][1]
    return {'difference': s['difference'], 'interval': s['interval'], 'writing_erased_rise': s['writing_erased_rise'],
            'verdict': 'useful' if hi < 0 and s['writing_erased_rise'] <= .02 else ('cannot_tell' if s['interval'][0] < 0 < hi else ('worse' if s['interval'][0] > 0 else 'not_useful'))}
rules = {'1_synthetic_useful (RS - R)': useful('switch_R_RS.json'), '2_useful_over_distillation (MS - M)': useful('switch_M_MS.json'),
         '3_cleaning_useful (RS - RS0)': useful('switch_RS0_RS.json')}
rules['3_cleaning_useful (RS - RS0)']['verdict'] = 'useful' if load('switch_RS0_RS.json')['interval'][1] < 0 else 'cannot_tell_cleaning'
passed = [m for m, k in (('RS', '1_synthetic_useful (RS - R)'), ('MS', '2_useful_over_distillation (MS - M)')) if rules[k]['verdict'] == 'useful']
scores = json.loads((R/'held_out/scores.json').read_text())
if passed:
    best = min(passed, key=lambda m: scores[m]['sets']['held_out']['summary']['blocks'])
    sw, un = load(f'switch_library_{best}.json'), load(f'unseen_library_{best}.json')
    rules['4_candidate_for_the_library'] = {'model': best, 'switch': {k: sw[k] for k in ('difference', 'interval', 'writing_erased_rise', 'replace')},
        'unseen_10': {k: un[k] for k in ('difference', 'interval', 'regions', 'writers')}, 'switchable': bool(sw['replace'] and un['difference'] <= 0)}
else:
    rules['4_candidate_for_the_library'] = {'model': None, 'switchable': False}
rules['5_heavy'] = {m: load('tiers.json')['models'][m]['strike_share']['heavy'] for m in load('tiers.json')['models']}
rules['summaries'] = {m: scores[m]['sets']['held_out']['summary'] for m in scores}
unseen = lambda m: [r for r in scores[m]['sets']['held_out']['regions'] if r['writer'] not in seen]
rules['unseen_10_blocks'] = {m: {'regions': len(unseen(m)), 'writers': len({r['writer'] for r in unseen(m)}),
                                 'blocks': sum(r['blocks'] for r in unseen(m))/max(1, len(unseen(m)))} for m in scores}
json.dump(rules, open(R/'rules.json', 'x'), indent=1); print(json.dumps({k: v.get('verdict', v.get('switchable')) for k, v in rules.items() if k[0] in '1234'}))
PY
    stamp "stage two scored"; touch "$R/SCORE_COMPLETE"
fi
