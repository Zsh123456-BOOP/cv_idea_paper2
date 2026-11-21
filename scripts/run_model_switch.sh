#!/usr/bin/env bash
# The plug-in on other models, stage one (plan of 2026-11-21, 换模型验证插件; the user: 「可以，开工，4点都听你的」): Qwen3.5-9B (the new
# env) and InternVL3.5-8B (the hme env, its bundled code under the safeguards; models.internvl), each with Qwen3-VL-8B's recipe:
#   H   a reader taught on the annotator's erased crops (context_revision, every region of the other folds, 100 steps), reading them;
#   R0  a reader taught on the scans (raw_context_revision), reading them;
#   V   H's reader with the tower adapter (V-修's recipe: real and 3,000 synthetic pairs, half each, at most 2,000 steps, the lowest
#       validation loss kept) seeing the second and third images of the scan's views.
# PART=check1 (once, Qwen3-VL-8B): the new env and the changed code reproduce what there is -- fold 0's H readings (at most 2% differ in
# the new env, every one the same in the hme env), the adapter's step-0 validation loss (within 0.001 of the sixth round's in the new
# env, the same in the hme env); V's fold-0 readings in the new env beside the sixth round's, reported.
# PART=smoke MODEL=...: the plan's checks 2-6 and 8 -- the untaught adapter's features equal the teacher's, the saved files apart, the
# folds and validation writers as Qwen3-VL-8B's, the paired images' token grids alike, no thinking written (Qwen3.5), ten steps of an
# adapter and of each reader, eight regions read each way.  PART=main MODEL=...: five folds of everything, the readings, the window
# cosines, eval.transcription against R0 and H, eval.model_switch.  Cards 1 and 2, at most two of our jobs on a card; 0 and 3 not touched.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; B=${N}_model_switch20261121; PART=${PART:-main}
NEW=/home/zsh/hme/envs/paper2_newmodels_20261121/bin/python; HME=$PY; MODELS=/home/zsh/hme/models; QWEN=$MODELS/Qwen3-VL-8B-Instruct
P2=${N}_nafnet_folds20261119; V=${N}_revision_folds_human_masked20261117b_views; EV=${P2}_erased_views; RAWV=${P2}_raw_views
SYN=${N}_privileged_view20261120_synth; FIX=${N}_covered_view20261121; SH=${P2}_reader_shared_f; R0R=${P2}_reader_R0_f
case "$PART" in
    check1) R=${B}_check1 ;;
    smoke|main) case "${MODEL:-}" in
                    Qwen3.5-9B) X=$NEW; SLUG=qwen35 ;;
                    InternVL3_5-8B) X=$HME; SLUG=internvl ;;
                    *) echo 'MODEL is Qwen3.5-9B or InternVL3_5-8B'; exit 2 ;; esac
                M=$MODELS/$MODEL; R=${B}_$SLUG; [ "$PART" = main ] || R=${R}_smoke ;;
    *) echo "PART is check1, smoke or main"; exit 2 ;; esac
case "$PART" in main) STEPS=100; P_STEPS=2000; P_EVAL=50; LIMIT=(); READ=(0 1 2 3 4) ;; *) STEPS=10; P_STEPS=10; P_EVAL=10; LIMIT=(--limit 8); READ=(0) ;; esac
mkdir -p "$R/checks"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
CARDS=("${ALL_CARDS[1]}" "${ALL_CARDS[2]}")
place() { place_on_cards "$@"; }
job() { local out="$1"; shift 2; bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1; }
teach() {   # OUT PYTHON MODEL CONDITION PACK FOLD: a reader taught on every region of the other folds (the second round's recipe)
    [ -e "$1/complete.json" ] || job "$1" -- "$2" -m revrender.train_reader --pack "$5" --condition "$4" --fold "$6" --all-rest --model "$3" --steps "$STEPS" \
        --batch-size 2 --accumulate 8 --warmup 10 --eval-every 10 --checkpointing --workers 2 --answer latex --output-dir "$1"
}
plugin() {   # OUT PYTHON MODEL FOLD [MORE...]: a tower adapter, V-修's recipe
    local out="$1" py="$2" model="$3" fold="$4"; shift 4
    [ -e "$out/complete.json" ] || job "$out" -- "$py" -m revrender.train_clean_view train --model "$model" --pack "$V" --scan-pack "$RAWV" --fold "$fold" \
        --synthetic "$SYN" --steps "$P_STEPS" --eval-every "$P_EVAL" "$@" --output-dir "$out"
}
read_with() {   # OUT PYTHON MODEL CONDITION PACK ADAPTER FOLD [MORE...]: the fold read, loops stopped
    local out="$1" py="$2" model="$3" cond="$4" pack="$5" adapter="$6" fold="$7"; shift 7
    [ -e "$out/complete.json" ] || job "$out" -- "$py" -m revrender.recognize_lines --pack "$pack" --model "$model" --backend hf --adapter "$adapter" \
        --batch-size 8 --conditions "$cond" --fold "$fold" --loop-limit 4 "${LIMIT[@]}" "$@" --output-dir "$out"
}
distance() {   # OUT PYTHON MODEL STUDENT FOLD
    [ -e "$1/distances.jsonl" ] || job "$1" -- "$2" -m revrender.train_clean_view distance --model "$3" --student "$4" --pack "$V" --raw-pack "$RAWV" \
        --erased-pack "$EV" --fold "$5" --output-dir "$1"
}
compare_reads() {   # MINE THEIRS OUT LIMIT: the readings that differ, listed
    "$HME" - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
from pathlib import Path
read = lambda d: {g['id']: g for g in map(json.loads, (Path(d)/'predictions.jsonl').read_text().splitlines())}
mine, theirs = read(sys.argv[1]), read(sys.argv[2]); ids = sorted(mine)
differ = [{'id': k, 'mine': mine[k]['text'], 'theirs': theirs[k]['text']} for k in ids if mine[k]['text'] != theirs[k]['text']]
r = {'regions': len(ids), 'differ': len(differ), 'share': len(differ)/len(ids), 'limit': float(sys.argv[4]), 'passed': len(differ)/len(ids) <= float(sys.argv[4]), 'readings': differ}
Path(sys.argv[3]).write_text(json.dumps(r, ensure_ascii=False, indent=2)); print(json.dumps({k: r[k] for k in ('regions', 'differ', 'share', 'passed')}))
PY
}
step0() {   # LOG REFERENCE TOLERANCE OUT: an adapter's step-0 validation loss against the sixth round's V-修 fold 0
    "$HME" -c 'import json,sys; a=json.loads(open(sys.argv[1]).readline())["val_loss"]; b=json.loads(open(sys.argv[2]).readline())["val_loss"]; t=float(sys.argv[3]); r={"mine":a,"sixth_round":b,"difference":a-b,"tolerance":t,"passed":abs(a-b)<=t}; json.dump(r,open(sys.argv[4],"w"),indent=2); print(json.dumps(r))' "$@"
}

if [ "$PART" = check1 ]; then
    for env in new hme; do py=$NEW; [ $env = new ] || py=$HME
        place 22000 1 read_with "$R/read_H_f0_$env" "$py" "$QWEN" context_revision "$V" "${SH}0/adapter" 0
        place 12000 0 plugin "$R/plugin_f0_$env" "$py" "$QWEN" 0 --steps 2000 --stop-after 1 --eval-every 50
    done
    place 22000 1 read_with "$R/read_V_f0_new" "$NEW" "$QWEN" context_revision "$RAWV" "${SH}0/adapter" 0 --view-student "${FIX}_view_fix_f0/student" --student-images 2 3
    wait
    compare_reads "$R/read_H_f0_new" "$P2/read_H_f0" "$R/checks/check1_H_new.json" 0.02
    compare_reads "$R/read_H_f0_hme" "$P2/read_H_f0" "$R/checks/check1_H_hme.json" 0
    compare_reads "$R/read_V_f0_new" "$FIX/read_fix_f0" "$R/checks/check1_V_new_reported.json" 1
    step0 "$R/plugin_f0_new/train_log.jsonl" "${FIX}_view_fix_f0/train_log.jsonl" 0.001 "$R/checks/check1_step0_new.json"
    step0 "$R/plugin_f0_hme/train_log.jsonl" "${FIX}_view_fix_f0/train_log.jsonl" 0 "$R/checks/check1_step0_hme.json"
    touch "$R/COMPLETE"; stamp "check1 complete"; exit 0
fi

# InternVL: the most tiles an image, set by memory alone before anything is read (the planning session's rule): the worst batch of each
# part (reader training, adapter training, reading) at 12, then 6, then 4 tiles; the largest that fits for all three, one value for
# all; the main run takes the smoke's.
probe() {   # OUT TILES PART
    [ -e "$1/result.json" ] || INTERNVL_MAX_TILES=$2 job "$1" -- "$X" -m revrender.models.internvl memory --model "$M" --clean-pack "$V" --scan-pack "$RAWV" \
        --part "$3" --output-dir "$1"
}
if [ "${SLUG:-}" = internvl ]; then
    if [ "$PART" = smoke ] && ! done_ tiles; then
        for t in 12 6 4; do
            for part in reader plugin read; do place 23000 1 probe "$R/memory_${t}_$part" "$t" "$part"; done; wait
            if "$HME" -c 'import json,sys; sys.exit(0 if all(json.load(open(f"{sys.argv[1]}/memory_{sys.argv[2]}_{p}/result.json"))["fitted"] for p in ("reader","plugin","read")) else 1)' "$R" "$t"; then
                echo "$t" > "$R/checks/max_tiles"; break; fi
        done
        test -s "$R/checks/max_tiles"
        mark tiles; fi
    TILES_FROM=$R; [ "$PART" = smoke ] || TILES_FROM=${B}_internvl_smoke
    export INTERNVL_MAX_TILES=$(cat "$TILES_FROM/checks/max_tiles"); stamp "InternVL at most $INTERNVL_MAX_TILES tiles an image"
fi

# The CPU checks: 5 the paired images' token grids alike (every region, the model's processor), 4 the folds and the validation writers.
if ! done_ cpu_checks; then
    "$X" - "$M" "$V" "$RAWV" "$R/checks/check5_grids.json" <<'PY'
import json, sys
from pathlib import Path
from PIL import Image
from revrender.models.vlm_tokens import processor_for
p = processor_for(sys.argv[1]); clean, scan = Path(sys.argv[2]), Path(sys.argv[3])
rows = {r['id']: r for r in json.loads((clean/'pack.json').read_text())['rows']}; scans = {r['id']: r for r in json.loads((scan/'pack.json').read_text())['rows']}
grid = lambda path: p.image_processor(images=[Image.open(path).convert('RGB')], return_tensors='pt')['image_grid_thw'][0].tolist()
bad = [(k, key) for k in rows for key in ('erased', 'context') if grid(clean/rows[k]['images'][key]) != grid(scan/scans[k]['images'][key])]
r = {'regions': len(rows), 'pairs': 2*len(rows), 'differ': bad, 'passed': not bad}; Path(sys.argv[4]).write_text(json.dumps(r, indent=2)); print(json.dumps({k: r[k] for k in ('pairs', 'passed')}))
sys.exit(0 if r['passed'] else 'paired images give different token grids')
PY
    mark cpu_checks; fi

if ! done_ teach; then
    for k in "${READ[@]}"; do
        place 22000 1 teach "$R/reader_H_f$k" "$X" "$M" context_revision "$V" "$k"
        place 22000 1 teach "$R/reader_R0_f$k" "$X" "$M" raw_context_revision "$RAWV" "$k"
        place 12000 0 plugin "$R/plugin_f$k" "$X" "$M" "$k"
    done; wait
    for k in "${READ[@]}"; do for d in reader_H reader_R0 plugin; do test -e "$R/${d}_f$k/complete.json"; done; done
    mark teach; fi

# Checks 2-4 on what was made: the untaught adapter's features (the frozen tower with a fresh adapter) equal the teacher's bit for bit;
# the reader's files hold no tower weight and the adapter's no language weight; the validation writers as Qwen3-VL-8B's fold by fold.
if ! done_ made_checks; then
    "$X" - "$M" "$V" "$R" "$FIX" "${READ[@]}" <<'PY'
import json, sys
from pathlib import Path
import torch
from PIL import Image
from safetensors import safe_open
from revrender.models.vlm_tokens import processor_for
from revrender.train_clean_view import encode, levels, vision_tower, with_student
model, pack, R, fix, folds = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], [int(k) for k in sys.argv[5:]]
rows = json.loads((pack/'pack.json').read_text())['rows'][:8]
p = processor_for(model); x, g = encode(p, [Image.open(pack/r['images']['erased']).convert('RGB') for r in rows], 'cuda')
student = with_student(vision_tower(model).to('cuda')).eval()
with torch.no_grad():
    s = levels(student, x, g)
    with student.disable_adapter(): t = levels(student, x, g)
check2 = {'images': len(rows), 'levels': len(s), 'equal': all(torch.equal(a, b) for a, b in zip(s, t))}
keys = lambda d: [k for f in sorted(Path(d).glob('*.safetensors')) for k in safe_open(str(f), 'pt').keys()]
check3 = {}
for k in folds:
    reader = keys(R/f'reader_H_f{k}/adapter')+keys(R/f'reader_R0_f{k}/adapter'); tower = keys(R/f'plugin_f{k}/student')
    check3[k] = {'reader_tensors': len(reader), 'reader_tower_tensors': [n for n in reader if 'visual' in n or 'vision_model' in n or 'mlp1' in n][:3],
                 'adapter_tensors': len(tower), 'adapter_language_tensors': [n for n in tower if 'language_model' in n][:3]}
check4 = {k: {'validation_writers_as_qwen3_vl': json.loads((R/f'plugin_f{k}'/'run_manifest.json').read_text())['validation_writers'] ==
                  json.loads((Path(f'{fix}_view_fix_f{k}')/'run_manifest.json').read_text())['validation_writers'],
              'fold': json.loads((R/f'plugin_f{k}'/'run_manifest.json').read_text())['config']['fold']} for k in folds}
r = {'check2_untaught_equals_teacher': check2, 'check3_files_apart': check3, 'check4_folds': check4,
     'passed': check2['equal'] and all(not v['reader_tower_tensors'] and not v['adapter_language_tensors'] and v['reader_tensors'] and v['adapter_tensors'] for v in check3.values())
               and all(v['validation_writers_as_qwen3_vl'] for v in check4.values())}
(R/'checks'/'check2_3_4.json').write_text(json.dumps(r, indent=2)); print(json.dumps({'passed': r['passed'], 'check2': check2}))
sys.exit(0 if r['passed'] else 'checks 2-4 failed')
PY
    mark made_checks; fi

if ! done_ reads; then
    for k in "${READ[@]}"; do
        place 22000 1 read_with "$R/read_H_f$k" "$X" "$M" context_revision "$V" "$R/reader_H_f$k/adapter" "$k"
        place 22000 1 read_with "$R/read_R0_f$k" "$X" "$M" raw_context_revision "$RAWV" "$R/reader_R0_f$k/adapter" "$k"
        place 22000 1 read_with "$R/read_V_f$k" "$X" "$M" context_revision "$RAWV" "$R/reader_H_f$k/adapter" "$k" --view-student "$R/plugin_f$k/student" --student-images 2 3
    done; wait
    for k in "${READ[@]}"; do for a in H R0 V; do test -e "$R/read_${a}_f$k/complete.json"; done; done
    # check 6: no thinking written, every reading as the answer's form
    "$HME" - "$R" <<'PY'
import json, sys
from pathlib import Path
rows = [g for d in sorted(Path(sys.argv[1]).glob('read_*')) if d.is_dir() for g in map(json.loads, (d/'predictions.jsonl').read_text().splitlines())]
think = [g['id'] for g in rows if '<think>' in g['raw'] or '</think>' in g['raw']]
r = {'readings': len(rows), 'with_thinking': think, 'finish': {f: sum(g['finish_reason'] == f for g in rows) for f in sorted({g['finish_reason'] for g in rows})},
     'passed': not think}
(Path(sys.argv[1])/'checks'/'check6_no_thinking.json').write_text(json.dumps(r, indent=2)); print(json.dumps(r))
sys.exit(0 if r['passed'] else 'a reading wrote thinking')
PY
    mark reads; fi

if [ "$PART" = smoke ]; then touch "$R/COMPLETE"; stamp "smoke complete"; exit 0; fi

if ! done_ distances; then
    for k in "${READ[@]}"; do place 8000 0 distance "$R/distance_f$k" "$X" "$M" "$R/plugin_f$k/student" "$k"; done; wait
    for k in "${READ[@]}"; do test -e "$R/distance_f$k/distances.jsonl"; done
    mark distances; fi

dirs() { local k s=; for k in "${READ[@]}"; do s+="${s:+,}$1_f$k"; done; echo "$s"; }
if ! done_ score; then
    ARMS_=(--arm "H=$(dirs "$R/read_H")" --arm "R0=$(dirs "$R/read_R0")" --arm "V=$(dirs "$R/read_V")")
    for ref in R0 H; do [ -e "$R/scores_vs_$ref.json" ] || "$HME" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference "$ref" \
        --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
    [ -e "$R/analysis/analysis.json" ] || "$HME" -m revrender.eval.model_switch --pack "$V" --model "$MODEL" "${ARMS_[@]}" --scores "R0=$R/scores_vs_R0.json" \
        --scores "H=$R/scores_vs_H.json" --shares "$FIX/checks/strike_shares.json" --distances "$(dirs "$R/distance")" --output-dir "$R/analysis" > "$R/analysis.log" 2>&1
    mark score; fi
touch "$R/COMPLETE"; stamp "main complete"
