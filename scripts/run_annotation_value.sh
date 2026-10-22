#!/usr/bin/env bash
# Does the second paper's annotation help every restorer, not only ours?  (user 2026-10-22)  Unattended; a step whose marker is in
# $R is skipped.
#   run_annotation_value.sh
# 1. The box remover (RevRender) trained on the very synthetic pairs the seven restorers trained on (data.restoration_pairs boxes:
#    the pairs cut as its strips; configs/revrender_strokes.toml from ImageNet), so all eight have the same synthetic data.
# 2. Each model's synthetic version chosen on validation read-back (eval.unified_restoration choose): among the versions whose pixel
#    F1 on the 16 fixed regions improved and the last (what a restorer's training keeps; the box remover's every 2,000 steps).
# 3. The same 201 human regions with the same synthetic replay for every model (data.review_set with the pairs' strips as replay;
#    data.restoration_pairs human for the restorers' tiles), each fine-tuned from its chosen version (configs/restoration.toml
#    [finetune]; the box remover's revrender_strokes_human.toml), the version chosen again the same way.
# 4. Every model's paired read-back difference (eval.unified_restoration compare) and where the errors fall (eval.restoration_errors).
# Cards 1 and 2; card 3 only while nobody else is on it.  Validation only: the test set is read once, at the end.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_annotation_value20261022
V4=${N}_restoration_pairs20261007_v4; BOXES=${N}_v4_boxes; OURS=${N}_v4_control
SET=${N}_v4_finetune_set; TILES=${N}_v4_finetune_tiles; FT=${N}_v4_finetune
SEL1=${N}_v4_versions_synthetic; SEL2=${N}_v4_versions_finetuned
VAL=$O/paper2_fixed_validation20261007; UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
MAP=$O/paper2_splits/index_manifest.json; PAIRS=$O/paper2_splits/same_student_pairs.json
FT_IN=$O/paper2_pixel_finetune_inputs20261007; EXPORT=$O/paper2_pixel_round_struck20261012_inputs/training_annotations.json
C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
SUITE=${N}_restoration_suite
declare -A RUN=([COR-Generator]=$SUITE/lane0/COR-Generator_gold_pixels [Restormer]=$SUITE/lane0/Restormer_gold_pixels
                [NAFNet]=$SUITE/lane2/NAFNet_gold_pixels [DE-GAN]=$SUITE/lane2/DE-GAN_gold_pixels
                [Strikethrough-CycleGAN]=$SUITE/lane3/Strikethrough-CycleGAN_gold_pixels [DocDiff]=$SUITE/lane3/DocDiff_gold_pixels
                [StainRestorer]=${N}_stainrestorer/StainRestorer_gold_pixels)
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
sha() { sha256sum "$1/run_manifest.json" | cut -d' ' -f1; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
upstream() { [ "$1" = StainRestorer ] && echo "$STAIN_UP" || echo "$UP"; }

candidates() {   # STAGE OUT.json: the versions to choose from (stage synthetic | finetuned), every model
    local pairs=() k
    for k in "${!RUN[@]}"; do pairs+=("$k" "${RUN[$k]}"); done
    "$PY" - "$1" "$2" "$OURS" "$FT" "$UP" "$STAIN_UP" "${pairs[@]}" <<'PY'
import json, re, sys
from pathlib import Path
stage, out, ours, ft, up, stain_up, *pairs = sys.argv[1:]
runs = dict(zip(pairs[::2], pairs[1::2]))
def step_of(name):          # step_<n>[_raw|_ema]: an EMA version after the raw one of its step
    m = re.fullmatch(r'step_(\d+)(?:_(raw|ema))?(?:\.pt)?', name); return int(m[1])*10+(m[2] == 'ema')
label = lambda n: f'{n//10}'+('ema' if n % 10 else '')
specs = []
mine = sorted(Path(ours+'/formal').glob('step_*.pt'), key=lambda p: step_of(p.name)) if stage == 'synthetic' else \
       sorted(Path(ft+'_Ours/train/formal').glob('step_*.pt'), key=lambda p: step_of(p.name))
if stage == 'synthetic':    # every 2,000 steps and the last, as a restorer keeps a version every evaluation it improves
    mine = [p for p in mine if step_of(p.name)//10 % 2000 == 0] + [mine[-1]]*(step_of(mine[-1].name)//10 % 2000 != 0)
for p in mine:
    specs.append(dict(name=f'Ours-s{label(step_of(p.name))}', family='Ours', step=step_of(p.name), type='own', checkpoint=str(p)))
for family, run in sorted(runs.items()):
    root = Path(run if stage == 'synthetic' else ft+'_'+family)
    for d in sorted((d for d in root.glob('step_*') if d.is_dir()), key=lambda d: step_of(d.name)):
        specs.append(dict(name=f'{family}-s{label(step_of(d.name))}', family=family, step=step_of(d.name), type='baseline', run=str(root),
                          upstream=stain_up if family == 'StainRestorer' else up, checkpoint=d.name))
json.dump(specs, open(out, 'x'), indent=1)
print(len(specs), 'versions')
PY
}
second_card() {   # card 2 when it has room, else card 3 when nobody is on it -> its uuid
    while true; do
        [ "$(nvidia-smi -i "$C2" --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 15000 ] || { echo "$C2"; return; }
        [ -n "$(nvidia-smi -i "$C3" --query-compute-apps=pid --format=csv,noheader)" ] || { echo "$C3"; return; }
        sleep 60
    done
}
versions() {   # STAGE OUT: evaluate every candidate version on validation together, then choose
    local stage="$1" out="$2"
    if [ ! -e "$out/COMPLETED" ]; then
        [ -e "$R/candidates_$stage.json" ] || candidates "$stage" "$R/candidates_$stage.json" >> "$R/$stage.log" 2>&1
        MODELS="$R/candidates_$stage.json" GPU_A=$C1 GPU_B=$(second_card) bash "$ROOT/scripts/eval_unified_restoration.sh" "$out" >> "$R/$stage.log" 2>&1
    fi
    [ -e "$out/chosen/selection.json" ] || "$PY" -m revrender.eval.unified_restoration choose --evaluation "$out" --output-dir "$out/chosen" >> "$R/$stage.log" 2>&1
}
finetune() {   # FAMILY CARD NUMA: a restorer fine-tuned from its chosen synthetic version
    local family="$1" card="$2" numa="$3" out="${FT}_$1" init runtime=$PY
    [ ! -e "$out/selection.json" ] || return 0
    init=$("$PY" -c 'import json,sys; s=json.load(open(sys.argv[1]))[sys.argv[2]]["spec"]; print(s["run"]+"/"+s["checkpoint"])' "$SEL1/chosen/selection.json" "$family")
    (
        export PYTHONPATH="$ROOT"; unset LD_LIBRARY_PATH
        if [ "$family" = DE-GAN ]; then
            runtime=/home/zsh/hme/envs/paper2_restoration_tf/bin/python; export TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=""
            for d in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; done
        elif [ "$family" = StainRestorer ]; then export PYTHONPATH="$ROOT:/home/zsh/hme/envs/paper2_stain_deps"; fi
        idle=0; if [ "$card" = "$C3" ]; then idle=1; export ALLOW_GPU3=1; fi
        wait_for_gpu_memory "$card" 20000 "$idle"
        bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$out" -- "$runtime" -m revrender.baselines.restoration --model "$family" \
            --arm gold_pixels --protocol "$ROOT/configs/restoration.toml" --data "$TILES" --upstream "$(upstream "$family")" \
            --human-validation "$VAL/validation_native.json" --init "$init" --output-dir "$out" > "$out.log" 2>&1 < /dev/null
    )
}

# ---- 1. the same synthetic data for the box remover; the shared fine-tuning set --------------------------------------------------
done_ boxes || { "$PY" -m revrender.data.restoration_pairs boxes --source "$V4" --output-dir "$BOXES" > "$R/boxes.log" 2>&1; mark boxes; }
if ! done_ set; then
    "$PY" -m revrender.data.review_set --review-export "$EXPORT" --pages "$O/paper2_full_review20261010/images" \
        --mapping "$FT_IN/index_manifest.json" --same-student "$FT_IN/same_student_pairs.json" --replay "$BOXES" --fixed-validation "$VAL" \
        --output-dir "$SET" > "$R/set.log" 2>&1
    "$PY" -m revrender.data.restoration_pairs human --review-set "$SET" --source "$V4" --output-dir "$TILES" >> "$R/set.log" 2>&1
    "$PY" -m revrender.eval.restoration_errors strokes --review-set "$SET" --pairs "$V4" --output "$R/strokes.json" >> "$R/set.log" 2>&1
    mark set; fi
if ! done_ ours; then
    SYNTHETIC=1 MAPPING=$MAP SAME_STUDENT=$PAIRS RECIPE=$ROOT/configs/revrender_strokes.toml bash "$ROOT/scripts/launch_train_arm.sh" \
        control "$C1" 0 "$BOXES" "$(sha "$BOXES")" - - "$OURS" > "$R/ours.log" 2>&1
    mark ours; fi

# ---- 2. synthetic versions, chosen on read-back ---------------------------------------------------------------------------------
done_ synthetic || { versions synthetic "$SEL1"; "$PY" -m revrender.eval.restoration_errors zones --evaluation "$SEL1" --output "$R/zones_synthetic.json" >> "$R/synthetic.log" 2>&1; mark synthetic; }

# ---- 3. the 201 human regions for every model -----------------------------------------------------------------------------------
if ! done_ finetune; then
    if [ ! -e "${FT}_Ours/COMPLETE" ]; then
        INIT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["Ours"]["spec"]["checkpoint"])' "$SEL1/chosen/selection.json")
        bash "$ROOT/scripts/run_review_round.sh" "$SET" "$INIT" "$C1" 0 "${FT}_Ours" > "$R/finetune_Ours.log" 2>&1
    fi
    ( finetune NAFNet "$C1" 0; finetune DE-GAN "$C1" 0; finetune Restormer "$C1" 0 ) > "$R/finetune_lane1.log" 2>&1 & A=$!
    ( finetune StainRestorer "$C2" 1; finetune COR-Generator "$C2" 1 ) > "$R/finetune_lane2.log" 2>&1 & B=$!
    ( finetune DocDiff "$C3" 1; finetune Strikethrough-CycleGAN "$C3" 1 ) > "$R/finetune_lane3.log" 2>&1 & C=$!
    wait $A; wait $B; wait $C
    mark finetune; fi
done_ finetuned || { versions finetuned "$SEL2"; "$PY" -m revrender.eval.restoration_errors zones --evaluation "$SEL2" --output "$R/zones_finetuned.json" >> "$R/finetuned.log" 2>&1; mark finetuned; }

# ---- 4. the paired difference ---------------------------------------------------------------------------------------------------
done_ compare || { "$PY" -m revrender.eval.unified_restoration compare --before "$SEL1/chosen" --after "$SEL2/chosen" --output-dir "$R/compare" > "$R/compare.log" 2>&1; mark compare; }
touch "$R/COMPLETE"; stamp "complete"
