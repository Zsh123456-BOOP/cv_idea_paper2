#!/usr/bin/env bash
# RevSep with the writing gate (models.separation --writing-gate; plan work/model_plan20261025/保字擦线_保字门_方案_2026-10-25.md).
# Declared before any result: the method, RevSep-W, is the arm whose struck-writing weight (--near-weight 2, 4 or 8) reads best on the
# validation strikes alone, at the thresholds (erase 0.5/0.7/0.9 x writing 0.3/0.5/0.7) that read best there; the ablations at weight
# 4 -- symmetric (weight 1), the hard gate while training, no hint -- with their own thresholds chosen the same way, the method's arm
# with its thresholds chosen by pixel F1 instead, and RevSep without the gate (the stroke version's deciding batch's).  The recipe of
# (b) for every arm (synthetic pairs, version chosen on validation and the development half at thresholds 0.5, the 201 human regions
# alone [finetune_long], chosen again), then the threshold grid (validation only), then one deciding batch with RevSep and the seven
# restorers' (b) versions chosen in that batch of the stroke version, with the error zones, the struck writing's erased share
# (restoration_diagnostics writing) and RevSep-W paired with every other family.  Rules: reaches the bar if its paired differences
# with RevSep and with the best baseline are both above 0 and its writer interval against RevSep starts above -2; the mechanism holds
# only if its struck writing's erased share is below RevSep's and its strike missed next to writing at most 31.1 + 3.
# Unattended; a step whose marker is in $R is skipped.  Every job goes to whichever card has room (_common.sh open_card).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_revsep_gate20261025
V4=${N}_restoration_pairs20261007_v4; TILES=${N}_revsep_tiles_human
VAL=$O/paper2_fixed_validation20261007; VAL_HINTS=${N}_revsep_validation_hints/hints.npz
UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream; STROKE_FINAL=${N}_revsep_stroke_versions_final
export DETECTOR=$O/paper2_pages_clean_run_unlabeled3976_tight
C0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
export DEV_EXPORT=$O/paper2_test_export20261022/export.json DEV_PAGES=$O/paper2_test_text20261018_page/site/images CHOOSE_SPLITS="val dev"
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
source "$ROOT/scripts/_restorers.sh"
ARMS=(RevSep-W-n2 RevSep-W-n4 RevSep-W-n8 RevSep-W-sym RevSep-W-hard RevSep-W-nohint)
BASELINES=(COR-Generator NAFNet DE-GAN Restormer DocDiff Strikethrough-CycleGAN StainRestorer)
flags() { case "$1" in RevSep-W-n2) echo --near-weight 2 ;; RevSep-W-n4) echo --near-weight 4 ;; RevSep-W-n8) echo --near-weight 8 ;;
                       RevSep-W-sym) echo --near-weight 1 ;; RevSep-W-hard) echo --near-weight 4 --gate-hard ;; RevSep-W-nohint) echo --near-weight 4 --no-hint ;; esac; }

arm_job() {   # CARD NUMA ARM STAGE INIT DATA BUDGET: one arm trained
    local card="$1" numa="$2" arm="$3" stage="$4" init="$5" data="$6" budget="$7"
    RESTORER_MEMORY=10000 RESTORER_FLAGS="--writing-gate $(flags "$arm")" finetune_restorer RevSep "$card" "$numa" "$init" "$data" \
        "${N}_revsep_${stage}_$arm" $budget > "$R/train_${stage}_$arm.log" 2>&1
}
train() {   # STAGE CHOSEN DATA BUDGET: every arm at once, each on a card with room
    local stage="$1" chosen="$2" data="$3" budget="$4"
    for arm in "${ARMS[@]}"; do
        local init=-; [ "$chosen" = - ] || init=$(chosen_of "$chosen" "$arm")
        place_job 10000 arm_job "$arm" "$stage" "$init" "$data" "$budget"
    done
    wait
    for arm in "${ARMS[@]}"; do test -e "${N}_revsep_${stage}_$arm/selection.json"; done
}
versions() {   # STAGE: every kept version of the arms evaluated together and chosen (thresholds 0.5)
    local stage="$1" runs=(); for arm in "${ARMS[@]}"; do runs+=(--run "$arm=${N}_revsep_${stage}_$arm"); done
    [ -e "$R/candidates_$stage.json" ] || "$PY" -m revrender.eval.unified_restoration candidates "${runs[@]}" --upstream "$UP" \
        --stain-upstream "$STAIN_UP" --output "$R/candidates_$stage.json" >> "$R/versions_$stage.log" 2>&1
    evaluate_versions "$R/candidates_$stage.json" "${N}_revsep_gate_versions_$stage" "$R/versions_$stage.log"
}
done_ train_synthetic || { train synthetic - "$V4" ""; mark train_synthetic; }
done_ versions_synthetic || { versions synthetic; mark versions_synthetic; }
done_ train_finetune_long || { train finetune_long "${N}_revsep_gate_versions_synthetic/chosen" "$TILES" finetune_long; mark train_finetune_long; }
done_ versions_finetune_long || { versions finetune_long; mark versions_finetune_long; }
if ! done_ thresholds; then
    G=${N}_revsep_gate_thresholds
    [ -e "$R/candidates_thresholds.json" ] || "$PY" -c 'import json,sys
chosen=json.load(open(sys.argv[1])); specs=[]
for family, c in sorted(chosen.items()):
    for e in (.5, .7, .9):
        for w in (.3, .5, .7):
            specs.append(dict(c["spec"], name=f"{family}-e{int(10*e)}w{int(10*w)}", erase_threshold=e, writing_threshold=w))
json.dump(specs, open(sys.argv[2], "w"), indent=1)' "${N}_revsep_gate_versions_finetune_long/chosen/selection.json" "$R/candidates_thresholds.json"
    evaluate_versions "$R/candidates_thresholds.json" "$G" "$R/thresholds.log"
    [ -e "$G/by_read_back" ] || "$PY" -m revrender.eval.unified_restoration grid --evaluation "$G" --splits val --output-dir "$G/by_read_back" >> "$R/thresholds.log" 2>&1
    [ -e "$G/by_pixel_f1" ] || "$PY" -m revrender.eval.unified_restoration grid --evaluation "$G" --splits val --by pixel_f1 --output-dir "$G/by_pixel_f1" >> "$R/thresholds.log" 2>&1
    mark thresholds; fi
if ! done_ versions_final; then
    G=${N}_revsep_gate_thresholds; E=${N}_revsep_gate_versions_final
    [ -e "$R/candidates_final.json" ] || "$PY" -c 'import json,sys
by_read, by_f1, stroke = (json.load(open(p)) for p in sys.argv[1:4])
weights = ("RevSep-W-n2", "RevSep-W-n4", "RevSep-W-n8")
method = max(weights, key=lambda f: by_read[f]["chosen"]["read_back"])     # the struck-writing weight that reads best on validation
named = lambda family, spec: dict(spec, family=family, name=family+"-"+spec["name"].rsplit("-", 1)[-1])
specs = [named("RevSep-W", by_read[method]["spec"]), named("RevSep-W-F1thr", by_f1[method]["spec"])]
specs += [named(f, by_read[f]["spec"]) for f in ("RevSep-W-sym", "RevSep-W-hard", "RevSep-W-nohint")]
specs += [named(f, stroke[f]["spec"]) for f in ["RevSep", *sys.argv[5].split()]]
json.dump(specs, open(sys.argv[4], "w"), indent=1)
json.dump({"method": method, "near_weight": {"RevSep-W-n2": 2, "RevSep-W-n4": 4, "RevSep-W-n8": 8}[method],
           "validation_read_back": {f: by_read[f]["chosen"]["read_back"] for f in weights}}, open(sys.argv[4].replace("candidates_final", "method"), "w"), indent=1)' \
        "$G/by_read_back/selection.json" "$G/by_pixel_f1/selection.json" "$STROKE_FINAL/chosen/selection.json" "$R/candidates_final.json" "${BASELINES[*]}"
    evaluate_versions "$R/candidates_final.json" "$E" "$R/versions_final.log"
    [ -e "$R/zones_final.json" ] || "$PY" -m revrender.eval.restoration_errors zones --evaluation "$E" --splits val dev --output "$R/zones_final.json" >> "$R/versions_final.log" 2>&1
    [ -e "$R/writing_final.json" ] || "$PY" -m revrender.eval.restoration_diagnostics writing --evaluation "$E" --dev-export "$DEV_EXPORT" --dev-pages "$DEV_PAGES" \
        --output "$R/writing_final.json" >> "$R/versions_final.log" 2>&1
    for other in "${BASELINES[@]}" RevSep RevSep-W-F1thr RevSep-W-sym RevSep-W-hard RevSep-W-nohint; do
        [ -e "$R/compare_final_$other" ] || "$PY" -m revrender.eval.unified_restoration compare --before "$E/chosen" --after "$E/chosen" \
            --pair RevSep-W="$other" --output-dir "$R/compare_final_$other" >> "$R/versions_final.log" 2>&1
    done
    mark versions_final; fi
touch "$R/COMPLETE"; stamp complete
