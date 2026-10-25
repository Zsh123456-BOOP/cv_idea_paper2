#!/usr/bin/env bash
# A NAFNet student on the unlabelled library (data.pseudo_regions; Noisy Student, CVPR 2020, with U2PL's unreliable pixels): the three
# (b) restorers chosen on the wider validation (paper2_clean_wide_versions_finetune_long) each restore every eligible region (training
# pages, writers apart from the wider and the fixed validation, nothing the user confirmed); ink at least AGREE of them erase is the
# strike, ink only fewer erase weighs nothing in the loss; the 201 human regions ride along HUMAN_COPIES times.  The student starts
# from NAFNet's chosen synthetic version, trains with [pseudo] (configs/restoration.toml) and every kept version is read back on the
# wider validation like the other arms.  Unattended; a step whose marker is in $R is skipped.  TRAIN_CARD (a uuid) labels and trains;
# the read-back uses cards 1 and 2 (or 3 when nobody is on it).  VARIANT names another run, RESTORER_ARGS its student's arguments,
# COMBINE_ARGS its labelling (e.g. --soft: the teachers' mean image as the target); a variant reuses another run's teacher_* directories
# when they are linked into its own directory before it starts.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_pseudo_student20261025${VARIANT:+_$VARIANT}
V4=${N}_restoration_pairs20261007_v4; SET=${N}_v4_finetune_set; LIB=${N}_full_review_prefill_nafnet; EV=${N}_wide_versions_finetune_long
SEL1=${N}_v4_versions_synthetic; VAL=$O/paper2_fixed_validation20261007; UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
export DEV_EXPORT=$O/paper2_test_export20261022/export.json DEV_PAGES=$O/paper2_test_text20261018_page/site/images CHOOSE_SPLITS="val dev"
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
source "$ROOT/scripts/_restorers.sh"
TRAIN_CARD=${TRAIN_CARD:-$C1}; NUMA=0; case "$TRAIN_CARD" in *56971197*|*314fdf43*) NUMA=1 ;; esac
export RESTORER_ARGS="${RESTORER_ARGS:-}" RESTORER_MEMORY="${RESTORER_MEMORY:-8000}"; HUMAN_COPIES=${HUMAN_COPIES:-10}; AGREE=${AGREE:-2}; COMBINE_ARGS=${COMBINE_ARGS:-}
PSET=${N}_pseudo_set${VARIANT:+_$VARIANT}; TILES=${N}_pseudo_tiles${VARIANT:+_$VARIANT}; FT=${N}_pseudo_student${VARIANT:+_$VARIANT}
test -e "$EV/chosen/selection.json" || { echo "missing the teachers' selection: $EV"; exit 2; }

label() {   # MODEL: the (b) teacher's restoration of every eligible region, in $R/teacher_MODEL
    local model="$1" out="$R/teacher_$1" step runtime=$PY
    [ ! -e "$out/run_manifest.json" ] || return 0
    step="$(chosen_of "$EV/chosen" "$model")"
    (
        unset LD_LIBRARY_PATH
        if [ "$model" = DE-GAN ]; then
            runtime=/home/zsh/hme/envs/paper2_restoration_tf/bin/python; export TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=""
            for d in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; done
        fi
        idle=0; if [ "$TRAIN_CARD" = "$C3" ]; then idle=1; export ALLOW_GPU3=1; fi
        wait_for_gpu_memory "$TRAIN_CARD" "$RESTORER_MEMORY" "$idle"
        bash "$ROOT/scripts/launch_gpu_job.sh" "$TRAIN_CARD" "$NUMA" "$out" -- "$runtime" -m revrender.data.pseudo_regions predict --library "$LIB" \
            --run "$(dirname "$step")" --checkpoint "$(basename "$step")" --upstream "$UP" --evaluation "$EV" --review-set "$SET" \
            --output-dir "$out" > "$out.log" 2>&1 < /dev/null
    )
}

for m in NAFNet DE-GAN COR-Generator; do done_ "label_$m" || { label "$m"; mark "label_$m"; }; done
done_ combine || { "$PY" -m revrender.data.pseudo_regions combine --predictions "$R/teacher_NAFNet" "$R/teacher_DE-GAN" "$R/teacher_COR-Generator" \
    --review-set "$SET" --source "$V4" --agree "$AGREE" --human-copies "$HUMAN_COPIES" ${COMBINE_ARGS} --output-dir "$PSET" > "$R/combine.log" 2>&1; mark combine; }
done_ tiles || { "$PY" -m revrender.data.restoration_pairs human --review-set "$PSET" --source "$V4" --no-replay --output-dir "$TILES" > "$R/tiles.log" 2>&1; mark tiles; }
done_ finetune || { finetune_restorer NAFNet "$TRAIN_CARD" $NUMA "$(chosen_of "$SEL1/chosen" NAFNet)" "$TILES" "$FT" pseudo > "$R/train.log" 2>&1; mark finetune; }
if ! done_ versions; then
    [ -e "$R/candidates.json" ] || "$PY" -m revrender.eval.unified_restoration candidates --run "NAFNet-pseudo${VARIANT:+-$VARIANT}=$FT" \
        --upstream "$UP" --stain-upstream "$STAIN_UP" --output "$R/candidates.json" >> "$R/versions.log" 2>&1
    evaluate_versions "$R/candidates.json" "${N}_wide_versions_pseudo_student${VARIANT:+_$VARIANT}" "$R/versions.log"
    mark versions; fi
touch "$R/COMPLETE"; stamp complete
