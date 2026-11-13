#!/usr/bin/env bash
# The layer reader, first trial (user 2026-11-13): does telling a taught Qwen3-VL-8B which ink is the strike help it read the struck text?
# PART=human (the default):
#  0. eval.transcription over the readings of the pilot of 2026-11-08 (no reading anew): the same four ways of reading the 122 strikes,
#     now in the literature's scores (AR charges what a reading adds; the pilot's read-back is CR).
#  1. data.struck_pack folds: the confirmed regions outside the test page, five folds of writers; each region's scan, erased crop and
#     layer map come from the annotator's own pixels -- the best a separator could give, so this part asks whether the condition can
#     help at all.
#  2. smoke: two steps of the mapped reader on fold 0, its projection no longer zero, eight regions read with it and not all alike.
#  3. per fold and arm -- raw (the scan), erased (the erased crop), pair (both, two images), mapped (the scan and the layer map into the
#     patch embedding) -- train_reader --fold K --keep-best, then recognize_lines --fold K on the writers it never saw.
#  4. the untaught 8B reads every region from the scan and from the erased crop.
#  5. eval.transcription, the reference the taught reader of the scan alone.
#  The rule, written before any reading: the layer map is worth a separator of its own when the mapped reader beats the taught reader of
#  the scan alone by more than 3 points of AR with its writer interval above 0, exact not lower.
# PART=separator (beside it; its reader jobs wait for the human part to finish):
#  1. data.struck_pack fold-sets, data.restoration_pairs human: each fold's review set and tiles, real regions only.
#  2. a NAFNet per fold from its own random start ([real_only]) on the other folds' regions (cards SEP_CARDS, default "0").
#  3. data.struck_pack separated: the pack again, erased crop and layer map now each fold's separator's, on writers it never trained on.
#  4. the erased, pair and mapped readers again on that pack; the untaught 8B on its erased crops (erase, then read).
#  5. eval.transcription against each control in turn: the taught reader of the scan (the human part's), erase-then-read, the pair.
#  The rule, written before any reading: the layer reader stands when the mapped reader beats each of those three by more than 3 points
#  of AR with its writer interval above 0, exact not lower.
# PART=views (after the separator part; the user's two questions and the first block of the design of 2026-11-13, each against the pair):
#  1. data.struck_pack views: the separated pack with a mild and a strong erasing of each region and the line it stands in.
#  2. smoke: two steps and eight regions of each new reader.
#  3. three readers, five folds each: context (the pair and the line, the region boxed), pair_mapped (the pair and the layer map into
#     the scan's patch embedding), bracket (the scan with the mild and the strong erasing).
#  4. eval.transcription, the reference the pair's reader of the separator part.
#  The rule, written before any reading, for each of the three: it adds to the method when it beats the pair by more than 3 points of AR
#  with its writer interval above 0, exact and NED no worse.
# PART=distilled (after the views part; does a better separator read better?  the second step of the design):
#  1. fold sets again, each with what its fold keeps apart; three teachers a fold (NAFNet, DE-GAN, COR-Generator from their chosen
#     synthetic versions, [finetune_long] on the other folds' real regions); each teacher restores the unlabelled library's regions.
#  2. data.pseudo_regions combine --soft and a NAFNet student from its own random start ([prefill], versions kept by blocks to fix):
#     the recipe of the prefill model, fold by fold.
#  3. data.struck_pack separated with the students; the pair's reader on it; eval.transcription against the pair of the plain separators.
#  4. --agreement: do the readings with and without the line differing mark the errors (the fourth block of the design)?
#  The rule, written before any reading: the separator is worth more work when the pair's reader on the students' crops beats the pair's
#  reader on the plain separators' by more than 3 points of AR with its writer interval above 0, exact and NED no worse.
# PART=joint (after the distilled part; the second block of the design: the separator taught by the reading):
#  1. models.hint_separator: its cut of an image into patches is the reader's image processor's; a two-step smoke in which it learns.
#  2. per fold, two pair readers whose erased crop the fold's student makes while they train: learned (from step 40 the student learns
#     from the loss on the reading, anchored to the crops a person erased) and frozen (it never does); each reads its fold with the
#     hints its separator ends with.
#  3. transfer: a fresh pair reader taught on the learned separator's hints -- do they help a reader that did not shape them?
#  4. eval.transcription, the reference the frozen arm.
#  The rule, written before any reading: the reading may teach the separator when the learned arm beats the frozen one by more than 3
#  points of AR with its writer interval above 0, exact and NED no worse, and the fresh reader keeps at least half of that gain.
# Unattended; a step whose marker is in $R is skipped.  CARDS and SEP_CARDS (default "1": the user's word of 2026-11-13, card 1 only)
# are the cards, by index, a reader job and a separator's training may go to; a reader job has a card to itself.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_layer_reader20261113
DATA=$O/paper2_struck_dataset20261113; PACK=${N}_revision_folds_human; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
PILOT=$O/paper2_struck_reader_pilot20261108; ARMS=(raw erased pair mapped)
PART=${PART:-human}; SETS=${N}_revision_fold_sets; SEP=${N}_revision_folds_separator; UP=$O/paper2_restoration_sources20261008; TAG=
VIEWS=${N}_revision_folds_views
SETS2=${N}_revision_fold_sets_apart; DISTILLED=${N}_revision_folds_distilled; SEL1=${N}_v4_versions_synthetic; LIB=${N}_full_review_prefill_soft
failure() { case "$PART" in human) echo FAILED ;; separator) echo SEPARATOR_FAILED ;; views) echo VIEWS_FAILED ;; distilled) echo DISTILLED_FAILED ;; *) echo JOINT_FAILED ;; esac; }
cards() { local i; PICK=(); for i in $1; do PICK+=("${CARDS_ALL[$i]}"); done; ALL_CARDS=("${PICK[@]}"); }
mkdir -p "$R"
exec 8>"$R/$([ "$PART" = human ] && echo lock || echo "$PART.lock")"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/$(failure)"' ERR

CARDS_ALL=("${ALL_CARDS[@]}"); cards "${CARDS:-1}"
alone() {   # FUNCTION ARGS...: run FUNCTION CARD NUMA ARGS... in the background on a card of ours with room and no other job of ours (the 8B fills a card)
    local card c
    until card=$(for c in "${ALL_CARDS[@]}"; do [ "$(ours_on "$c")" -eq 0 ] && [ "$(nvidia-smi -i "$c" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 22000 ] && { echo "$c"; break; }; done; true) && [ -n "$card" ]; do sleep 30; done
    ( "$1" "$card" "$(numa_of "$card")" "${@:2}" ) & PLACED_PIDS+=($!); PLACED_CARDS+=("$card")
}
job() {   # CARD NUMA OUT -- COMMAND...: one GPU job
    local card="$1" numa="$2" out="$3"; shift 4
    bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$out" -- "$@" > "$out.log" 2>&1
}
when_mapped() {   # ARM WORDS...: the words, for an arm whose reader takes the layer map
    case "$1" in (*mapped) echo "${@:2}" ;; esac
}
teach() {   # OUT ARM FOLD [more arguments of train_reader]: a reader of the arm, fold left out
    local out="$1" arm="$2" k="$3"; shift 3
    echo "$PY" -m revrender.train_reader --pack "$PACK" --condition "${arm}_revision" --fold "$k" --model "$MODEL" --steps 100 --batch-size 2 --accumulate 8 \
        --warmup 10 --eval-every 10 --keep-best --checkpointing --workers 2 $(when_mapped "$arm" --layer-map) "$@" --output-dir "$out"
}
read_with() {   # OUT ARM READER [more arguments of recognize_lines]
    local out="$1" arm="$2" reader="$3"; shift 3
    echo "$PY" -m revrender.recognize_lines --pack "$PACK" --model "$MODEL" --backend hf --adapter "$reader/adapter" --batch-size 8 --conditions "${arm}_revision" \
        $(when_mapped "$arm" --layer-map "$reader/layer_map.pt") "$@" --output-dir "$out"
}
arm_job() {   # CARD NUMA ARM FOLD: teach, then read the fold left out
    local reader="${N}_revision_reader_${TAG}$3_f$4" reading="$R/read_${TAG}$3_f$4"
    [ -e "$reader/complete.json" ] || job "$1" "$2" "$reader" -- $(teach "$reader" "$3" "$4")
    [ -e "$reading/complete.json" ] || job "$1" "$2" "$reading" -- $(read_with "$reading" "$3" "$reader" --fold "$4")
}
folds_of() { local a="$1" k s=; for k in 0 1 2 3 4; do s+="${s:+,}$R/read_${a}_f$k"; done; echo "$s"; }
if [ "$PART" = human ]; then
done_ rescore || { "$PY" -m revrender.eval.transcription --pack "${N}_struck_reader_eval_pack" --arm "erase_then_read=$PILOT/read_untaught@erased_neutral" \
    --arm "untaught_scan=$PILOT/read_untaught@raw_neutral" --arm "untaught_struck=$PILOT/read_untaught@raw_struck" --arm "taught_struck=$PILOT/read_taught" \
    --reference erase_then_read --output "$R/rescore_pilot.json" > "$R/rescore.log" 2>&1; mark rescore; }
if ! done_ pack; then
    [ -e "$PACK/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack folds --dataset "$DATA/regions.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --hold-out test-page --output-dir "$PACK" > "$R/pack.log" 2>&1
    mark pack; fi
if ! done_ smoke; then
    smoke_job() {   # CARD NUMA
        job "$1" "$2" "$R/smoke_reader" -- $(teach "$R/smoke_reader" mapped 0 --steps 2 --eval-every 1 --warmup 1)
        "$PY" -c "import sys, torch; w = torch.load(sys.argv[1])['weight']; assert float(w.abs().max()) > 0, 'the layer map was not taught'; print('layer map', tuple(w.shape), float(w.abs().max()))" "$R/smoke_reader/layer_map.pt"
        job "$1" "$2" "$R/smoke_read" -- $(read_with "$R/smoke_read" mapped "$R/smoke_reader" --fold 0 --limit 8)
        "$PY" -c "import json, sys; texts = {json.loads(l)['text'] for l in open(sys.argv[1])}; assert len(texts) > 2, f'eight regions read as {texts}: the reader no longer sees the image'" "$R/smoke_read/predictions.jsonl"
    }
    alone smoke_job; wait; test -e "$R/smoke_read/complete.json"; mark smoke; fi
if ! done_ arms; then
    for k in 0 1 2 3 4; do for arm in "${ARMS[@]}"; do [ -e "$R/read_${arm}_f$k/complete.json" ] || alone arm_job "$arm" "$k"; done; done
    wait; for k in 0 1 2 3 4; do for arm in "${ARMS[@]}"; do test -e "$R/read_${arm}_f$k/complete.json"; done; done
    mark arms; fi
if ! done_ untaught; then
    [ -e "$R/read_untaught/complete.json" ] || alone job "$R/read_untaught" -- "$PY" -m revrender.recognize_lines --pack "$PACK" --model "$MODEL" --backend hf \
        --batch-size 8 --conditions raw_revision erased_revision --output-dir "$R/read_untaught"
    wait; test -e "$R/read_untaught/complete.json"; mark untaught; fi
done_ score || { "$PY" -m revrender.eval.transcription --pack "$PACK" --arm "taught_scan=$(folds_of raw)" --arm "taught_erased=$(folds_of erased)" \
    --arm "taught_pair=$(folds_of pair)" --arm "taught_mapped=$(folds_of mapped)" --arm "untaught_scan=$R/read_untaught@raw_revision" \
    --arm "untaught_erased=$R/read_untaught@erased_revision" --reference taught_scan --output "$R/scores.json" > "$R/score.log" 2>&1; mark score; }
touch "$R/COMPLETE"; stamp complete; exit 0; fi

if [ "$PART" = separator ]; then
source "$ROOT/scripts/_restorers.sh"
if ! done_ sep_sets; then
    [ -e "$SETS/fold_4/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack fold-sets --dataset "$DATA/regions.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --hold-out test-page --output-dir "$SETS" > "$R/sep_sets.log" 2>&1
    for k in 0 1 2 3 4; do [ -e "${N}_revision_separator_tiles_f$k/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "$SETS/fold_$k" \
        --no-replay --output-dir "${N}_revision_separator_tiles_f$k" >> "$R/sep_sets.log" 2>&1; done
    mark sep_sets; fi
sep_job() {   # CARD NUMA FOLD: the fold's separator
    VAL="$SETS/fold_$3" RESTORER_MEMORY=10000 finetune_restorer NAFNet "$1" "$2" - "${N}_revision_separator_tiles_f$3" "${N}_revision_separator_f$3" real_only > "$R/sep_train_f$3.log" 2>&1
}
if ! done_ sep_train; then
    cards "${SEP_CARDS:-1}"
    for k in 0 1 2 3 4; do [ -e "${N}_revision_separator_f$k/selection.json" ] || place_job 10000 sep_job "$k"; done
    wait; for k in 0 1 2 3 4; do test -e "${N}_revision_separator_f$k/selection.json"; done
    mark sep_train; fi
if ! done_ sep_pack; then
    cards "${SEP_CARDS:-1}"
    [ -e "$SEP/run_manifest.json" ] || place_job 6000 job "$SEP" -- "$PY" -m revrender.data.struck_pack separated --pack "$PACK" --sets "$SETS" \
        $(for k in 0 1 2 3 4; do echo --separator "${N}_revision_separator_f$k"; done) --upstream "$UP" --output-dir "$SEP"
    wait; test -e "$SEP/run_manifest.json"; mark sep_pack; fi
until [ -e "$R/COMPLETE" ]; do [ ! -e "$R/FAILED" ] || { echo 'the human part failed'; exit 1; }; sleep 60; done      # the reader jobs fill a card: after the human part's
cards "${CARDS:-1}"; HUMAN=$PACK; PACK=$SEP; TAG=sep_
if ! done_ sep_arms; then
    for k in 0 1 2 3 4; do for arm in erased pair mapped; do [ -e "$R/read_sep_${arm}_f$k/complete.json" ] || alone arm_job "$arm" "$k"; done; done
    wait; for k in 0 1 2 3 4; do for arm in erased pair mapped; do test -e "$R/read_sep_${arm}_f$k/complete.json"; done; done
    mark sep_arms; fi
if ! done_ sep_untaught; then
    [ -e "$R/read_sep_untaught/complete.json" ] || alone job "$R/read_sep_untaught" -- "$PY" -m revrender.recognize_lines --pack "$PACK" --model "$MODEL" --backend hf \
        --batch-size 8 --conditions erased_revision --output-dir "$R/read_sep_untaught"
    wait; test -e "$R/read_sep_untaught/complete.json"; mark sep_untaught; fi
if ! done_ sep_score; then
    for reference in taught_scan erase_then_read taught_pair; do
        [ -e "$R/scores_separator_vs_$reference.json" ] || "$PY" -m revrender.eval.transcription --pack "$PACK" --arm "taught_scan=$(folds_of raw)" \
            --arm "erase_then_read=$R/read_sep_untaught" --arm "taught_erased=$(folds_of sep_erased)" --arm "taught_pair=$(folds_of sep_pair)" \
            --arm "taught_mapped=$(folds_of sep_mapped)" --arm "untaught_scan=$R/read_untaught@raw_revision" --reference "$reference" \
            --output "$R/scores_separator_vs_$reference.json" > "$R/sep_score_$reference.log" 2>&1
    done; mark sep_score; fi
touch "$R/SEPARATOR_COMPLETE"; stamp 'separator part complete'; exit 0; fi

if [ "$PART" = views ]; then test -e "$R/SEPARATOR_COMPLETE"
if ! done_ views_pack; then
    [ -e "$VIEWS/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack views --pack "$SEP" --sets "$SETS" --dataset "$DATA/regions.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --hold-out test-page --output-dir "$VIEWS" > "$R/views_pack.log" 2>&1
    mark views_pack; fi
PACK=$VIEWS; TAG=views_; NEW=(context pair_mapped bracket)
if ! done_ views_smoke; then
    views_smoke_job() {   # CARD NUMA: two steps and eight regions of each new reader
        local arm
        for arm in "${NEW[@]}"; do
            job "$1" "$2" "$R/views_smoke_reader_$arm" -- $(teach "$R/views_smoke_reader_$arm" "$arm" 0 --steps 2 --eval-every 1 --warmup 1)
            job "$1" "$2" "$R/views_smoke_read_$arm" -- $(read_with "$R/views_smoke_read_$arm" "$arm" "$R/views_smoke_reader_$arm" --fold 0 --limit 8)
            "$PY" -c "import json, sys; texts = {json.loads(l)['text'] for l in open(sys.argv[1])}; assert len(texts) > 2, f'eight regions read as {texts}'" "$R/views_smoke_read_$arm/predictions.jsonl"
        done
    }
    alone views_smoke_job; wait; for arm in "${NEW[@]}"; do test -e "$R/views_smoke_read_$arm/complete.json"; done; mark views_smoke; fi
if ! done_ views_arms; then
    for arm in "${NEW[@]}"; do for k in 0 1 2 3 4; do [ -e "$R/read_views_${arm}_f$k/complete.json" ] || alone arm_job "$arm" "$k"; done; done
    wait; for arm in "${NEW[@]}"; do for k in 0 1 2 3 4; do test -e "$R/read_views_${arm}_f$k/complete.json"; done; done
    mark views_arms; fi
done_ views_score || { "$PY" -m revrender.eval.transcription --pack "$PACK" --arm "taught_pair=$(folds_of sep_pair)" --arm "taught_scan=$(folds_of raw)" \
    --arm "context=$(folds_of views_context)" --arm "pair_mapped=$(folds_of views_pair_mapped)" --arm "bracket=$(folds_of views_bracket)" \
    --reference taught_pair --output "$R/scores_views.json" > "$R/views_score.log" 2>&1; mark views_score; }
touch "$R/VIEWS_COMPLETE"; stamp 'views part complete'; exit 0; fi

if [ "$PART" = distilled ]; then
source "$ROOT/scripts/_restorers.sh"
if ! done_ distilled_sets; then
    [ -e "$SETS2/fold_4/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack fold-sets --dataset "$DATA/regions.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --hold-out test-page --output-dir "$SETS2" > "$R/distilled_sets.log" 2>&1
    mark distilled_sets; fi
until [ -e "$R/VIEWS_COMPLETE" ]; do [ ! -e "$R/VIEWS_FAILED" ] || { echo 'the views part failed'; exit 1; }; sleep 60; done      # one card: after the views part
cards "${SEP_CARDS:-1}"; TEACHERS=(NAFNet DE-GAN COR-Generator)
teacher_job() {   # CARD NUMA FAMILY FOLD: a teacher of the fold, from its family's chosen synthetic version
    VAL="$SETS2/fold_$4" RESTORER_MEMORY=10000 finetune_restorer "$3" "$1" "$2" "$(chosen_of "$SEL1/chosen" "$3")" "${N}_revision_separator_tiles_f$4" \
        "${N}_revision_teacher_$3_f$4" finetune_long > "$R/distilled_teacher_$3_f$4.log" 2>&1
}
if ! done_ distilled_teachers; then
    for k in 0 1 2 3 4; do for t in "${TEACHERS[@]}"; do [ -e "${N}_revision_teacher_${t}_f$k/selection.json" ] || place_job 10000 teacher_job "$t" "$k"; done; done
    wait; for k in 0 1 2 3 4; do for t in "${TEACHERS[@]}"; do test -e "${N}_revision_teacher_${t}_f$k/selection.json"; done; done
    mark distilled_teachers; fi
label_job() {   # CARD NUMA FAMILY FOLD: the teacher's restoration of the library's regions this fold may learn from
    local run="${N}_revision_teacher_$3_f$4" out="$R/distilled_labels_$3_f$4" runtime=$PY step
    step=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/selection.json")
    (
        unset LD_LIBRARY_PATH
        if [ "$3" = DE-GAN ]; then
            runtime=/home/zsh/hme/envs/paper2_restoration_tf/bin/python; export TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=""
            for d in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; done
        fi
        bash "$ROOT/scripts/launch_gpu_job.sh" "$1" "$2" "$out" -- "$runtime" -m revrender.data.pseudo_regions predict --library "$LIB" --run "$run" --checkpoint "$step" \
            --upstream "$UP" --prefill-set "$SETS2/fold_$4" --output-dir "$out" > "$out.log" 2>&1
    )
}
if ! done_ distilled_labels; then
    for k in 0 1 2 3 4; do for t in "${TEACHERS[@]}"; do [ -e "$R/distilled_labels_${t}_f$k/run_manifest.json" ] || place_job 10000 label_job "$t" "$k"; done; done
    wait; for k in 0 1 2 3 4; do for t in "${TEACHERS[@]}"; do test -e "$R/distilled_labels_${t}_f$k/run_manifest.json"; done; done
    mark distilled_labels; fi
if ! done_ distilled_data; then
    for k in 0 1 2 3 4; do
        [ -e "${N}_revision_distil_set_f$k/run_manifest.json" ] || "$PY" -m revrender.data.pseudo_regions combine \
            --predictions $(for t in "${TEACHERS[@]}"; do echo "$R/distilled_labels_${t}_f$k"; done) --review-set "$SETS2/fold_$k" --soft --human-share .25 --pseudo-weight .5 \
            --output-dir "${N}_revision_distil_set_f$k" >> "$R/distilled_data.log" 2>&1
        [ -e "${N}_revision_distil_tiles_f$k/run_manifest.json" ] || "$PY" -m revrender.data.restoration_pairs human --review-set "${N}_revision_distil_set_f$k" --no-replay \
            --output-dir "${N}_revision_distil_tiles_f$k" >> "$R/distilled_data.log" 2>&1
    done; mark distilled_data; fi
student_job() {   # CARD NUMA FOLD: the fold's student
    VAL="$SETS2/fold_$3" RESTORER_MEMORY=10000 RESTORER_ARGS="--select blocks" finetune_restorer NAFNet "$1" "$2" - "${N}_revision_distil_tiles_f$3" \
        "${N}_revision_distilled_f$3" prefill > "$R/distilled_student_f$3.log" 2>&1
}
if ! done_ distilled_students; then
    for k in 0 1 2 3 4; do [ -e "${N}_revision_distilled_f$k/selection.json" ] || place_job 10000 student_job "$k"; done
    wait; for k in 0 1 2 3 4; do test -e "${N}_revision_distilled_f$k/selection.json"; done
    mark distilled_students; fi
if ! done_ distilled_pack; then
    [ -e "$DISTILLED/run_manifest.json" ] || place_job 6000 job "$DISTILLED" -- "$PY" -m revrender.data.struck_pack separated --pack "$PACK" --sets "$SETS2" \
        $(for k in 0 1 2 3 4; do echo --separator "${N}_revision_distilled_f$k"; done) --upstream "$UP" --output-dir "$DISTILLED"
    wait; test -e "$DISTILLED/run_manifest.json"; mark distilled_pack; fi
cards "${CARDS:-1}"; PACK=$DISTILLED; TAG=distilled_
if ! done_ distilled_arms; then
    for k in 0 1 2 3 4; do [ -e "$R/read_distilled_pair_f$k/complete.json" ] || alone arm_job pair "$k"; done
    [ -e "$R/read_distilled_untaught/complete.json" ] || alone job "$R/read_distilled_untaught" -- "$PY" -m revrender.recognize_lines --pack "$PACK" --model "$MODEL" --backend hf \
        --batch-size 8 --conditions erased_revision --output-dir "$R/read_distilled_untaught"
    wait; for k in 0 1 2 3 4; do test -e "$R/read_distilled_pair_f$k/complete.json"; done; test -e "$R/read_distilled_untaught/complete.json"
    mark distilled_arms; fi
done_ distilled_score || { "$PY" -m revrender.eval.transcription --pack "$PACK" --arm "taught_pair=$(folds_of sep_pair)" --arm "distilled_pair=$(folds_of distilled_pair)" \
    --arm "distilled_erase_then_read=$R/read_distilled_untaught" --arm "erase_then_read=$R/read_sep_untaught" --arm "taught_scan=$(folds_of raw)" \
    --arm "human_pair=$(folds_of pair)" --arm "context=$(folds_of views_context)" --reference taught_pair --agreement context=taught_pair --agreement taught_pair=taught_scan \
    --output "$R/scores_distilled.json" > "$R/distilled_score.log" 2>&1; mark distilled_score; }
touch "$R/DISTILLED_COMPLETE"; stamp 'distilled part complete'; exit 0; fi

test "$PART" = joint
until [ -e "$R/DISTILLED_COMPLETE" ]; do [ ! -e "$R/DISTILLED_FAILED" ] || { echo 'the distilled part failed'; exit 1; }; sleep 60; done      # one card, and its students
cards "${CARDS:-1}"; HUMAN=$PACK; PACK=$DISTILLED
joint_job() {   # KIND FOLD [more of train_reader]: a pair reader whose separator learns from the reading (learned) or never does (frozen), then its fold read with that separator's hints
    local kind="$1" k="$2"; shift 2; local reader="${N}_revision_reader_joint_${kind}_f$k" reading="$R/read_joint_${kind}_f$k" lr=()
    [ "$kind" != frozen ] || lr=(--separator-lr 0)
    [ -e "$reader/complete.json" ] || job "$CARD_" "$NUMA_" "$reader" -- $(teach "$reader" pair "$k" --separator "${N}_revision_distilled_f$k" --upstream "$UP" \
        --anchor-pack "$HUMAN" ${lr[@]+"${lr[@]}"} "$@")
    [ -e "$reading/complete.json" ] || job "$CARD_" "$NUMA_" "$reading" -- $(read_with "$reading" pair "$reader" --fold "$k" --erased "$reader/hints")
}
on_card() { CARD_="$1"; NUMA_="$2"; "${@:3}"; }   # CARD NUMA FUNCTION ARGS...: the function with the card it was given in CARD_ and NUMA_
transfer_job() {   # FOLD: a fresh pair reader taught on the learned separator's hints
    local reader="${N}_revision_reader_joint_transfer_f$1" reading="$R/read_joint_transfer_f$1" hints="${N}_revision_reader_joint_learned_f$1/hints"
    [ -e "$reader/complete.json" ] || job "$CARD_" "$NUMA_" "$reader" -- $(teach "$reader" pair "$1" --erased "$hints")
    [ -e "$reading/complete.json" ] || job "$CARD_" "$NUMA_" "$reading" -- $(read_with "$reading" pair "$reader" --fold "$1" --erased "$hints")
}
if ! done_ joint_smoke; then
    "$PY" -m revrender.models.hint_separator --model "$MODEL" --pack "$PACK" > "$R/joint_check.log" 2>&1
    joint_smoke_job() {   # two steps, the separator learning from the first
        local reader="$R/joint_smoke_reader"
        job "$CARD_" "$NUMA_" "$reader" -- $(teach "$reader" pair 0 --separator "${N}_revision_distilled_f0" --upstream "$UP" --anchor-pack "$HUMAN" --steps 2 --eval-every 1 --warmup 1 --separator-from 0)
        "$PY" -c "import json, sys, torch; a = torch.load(sys.argv[1], map_location='cpu')['model']; b = torch.load(sys.argv[2], map_location='cpu', weights_only=False)['model']; moved = max(float((a[k].float()-b[k].float()).abs().max()) for k in a); assert moved > 0, 'the separator did not learn'; print('separator moved', moved)" \
            "$reader/separator.pt" "${N}_revision_distilled_f0/$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${N}_revision_distilled_f0/selection.json")/weights.pt"
        job "$CARD_" "$NUMA_" "$R/joint_smoke_read" -- $(read_with "$R/joint_smoke_read" pair "$reader" --fold 0 --limit 8 --erased "$reader/hints")
        "$PY" -c "import json, sys; texts = {json.loads(l)['text'] for l in open(sys.argv[1])}; assert len(texts) > 2, f'eight regions read as {texts}'" "$R/joint_smoke_read/predictions.jsonl"
    }
    alone on_card joint_smoke_job; wait; test -e "$R/joint_smoke_read/complete.json"; mark joint_smoke; fi
if ! done_ joint_arms; then
    for k in 0 1 2 3 4; do for kind in learned frozen; do [ -e "$R/read_joint_${kind}_f$k/complete.json" ] || alone on_card joint_job "$kind" "$k"; done; done
    wait; for k in 0 1 2 3 4; do for kind in learned frozen; do test -e "$R/read_joint_${kind}_f$k/complete.json"; done; done
    mark joint_arms; fi
if ! done_ joint_transfer; then
    for k in 0 1 2 3 4; do [ -e "$R/read_joint_transfer_f$k/complete.json" ] || alone on_card transfer_job "$k"; done
    wait; for k in 0 1 2 3 4; do test -e "$R/read_joint_transfer_f$k/complete.json"; done
    mark joint_transfer; fi
done_ joint_score || { "$PY" -m revrender.eval.transcription --pack "$PACK" --arm "frozen=$(folds_of joint_frozen)" --arm "learned=$(folds_of joint_learned)" \
    --arm "transfer=$(folds_of joint_transfer)" --arm "distilled_pair=$(folds_of distilled_pair)" --arm "taught_pair=$(folds_of sep_pair)" --arm "human_pair=$(folds_of pair)" \
    --reference frozen --output "$R/scores_joint.json" > "$R/joint_score.log" 2>&1; mark joint_score; }
touch "$R/JOINT_COMPLETE"; stamp 'joint part complete'
