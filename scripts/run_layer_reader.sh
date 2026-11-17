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
#  1. data.struck_pack views: the separated pack with the line each region stands in.
#  2. smoke: two steps and eight regions of each new reader.
#  3. two readers, five folds each: context (the pair and the line, the region boxed), pair_mapped (the pair and the layer map into
#     the scan's patch embedding).
#  4. eval.transcription, the reference the pair's reader of the separator part.
#  The rule, written before any reading, for each of the two: it adds to the method when it beats the pair by more than 3 points of AR
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
# PART=evidence (after the distilled part; the plan of 2026-11-14, two lines on the students' crops, loops stopped in every reading):
#  1. data.struck_pack counterfactual: each region's evidence taken apart by the annotator's strike pixels (strike off; writing off).
#  2. per fold two readers: the counterfactual pair reader (train_reader --counterfactual: a strike taken off changes nothing, the
#     struck writing taken off leaves nothing to read) and a reader of the erased crop alone; then the fold read by the scan's reader,
#     the erased crop's, the pair's and the counterfactual one, and by the last two with the strike and with the writing taken off.
#  3. eval.transcription; router (critics of each source's reading, the most trusted one given) over scan / erased / pair, and with
#     the counterfactual reader as a fourth source.
#  The rules, written before any reading.  The counterfactual reader stands when it beats the pair by more than 3 points of AR with its
#  writer interval above 0, exact and NED no worse, and, the struck writing gone, refuses at least twice as often as the pair's reader.
#  The router stands when its readings beat the pair's by more than 3 points of AR with the writer interval above 0, exact no worse, and
#  its area under the risk-coverage curve is below the pair's by its own confidence.
# PART=masked (after the evidence part; the plan of 2026-11-15): the crops a reader sees, white outside the annotator's polygon.
#  1. data.struck_pack folds, separated (the distilled students as they are, nothing taught anew) and counterfactual again: new packs
#     ..._masked; then nothing more until the crops are looked at by hand (touch $R/masked_checked.done).
#  2. per fold, taught anew on the new crops with the same recipe: the scan's reader, the human pair (the ceiling), the distilled
#     erased crop's reader and the distilled pair; read with loops stopped, the distilled pair also with the strike taken off; the
#     untaught 8B reads the scan and the human erased crop.
#  3. eval.transcription: every way of reading on the new crops paired by writer with the same way on the old crops.
#  The rule, written before any reading.  The new crops take the place of the old for every number after this one unless, for the
#  distilled pair's reader, the writer interval of its AR difference or of its exact difference lies wholly below 0; then the old
#  crops stay (the change is reverted) and the new numbers are only recorded.  A correction of the input, not a method.
# PART=ceiling (the plan of 2026-11-16, work/literature_revision_reader20261113/人工像素上限重训_方案与判定规则_2026-11-16.md): the ceiling
#  (the human pair) taught anew on the data the user has gone through.
#  1. data.struck_pack folds on paper2_clean_line_dataset20261116b, each writer keeping its fold and val mark of the masked pack
#     (--folds-from); data.line_set unchanged: the regions whose box, polygon and pixels (and text) the corrections left alone.  Then
#     nothing more until the crops are looked at by hand (touch $R/ceiling_checked.done).
#  2. per fold, the pair's reader of the masked part, its recipe unchanged, taught on the new pack and reading the fold it never saw.
#  3. eval.transcription: the old readings and the new on the unchanged regions (pixels; pixels and text), the new against the untaught
#     reader of the line part (R0) with every region's readings written out; eval.line_reading: the new readings line by line.
#  The rules, written before the readings, are the plan's: the old readings first give 85.3 / 59.1 again on the old pack and labels (the
#  spelling of 876943d, the code table 4 was scored with); the ceiling is the reader taught anew whichever way it moves; it is marked
#  worse after teaching when, on the regions nothing changed, its AR or exact difference from the old readings lies wholly below 0;
#  taught beats untaught when the lower end of its exact interval against R0 is above 0; differences under 2 AR are not read.
# PART=ceiling2 (the plan of 2026-11-17, work/literature_revision_reader20261113/人工像素上限第二轮_方案与判定规则_2026-11-17.md; each
#  job on the card of the four with the most room, the user's word): the ceiling's teaching and input changed, the scores not.  The pack
#  of the ceiling again with each region's
#  line (struck_pack views) and its question; struck_pack answers: the LaTeX answers come back to the targets.  Then, each a reader per
#  fold with the ceiling's recipe on the pack, read with loops stopped: A the pair taught its answers in LaTeX; B1 that and the question
#  in the words (only if 95% of the regions have one); B2 that and the line as a third image; B12 both, only if B1 and B2 each stand;
#  C the reading chosen, 300 steps a fold.  The ceiling's readings (U0) are not read again.  The rules, written before the readings, are
#  the plan's section two; ceiling2_rules.json holds every verdict and the arms' scores, readings_raw.jsonl every reading and its raw output.
# PART=ceiling3 (the plan of 2026-11-17, work/literature_revision_reader20261113/人工像素上限第三轮_方案与判定规则_2026-11-17.md; cards 1, 2
#  and 3, the user's word, each job on the one with the most room): on the pack with the user's answers, D1 the ceiling (D0, the second
#  round's B2 taught again on that pack, not read again) with the question in its words as well; N8 the untaught 8B and N32 the untaught
#  32B (FP8, two cards) reading the ceiling's three images and words, both with vLLM and the same decoding.  The rules, written before the
#  readings, are the plan's section three: ceiling3_rules.json holds the verdicts and the arms' scores, readings_raw.jsonl every reading.
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
failure() { case "$PART" in human) echo FAILED ;; separator) echo SEPARATOR_FAILED ;; views) echo VIEWS_FAILED ;; distilled) echo DISTILLED_FAILED ;; masked) echo MASKED_FAILED ;; ceiling) echo CEILING_FAILED ;; ceiling2) echo CEILING2_FAILED ;; ceiling3) echo CEILING3_FAILED ;; prefill) echo PREFILL_FAILED ;; *) echo EVIDENCE_FAILED ;; esac; }
CF=${N}_revision_folds_counterfactual
cards() { local i; PICK=(); for i in $1; do PICK+=("${CARDS_ALL[$i]}"); done; ALL_CARDS=("${PICK[@]}"); }
mkdir -p "$R"
exec 8>"$R/$([ "$PART" = human ] && echo lock || echo "$PART.lock")"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/$(failure)"' ERR

CARDS_ALL=("${ALL_CARDS[@]}"); cards "${CARDS:-1}"
alone() {   # FUNCTION ARGS...: run FUNCTION CARD NUMA ARGS... in the background on the card of ours with the most room of those with room and no
            # other job of ours (the 8B fills a card); card 3 only while nobody else is on it
    local card c free best
    until card=$(best=0; for c in "${ALL_CARDS[@]}"; do free=$(nvidia-smi -i "$c" --query-gpu=memory.free --format=csv,noheader,nounits)
                     [ "$(ours_on "$c")" -eq 0 ] && [ "$free" -ge 22000 ] && [ "$free" -gt "$best" ] || continue
                     if [ "$c" = "$CARD3" ] && card_users "$c" | grep -qvx "$ME"; then continue; fi; best=$free; echo "$c"; done | tail -1) && [ -n "$card" ]; do sleep 30; done
    ( [ "$card" != "$CARD3" ] || export ALLOW_GPU3=1; "$1" "$card" "$(numa_of "$card")" "${@:2}" ) & PLACED_PIDS+=($!); PLACED_CARDS+=("$card")
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
on_card() { CARD_="$1"; NUMA_="$2"; "${@:3}"; }   # CARD NUMA FUNCTION ARGS...: the function with the card it was given in CARD_ and NUMA_
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
    [ -e "$VIEWS/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack views --pack "$SEP" --dataset "$DATA/regions.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --hold-out test-page --output-dir "$VIEWS" > "$R/views_pack.log" 2>&1
    mark views_pack; fi
PACK=$VIEWS; TAG=views_; NEW=(context pair_mapped)
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
    --arm "context=$(folds_of views_context)" --arm "pair_mapped=$(folds_of views_pair_mapped)" \
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

if [ "$PART" = masked ]; then test -e "$R/EVIDENCE_COMPLETE"
HM=${N}_revision_folds_human_masked; DM=${N}_revision_folds_distilled_masked; CM=${N}_revision_folds_counterfactual_masked
if ! done_ masked_packs; then
    cards "${SEP_CARDS:-1}"
    [ -e "$HM/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack folds --dataset "$DATA/regions.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --hold-out test-page --output-dir "$HM" > "$R/masked_pack.log" 2>&1
    [ -e "$DM/run_manifest.json" ] || place_job 6000 job "$DM" -- "$PY" -m revrender.data.struck_pack separated --pack "$HM" --sets "$SETS2" \
        $(for k in 0 1 2 3 4; do echo --separator "${N}_revision_distilled_f$k"; done) --upstream "$UP" --output-dir "$DM"
    wait; test -e "$DM/run_manifest.json"
    [ -e "$CM/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack counterfactual --pack "$DM" --human "$HM" --output-dir "$CM" >> "$R/masked_pack.log" 2>&1
    mark masked_packs; fi
done_ masked_checked || { stamp 'the new packs are built: look at the crops, then touch masked_checked.done and run again'; exit 0; }
cards "${CARDS:-1}"
masked_teach() {   # NAME ARM PACK FOLD: the fold's reader NAME on the new crops
    local name="$1" arm="$2" PACK="$3" k="$4" reader="${N}_revision_reader_masked_$1_f$4"
    [ -e "$reader/complete.json" ] || job "$CARD_" "$NUMA_" "$reader" -- $(teach "$reader" "$arm" "$k")
}
masked_read() {   # OUT NAME ARM FOLD PACK [more of recognize_lines]: the fold read by the reader NAME, loops stopped
    local out="$R/masked_$1_f$4" name="$2" arm="$3" k="$4" PACK="$5"; shift 5
    [ -e "$out/complete.json" ] || job "$CARD_" "$NUMA_" "$out" -- $(read_with "$out" "$arm" "${N}_revision_reader_masked_${name}_f$k" --fold "$k" --loop-limit 4 "$@")
}
masked_fold() {   # FOLD: its four readers, then every reading of the fold
    local k="$1"
    masked_teach raw raw "$HM" "$k"; masked_teach pair pair "$HM" "$k"; masked_teach distilled_erased erased "$DM" "$k"; masked_teach distilled_pair pair "$DM" "$k"
    masked_read scan raw raw "$k" "$HM"; masked_read human_pair pair pair "$k" "$HM"; masked_read erased distilled_erased erased "$k" "$DM"
    masked_read pair distilled_pair pair "$k" "$CM"; masked_read pair_strike_off distilled_pair pair "$k" "$CM" --intervention strike_off
}
MREAD=(scan human_pair erased pair pair_strike_off)
if ! done_ masked_folds; then
    for k in 0 1 2 3 4; do alone on_card masked_fold "$k"; done
    [ -e "$R/masked_untaught/complete.json" ] || alone job "$R/masked_untaught" -- "$PY" -m revrender.recognize_lines --pack "$HM" --model "$MODEL" --backend hf \
        --batch-size 8 --conditions raw_revision erased_revision --loop-limit 4 --output-dir "$R/masked_untaught"
    wait; for k in 0 1 2 3 4; do for name in "${MREAD[@]}"; do test -e "$R/masked_${name}_f$k/complete.json"; done; done; test -e "$R/masked_untaught/complete.json"
    mark masked_folds; fi
five() { local a="$1" k s=; for k in 0 1 2 3 4; do s+="${s:+,}$R/${a}_f$k"; done; echo "$s"; }
if ! done_ masked_score; then
    pairs=("untaught_scan|$R/reread_untaught@raw_revision|$R/masked_untaught@raw_revision" "untaught_human_erased|$R/reread_untaught@erased_revision|$R/masked_untaught@erased_revision"
           "taught_scan|$(five reread_raw)|$(five masked_scan)" "taught_erased|$(five evidence_erased)|$(five masked_erased)" "taught_pair|$(five reread_distilled_pair)|$(five masked_pair)"
           "pair_strike_off|$(five evidence_pair_strike_off)|$(five masked_pair_strike_off)" "human_pair|$(five reread_pair)|$(five masked_human_pair)")
    for p in "${pairs[@]}"; do IFS='|' read -r name old new <<< "$p"
        [ -e "$R/scores_masked_$name.json" ] || "$PY" -m revrender.eval.transcription --pack "$CM" --arm "old=$old" --arm "new=$new" --reference old \
            --output "$R/scores_masked_$name.json" > "$R/masked_score_$name.log" 2>&1
    done
    [ -e "$R/scores_masked.json" ] || "$PY" -m revrender.eval.transcription --pack "$CM" --arm "untaught_scan=$R/masked_untaught@raw_revision" \
        --arm "untaught_human_erased=$R/masked_untaught@erased_revision" $(for name in "${MREAD[@]}"; do echo --arm "$name=$(five "masked_$name")"; done) \
        --reference pair --output "$R/scores_masked.json" > "$R/masked_score.log" 2>&1
    "$PY" - "$R/scores_masked_taught_pair.json" "$R/masked_decision.json" <<'PYTHON'
import json, sys
new = json.load(open(sys.argv[1]))['arms']['new']['against_reference']
below = [m for m in ('ar', 'exact') if new[m]['interval'][1] < 0]
json.dump({'adopted': not below, 'below_zero': below, **{m: new[m] for m in ('ar', 'exact', 'ned')}}, open(sys.argv[2], 'x'), indent=1)
print('the new crops', 'stay out: ' + ', '.join(below) if below else 'take the place of the old')
PYTHON
    mark masked_score; fi
touch "$R/MASKED_COMPLETE"; stamp 'masked part complete'; exit 0; fi

if [ "$PART" = prefill ]; then cards "${CARDS:-2}"
# The prefill for the whole library (user 2026-11-16), after run_prefill_model.sh ROUND=2: the pixel model its switch rule chose (round
# 2's main, else its real, else round 1's); every detection redrawn tight round the strike the model finds (tight_region) and the model
# again on that crop; a pair reader taught on every confirmed region but the prefill set's held-out writers (struck_pack folds
# --holdout-share .1: fold 0 those writers) writes each region's starting text.  Measured first on the held-out writers' pages, inferred
# afresh without the annotator's records: the tight polygons against the annotator's (and those from the annotator's own strike, the
# best the model's could give), and the text read from the model's crops against the annotator's (and from the annotator's crops).  Then
# the library: the same pages as round 1's, every export's records kept (later exports first, the line page's corrections over them),
# a record only moved or reshaped refilled in its own polygon, bundled with the drafts and one shared KaTeX.  Card 2 only.
P=$O/paper2_prefill_text20261116; M2=$O/paper2_prefill_model20261116; MERGED=${N}_prefill_data20261116; SET2=${N}_prefill_set_r2
LIB1=${N}_full_review_prefill_r1; EXPORTS=$O/paper2_annotation_exports20261116; KATEX=$O/paper2_katex_0.16.25
SPL=$O/paper2_pixel_finetune_inputs20261007; RP=${N}_prefill_reader_pack20261116; RD=${N}_prefill_reader20261116
LH=${N}_prefill_reader_heldout20261116; LF=${N}_prefill_reader_library20261116; mkdir -p "$P"; test -e "$M2/COMPLETE"
if ! done_ prefill_choice; then
    "$PY" - "$M2" "$N" "$LIB1" "$MERGED/regions.json" "$SET2/summary.json" "$P" <<'PYTHON'
import json, sys
m2, n, lib, merged, summary, out = sys.argv[1:]
for arm in ('main', 'real'):
    if json.load(open(f'{m2}/switch_{arm}.json'))['replace']: run = f'{n}_prefill_model_r2_{arm}'; break
else: run = f'{n}_prefill_model_main'
open(f'{out}/choice.txt', 'x').write(f"{run} {json.load(open(run+'/selection.json'))['checkpoint']}\n")
pages = [p['id'] for p in json.load(open(lib+'/pack.json'))['pages']]; open(f'{out}/library_pages.txt', 'x').write('\n'.join(pages)+'\n')
held = set(json.load(open(summary))['validation_writers'])
mine = sorted({r['page'] for r in json.load(open(merged)) if r['writer'] in held} & set(pages)); open(f'{out}/held_out_pages.txt', 'x').write('\n'.join(mine)+'\n')
PYTHON
    mark prefill_choice; fi
read -r RUN STEP < "$P/choice.txt"
infer_job() {   # CARD NUMA OUT NAME IDS [more of pixel_library infer]
    local card="$1" numa="$2" out="$3" name="$4" ids="$5"; shift 5
    job "$card" "$numa" "$out" -- "$PY" -m revrender.tools.annotate.pixel_library infer --restorer "$RUN" --restorer-checkpoint "$STEP" --upstream "$UP" \
        --pages "$O/paper2_pages_clean_unlabeled3976" --detector "$O/paper2_pages_clean_run_unlabeled3976_tight" --mapping "$SPL/index_manifest.json" \
        --same-student "$SPL/same_student_pairs.json" --validation "$O/paper2_fixed_validation20261007/validation_native.json" --ids "$ids" --tight --name "$name" "$@" --output-dir "$out"
}
reader_job() {   # CARD NUMA: the pair reader, the held-out writers left out
    [ -e "$RD/complete.json" ] || PACK=$RP job "$1" "$2" "$RD" -- $(PACK=$RP teach "$RD" pair 0)
}
read_job() {   # CARD NUMA OUT PACK
    [ -e "$3/complete.json" ] || job "$1" "$2" "$3" -- $(PACK=$4 read_with "$3" pair "$RD" --fold 0 --loop-limit 4)
}
if ! done_ prefill_heldout; then
    [ -e "$P/heldout_lib/COMPLETE" ] || { alone infer_job "$P/heldout_lib" paper2-full-review-heldout "$P/held_out_pages.txt"; wait; }
    [ -e "$RP/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack folds --dataset "$MERGED/regions.json" --pages "$O/paper2_full_review20261010/images" \
        --holdout-share .1 --output-dir "$RP" > "$P/reader_pack.log" 2>&1
    [ -e "$LH/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack library --library "$P/heldout_lib" --dataset "$MERGED/regions.json" --holdout-share .1 \
        --output-dir "$LH" >> "$P/reader_pack.log" 2>&1
    alone reader_job; wait; test -e "$RD/complete.json"
    alone read_job "$P/read_human" "$RP"; wait; alone read_job "$P/read_heldout" "$LH"; wait
    test -e "$P/read_human/complete.json"; test -e "$P/read_heldout/complete.json"
    for how in oracle predicted; do [ -e "$P/tight_$how.json" ] || "$PY" -m revrender.tools.annotate.tight_region score --dataset "$MERGED/regions.json" \
        --library "$P/heldout_lib" --writers "$SET2/summary.json" $([ $how = predicted ] && echo --predicted) --output "$P/tight_$how.json" > "$P/tight_$how.log" 2>&1; done
    [ -e "$P/scores_heldout.json" ] || "$PY" -m revrender.eval.transcription --pack "$LH" --only "$LH/held_out.json:ids" --arm "model_pixels=$P/read_heldout@pair_revision" \
        --arm "human_pixels=$P/read_human@pair_revision" --reference human_pixels --readings "$P/readings_heldout.jsonl" --output "$P/scores_heldout.json" > "$P/scores_heldout.log" 2>&1
    mark prefill_heldout; fi
if ! done_ prefill_library; then
    exports=(); for f in "$EXPORTS"/*.json; do exports+=(--review-export "$f"); done
    [ -e "$P/library/COMPLETE" ] || { alone infer_job "$P/library" paper2-full-review "$P/library_pages.txt" "${exports[@]}" --dataset "$MERGED/regions.json"; wait; }
    test -e "$P/library/COMPLETE"
    [ -e "$LF/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack library --library "$P/library" --output-dir "$LF" > "$P/library_pack.log" 2>&1
    alone read_job "$P/read_library" "$LF"; wait; test -e "$P/read_library/complete.json"
    [ -e "$P/site/run_manifest.json" ] || "$PY" -m revrender.tools.annotate.pixel_library bundle --pack "$P/library" --readings "$P/read_library/predictions.jsonl" \
        --condition pair_revision --katex "$KATEX" --output-dir "$P/site" > "$P/bundle.log" 2>&1
    mark prefill_library; fi
# The reader taught on the model's erased crops (2026-11-16): taught on the annotator's, it read the model's crops at 44 AR on the held-out
# writers (the annotator's: 86), the strike the model leaves unerased being new to it.  The same recipe on crops whose erased image is
# the pixel model's strike on the annotator's polygon (struck_pack folds --erased-by); it takes the library's drafts if its exact on the
# held-out writers' model crops is the higher of the two readers'.
RPM=${N}_prefill_reader_pack_model20261116; RDM=${N}_prefill_reader_model20261116
reader_model_job() {   # CARD NUMA
    [ -e "$RDM/complete.json" ] || job "$1" "$2" "$RDM" -- $(PACK=$RPM teach "$RDM" pair 0)
}
read_model_job() {   # CARD NUMA OUT PACK
    [ -e "$3/complete.json" ] || job "$1" "$2" "$3" -- $(PACK=$4 read_with "$3" pair "$RDM" --fold 0 --loop-limit 4)
}
pack_model_job() {   # CARD NUMA
    job "$1" "$2" "$RPM" -- "$PY" -m revrender.data.struck_pack folds --dataset "$MERGED/regions.json" --pages "$O/paper2_full_review20261010/images" \
        --holdout-share .1 --erased-by "$RUN" --erased-step "$STEP" --upstream "$UP" --output-dir "$RPM"
}
if ! done_ prefill_reader_model; then
    [ -e "$RPM/run_manifest.json" ] || { alone pack_model_job; wait; }
    alone reader_model_job; wait; test -e "$RDM/complete.json"
    alone read_model_job "$P/read_heldout_m" "$LH"; wait; test -e "$P/read_heldout_m/complete.json"
    [ -e "$P/scores_heldout_m.json" ] || "$PY" -m revrender.eval.transcription --pack "$LH" --only "$LH/held_out.json:ids" --arm "taught_on_model=$P/read_heldout_m@pair_revision" \
        --arm "taught_on_annotator=$P/read_heldout@pair_revision" --arm "human_pixels=$P/read_human@pair_revision" --reference taught_on_annotator \
        --readings "$P/readings_heldout_m.jsonl" --output "$P/scores_heldout_m.json" > "$P/scores_heldout_m.log" 2>&1
    mark prefill_reader_model; fi
if ! done_ prefill_library_m; then
    better=$("$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["arms"]; print("m" if a["taught_on_model"]["exact"] > a["taught_on_annotator"]["exact"] else "")' "$P/scores_heldout_m.json")
    if [ "$better" = m ]; then
        alone read_model_job "$P/read_library_m" "$LF"; wait; test -e "$P/read_library_m/complete.json"
        [ -e "$P/site_m/run_manifest.json" ] || "$PY" -m revrender.tools.annotate.pixel_library bundle --pack "$P/library" --readings "$P/read_library_m/predictions.jsonl" \
            --condition pair_revision --katex "$KATEX" --output-dir "$P/site_m" > "$P/bundle_m.log" 2>&1
    fi
    mark prefill_library_m; fi
touch "$P/COMPLETE"; stamp 'prefill part complete'; exit 0; fi

if [ "$PART" = ceiling ]; then cards "${CARDS:-1}"
C=$O/paper2_ceiling_reader20261116; OLDHM=${N}_revision_folds_human_masked; HM=${N}_revision_folds_human_masked20261116b
MERGED=${N}_line_dataset20261116b; LP=${N}_line_pack20261116b; R0=$O/paper2_line_reader20261116b; mkdir -p "$C"
if ! done_ ceiling_pack; then
    [ -e "$HM/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack folds --dataset "$MERGED/regions.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --hold-out test-page --folds-from "$OLDHM" --output-dir "$HM" > "$C/pack.log" 2>&1
    [ -e "$C/unchanged.json" ] || "$PY" -m revrender.data.line_set unchanged --old "$DATA/regions.json" --new "$MERGED/regions.json" --output "$C/unchanged.json" >> "$C/pack.log" 2>&1
    mark ceiling_pack; fi
done_ ceiling_checked || { stamp 'the new folds pack is built: look at the crops, then touch ceiling_checked.done and run again'; exit 0; }
ceiling_fold() {   # FOLD: the pair's reader taught anew on the new pack (the recipe of the masked part), then the fold it never saw
    local k="$1" reader="${N}_revision_reader_ceiling20261116_pair_f$1" PACK="$HM"
    [ -e "$reader/complete.json" ] || job "$CARD_" "$NUMA_" "$reader" -- $(teach "$reader" pair "$k")
    [ -e "$C/pair_f$k/complete.json" ] || job "$CARD_" "$NUMA_" "$C/pair_f$k" -- $(read_with "$C/pair_f$k" pair "$reader" --fold "$k" --loop-limit 4)
}
if ! done_ ceiling_folds; then
    for k in 0 1 2 3 4; do alone on_card ceiling_fold "$k"; wait; test -e "$C/pair_f$k/complete.json"; done
    mark ceiling_folds; fi
five_in() { local d="$1" a="$2" k s=; for k in 0 1 2 3 4; do s+="${s:+,}$d/${a}_f$k"; done; echo "$s"; }
if ! done_ ceiling_score; then
    OLD="$(five_in "$R" masked_human_pair)"; U="$(five_in "$C" pair)"
    for only in pixels all; do [ -e "$C/scores_unchanged_$only.json" ] || "$PY" -m revrender.eval.transcription --pack "$HM" --only "$C/unchanged.json:$only" \
        --arm "old=$OLD" --arm "U=$U" --reference old --output "$C/scores_unchanged_$only.json" > "$C/score_unchanged_$only.log" 2>&1; done
    [ -e "$C/scores.json" ] || "$PY" -m revrender.eval.transcription --pack "$HM" --arm "U=$U" --arm "R0=$R0/read_regions@erased_revision" --reference R0 \
        --readings "$C/readings.jsonl" --output "$C/scores.json" > "$C/score.log" 2>&1
    [ -e "$C/lines" ] || "$PY" -m revrender.eval.line_reading --pack "$LP" --regions "$U@pair_revision" --lines "$R0/read_lines" --output-dir "$C/lines" > "$C/lines.log" 2>&1
    mark ceiling_score; fi
touch "$C/COMPLETE"; stamp 'ceiling part complete'; exit 0; fi

if [ "$PART" = ceiling2 ]; then cards "${CARDS:-0 1 2 3}"
C=$O/paper2_ceiling_reader20261116; C2=$O/paper2_ceiling2_reader20261117; HM=${N}_revision_folds_human_masked20261116b
MERGED=${N}_line_dataset20261116b; V=${N}_revision_folds_human_masked20261117_views; mkdir -p "$C2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True      # three images and the question come close to a card that another user shares
TABLE=/home/zsh/hme/code_roi725_v1/outputs; QUESTIONS=$TABLE/line_edit_question_answer_authority_20260814_v1/questions_with_answers.jsonl
QGROUPS=$TABLE/line_edit_response_index_20260814_v1/answer_groups.jsonl
if ! done_ ceiling2_pack; then
    [ -e "$V/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack views --pack "$HM" --dataset "$MERGED/regions.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --hold-out test-page --questions "$QUESTIONS" --answer-groups "$QGROUPS" --output-dir "$V" > "$C2/pack.log" 2>&1
    [ -e "$C2/answers.json" ] || "$PY" -m revrender.data.struck_pack answers --pack "$V" --output "$C2/answers.json" >> "$C2/pack.log" 2>&1
    [ -e "$C2/subsets.json" ] || "$PY" - "$V" "$C2/subsets.json" <<'PY'
import json, sys
rows = [r for r in json.load(open(sys.argv[1]+'/pack.json'))['rows'] if r['readable']]
json.dump({'fraction': [r['id'] for r in rows if '/' in r['target']], 'single': [r['id'] for r in rows if len(r['target']) == 1]}, open(sys.argv[2], 'x'))
PY
    mark ceiling2_pack; fi
PROBLEMS=$("$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print("yes" if m["with_problem"] >= .95*m["regions"] else "no")' "$V/run_manifest.json")
c2_fold() {   # FOLD NAME ARM [more of train_reader]: the fold's reader NAME on the views pack, then the fold it never saw, loops stopped
    local k="$1" name="$2" arm="$3"; shift 3; local reader="${N}_revision_reader_ceiling2_${name}_f$k" PACK="$V"
    [ -e "$reader/complete.json" ] || job "$CARD_" "$NUMA_" "$reader" -- $(teach "$reader" "$arm" "$k" "$@")
    [ -e "$C2/${name}_f$k/complete.json" ] || job "$CARD_" "$NUMA_" "$C2/${name}_f$k" -- $(read_with "$C2/${name}_f$k" "$arm" "$reader" --fold "$k" --loop-limit 4)
}
c2_arm() {   # NAME ARM [more of train_reader]: five folds, one after another
    local k; for k in 0 1 2 3 4; do alone on_card c2_fold "$k" "$@"; wait; test -e "$C2/$1_f$k/complete.json"; done; mark "ceiling2_$1"
}
five() { local a="$1" k s=; for k in 0 1 2 3 4; do s+="${s:+,}$C2/${a}_f$k"; done; echo "$s"; }
U0="$C/pair_f0,$C/pair_f1,$C/pair_f2,$C/pair_f3,$C/pair_f4"
arm_of() { case "$1" in U0) echo "U0=$U0" ;; *) echo "$1=$(five "$1")" ;; esac; }
score() {   # OUT REFERENCE ARM...: eval.transcription of the arms against the reference
    local out="$1" ref="$2" a; shift 2; [ -e "$C2/$out.json" ] && return
    "$PY" -m revrender.eval.transcription --pack "$V" $(for a in "$ref" "$@"; do echo --arm "$(arm_of "$a")"; done) --reference "$ref" --output "$C2/$out.json" > "$C2/$out.log" 2>&1
}
verdict() {   # SCORES ARM RULE: teach (rule 1), context (rule 2), steps (rule 4)
    "$PY" - "$@" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))['arms'][sys.argv[2]]['against_reference']; lo, hi = a['exact']['interval']; rule = sys.argv[3]
print({'teach': 'better' if lo > 0 else 'worse' if hi < 0 else 'cannot_tell', 'context': 'stands' if lo > 0 and a['ar']['difference'] >= -.02 else 'not',
       'steps': 'more' if lo > 0 else 'same'}[rule])
PY
}
if ! done_ ceiling2_smoke; then
    smoke_job() {   # CARD NUMA: two steps of the question's reader with LaTeX answers, eight regions read, the question in its words
        local PACK="$V"
        job "$1" "$2" "$C2/smoke_reader" -- $(teach "$C2/smoke_reader" context_problem 0 --steps 2 --eval-every 1 --warmup 1 --answer latex)
        job "$1" "$2" "$C2/smoke_read" -- $(read_with "$C2/smoke_read" context_problem "$C2/smoke_reader" --fold 0 --limit 8 --loop-limit 4)
        "$PY" -c "import json, sys; m = json.load(open(sys.argv[1])); assert '【题目】' in m['conditions']['context_problem_revision']['example_prompt']; texts = {json.loads(l)['text'] for l in open(sys.argv[2])}; assert len(texts) > 2, texts" \
            "$C2/smoke_read/run_manifest.json" "$C2/smoke_read/predictions.jsonl"
    }
    alone smoke_job; wait; test -e "$C2/smoke_read/complete.json"; mark ceiling2_smoke; fi
done_ ceiling2_A || c2_arm A pair --answer latex
score scores_A U0 A
TEACH=$(verdict "$C2/scores_A.json" A teach)
if [ "$TEACH" = worse ]; then BASE=U0; ANSWER=(); else BASE=A; ANSWER=(--answer latex); fi
[ "$PROBLEMS" = yes ] && { done_ ceiling2_B1 || c2_arm B1 problem "${ANSWER[@]}"; }
done_ ceiling2_B2 || c2_arm B2 context "${ANSWER[@]}"
CONTEXTS=(B2); [ "$PROBLEMS" = yes ] && CONTEXTS=(B1 B2)
score scores_B "$BASE" "${CONTEXTS[@]}"
V1=$([ "$PROBLEMS" = yes ] && verdict "$C2/scores_B.json" B1 context || echo not_run); V2=$(verdict "$C2/scores_B.json" B2 context)
if [ "$V1" = stands ] && [ "$V2" = stands ]; then
    done_ ceiling2_B12 || c2_arm B12 context_problem "${ANSWER[@]}"
    score scores_B12 "$BASE" B1 B2 B12
    SEL=$("$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["arms"]; print("B12" if a["B12"]["exact"] >= max(a["B1"]["exact"], a["B2"]["exact"]) else max(("B1","B2"), key=lambda n: a[n]["exact"]))' "$C2/scores_B12.json")
elif [ "$V1" = stands ]; then SEL=B1
elif [ "$V2" = stands ]; then SEL=B2
else SEL=$BASE; fi
case "$SEL" in U0|A) SEL_ARM=pair ;; B1) SEL_ARM=problem ;; B2) SEL_ARM=context ;; B12) SEL_ARM=context_problem ;; esac
done_ ceiling2_C || c2_arm C "$SEL_ARM" "${ANSWER[@]}" --steps 300
score scores_C "$SEL" C
STEPS=$(verdict "$C2/scores_C.json" C steps); [ "$STEPS" = more ] && CEILING=C || CEILING=$SEL
if ! done_ ceiling2_score; then
    ARMS=(U0 A $([ "$PROBLEMS" != yes ] || echo B1) B2 $([ ! -e "$R/ceiling2_B12.done" ] || echo B12) C)
    [ -e "$C2/scores_all.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" $(for a in "${ARMS[@]}"; do echo --arm "$(arm_of "$a")"; done) --reference U0 \
        --readings "$C2/readings.jsonl" --output "$C2/scores_all.json" > "$C2/scores_all.log" 2>&1
    for only in fraction single; do [ -e "$C2/scores_$only.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" --only "$C2/subsets.json:$only" \
        $(for a in "${ARMS[@]}"; do echo --arm "$(arm_of "$a")"; done) --reference U0 --output "$C2/scores_$only.json" > "$C2/scores_$only.log" 2>&1; done
    [ -e "$C2/ceiling2_rules.json" ] || "$PY" - "$C2" "$TEACH" "$V1" "$V2" "$SEL" "$STEPS" "$CEILING" "$U0" "${ARMS[@]}" <<'PY'
import json, sys
from pathlib import Path
C2 = Path(sys.argv[1]); teach, v1, v2, sel, steps, ceiling, u0 = sys.argv[2:9]; arms = sys.argv[9:]
load = lambda n: json.loads((C2/f'{n}.json').read_text()) if (C2/f'{n}.json').exists() else None
every, fraction, single = load('scores_all')['arms'], load('scores_fraction')['arms'], load('scores_single')['arms']
pair = lambda scores, arm: {m: scores['arms'][arm]['against_reference'][m] for m in ('ar', 'exact', 'ned')} if scores and arm in scores['arms'] else None
out = {'verdicts': {'1_teach_latex (A - U0)': teach, '2_B1_question (B1 - base)': v1, '2_B2_line (B2 - base)': v2, '3_chosen_with_context': sel,
                    '4_steps_300 (C - chosen at 100)': steps, '5_new_ceiling': ceiling},
       'differences': {'A - U0': pair(load('scores_A'), 'A'), 'B1 - base': pair(load('scores_B'), 'B1'), 'B2 - base': pair(load('scores_B'), 'B2'),
                       'B12 - base': pair(load('scores_B12'), 'B12'), 'C - chosen': pair(load('scores_C'), 'C'), **{f'{a} - U0': every[a].get('against_reference') for a in arms if a != 'U0'}},
       'arms': {a: {'ar': every[a]['ar'], 'exact': every[a]['exact'], 'cer': every[a]['cer'], 'ned': every[a]['ned'], 'inserted': every[a]['inserted'],
                    'exact_by_length': every[a]['exact_by_length'], 'left_wrong': every[a]['left_wrong'], 'refusal': every[a]['refusal'],
                    'fraction_exact': fraction[a]['exact'], 'fraction_regions': fraction[a]['readable'],
                    'single_refused': single[a]['left_wrong']['refused'], 'single_regions': single[a]['readable']} for a in arms}}
(C2/'ceiling2_rules.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
folders = {a: (u0 if a == 'U0' else ','.join(str(C2/f'{a}_f{k}') for k in range(5))).split(',') for a in arms}
read = {a: {g['id']: g for f in fs for g in map(json.loads, open(Path(f)/'predictions.jsonl', encoding='utf-8'))} for a, fs in folders.items()}
with (C2/'readings_raw.jsonl').open('x', encoding='utf-8') as stream:
    for row in map(json.loads, open(C2/'readings.jsonl', encoding='utf-8')):
        stream.write(json.dumps({**row, **{f'{a}_raw': read[a][row['region']]['raw'] for a in arms if row['region'] in read[a]}}, ensure_ascii=False)+'\n')
print(json.dumps(out['verdicts'], ensure_ascii=False))
PY
    mark ceiling2_score; fi
touch "$C2/COMPLETE"
# The user's answers of 2026-11-17 (paper2_answer_review_returned20261117, 7 regions): the pack again with them (line_set merge with both
# corrections, then struck_pack folds and views, built by hand on CPU), the new ceiling's reading taught once more on it, split as table 6:
# the old readings against the new answers (every readable region; those whose answer and pixels stayed), then the new readings.
V2=${N}_revision_folds_human_masked20261117b_views; NEW=${CEILING}new
if [ -e "$V2/pack.json" ] && ! done_ ceiling2_answers; then
    MORE=(); [ "$CEILING" != C ] || MORE=(--steps 300)
    OLDV=$V; V=$V2; done_ "ceiling2_$NEW" || c2_arm "$NEW" "$SEL_ARM" "${ANSWER[@]}" "${MORE[@]}"; V=$OLDV
    [ -e "$C2/answers_subsets.json" ] || "$PY" - "$V" "$V2" "$C2/answers_subsets.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
old, new = ({r['id']: r for r in json.loads((Path(d)/'pack.json').read_text())['rows']} for d in sys.argv[1:3])
same = lambda r, s: all(hashlib.sha256((Path(sys.argv[1])/r['images'][k]).read_bytes()).digest() == hashlib.sha256((Path(sys.argv[2])/s['images'][k]).read_bytes()).digest() for k in r['images'])
kept = [k for k, r in new.items() if r['readable'] and old[k]['readable'] and r['target'] == old[k]['target'] and same(old[k], r)]
json.dump({'unchanged': kept}, open(sys.argv[3], 'x')); print(len(kept), 'readable regions kept their answer and crops')
PY
    for only in all unchanged; do [ -e "$C2/scores_answers_$only.json" ] || "$PY" -m revrender.eval.transcription --pack "$V2" \
        $([ "$only" = all ] || echo --only "$C2/answers_subsets.json:unchanged") --arm "$(arm_of "$CEILING")" --arm "$(arm_of "$NEW")" --reference "$CEILING" \
        $([ "$only" = unchanged ] || echo --readings "$C2/readings_answers.jsonl") --output "$C2/scores_answers_$only.json" > "$C2/scores_answers_$only.log" 2>&1; done
    mark ceiling2_answers; touch "$C2/ANSWERS_COMPLETE"; fi
stamp 'ceiling2 part complete'; exit 0; fi

if [ "$PART" = ceiling3 ]; then cards "${CARDS:-1 2 3}"
C2=$O/paper2_ceiling2_reader20261117; C3=$O/paper2_ceiling3_reader20261117; V=${N}_revision_folds_human_masked20261117b_views
M32=/home/zsh/hme/models/Qwen3-VL-32B-Instruct-FP8; mkdir -p "$C3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" - "$V" "$C3/subsets.json" <<'PY'
import json, os, sys
rows = json.load(open(sys.argv[1]+'/pack.json'))['rows']; can = [r for r in rows if r['readable']]
assert (len(rows), len(can)) == (610, 2*257), (len(rows), len(can))      # the regions and the readable ones of the pack
if not os.path.exists(sys.argv[2]):
    json.dump({'fraction': [r['id'] for r in can if '/' in r['target']], 'single': [r['id'] for r in can if len(r['target']) == 1]}, open(sys.argv[2], 'x'))
PY
two_cards() {   # FUNCTION ARGS...: run FUNCTION "CARD,CARD" NUMA ARGS... in the background on the two cards of ours with the most room of those
                # with room and no job of ours; card 3 only while nobody else is on it
    local pair c free
    until pair=$(for c in "${ALL_CARDS[@]}"; do free=$(nvidia-smi -i "$c" --query-gpu=memory.free --format=csv,noheader,nounits)
                     [ "$(ours_on "$c")" -eq 0 ] && [ "$free" -ge 22000 ] || continue
                     if [ "$c" = "$CARD3" ] && card_users "$c" | grep -qvx "$ME"; then continue; fi; echo "$free $c"; done | sort -rn | head -2 | awk '{print $2}' | paste -sd,) \
          && [ "$(tr ',' '\n' <<< "$pair" | grep -c .)" = 2 ]; do sleep 30; done
    ( case "$pair" in *"$CARD3"*) export ALLOW_GPU3=1 ;; esac; "$1" "$pair" "$(numa_of "${pair%%,*}")" "${@:2}" ) & PLACED_PIDS+=($! $!); PLACED_CARDS+=("${pair%%,*}" "${pair##*,}")
}
untaught() {   # OUT MODEL TP [more of recognize_lines]: the untaught model on the ceiling's images and words, vLLM, loops stopped
    local out="$1" model="$2" tp="$3"; shift 3
    [ -e "$out/complete.json" ] || job "$CARD_" "$NUMA_" "$out" -- "$PY" -m revrender.recognize_lines --pack "$V" --model "$model" --backend vllm \
        --conditions context_revision --tp "$tp" --gpu-memory .85 --loop-limit 4 "$@" --output-dir "$out"
}
c3_fold() {   # FOLD: D1, the fold's reader of the ceiling's recipe with the question in its words, then the fold it never saw
    local k="$1" reader="${N}_revision_reader_ceiling3_D1_f$1" PACK="$V"
    [ -e "$reader/complete.json" ] || job "$CARD_" "$NUMA_" "$reader" -- $(teach "$reader" context_problem "$k" --answer latex)
    [ -e "$C3/D1_f$k/complete.json" ] || job "$CARD_" "$NUMA_" "$C3/D1_f$k" -- $(read_with "$C3/D1_f$k" context_problem "$reader" --fold "$k" --loop-limit 4)
}
if ! done_ ceiling3_smoke; then     # eight regions each: the 8B and the 32B read three images with vLLM
    alone on_card untaught "$C3/smoke_N8" "$MODEL" 1 --limit 8; wait
    two_cards on_card untaught "$C3/smoke_N32" "$M32" 2 --limit 8; wait
    for d in smoke_N8 smoke_N32; do "$PY" -c "import json, sys; m = json.load(open(sys.argv[1]+'/run_manifest.json')); assert m['conditions']['context_revision']['image'] == ['raw', 'erased', 'context'] and m['conditions']['context_revision']['rows'] == 8; texts = [json.loads(l)['text'] for l in open(sys.argv[1]+'/predictions.jsonl')]; assert len(texts) == 8 and len(set(texts)) > 2, texts" "$C3/$d"; done
    mark ceiling3_smoke; fi
if ! done_ ceiling3_read; then
    [ -e "$C3/N32/complete.json" ] || { two_cards on_card untaught "$C3/N32" "$M32" 2; wait; }
    [ -e "$C3/N8/complete.json" ] || alone on_card untaught "$C3/N8" "$MODEL" 1
    for k in 0 1 2 3 4; do [ -e "$C3/D1_f$k/complete.json" ] || alone on_card c3_fold "$k"; done; wait
    for d in N32 N8 D1_f0 D1_f1 D1_f2 D1_f3 D1_f4; do test -e "$C3/$d/complete.json"; done
    mark ceiling3_read; fi
five3() { local d="$1" a="$2" k s=; for k in 0 1 2 3 4; do s+="${s:+,}$d/${a}_f$k"; done; echo "$s"; }
ARMS=(--arm "D0=$(five3 "$C2" B2new)" --arm "D1=$(five3 "$C3" D1)" --arm "N8=$C3/N8" --arm "N32=$C3/N32")
if ! done_ ceiling3_score; then
    [ -e "$C3/scores_all.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" "${ARMS[@]}" --reference D0 --readings "$C3/readings.jsonl" \
        --output "$C3/scores_all.json" > "$C3/scores_all.log" 2>&1
    [ -e "$C3/scores_untaught.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" --arm "N8=$C3/N8" --arm "N32=$C3/N32" --reference N8 \
        --output "$C3/scores_untaught.json" > "$C3/scores_untaught.log" 2>&1
    for only in fraction single; do [ -e "$C3/scores_$only.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" --only "$C3/subsets.json:$only" \
        "${ARMS[@]}" --reference D0 --output "$C3/scores_$only.json" > "$C3/scores_$only.log" 2>&1; done
    [ -e "$C3/ceiling3_rules.json" ] || "$PY" - "$C2" "$C3" <<'PY'
import json, sys
from pathlib import Path
C2, C3 = map(Path, sys.argv[1:3]); load = lambda n: json.loads((C3/f'{n}.json').read_text())['arms']
every, untaught, fraction, single = load('scores_all'), load('scores_untaught'), load('scores_fraction'), load('scores_single')
against = lambda scores, arm: {m: scores[arm]['against_reference'][m] for m in ('ar', 'exact', 'ned')}
d1, n32, n32_d0 = against(every, 'D1'), against(untaught, 'N32'), against(every, 'N32')
low = lambda d: d['exact']['interval'][0]
v1 = 'stands' if low(d1) > 0 and d1['ar']['difference'] >= -.02 else 'not'
v2 = 'stronger' if low(n32) > 0 else 'worse' if n32['exact']['interval'][1] < 0 else 'cannot_tell'
v3 = 'beats_D0' if low(n32_d0) > 0 else 'reported_only'
ceiling = ('D1 and N32 both beat D0; the plan does not order them' if v1 == 'stands' and v3 == 'beats_D0' else
           'N32 (untaught)' if v3 == 'beats_D0' else 'D1' if v1 == 'stands' else 'D0')
out = {'verdicts': {'1_D1_question (D1 - D0)': v1, '2_N32_base (N32 - N8)': v2, '3_N32_against_ceiling (N32 - D0)': v3, 'ceiling': ceiling},
       'differences': {'D1 - D0': d1, 'N32 - N8': n32, 'N32 - D0': n32_d0, 'N8 - D0': against(every, 'N8')},
       'arms': {a: {'ar': every[a]['ar'], 'exact': every[a]['exact'], 'cer': every[a]['cer'], 'ned': every[a]['ned'], 'inserted': every[a]['inserted'],
                    'exact_by_length': every[a]['exact_by_length'], 'left_wrong': every[a]['left_wrong'], 'refusal': every[a]['refusal'],
                    'fraction_exact': fraction[a]['exact'], 'fraction_regions': fraction[a]['readable'],
                    'single_exact': single[a]['exact'], 'single_refused': single[a]['left_wrong']['refused'], 'single_regions': single[a]['readable']}
                for a in ('D0', 'D1', 'N8', 'N32')}}
(C3/'ceiling3_rules.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
folders = {'D0': [C2/f'B2new_f{k}' for k in range(5)], 'D1': [C3/f'D1_f{k}' for k in range(5)], 'N8': [C3/'N8'], 'N32': [C3/'N32']}
read = {a: {g['id']: g for f in fs for g in map(json.loads, open(f/'predictions.jsonl', encoding='utf-8'))} for a, fs in folders.items()}
with (C3/'readings_raw.jsonl').open('x', encoding='utf-8') as stream:
    for row in map(json.loads, open(C3/'readings.jsonl', encoding='utf-8')):
        stream.write(json.dumps({**row, **{f'{a}_raw': read[a][row['region']]['raw'] for a in read if row['region'] in read[a]},
                                 **{f'{a}_finish': read[a][row['region']]['finish_reason'] for a in read if row['region'] in read[a]}}, ensure_ascii=False)+'\n')
print(json.dumps(out['verdicts'], ensure_ascii=False))
PY
    mark ceiling3_score; fi
touch "$C3/COMPLETE"; stamp 'ceiling3 part complete'; exit 0; fi

test "$PART" = evidence; test -e "$R/DISTILLED_COMPLETE"
HUMAN=$PACK; cards "${CARDS:-1}"
if ! done_ evidence_pack; then
    [ -e "$CF/run_manifest.json" ] || "$PY" -m revrender.data.struck_pack counterfactual --pack "$DISTILLED" --human "$HUMAN" --output-dir "$CF" > "$R/evidence_pack.log" 2>&1
    mark evidence_pack; fi
PACK=$CF
taught() {   # NAME ARM FOLD [more of train_reader]: the fold's reader NAME, taught in the words of ARM
    local name="$1" arm="$2" k="$3"; shift 3; local reader="${N}_revision_reader_${name}_f$k"
    [ -e "$reader/complete.json" ] || job "$CARD_" "$NUMA_" "$reader" -- $(teach "$reader" "$arm" "$k" "$@")
}
read_by() {   # OUT NAME ARM FOLD [more of recognize_lines]: the fold read by the reader NAME, loops stopped
    local out="$R/evidence_$1_f$4" name="$2" arm="$3" k="$4"; shift 4
    [ -e "$out/complete.json" ] || job "$CARD_" "$NUMA_" "$out" -- $(read_with "$out" "$arm" "${N}_revision_reader_${name}_f$k" --fold "$k" --loop-limit 4 "$@")
}
evidence_fold() {   # FOLD: its two readers, then every reading of the fold
    local k="$1" how
    taught counterfactual_pair pair "$k" --counterfactual .2 .15; taught distilled_erased erased "$k"
    read_by scan raw raw "$k"; read_by erased distilled_erased erased "$k"; read_by pair distilled_pair pair "$k"; read_by counterfactual counterfactual_pair pair "$k"
    for how in strike_off writing_off; do
        read_by "pair_$how" distilled_pair pair "$k" --intervention "$how"; read_by "counterfactual_$how" counterfactual_pair pair "$k" --intervention "$how"
    done
}
READ=(scan erased pair counterfactual pair_strike_off pair_writing_off counterfactual_strike_off counterfactual_writing_off)
if ! done_ evidence_folds; then
    for k in 0 1 2 3 4; do alone on_card evidence_fold "$k"; done
    wait; for k in 0 1 2 3 4; do for name in "${READ[@]}"; do test -e "$R/evidence_${name}_f$k/complete.json"; done; done
    mark evidence_folds; fi
of() { local a="$1" k s=; for k in 0 1 2 3 4; do s+="${s:+,}$R/evidence_${a}_f$k"; done; echo "$s"; }
if ! done_ evidence_score; then
    [ -e "$R/scores_evidence.json" ] || "$PY" -m revrender.eval.transcription --pack "$CF" $(for name in "${READ[@]}"; do echo --arm "$name=$(of "$name")"; done) \
        --reference pair --gone pair_writing_off --gone counterfactual_writing_off --agreement pair_strike_off=pair --agreement counterfactual_strike_off=counterfactual \
        --output "$R/scores_evidence.json" > "$R/evidence_score.log" 2>&1
    [ -e "$R/router_three/router.json" ] || "$PY" -m revrender.router --pack "$CF" --source "scan=$(of scan)" --source "erased=$(of erased)" --source "pair=$(of pair)" \
        --output-dir "$R/router_three" > "$R/router_three.log" 2>&1
    [ -e "$R/router_four/router.json" ] || "$PY" -m revrender.router --pack "$CF" --source "scan=$(of scan)" --source "erased=$(of erased)" --source "pair=$(of pair)" \
        --source "counterfactual=$(of counterfactual)" --output-dir "$R/router_four" > "$R/router_four.log" 2>&1
    [ -e "$R/scores_router.json" ] || "$PY" -m revrender.eval.transcription --pack "$CF" --arm "pair=$(of pair)" --arm "scan=$(of scan)" --arm "erased=$(of erased)" \
        --arm "counterfactual=$(of counterfactual)" --arm "routed=$R/router_three" --arm "routed_with_counterfactual=$R/router_four" --reference pair \
        --output "$R/scores_router.json" > "$R/router_score.log" 2>&1
    mark evidence_score; fi
touch "$R/EVIDENCE_COMPLETE"; stamp 'evidence part complete'
