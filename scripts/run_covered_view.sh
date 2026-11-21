#!/usr/bin/env bash
# The method stage's sixth round (plan of 2026-11-21, 第六轮_专补涂满_学习曲线_拒读): on the second round's 610 regions,
#   V-修     the fourth round's real+synthetic adapters taught again on the tower with its positions mended (train_clean_view
#            vision_tower keeps inv_freq in float32; section 4.1);
#   V-涂满   V-修's recipe with covered synthetic regions (data.synth_views covered): real 1/2, synthetic 1/4, covered 1/4 (4.2);
#   L25, L50 V-修's recipe on 25 % / 50 % of the real training writers (train_clean_view --real-share; 4.3);
# each read by the shared readers, images 2 and 3 through the fold's adapter (the fourth round's V reading); the feature distances;
# the confidence head (train_clean_view head, on V-修's adapters); eval.transcription, eval.privileged (rules F and A, the learning
# curve), eval.selective (B1, rule B2).  PART=smoke (V-修 all five folds, the rest fold 0, a few steps, eight regions read; the covered
# regions of all five folds built and checked) also runs the plan's section-six checks: 1 the mended tower against the whole model's
# (same-tower), 2 V-修's step-0 validation loss of fold 0 against the fourth round's 0.5101, 3 the covered regions' acceptance (in
# the builder), 4 the learning-curve subsets (subsets), 5 the reading path with the adapter off against H, 6 the head's inputs (its
# run manifest).  The strike shares are checked region by region against the planning session's count ($B_inputs/region_stats.json).
# Cards 1 and 2 (the user's word, 2026-11-21: 「那就用两张卡，1，2」); cards 0 and 3 are not touched,
# ALLOW_GPU3 is not set.  ARMS="fix l25 l50": only those arms are taught, read and scored (the planning session, 2026-11-21: V-涂满
# waits for the user's word; rule A then is not judged).  RUN=maa (the sixth round's update, 第六轮更新稿_MAA迁移与提示前提试验,
# beside the run above, in paper2_clean_maa_view20261121): V-判别 (train_clean_view --discriminator; rule G against the run's V-修)
# and the boards' premise test (data.hint_image boards; adapters C on blank boards and T on the answer, windows only, real pairs;
# train_clean_view boards-distance; rule H), the spread of V-修 and V-判别 (MMD squared, energy distance); its smoke checks: V-修's
# recipe on the new code (fold 0's step-0 validation loss the same as the run's, its first four losses beside), the discriminator
# never saved, the boards read back (in the builder).  RUN=disc (plan 判别器调强, in paper2_clean_disc_strong20261121): V-判别-10 and
# V-判别-100, V-判别 with the discriminator at 1e-5 and 1e-4 (train_clean_view --discriminator-lr), read, measured and scored against
# V-修 on 97.5 % intervals; its smoke checks: the configuration differs from the sixth round's V-判别 in the rate and the output alone,
# the new code at 1e-6 beside the sixth round's first four steps, ten steps of each rate, no discriminator in a saved adapter.
# Unattended; a step whose marker is in $R is skipped.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; B=${N}_covered_view20261121; PART=${PART:-main}
case "$PART" in main) TAG=; P_STEPS=2000; P_EVAL=50; LIMIT=(); READ=(0 1 2 3 4); FIX_FOLDS=(0 1 2 3 4) ;;
    smoke) TAG=_smoke; P_STEPS=10; P_EVAL=10; LIMIT=(--limit 8); READ=(0); FIX_FOLDS=(0 1 2 3 4) ;;
    *) echo "PART is main or smoke"; exit 2 ;; esac
FIX=$B$TAG; R=$FIX; MAA=${N}_maa_view20261121$TAG
case "${RUN:-}" in maa) R=$MAA ;; disc) R=${N}_disc_strong20261121$TAG ;; esac; mkdir -p "$R/checks"
P2=${N}_nafnet_folds20261119; V=${N}_revision_folds_human_masked20261117b_views; EV=${P2}_erased_views; RAWV=${P2}_raw_views
P4=${N}_privileged_view20261120; SYN=${P4}_synth; COV=${B}_covered; SH=${P2}_reader_shared_f; SHARES=$FIX/checks/strike_shares.json
MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
test -e "$P2/COMPLETE" && test -e "$P4/COMPLETE" && test -e "${B}_inputs/region_stats.json" || { echo 'the second or fourth round, or the planning session'"'"'s strike shares, missing'; exit 2; }
CARDS=("${ALL_CARDS[1]}" "${ALL_CARDS[2]}")
place() { place_on_cards "$@"; }   # MiB WHOLE FUNCTION ARGS...: on card 1 or 2, whichever opens first (_common.sh)
job() {   # OUT -- COMMAND...: one GPU job on $CARD
    local out="$1"; shift 2
    bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1
}
plugin() {   # OUT FOLD [MORE...]: an adapter on the mended tower, the fourth round's real+synthetic recipe
    local out="$1" fold="$2"; shift 2
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.train_clean_view train --model "$MODEL" --pack "$V" --scan-pack "$RAWV" --fold "$fold" \
        --synthetic "$SYN" --steps "$P_STEPS" --eval-every "$P_EVAL" "$@" --output-dir "$out"
}
read_with() {   # OUT PACK ADAPTER FOLD [MORE...]: the fold read under context_revision, loops stopped
    local out="$1" pack="$2" adapter="$3" fold="$4"; shift 4
    [ -e "$out/complete.json" ] || job "$out" -- "$PY" -m revrender.recognize_lines --pack "$pack" --model "$MODEL" --backend hf --adapter "$adapter" \
        --batch-size 8 --conditions context_revision --fold "$fold" --loop-limit 4 "$@" --output-dir "$out"
}
distance() {   # OUT STUDENT FOLD
    [ -e "$1/distances.jsonl" ] || job "$1" -- "$PY" -m revrender.train_clean_view distance --model "$MODEL" --student "$2" --pack "$V" --raw-pack "$RAWV" \
        --erased-pack "$EV" --fold "$3" --output-dir "$1"
}
compare_reads() {   # MINE THEIRS OUT LABEL: the share of readings that differ (at most 2%)
    "$PY" - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
from pathlib import Path
read = lambda d: {g['id']: g['text'] for g in map(json.loads, (Path(d)/'predictions.jsonl').read_text().splitlines())}
mine, theirs = read(sys.argv[1]), read(sys.argv[2]); ids = sorted(mine)
differ = [k for k in ids if mine[k] != theirs[k]]; share = len(differ)/len(ids)
Path(sys.argv[3]).write_text(json.dumps({'regions': len(ids), 'differ': len(differ), 'share': share, 'ids': differ, 'passed': share <= .02}, indent=2))
print(json.dumps({'check': sys.argv[4], 'regions': len(ids), 'differ': len(differ), 'share': share}))
sys.exit(0 if share <= .02 else f'{sys.argv[4]} failed: more than 2% of the readings differ')
PY
}
dirs() { local k s=; for k in "${READ[@]}"; do s+="${s:+,}$1_f$k"; done; echo "$s"; }   # PREFIX -> the folds' folders
read -ra ARMS <<< "${ARMS:-fix covered l25 l50}"; has() { [[ " ${ARMS[*]} " == *" $1 "* ]]; }
[ "${ARMS[0]}" = fix ] || { echo 'ARMS starts with fix (the head and the rules need V-修)'; exit 2; }
declare -A NAME=([fix]=V-修 [covered]=V-涂满 [l25]=L25 [l50]=L50)
declare -A MORE=([fix]="" [covered]="--covered $COV" [l25]="--real-share 0.25" [l50]="--real-share 0.5")
view() { echo "${B}${TAG}_view_$1_f$2"; }

if [ "${RUN:-}" = maa ]; then
    BOARDS=${N}_maa_view20261121_boards; mine() { echo "${R}_view_$1_f$2"; }
    window_plugin() {   # OUT FOLD KIND: an adapter on the windows with a board of KIND beside them, real pairs only (V-真's recipe)
        [ -e "$1/complete.json" ] || job "$1" -- "$PY" -m revrender.train_clean_view train --model "$MODEL" --pack "$BOARDS/$3/teacher" --scan-pack "$BOARDS/$3/scan" \
            --fold "$2" --window-only --steps "$P_STEPS" --eval-every "$P_EVAL" --output-dir "$1"
    }
    if ! done_ data; then
        [ -e "$BOARDS/summary.json" ] || "$PY" -m revrender.data.hint_image boards --pack "$V" --model "$MODEL" --output-dir "$BOARDS" > "$BOARDS.log" 2>&1
        "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if all(v["passed"] for v in s.values()) else "boards not as made")' "$BOARDS/summary.json"
        mark data; fi
    if ! done_ plugins; then
        if [ "$PART" = smoke ]; then   # V-修's recipe on the new code: fold 0's step-0 validation loss as the run's, its first four losses beside
            place 12000 0 plugin "$R/check_replay" 0 --steps 2000 --stop-after 4; wait     # the run's schedule
            "$PY" -c 'import json,sys; read=lambda p:[json.loads(l) for l in open(p)][:5]; a,b=read(sys.argv[1]),read(sys.argv[2]); r={"step0":[a[0]["val_loss"],b[0]["val_loss"]],"losses_new":[x["loss"] for x in a[1:]],"losses_run":[x["loss"] for x in b[1:]],"step0_same":a[0]["val_loss"]==b[0]["val_loss"]}; json.dump(r,open(sys.argv[3],"w"),indent=2); print(json.dumps(r)); sys.exit(0 if r["step0_same"] else "the step-0 validation loss moved")' \
                "$R/check_replay/train_log.jsonl" "${B}_view_fix_f0/train_log.jsonl" "$R/checks/check_replay.json"
        fi
        for k in "${READ[@]}"; do place 12000 0 plugin "$(mine disc "$k")" "$k" --discriminator; done
        for k in "${READ[@]}"; do place 12000 0 window_plugin "$(mine hint_c "$k")" "$k" blank; place 12000 0 window_plugin "$(mine hint_t "$k")" "$k" answer; done; wait
        for arm in disc hint_c hint_t; do for k in "${READ[@]}"; do test -e "$(mine $arm "$k")/student/adapter_config.json"; done; done
        "$PY" - "$(mine disc "${READ[0]}")/student" "$(view fix "${READ[0]}")/student" "$R/checks/check_discriminator_not_saved.json" <<'PY'
import json, sys
from safetensors import safe_open
keys = lambda d: sorted(safe_open(d+'/adapter_model.safetensors', 'pt').keys())
mine, fix = keys(sys.argv[1]), keys(sys.argv[2])
r = {'tensors': len(mine), 'same_keys_as_V-修': mine == fix, 'discriminator_like': [k for k in mine if 'lora_' not in k]}
json.dump(r, open(sys.argv[3], 'w'), indent=2); print(json.dumps(r)); sys.exit(0 if r['same_keys_as_V-修'] and not r['discriminator_like'] else 'the saved adapter holds more than the adapter')
PY
        mark plugins; fi
    if ! done_ reads; then
        for k in "${READ[@]}"; do place 22000 1 read_with "$R/read_disc_f$k" "$RAWV" "${SH}$k/adapter" "$k" "${LIMIT[@]}" --view-student "$(mine disc "$k")/student" --student-images 2 3; done; wait
        for k in "${READ[@]}"; do test -e "$R/read_disc_f$k/complete.json"; done
        mark reads; fi
    if ! done_ measures; then
        for k in "${READ[@]}"; do
            place 6000 0 distance "$R/distance_disc_f$k" "$(mine disc "$k")/student" "$k"
            [ -e "$R/boards_c_f$k/distances.jsonl" ] || place 6000 0 job "$R/boards_c_f$k" -- "$PY" -m revrender.train_clean_view boards-distance --model "$MODEL" --student "$(mine hint_c "$k")/student" --boards "$BOARDS" \
                --kinds blank --fold "$k" --output-dir "$R/boards_c_f$k"
            [ -e "$R/boards_t_f$k/distances.jsonl" ] || place 6000 0 job "$R/boards_t_f$k" -- "$PY" -m revrender.train_clean_view boards-distance --model "$MODEL" --student "$(mine hint_t "$k")/student" --boards "$BOARDS" \
                --kinds answer wrong blank --fold "$k" --output-dir "$R/boards_t_f$k"
        done
        [ -e "$R/spread.json" ] || place 12000 1 job "$R/spread.json" -- "$PY" -m revrender.train_clean_view spread --model "$MODEL" --pack "$V" \
            --student "V-修=${B}${TAG}_view_fix_f{fold}/student" --student "V-判别=${R}_view_disc_f{fold}/student" --folds "${READ[@]}" --output "$R/spread.json"; wait
        for k in "${READ[@]}"; do test -e "$R/distance_disc_f$k/distances.jsonl"; test -e "$R/boards_c_f$k/distances.jsonl"; test -e "$R/boards_t_f$k/distances.jsonl"; done; test -e "$R/spread.json"
        mark measures; fi
    if ! done_ score; then
        ARMS_=(--arm "V-判别=$(dirs "$R/read_disc")" --arm "V-修=$(dirs "$FIX/read_fix")" --arm "V-真+合=$(dirs "$P4/read_V-synth")" --arm "X=$(dirs "$P4/read_X")"
               --arm "E-NAFNet=$(dirs "$P2/read_E-NAFNet")" --arm "R0=$(dirs "$P2/read_R0")" --arm "H=$(dirs "$P2/read_H")")
        REFS=(V-修 V-真+合 H R0 X E-NAFNet)
        for ref in "${REFS[@]}"; do [ -e "$R/scores_vs_$ref.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference "$ref" \
            $([ "$ref" != V-修 ] || echo --readings "$R/readings.jsonl") --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
        [ -e "$R/analysis/analysis.json" ] || "$PY" -m revrender.eval.privileged analyse --pack "$V" "${ARMS_[@]}" --shares "$SHARES" \
            $(for ref in "${REFS[@]}"; do echo --scores "$ref=$R/scores_vs_$ref.json"; done) --distances "V-判别=$(dirs "$R/distance_disc")" --output-dir "$R/analysis" > "$R/analysis.log" 2>&1
        [ -e "$R/hint/hint.json" ] || "$PY" -m revrender.eval.privileged hint --control "$(dirs "$R/boards_c")" --hinted "$(dirs "$R/boards_t")" --shares "$SHARES" \
            --output-dir "$R/hint" > "$R/hint.log" 2>&1
        mark score; fi
    touch "$R/COMPLETE"; stamp "$PART maa complete"; exit 0
fi

if [ "${RUN:-}" = disc ]; then
    mine() { echo "${R}_view_$1_f$2"; }; declare -A RATE=([disc10]=1e-5 [disc100]=1e-4)
    if ! done_ plugins; then
        if [ "$PART" = smoke ]; then   # the new code at the sixth round's 1e-6, on the run's schedule: its first four steps beside the record
            [ -e "$R/check_replay/complete.json" ] || { place 12000 0 plugin "$R/check_replay" 0 --discriminator --discriminator-lr 1e-6 --steps 2000 --stop-after 4; wait; }
            "$PY" -c 'import json,sys; read=lambda p:[json.loads(l) for l in open(p)][:5]; a,b=read(sys.argv[1]),read(sys.argv[2]); keys=("loss","align","adversarial","discriminator"); r={"step0":[a[0]["val_loss"],b[0]["val_loss"]],"new":[{k:x[k] for k in keys} for x in a[1:]],"sixth_round":[{k:x[k] for k in keys} for x in b[1:]],"step1_same":a[1]["loss"]==b[1]["loss"]}; json.dump(r,open(sys.argv[3],"w"),indent=2); print(json.dumps(r))' \
                "$R/check_replay/train_log.jsonl" "${N}_maa_view20261121_view_disc_f0/train_log.jsonl" "$R/checks/check_replay.json"
        fi
        for arm in disc10 disc100; do for k in "${READ[@]}"; do place 12000 0 plugin "$(mine $arm "$k")" "$k" --discriminator --discriminator-lr "${RATE[$arm]}"; done; done; wait
        for arm in disc10 disc100; do for k in "${READ[@]}"; do test -e "$(mine $arm "$k")/student/adapter_config.json"; done; done
        "$PY" - "$(mine disc10 "${READ[0]}")" "$(mine disc100 "${READ[0]}")" "${N}_maa_view20261121${TAG}_view_disc_f${READ[0]}" "$(view fix "${READ[0]}")/student" "$R/checks/check_config_and_saved.json" <<'PY'
import json, sys
from safetensors import safe_open
runs, old, fix, out = sys.argv[1:3], sys.argv[3], sys.argv[4], sys.argv[5]
config = lambda d: json.load(open(d+'/run_manifest.json'))['config']
keys = lambda d: sorted(safe_open(d+'/adapter_model.safetensors', 'pt').keys())
r = {}
for run in runs:
    a, b = config(run), config(old)
    r[run] = {'config_differs_in': sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)), 'lr': a.get('discriminator_lr'),
              'saved_keys_as_V-修': keys(run+'/student') == keys(fix), 'non_lora_tensors': [k for k in keys(run+'/student') if 'lora_' not in k]}
r['passed'] = all(set(v['config_differs_in']) <= {'discriminator_lr', 'output_dir'} and v['saved_keys_as_V-修'] and not v['non_lora_tensors'] for v in r.values())
json.dump(r, open(out, 'w'), indent=2); print(json.dumps(r)); sys.exit(0 if r['passed'] else 'the configuration or the saved adapter is not as planned')
PY
        mark plugins; fi
    if ! done_ reads; then
        for arm in disc10 disc100; do for k in "${READ[@]}"; do
            place 22000 1 read_with "$R/read_${arm}_f$k" "$RAWV" "${SH}$k/adapter" "$k" "${LIMIT[@]}" --view-student "$(mine $arm "$k")/student" --student-images 2 3
        done; done; wait
        for arm in disc10 disc100; do for k in "${READ[@]}"; do test -e "$R/read_${arm}_f$k/complete.json"; done; done
        mark reads; fi
    if ! done_ measures; then
        for arm in disc10 disc100; do for k in "${READ[@]}"; do place 6000 0 distance "$R/distance_${arm}_f$k" "$(mine $arm "$k")/student" "$k"; done; done
        [ -e "$R/spread.json" ] || place 12000 1 job "$R/spread.json" -- "$PY" -m revrender.train_clean_view spread --model "$MODEL" --pack "$V" \
            --student "V-修=${B}${TAG}_view_fix_f{fold}/student" --student "V-判别=${N}_maa_view20261121${TAG}_view_disc_f{fold}/student" \
            --student "V-判别-10=${R}_view_disc10_f{fold}/student" --student "V-判别-100=${R}_view_disc100_f{fold}/student" --folds "${READ[@]}" --output "$R/spread.json"; wait
        for arm in disc10 disc100; do for k in "${READ[@]}"; do test -e "$R/distance_${arm}_f$k/distances.jsonl"; done; done; test -e "$R/spread.json"
        mark measures; fi
    if ! done_ score; then
        ARMS_=(--arm "V-判别-10=$(dirs "$R/read_disc10")" --arm "V-判别-100=$(dirs "$R/read_disc100")" --arm "V-修=$(dirs "$FIX/read_fix")" --arm "V-判别=$(dirs "$MAA/read_disc")"
               --arm "V-真+合=$(dirs "$P4/read_V-synth")" --arm "X=$(dirs "$P4/read_X")" --arm "E-NAFNet=$(dirs "$P2/read_E-NAFNet")" --arm "R0=$(dirs "$P2/read_R0")" --arm "H=$(dirs "$P2/read_H")")
        REFS=(V-修 V-判别 V-真+合 H R0 X E-NAFNet)
        for ref in "${REFS[@]}"; do [ -e "$R/scores_vs_$ref.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference "$ref" \
            $([ "$ref" != V-修 ] || echo --readings "$R/readings.jsonl") --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
        [ -e "$R/analysis/analysis.json" ] || "$PY" -m revrender.eval.privileged analyse --pack "$V" "${ARMS_[@]}" --shares "$SHARES" \
            $(for ref in "${REFS[@]}"; do echo --scores "$ref=$R/scores_vs_$ref.json"; done) \
            --distances "V-判别-10=$(dirs "$R/distance_disc10")" --distances "V-判别-100=$(dirs "$R/distance_disc100")" --output-dir "$R/analysis" > "$R/analysis.log" 2>&1
        mark score; fi
    touch "$R/COMPLETE"; stamp "$PART disc complete"; exit 0
fi

# CPU: the strike shares (checked against the planning session's), the covered regions (check 3), the subsets (check 4).
if ! done_ data; then
    [ -e "$SHARES" ] || "$PY" -m revrender.eval.privileged shares --pack "$V" --compare "${B}_inputs/region_stats.json" --output "$SHARES" > "$SHARES.log" 2>&1
    ! has covered || [ -e "$COV/summary.json" ] || "$PY" -m revrender.data.synth_views covered --pack "$V" --synth "$O/paper2_clean_synth_regions_ink" \
        --same-student "$O/paper2_splits/same_student_pairs.json" --dataset "$O/paper2_clean_line_dataset20261117/regions.json" \
        --test-halves "$ROOT/configs/test_halves.json" --pages "$O/paper2_pages_clean_unlabeled3976" --take 1000 --output-dir "$COV" > "$COV.log" 2>&1
    ! has covered || "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); bad=[k for k,v in s["folds"].items() if not v["accepted"]]; sys.exit("covered regions not accepted in folds "+",".join(bad) if bad else 0)' "$COV/summary.json"
    [ -e "$R/checks/check4_subsets.json" ] || "$PY" -m revrender.train_clean_view subsets --pack "$V" --scan-pack "$RAWV" --shares 0.25 0.5 --output "$R/checks/check4_subsets.json"
    # the bar as revised before any reading (plan, section six, second revision): the five folds together within two points of the
    # share; each fold's own count reported (the first bar, each fold within 5 %, folds 1 and 2 missed)
    "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); t=sum(v["training_regions"] for v in s.values()); r={x:{"regions":sum(v[x]["regions"] for v in s.values()),"training":t,"share":sum(v[x]["regions"] for v in s.values())/t,"each_fold":{k:{"regions":v[x]["regions"],"writers":len(v[x]["writers"]),"off_target":v[x]["off_target"]} for k,v in s.items()}} for x in ("0.25","0.5")}; r["nested"]=all(v["nested"] for v in s.values()); r["first_bar_each_fold_5_percent"]=all(abs(v[x]["off_target"])<=.05 for v in s.values() for x in ("0.25","0.5")); r["passed"]=r["nested"] and all(abs(r[x]["share"]-float(x))<=.02 for x in ("0.25","0.5")); json.dump(r,open(sys.argv[2],"w"),indent=2); print(json.dumps({x:r[x]["share"] for x in ("0.25","0.5")}|{"passed":r["passed"]})); sys.exit(0 if r["passed"] else "check 4 failed")' \
        "$R/checks/check4_subsets.json" "$R/checks/check4_pooled.json"
    mark data; fi

# The checks that need a card (smoke only: the main run starts after the planning session has seen them).
if [ "$PART" = smoke ] && ! done_ checks; then
    [ -e "$R/checks/check1_same_tower.json" ] || place 22000 1 job "$R/checks/check1_same_tower.json" -- "$PY" -m revrender.train_clean_view same-tower \
        --model "$MODEL" --pack "$V" --output "$R/checks/check1_same_tower.json"; wait
    "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else "check 1 failed")' "$R/checks/check1_same_tower.json"
    mark checks; fi

# The adapters (V-修 first; the smoke run teaches V-修 on every fold, the head needs them, the rest on fold 0).
if ! done_ plugins; then
    for k in "${FIX_FOLDS[@]}"; do place 12000 0 plugin "$(view fix "$k")" "$k"; done
    for arm in "${ARMS[@]:1}"; do for k in "${READ[@]}"; do place 12000 0 plugin "$(view $arm "$k")" "$k" ${MORE[$arm]}; done; done; wait
    for k in "${FIX_FOLDS[@]}"; do test -e "$(view fix "$k")/student/adapter_config.json"; done
    for arm in "${ARMS[@]:1}"; do for k in "${READ[@]}"; do test -e "$(view $arm "$k")/student/adapter_config.json"; done; done
    if [ "$PART" = smoke ]; then   # 2: the mended tower's step-0 validation loss of fold 0 against the fourth round's
        "$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["initial_val_loss"]; b=json.load(open(sys.argv[2]))["initial_val_loss"]; r={"mended":a,"fourth_round":b,"difference":a-b,"passed":abs(a-b)<=.02}; json.dump(r,open(sys.argv[3],"w"),indent=2); print(json.dumps(r)); sys.exit(0 if r["passed"] else "check 2 failed")' \
            "$(view fix 0)/complete.json" "${P4}_view_real_f0/complete.json" "$R/checks/check2_step0.json"
        # 5: the reading path with the adapter off, the annotator's pack, the whole of fold 0: as H read
        [ -e "$R/checks/check5_wiring.json" ] || { place 22000 1 read_with "$R/check5_wiring" "$V" "${SH}0/adapter" 0 --view-student "$(view fix 0)/student" --student-images 2 3 --student-off; wait
            compare_reads "$R/check5_wiring" "$P2/read_H_f0" "$R/checks/check5_wiring.json" 'check 5'; }
    fi
    mark plugins; fi

# The readings: the shared readers, images 2 and 3 through the fold's adapter.
if ! done_ reads; then
    for arm in "${ARMS[@]}"; do for k in "${READ[@]}"; do
        place 22000 1 read_with "$R/read_${arm}_f$k" "$RAWV" "${SH}$k/adapter" "$k" "${LIMIT[@]}" --view-student "$(view $arm "$k")/student" --student-images 2 3
    done; done; wait
    for arm in "${ARMS[@]}"; do for k in "${READ[@]}"; do test -e "$R/read_${arm}_f$k/complete.json"; done; done
    mark reads; fi

# The feature distances and the confidence head (check 6: its run manifest names its inputs).
if ! done_ measures; then
    for arm in "${ARMS[@]}"; do for k in "${READ[@]}"; do place 6000 0 distance "$R/distance_${arm}_f$k" "$(view $arm "$k")/student" "$k"; done; done
    [ -e "$R/head/summary.json" ] || place 12000 1 job "$R/head" -- "$PY" -m revrender.train_clean_view head --model "$MODEL" --pack "$V" \
        --students "${B}${TAG}_view_fix_f{fold}/student" --output-dir "$R/head"; wait
    for arm in "${ARMS[@]}"; do for k in "${READ[@]}"; do test -e "$R/distance_${arm}_f$k/distances.jsonl"; done; done; test -e "$R/head/summary.json"
    "$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); r={k:m[k] for k in ("input_features","target","own_fold_adapter","trained_on")}; json.dump(r,open(sys.argv[2],"w"),indent=2); print(json.dumps(r))' \
        "$R/head/run_manifest.json" "$R/checks/check6_head_inputs.json"
    mark measures; fi

# The scores, the rules, the refusals.
ARMS_=(); for arm in "${ARMS[@]}"; do ARMS_+=(--arm "${NAME[$arm]}=$(dirs "$R/read_$arm")"); done
ARMS_+=(--arm "V-真+合=$(dirs "$P4/read_V-synth")" --arm "X=$(dirs "$P4/read_X")" --arm "E-NAFNet=$(dirs "$P2/read_E-NAFNet")" --arm "R0=$(dirs "$P2/read_R0")"
       --arm "H=$(dirs "$P2/read_H")")
REFS=(V-修 V-真+合 H R0 X E-NAFNet); ! has l25 || REFS+=(L25); ! has l50 || REFS+=(L50)
if ! done_ score; then
    for ref in "${REFS[@]}"; do [ -e "$R/scores_vs_$ref.json" ] || "$PY" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference "$ref" \
        $([ "$ref" != V-修 ] || echo --readings "$R/readings.jsonl") --output "$R/scores_vs_$ref.json" > "$R/scores_vs_$ref.log" 2>&1; done
    [ -e "$R/analysis/analysis.json" ] || "$PY" -m revrender.eval.privileged analyse --pack "$V" "${ARMS_[@]}" --shares "$SHARES" \
        $(for ref in "${REFS[@]}"; do echo --scores "$ref=$R/scores_vs_$ref.json"; done) \
        $(for arm in "${ARMS[@]}"; do echo --distances "${NAME[$arm]}=$(dirs "$R/distance_$arm")"; done) --output-dir "$R/analysis" > "$R/analysis.log" 2>&1
    # the refusals over every region: in the smoke run the fourth round's V-真+合 stands in for the arms read on eight regions only
    if [ "$PART" = main ]; then SEL=(--arm "V-修=$(dirs "$R/read_fix")"); ! has covered || SEL+=(--arm "V-涂满=$(dirs "$R/read_covered")"); else
        SEL=(--arm "V-真+合=${P4}/read_V-synth_f0,${P4}/read_V-synth_f1,${P4}/read_V-synth_f2,${P4}/read_V-synth_f3,${P4}/read_V-synth_f4"); fi
    [ -e "$R/selective/selective.json" ] || "$PY" -m revrender.eval.selective --pack "$V" "${SEL[@]}" \
        --other "R0=${P2}/read_R0_f0,${P2}/read_R0_f1,${P2}/read_R0_f2,${P2}/read_R0_f3,${P2}/read_R0_f4" \
        --other "X=${P4}/read_X_f0,${P4}/read_X_f1,${P4}/read_X_f2,${P4}/read_X_f3,${P4}/read_X_f4" --head "$R/head" --output-dir "$R/selective" > "$R/selective.log" 2>&1
    mark score; fi
touch "$R/COMPLETE"; stamp "$PART complete"
