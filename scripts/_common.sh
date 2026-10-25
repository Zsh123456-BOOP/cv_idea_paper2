# Shared by the launchers (source it; do not run it).  Server conventions of this project:
#  - code runs from a committed snapshot  /home/zsh/hme/code_paper2_<commit>  (scripts/deploy.sh), never from a work tree
#  - outputs go to /home/zsh/hme/outputs/paper2_*   (create-only: a launcher refuses an existing OUT)
#  - every job holds its card's lock SHARED (flock -s), so an exclusive holder elsewhere waits for all of us;
#    at most two of our jobs per card; card 3 is not ours unless ALLOW_GPU3=1 (the user's word)
#  - never kill another user's process; never install into a shared environment
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/zsh/hme/envs/hme/bin/python}"
NATIVE_PY="${NATIVE_PY:-/home/zsh/hme/envs/native_baselines_v1/bin/python}"
LOCKS="${LOCKS:-/home/zsh/hme/work/revision_speed_gpu_locks}"
WEIGHTS="${WEIGHTS:-/home/zsh/hme/work/native_baselines_weights_v1}"
UPSTREAM_V1="${UPSTREAM_V1:-/home/zsh/hme/work/native_baselines_sources_v1}"
UPSTREAM_V2="${UPSTREAM_V2:-/home/zsh/hme/work/native_baselines_sources_v2}"
CARD3=GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b

ALL_CARDS=(GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0 GPU-d0bcc564-5bd9-b943-f995-8536415088e3 GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0 GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b)
declare -A PLACED=()   # the jobs this shell placed with place_job: pid -> card
ME=$(id -un)
numa_of() { case "$1" in *56971197*|*314fdf43*) echo 1 ;; *) echo 0 ;; esac; }
card_users() {   # $1 = GPU uuid -> the user of every process on it, one per line
    local p; for p in $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader); do ps -o user= -p "$p" 2>/dev/null; done
}
ours_on() {   # $1 = GPU uuid -> our jobs on it: those this shell placed and still running, or this user's processes it shows, the more
    local n=0 p shown
    for p in "${!PLACED[@]}"; do kill -0 "$p" 2>/dev/null && [ "${PLACED[$p]}" = "$1" ] && n=$((n+1)); done
    shown=$(card_users "$1" | grep -cx "$ME" || true)
    echo $(( n > shown ? n : shown ))
}
open_card() {   # $1 = MiB -> a card with that much free and fewer than two of our jobs; card 3 only while nobody else is on it
    local c
    for c in "${ALL_CARDS[@]}"; do
        [ "$(nvidia-smi -i "$c" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$1" ] || continue
        [ "$(ours_on "$c")" -lt 2 ] || continue
        if [ "$c" = "$CARD3" ] && card_users "$c" | grep -qvx "$ME"; then continue; fi
        echo "$c"; return 0
    done
    return 1
}
place_job() {   # MiB FUNCTION ARGS...: run FUNCTION CARD NUMA ARGS... in the background on the first open card (waiting for one), recorded in PLACED
    local mib="$1" card; shift
    until card=$(open_card "$mib"); do sleep 30; done
    ( [ "$card" != "$CARD3" ] || export ALLOW_GPU3=1; "$1" "$card" "$(numa_of "$card")" "${@:2}" ) & PLACED[$!]=$card
    sleep 20      # until its process shows on the card
}

refuse_card3() {   # $1 = GPU uuid
    case "$1" in *314fdf43*) [ "${ALLOW_GPU3:-0}" = 1 ] || { echo 'GPU 3 is not ours (ALLOW_GPU3=1 only on the user'"'"'s word)'; exit 2; } ;; esac
}

require_clean_source() {   # a deployed snapshot carries deployment_origin.json instead of .git
    [ -f "$ROOT/deployment_origin.json" ] && return 0
    test -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" || { echo 'source dirty'; exit 2; }
}

HELD_FDS=()
hold_card() {      # $1 = GPU uuid (several joined by commas for a job that spans cards): take each card's lock shared and pin the environment
    local uuid fd
    for uuid in ${1//,/ }; do
        exec {fd}>>"$LOCKS/$uuid.lock"
        flock -s "$fd"
        HELD_FDS+=("$fd")
    done
    export CUDA_VISIBLE_DEVICES="$1" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
}

release_cards() {  # give back the locks hold_card took (the job's GPU work is done)
    local fd
    for fd in "${HELD_FDS[@]}"; do flock -u "$fd"; done
}

require_free_memory() {   # $1 = GPU uuid, $2 = MiB
    local free
    free="$(nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits)"
    test "$free" -ge "$2" || { echo "insufficient free GPU memory ($free < $2)"; exit 3; }
}

wait_for_gpu_memory() {  # $1 = UUID, $2 = MiB, $3 = require idle (0/1)
    local free waited=0 busy
    while true; do
        free="$(nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits)" || return 3
        busy=""
        if [ "${3:-0}" = 1 ]; then
            busy="$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader)" || return 3
        fi
        if [ "$free" -ge "$2" ] && [ -z "$busy" ]; then return 0; fi
        if [ "$waited" = 0 ]; then echo "waiting for GPU $1: free=$free MiB, need=$2 MiB, idle=${3:-0}"; fi
        if [ "$waited" -ge 172800 ]; then echo 'GPU wait exceeded 48 hours'; return 3; fi
        sleep 60
        waited=$((waited+60))
    done
}
