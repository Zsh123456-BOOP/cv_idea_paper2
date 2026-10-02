#!/usr/bin/env bash
# The job queue of the 4090 box: starts pending jobs on GPU 0, 1, 2 (and GPU 3 only with ALLOW_GPU3=1), at most two of our jobs per card.
# Paper 2 uses its own queue directory (default /home/zsh/hme/work/paper2_queue_v1); the paper-1 queue is left alone.
#
#   $Q/pending/NN_name.job   one job; lines KEY=VALUE:
#       MEM=8000             MiB that must be free on the card (plus MARGIN) before it starts
#       CARDS=0,2            optional: only these card indices
#       AFTER=/path          optional: start only once this path exists
#       CMD=bash ...         the command; {GPU} and {NUMA} are replaced by the card's UUID and NUMA node
#   $Q/running, done, failed hold the job file, its pid, its exit code and its log (logs/NN_name.log).
#
# A card's load is the number of our launchers holding its lock file in /home/zsh/hme/work/revision_speed_gpu_locks
# (a holder whose parent also holds it is the same job), so jobs started outside the queue count too.  After a start
# the card is left alone for SETTLE seconds so the new job's memory shows before the next check.  Touch $Q/STOP to end.
# GPU 3 was someone else's card; ALLOW_GPU3=1 adds it only when the user says so (2026-09-29: "use the 4 cards").
# This runner deliberately does NOT: kill anything; touch GPU 3 unless ALLOW_GPU3=1; or start a job whose command names another card.
set -Euo pipefail
Q="${1:-/home/zsh/hme/work/paper2_queue_v1}"
MAX_PER_CARD=2; MARGIN=1000; SETTLE=600; TICK=60
LOCKS=/home/zsh/hme/work/revision_speed_gpu_locks
CARDS_UUID=(GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0 GPU-d0bcc564-5bd9-b943-f995-8536415088e3 GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0)
CARDS_NUMA=(0 0 1)
FORBIDDEN=314fdf43
if [ "${ALLOW_GPU3:-0}" = 1 ]; then
    CARDS_UUID+=(GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b); CARDS_NUMA+=(1); FORBIDDEN=__none__
fi
mkdir -p "$Q"/{pending,running,done,failed,logs,state}
log() { echo "$(date -u +%FT%TZ) $*"; }

load_of() {  # our launchers holding the card's lock
    local holders p n=0
    holders=" $(lsof -t "$LOCKS/$1.lock" 2>/dev/null | tr '\n' ' ') "
    for p in $holders; do
        case "$holders" in *" $(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ') "*) ;; *) n=$((n+1)) ;; esac
    done
    echo "$n"
}

field() { sed -n "s/^$1=//p" "$2" | head -1; }

reap() {
    local pidfile name rc
    for pidfile in "$Q"/running/*.pid; do
        test -e "$pidfile" || continue
        name="$(basename "$pidfile" .pid)"
        kill -0 "$(cat "$pidfile")" 2>/dev/null && continue
        rc="$(cat "$Q/running/$name.rc" 2>/dev/null || echo lost)"
        if [ "$rc" = 0 ]; then dest=done; else dest=failed; fi
        mv "$Q/running/$name".* "$Q/$dest/"
        log "finished $name rc=$rc -> $dest"
    done
}

start() {  # job file, card index
    local job="$1" i="$2" name cmd
    name="$(basename "$job" .job)"
    cmd="$(field CMD "$job")"
    cmd="${cmd//\{GPU\}/${CARDS_UUID[i]}}"; cmd="${cmd//\{NUMA\}/${CARDS_NUMA[i]}}"
    mv "$job" "$Q/running/$name.job"
    echo "card=$i uuid=${CARDS_UUID[i]} started=$(date -u +%FT%TZ)" >> "$Q/running/$name.job"
    setsid -w nohup bash -c "$cmd; echo \$? > '$Q/running/$name.rc'" > "$Q/logs/$name.log" 2>&1 < /dev/null &
    echo $! > "$Q/running/$name.pid"
    date +%s > "$Q/state/${CARDS_UUID[i]}.last"
    log "started $name on GPU$i (pid $!)"
}

log "queue runner up at $Q; cards ${!CARDS_UUID[*]}; at most $MAX_PER_CARD of ours per card"
while [ ! -e "$Q/STOP" ]; do
    reap
    for job in $(ls "$Q"/pending/*.job 2>/dev/null | sort); do
        cmd="$(field CMD "$job")"; mem="$(field MEM "$job")"; after="$(field AFTER "$job")"; only="$(field CARDS "$job")"
        if [ -z "$cmd" ] || [ -z "$mem" ] || [[ "$cmd" == *"$FORBIDDEN"* ]]; then
            mv "$job" "$Q/failed/"; log "refused $(basename "$job"): no CMD/MEM, or it names GPU 3"; continue
        fi
        [ -n "$after" ] && [ ! -e "$after" ] && continue
        best=-1; best_key=""
        for i in "${!CARDS_UUID[@]}"; do
            [ -n "$only" ] && [[ ",$only," != *",$i,"* ]] && continue
            last="$(cat "$Q/state/${CARDS_UUID[i]}.last" 2>/dev/null || echo 0)"
            [ $(( $(date +%s) - last )) -lt "$SETTLE" ] && continue
            n="$(load_of "${CARDS_UUID[i]}")"
            [ "$n" -ge "$MAX_PER_CARD" ] && continue
            free="$(nvidia-smi -i "${CARDS_UUID[i]}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)"
            [ "$free" -ge $((mem+MARGIN)) ] || continue
            key="$(printf '%d %06d' "$n" $((999999-free)))"
            if [ "$best" -lt 0 ] || [[ "$key" < "$best_key" ]]; then best=$i; best_key="$key"; fi
        done
        [ "$best" -ge 0 ] && start "$job" "$best"
    done
    sleep "$TICK"
done
log "STOP found; runner exits (running jobs continue)"
