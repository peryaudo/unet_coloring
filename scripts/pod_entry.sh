#!/usr/bin/env bash
# pod_entry.sh - runs inside a RunPod Pod, from the repo root checked out at $GIT_SHA.
# Bundled with the runpod-train skill; copy to <repo>/scripts/pod_entry.sh. Version: 2026.11.26-1
#
# Contract (env set by runpod_api.py launch):
#   RUN_ID PROJECT GIT_SHA MAX_HOURS TRAIN_ENTRY      required
#   R2_ENDPOINT_URL R2_BUCKET R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY   required (keys via RunPod Secrets)
#   DATASET        datasets/<name>/LATEST | datasets/<name>/vNNN | hf:<repo>@<rev> | s3-public://... | none
#   CONFIG_PATH    repo-relative config uploaded as runs/<run_id>/config.yaml
#   RESUME_FROM    optional R2 key (project-relative or full) of a checkpoint to resume from
#   REQUIRE_GPU    default 1: fail before training if no CUDA GPU is visible (set 0 for CPU-only jobs)
#   RUNPOD_POD_ID, RUNPOD_API_KEY   injected by RunPod (the key is scoped to this Pod)
# Exported to the training command:
#   DATA_DIR       local directory holding the dataset
#   RUN_DIR        local dir for logs/ and metrics.jsonl (uploaded periodically and at exit);
#                  default <repo>/run/<run_id>
#   RESUME_CKPT    local path of the resume checkpoint, if RESUME_FROM was set
#
# Guarantees: the Pod is terminated on success, failure, crash of this script, and when
# MAX_HOURS is reached (watchdog), and status.json ends in done | failed | timeout.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
export RUN_DIR="${RUN_DIR:-$PWD/run/${RUN_ID:-unknown}}" DATA_DIR="${DATA_DIR:-$PWD/data}"
mkdir -p "$RUN_DIR/logs" "$DATA_DIR"
LOG="$RUN_DIR/logs/train.log"
STATE_FILE="$RUN_DIR/.final_state"
rm -f "$STATE_FILE"   # never inherit a final state from an earlier run (e.g. on a network volume)
R2=(python3 -m tools.r2)
TRAIN_PID="" WATCHDOG_PID="" UPLOADER_PID=""

log() { echo "[pod_entry $(date -u +%FT%TZ)] $*" | tee -a "$LOG" >&2; }

terminate_pod() {
  if [ -n "${POD_ENTRY_NO_TERMINATE:-}" ]; then log "POD_ENTRY_NO_TERMINATE set (local test): not terminating"; return 0; fi
  if [ -z "${RUNPOD_POD_ID:-}" ]; then log "RUNPOD_POD_ID unset: cannot terminate"; return 1; fi
  log "terminating pod $RUNPOD_POD_ID"
  for _ in 1 2 3; do
    if command -v runpodctl >/dev/null 2>&1; then
      runpodctl pod delete "$RUNPOD_POD_ID" >/dev/null 2>&1 && return 0
      runpodctl remove pod "$RUNPOD_POD_ID" >/dev/null 2>&1 && return 0   # older runpodctl
    fi
    if [ -n "${RUNPOD_API_KEY:-}" ] && command -v curl >/dev/null 2>&1; then
      curl -fsS -m 30 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" >/dev/null 2>&1 && return 0
      curl -fsS -m 30 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null 2>&1 && return 0
    fi
    sleep 10
  done
  log "terminate failed; trying stop (stopped pods still bill for disk)"
  runpodctl pod stop "$RUNPOD_POD_ID" >/dev/null 2>&1 || runpodctl stop pod "$RUNPOD_POD_ID" >/dev/null 2>&1 || true
  return 1
}

upload_outputs() {
  [ -s "$LOG" ] && "${R2[@]}" put-log "$LOG" >/dev/null 2>&1
  [ -s "$RUN_DIR/metrics.jsonl" ] && "${R2[@]}" put-metrics "$RUN_DIR/metrics.jsonl" >/dev/null 2>&1
  return 0
}

finish() {
  local code=$?
  trap - EXIT INT TERM
  [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
  [ -n "$UPLOADER_PID" ] && kill "$UPLOADER_PID" 2>/dev/null
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    log "stopping training (grace ${STOP_GRACE_SEC:-60}s)"
    kill -TERM -- "-$TRAIN_PID" 2>/dev/null || kill -TERM "$TRAIN_PID" 2>/dev/null
    for _ in $(seq "${STOP_GRACE_SEC:-60}"); do kill -0 "$TRAIN_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$TRAIN_PID" 2>/dev/null
  fi
  local state; state="$(cat "$STATE_FILE" 2>/dev/null || echo failed)"
  [ "$state" = timeout ] && FINAL_MESSAGE="MAX_HOURS=$MAX_HOURS reached"
  log "finishing: state=$state exit=$code"
  upload_outputs
  "${R2[@]}" write-status --state "$state" --message "${FINAL_MESSAGE:-exit $code}" >/dev/null 2>&1 \
    || log "could not write status.json"
  terminate_pod
  exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM INT

for v in RUN_ID PROJECT GIT_SHA MAX_HOURS TRAIN_ENTRY R2_ENDPOINT_URL R2_BUCKET R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; do
  if [ -z "${!v:-}" ]; then FINAL_MESSAGE="missing env $v"; log "$FINAL_MESSAGE"; exit 2; fi
done
case "$R2_ACCESS_KEY_ID$R2_SECRET_ACCESS_KEY" in
  *"{{"*) FINAL_MESSAGE="RunPod secret placeholder was not substituted (secret missing?)"; log "$FINAL_MESSAGE"; exit 2 ;;
esac

# Watchdog first, so nothing below (pip, downloads, training) can outlive MAX_HOURS.
WATCHDOG_SEC="$(python3 -c "import sys; print(int(float(sys.argv[1])*3600))" "$MAX_HOURS")" || exit 2
(
  sleep "$WATCHDOG_SEC"
  echo timeout > "$STATE_FILE"
  log "watchdog: MAX_HOURS=$MAX_HOURS reached"
  kill -TERM $$ 2>/dev/null
  sleep $(( ${STOP_GRACE_SEC:-60} + 120 ))   # main shell did not finish: hard stop
  "${R2[@]}" write-status --state timeout --message "watchdog hard stop" >/dev/null 2>&1
  terminate_pod
) &
WATCHDOG_PID=$!

log "run $RUN_ID project $PROJECT sha $GIT_SHA pod ${RUNPOD_POD_ID:-?} gpu ${RUN_GPU:-?} max_hours $MAX_HOURS"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

log "installing requirements"
python3 -m pip install -q --disable-pip-version-check boto3 >>"$LOG" 2>&1 || { FINAL_MESSAGE="pip install boto3 failed"; exit 1; }
if [ -f requirements.txt ]; then
  python3 -m pip install -q --disable-pip-version-check -r requirements.txt >>"$LOG" 2>&1 \
    || { FINAL_MESSAGE="pip install -r requirements.txt failed"; exit 1; }
fi

# Frameworks such as Lightning silently fall back to CPU; a paid GPU Pod must not train on CPU.
if [ "${REQUIRE_GPU:-1}" != 0 ]; then
  if python3 -c "import torch" 2>/dev/null; then
    gpu_check="$(python3 -c "import torch; print(torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" 2>&1)"
    log "torch: $gpu_check"
    case "$gpu_check" in *"available True"*) ;; *) FINAL_MESSAGE="no CUDA GPU visible to torch ($gpu_check)"; exit 1 ;; esac
  elif ! nvidia-smi -L >/dev/null 2>&1; then
    FINAL_MESSAGE="no GPU visible (nvidia-smi failed)"; exit 1
  fi
fi

"${R2[@]}" write-status --state running --step 0 --message "setup" >/dev/null || { FINAL_MESSAGE="cannot write to R2"; exit 1; }
[ -n "${CONFIG_PATH:-}" ] && [ -f "$CONFIG_PATH" ] && "${R2[@]}" put-config "$CONFIG_PATH" >/dev/null

if [ -n "${DATASET:-}" ] && [ "${DATASET}" != "none" ]; then
  log "fetching dataset $DATASET"
  DATA_DIR="$("${R2[@]}" pull-dataset "$DATASET" "$DATA_DIR" 2>>"$LOG" | tail -n1)" \
    || { FINAL_MESSAGE="dataset download failed: $DATASET"; exit 1; }
  export DATA_DIR
  log "dataset at $DATA_DIR ($(du -sh "$DATA_DIR" 2>/dev/null | cut -f1))"
fi

if [ -n "${RESUME_FROM:-}" ]; then
  key="$RESUME_FROM"; case "$key" in "$PROJECT"/*) key="${key#"$PROJECT"/}" ;; esac
  export RESUME_CKPT="$RUN_DIR/resume.pt"
  "${R2[@]}" get "$key" "$RESUME_CKPT" >/dev/null || { FINAL_MESSAGE="cannot fetch $RESUME_FROM"; exit 1; }
  log "resuming from $RESUME_FROM"
fi

( while sleep "${UPLOAD_EVERY_SEC:-300}"; do upload_outputs; done ) &
UPLOADER_PID=$!

log "starting: $TRAIN_ENTRY"
# New session so the whole training process group can be signalled; wait is interruptible.
setsid bash -c 'eval "$TRAIN_ENTRY" 2>&1 | tee -a "$0"; exit "${PIPESTATUS[0]}"' "$LOG" &
TRAIN_PID=$!
wait "$TRAIN_PID"; rc=$?
TRAIN_PID=""
if [ "$rc" -eq 0 ]; then
  echo "done" > "$STATE_FILE"; FINAL_MESSAGE="training finished"
else
  FINAL_MESSAGE="training exited with $rc"
fi
exit "$rc"
