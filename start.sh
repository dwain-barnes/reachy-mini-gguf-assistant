#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 the reachy-mini-gguf-assistant contributors
# SPDX-License-Identifier: Apache-2.0
#
# Start Reachy Mini: llama-server (Gemma 4 E2B, GPU) + llama-tts-server
# (Pocket TTS) + the assistant itself with its browser UI.
#
# Startup is deliberately sequential: each model is loaded and confirmed
# healthy before the next process is allowed anywhere near the GPU. On a Jetson
# that ordering is not a nicety, it is the difference between working and an
# OOM kill halfway through a model load.
#
# Leave this running. Ctrl+C shuts everything down.
#
#   ./start.sh                   web UI on every interface (browse from your laptop)
#   ./start.sh --localhost       bind the web UI to 127.0.0.1 only
#   ./start.sh --no-warmup       skip the warm-up requests
#   ./start.sh --no-app          just the two model servers
#   ./start.sh --no-drop-caches  do not free the page cache before each model load
#   ./start.sh --any-build       do not insist on the pinned llama.cpp commit
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO/logs"

BIND_ALL=1
WARMUP=1
RUN_APP=1
DROP_CACHES=1
CHECK_BUILD=1
while [ $# -gt 0 ]; do
    case "$1" in
        --localhost)       BIND_ALL=0; shift ;;
        --no-warmup)       WARMUP=0; shift ;;
        --no-app)          RUN_APP=0; shift ;;
        --no-drop-caches)  DROP_CACHES=0; shift ;;
        --any-build)       CHECK_BUILD=0; shift ;;
        -h|--help)         sed -n '5,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)                 echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'
    C_DIM=$'\033[90m'; C_HEAD=$'\033[1m'; C_CYAN=$'\033[36m'; C_OFF=$'\033[0m'
else
    C_OK=; C_WARN=; C_ERR=; C_DIM=; C_HEAD=; C_CYAN=; C_OFF=
fi
step() { printf '\n%s==> %s%s\n' "$C_HEAD" "$*" "$C_OFF"; }
ok()   { printf '    %s[ok]%s %s\n'   "$C_OK"   "$C_OFF" "$*"; }
info() { printf '    %s%s%s\n'        "$C_DIM"  "$*"     "$C_OFF"; }
warn() { printf '    %s[!]%s %s\n'    "$C_WARN" "$C_OFF" "$*"; }
die()  { printf '    %s[x]%s %s\n'    "$C_ERR"  "$C_OFF" "$*" >&2; exit 1; }

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------ config

[ -f "$REPO/config/servers.local.json" ] || \
    die "config/servers.local.json is missing. Run ./setup.sh first."

# Read the config once and eval it as shell assignments, so the rest of the
# script is plain variables rather than a python call per field.
eval "$(python3 - "$REPO/config/servers.json" "$REPO/config/servers.local.json" <<'PY'
import json, shlex, sys
cfg = {}
for path in sys.argv[1:]:
    try:
        with open(path, encoding="utf-8-sig") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass


def flag(value):
    if isinstance(value, str):
        return 1 if value.strip().lower() in ("1", "true", "yes", "on") else 0
    return 1 if value else 0


flat = {
    "LLM_PORT": cfg.get("llmPort", 8080),
    "TTS_PORT": cfg.get("ttsPort", 8100),
    "WEB_PORT": cfg.get("webUiPort", 8090),
    # 2048 is the safe default on an 8 GB Orin Nano, and the README explains
    # why raising it is how this stops fitting: the KV cache comes out of the
    # same pool as the weights, and Pocket TTS needs its share.
    "CTX": cfg.get("contextSize", 2048),
    "NGL": cfg.get("llmGpuLayers", 99),
    "TTS_THREADS": cfg.get("ttsThreads", 4),
    "TTS_ON_GPU": flag(cfg.get("ttsOnGpu", True)),
    "TTS_NGL": cfg.get("ttsGpuLayers", 99),
    "LLAMA_COMMIT": cfg.get("llamaCommit", "9f0d017"),
    "BIN_LLM": cfg.get("bin", {}).get("llamaServer", ""),
    "BIN_TTS": cfg.get("bin", {}).get("llamaTtsServer", ""),
    "M_LLM": cfg.get("models", {}).get("llm", ""),
    "M_LLM_MMPROJ": cfg.get("models", {}).get("llmMmproj", ""),
    "M_TTS": cfg.get("models", {}).get("tts", ""),
    "M_TTS_MMPROJ": cfg.get("models", {}).get("ttsMmproj", ""),
    "M_VOICE": cfg.get("models", {}).get("voice", ""),
}
for k, v in flat.items():
    print("%s=%s" % (k, shlex.quote(str(v))))
PY
)"

LLM_URL="http://127.0.0.1:$LLM_PORT"
TTS_URL="http://127.0.0.1:$TTS_PORT"

PYTHON="$REPO/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

# A Whisper transcript for the browser means a second request in flight while
# the conversation one is still streaming, so the server needs a second slot.
# pipeline.transcribe_after_reply does not: it only ever asks once the reply is
# finished and spoken, and a new utterance hangs up on it rather than queueing
# behind it. The single slot is the point there - it is what makes the
# conversation win.
PARALLEL=1
if "$PYTHON" - "$REPO/config/settings.yaml" <<'PY' 2>/dev/null
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
sys.exit(0 if (cfg.get("pipeline") or {}).get("transcribe_for_display") else 1)
PY
then
    PARALLEL=2
fi

# The prebuilt tarball ships libggml*.so / libllama*.so / libmtmd.so beside the
# two binaries rather than installing them, so the loader has to be told where
# to look. Harmless for a source build, where the same files live there anyway.
BIN_DIR="$(dirname "$BIN_LLM")"
export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo
printf '  %sReachy Mini GGUF Assistant%s\n' "$C_HEAD" "$C_OFF"
info 'starting up - the first run takes a couple of minutes while the models load'

step "Checking everything is where it should be"
for pair in "llama-server:$BIN_LLM" "llama-tts-server:$BIN_TTS"; do
    [ -x "${pair#*:}" ] || die "${pair%%:*} is missing or not executable: ${pair#*:} - run ./setup.sh"
done
for pair in "language model:$M_LLM" "vision/audio projector:$M_LLM_MMPROJ" \
            "speech model:$M_TTS" "speech projector:$M_TTS_MMPROJ" \
            "reference voice:$M_VOICE"; do
    [ -s "${pair#*:}" ] || die "the ${pair%%:*} is missing: ${pair#*:} - run ./setup.sh"
done
ok "all files present"

# The pinned build is not superstition. Server-side input_audio routing - the
# thing that lets the microphone reach Gemma with no speech-to-text model in
# between - and the llama-tts-server patch are both cut against this commit.
# A newer llama.cpp may still work; it has not been tested here, and the way it
# fails is a 500 on the first spoken turn rather than anything obvious.
BUILD_VERSION="$("$BIN_LLM" --version 2>&1 | head -n1 || true)"
if printf '%s' "$BUILD_VERSION" | grep -q "$LLAMA_COMMIT"; then
    ok "llama.cpp $LLAMA_COMMIT ($BUILD_VERSION)"
elif [ "$CHECK_BUILD" -eq 1 ]; then
    warn "these binaries are not the pinned llama.cpp $LLAMA_COMMIT:"
    warn "    $BUILD_VERSION"
    info "audio-in and the speech server are both tied to that commit."
    info "rebuild with ./setup.sh --build-from-source --force, or pass"
    info "--any-build to run with what you have."
    die "refusing to start on an untested build"
else
    warn "running on $BUILD_VERSION, not the pinned $LLAMA_COMMIT (--any-build)"
fi

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }
for port in "$LLM_PORT" "$TTS_PORT"; do
    ! port_busy "$port" || die "port $port is already in use - another copy may still be running"
done
ok "ports $LLM_PORT and $TTS_PORT are free"

# ------------------------------------------------------- memory / page cache

meminfo_mb() {  # meminfo_mb <MemFree|MemAvailable|...>
    awk -v k="$1:" '$1 == k {v = $2/1024} END {printf "%d", v}' /proc/meminfo 2>/dev/null \
        || printf '0'
}

DROP_CACHES_WARNED=0

# Free the page cache immediately before a CUDA allocation.
#
# This is the single most important line in this script on a Jetson. NvMap, the
# Tegra GPU memory allocator, will not reclaim the Linux page cache to satisfy
# a request. If MemFree is low, cudaMalloc fails with
#   NvMapMemAllocInternalTagged error 12
# even when MemAvailable is showing several free gigabytes, because the kernel
# counts reclaimable cache as "available" and NvMap does not. Dropping the
# cache first turns that available memory into genuinely free memory.
drop_caches() {  # drop_caches <what-is-about-to-load>
    local what="$1"
    [ "$DROP_CACHES" -eq 1 ] || return 0

    if [ "$(id -u)" -eq 0 ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    elif sudo -n true 2>/dev/null; then
        sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    else
        if [ "$DROP_CACHES_WARNED" -eq 0 ]; then
            DROP_CACHES_WARNED=1
            warn "cannot free the page cache: sudo wants a password and nobody is here to type it."
            warn "on a Jetson this is how model loads fail with 'NvMapMemAllocInternalTagged"
            warn "error 12' even though free -m says there are gigabytes available."
            info "do one of these, then start again:"
            info "    sudo -v && ./start.sh          # cache the sudo timestamp first"
            info "    sudo ./start.sh                # or run the whole thing as root"
            info "or pass --no-drop-caches to stop being told about it."
        fi
        return 0
    fi

    ok "page cache freed before loading $what (MemFree now $(meminfo_mb MemFree) MB)"
}

# A single drop before the load is not always enough: reading gigabytes of
# weights refills the page cache DURING the load, and NvMap can starve on an
# allocation that arrives mid-way. Some boots win that race, some lose it -
# same command, same free memory. The reliable form is a loop that keeps the
# cache empty for the whole load window.
DROP_LOOP_PID=""
start_drop_loop() {
    [ "$DROP_CACHES" -eq 1 ] || return 0
    if [ "$(id -u)" -eq 0 ]; then
        ( while :; do sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; sleep 2; done ) &
        DROP_LOOP_PID=$!
    elif sudo -n true 2>/dev/null; then
        sudo -n sh -c 'while :; do sync; echo 3 > /proc/sys/vm/drop_caches; sleep 2; done' 2>/dev/null &
        DROP_LOOP_PID=$!
    fi
}
stop_drop_loop() {
    if [ -n "$DROP_LOOP_PID" ]; then
        kill "$DROP_LOOP_PID" 2>/dev/null || sudo -n kill "$DROP_LOOP_PID" 2>/dev/null || true
        wait "$DROP_LOOP_PID" 2>/dev/null || true
        DROP_LOOP_PID=""
    fi
}

step "Making room for the first model"
info "MemFree $(meminfo_mb MemFree) MB, MemAvailable $(meminfo_mb MemAvailable) MB"
info "NvMap allocates out of MemFree only, so MemAvailable is not the number that matters"

# ---------------------------------------------------------------- shutdown

CHILDREN=()
SHUTTING_DOWN=0

shutdown() {
    [ "$SHUTTING_DOWN" -eq 0 ] || return
    SHUTTING_DOWN=1
    printf '\n  %sShutting down...%s\n' "$C_WARN" "$C_OFF"
    for pid in "${CHILDREN[@]:-}"; do
        [ -n "$pid" ] || continue
        kill -TERM "$pid" 2>/dev/null || true
    done
    # Give them a moment to close their sockets - and the app time to put the
    # robot to sleep - then insist.
    for _ in $(seq 1 20); do
        local alive=0
        for pid in "${CHILDREN[@]:-}"; do
            [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
        done
        [ "$alive" -eq 1 ] || break
        sleep 0.5
    done
    for pid in "${CHILDREN[@]:-}"; do
        [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null || true
    done
    printf '  %sStopped.%s\n\n' "$C_DIM" "$C_OFF"
}
trap 'shutdown; exit 0' INT TERM
trap 'shutdown' EXIT

# What to say when the language model fails to load. Almost always memory, and
# almost always something else on the board is holding it.
llm_failure_hint() {
    local errlog="$LOG_DIR/llm.err.log"
    warn "something else may be holding GPU memory, or there was not enough free"
    warn "memory for NvMap at the moment of the allocation."
    if grep -qi 'NvMapMemAllocInternalTagged\|cudaMalloc failed\|out of memory' "$errlog" 2>/dev/null; then
        warn "the log says exactly that:"
        grep -i 'NvMapMemAllocInternalTagged\|cudaMalloc failed\|out of memory' "$errlog" \
            2>/dev/null | tail -n 3 | sed 's/^/        /'
    fi
    info "things worth checking, in order:"
    info "  1. free the page cache and try again:"
    info "         sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
    info "  2. is another GPU workload running? many Jetsons ship with one"
    info "     (NanoOWL, jetson-inference, a stray container):"
    info "         systemctl list-units --type=service --state=running | grep -i 'nano\\|owl\\|jetson'"
    info "         docker ps"
    info "     stop it for the session with 'sudo systemctl stop <service>'."
    info "     do NOT disable it - that is someone's own setup - and do not pkill"
    info "     it either: services restart on failure and containers keep holding"
    info "     GPU memory after the visible process dies."
    info "  3. still tight? use a smaller quant: ./setup.sh --quant UD-Q2_K_XL --force"
    info "  4. close the desktop session, or run the board headless."
    info "see $errlog for the whole story."
}

wait_http() {   # wait_http <url> <timeout-s> <label> <pid>
    local url="$1" timeout="$2" label="$3" pid="$4" waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            printf '    %s[x]%s %s\n' "$C_ERR" "$C_OFF" \
                "$label stopped while loading - see $LOG_DIR/" >&2
            return 1
        fi
        # -f is not decoration. llama-server answers /health with 503 while it
        # is still loading, and a bare `curl -s` treats a 503 as success. Get
        # this wrong and the next server starts early, grabs GPU memory in the
        # middle of this one's load, and kills it.
        if curl -fs --max-time 2 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    printf '    %s[x]%s %s\n' "$C_ERR" "$C_OFF" \
        "$label did not come up within ${timeout}s - see $LOG_DIR/" >&2
    return 1
}

# ----------------------------------------------------------- language model

step "Starting the part that listens, sees and thinks"
info "Gemma 4 E2B on the GPU, $NGL layers, ${CTX} token context, $PARALLEL slot(s)"

drop_caches "the language model"
start_drop_loop   # keep it empty for the whole load - see the comment above

# --reasoning off is not optional. Gemma 4 is a thinking model; left on, it
# spends the whole token budget reasoning and returns empty content, which the
# speech server then refuses.
"$BIN_LLM" \
    -m "$M_LLM" \
    --mmproj "$M_LLM_MMPROJ" \
    -ngl "$NGL" \
    -c "$CTX" \
    -np "$PARALLEL" \
    --reasoning off \
    --host 127.0.0.1 \
    --port "$LLM_PORT" \
    >"$LOG_DIR/llm.out.log" 2>"$LOG_DIR/llm.err.log" &
PID_LLM=$!
CHILDREN+=("$PID_LLM")

T0=$(date +%s)
info "loading - expect 30 to 60 seconds. MemFree will dive to almost nothing"
info "on the way; that is fine as long as the cache was dropped first."
if ! wait_http "$LLM_URL/health" 600 "the thinking model" "$PID_LLM"; then
    stop_drop_loop
    llm_failure_hint
    exit 1
fi
stop_drop_loop
ok "thinking model loaded ($(( $(date +%s) - T0 ))s), MemFree $(meminfo_mb MemFree) MB"

# ------------------------------------------------------------- speech model
#
# Nothing below this line is allowed to start until the language model is
# healthy. The whole point of the sequencing is that the GPU is handed over one
# model at a time.

step "Starting the part that speaks"

drop_caches "the speech model"
start_drop_loop

if [ "$TTS_ON_GPU" -eq 1 ]; then
    info "Pocket TTS on the GPU, $TTS_NGL layers - roughly 30x faster than the CPU path"
    info "with the UD-Q2_K_XL quant this fits alongside Gemma with a little to spare"
    warn "it needs a second CUDA context (~0.6 GB, plus compute buffers). On a bigger"
    warn "quant, or with a desktop session running, it will not fit: set ttsOnGpu"
    warn "false in config/servers.local.json, or go back to UD-Q2_K_XL."

    "$BIN_TTS" \
        -m "$M_TTS" \
        --mmproj "$M_TTS_MMPROJ" \
        --tts-speaker-file "$M_VOICE" \
        -ngl "$TTS_NGL" \
        --threads "$TTS_THREADS" \
        --host 127.0.0.1 \
        --port "$TTS_PORT" \
        >"$LOG_DIR/tts.out.log" 2>"$LOG_DIR/tts.err.log" &
else
    info "Pocket TTS on the CPU, $TTS_THREADS of the 6 A78AE cores"
    info "expect about 0.16x realtime: a short sentence takes ~19s to speak."
    info "fine for testing, not for conversation."

    # An empty CUDA_VISIBLE_DEVICES hides the GPU from this child completely.
    #
    # -ngl 0 is NOT enough on its own. On a CUDA build the mtmd audio projector
    # allocates CUDA buffers regardless of the layer count, and when Gemma
    # already owns the GPU that allocation fails and takes the speech server
    # down with a ggml assert.
    CUDA_VISIBLE_DEVICES="" \
    "$BIN_TTS" \
        -m "$M_TTS" \
        --mmproj "$M_TTS_MMPROJ" \
        --tts-speaker-file "$M_VOICE" \
        -ngl 0 \
        --threads "$TTS_THREADS" \
        --host 127.0.0.1 \
        --port "$TTS_PORT" \
        >"$LOG_DIR/tts.out.log" 2>"$LOG_DIR/tts.err.log" &
fi
PID_TTS=$!
CHILDREN+=("$PID_TTS")

if ! wait_http "$TTS_URL/health" 600 "the speaking model" "$PID_TTS"; then
    stop_drop_loop
    if [ "$TTS_ON_GPU" -eq 1 ]; then
        warn "GPU speech did not fit alongside the language model. Set"
        warn "  \"ttsOnGpu\": false"
        warn "in config/servers.local.json and start again."
    else
        warn "if the log ends in a ggml assert about a CUDA buffer, the binary was"
        warn "built with CUDA and the audio projector tried to allocate on the GPU"
        warn "anyway. This script already starts it with CUDA_VISIBLE_DEVICES empty;"
        warn "check nothing in your environment is overriding that."
    fi
    exit 1
fi
stop_drop_loop
ok "speaking model loaded ($(( $(date +%s) - T0 ))s), MemFree $(meminfo_mb MemFree) MB"

# ----------------------------------------------------------------- warm-up

if [ "$WARMUP" -eq 1 ]; then
    step "Warming up"
    info "the first request of each kind allocates buffers and is much slower than"
    info "the rest; we pay that cost now rather than mid-conversation"

    W=$(date +%s)
    if curl -fs --max-time 300 -X POST "$LLM_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"gemma","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}' \
        >/dev/null; then
        ok "thinking model warm ($(( $(date +%s) - W ))s)"
    else
        warn "warm-up question failed - see $LOG_DIR/llm.err.log"
    fi

    # The speech warm-up matters more than the LLM one: the first synthesis
    # allocates the whole audio graph, and on ARM cores that is tens of seconds.
    W=$(date +%s)
    WARM_WAV="$LOG_DIR/warmup.wav"
    if curl -fs --max-time 300 -X POST "$TTS_URL/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -d '{"input":"Ready.","response_format":"wav","max_seconds":5}' \
        -o "$WARM_WAV" && [ "$(stat -c%s "$WARM_WAV" 2>/dev/null || echo 0)" -gt 44 ]; then
        ok "speaking model warm ($(( $(date +%s) - W ))s)"
    else
        warn "warm-up speech failed - the first real reply will be slow"
        warn "see $LOG_DIR/tts.err.log"
    fi
    rm -f "$WARM_WAV"
fi

# --------------------------------------------------------------- the robot

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ -n "$LAN_IP" ] || LAN_IP="127.0.0.1"
if [ "$BIND_ALL" -eq 1 ]; then
    WEB_HOST=0.0.0.0
    WEB_SHOWN="http://$LAN_IP:$WEB_PORT"
else
    WEB_HOST=127.0.0.1
    WEB_SHOWN="http://localhost:$WEB_PORT"
fi

if [ "$RUN_APP" -eq 0 ]; then
    step "Servers only (--no-app)"
    ok "language model on $LLM_URL"
    ok "speech on $TTS_URL"
    info "start the assistant yourself, as your normal user - not root, because"
    info "PulseAudio lives in the user session:"
    info "    export XDG_RUNTIME_DIR=/run/user/\$(id -u)"
    info "    $PYTHON run_web_vision_chat.py"
    echo
    while true; do
        kill -0 "$PID_LLM" 2>/dev/null || die "the thinking model stopped - see $LOG_DIR/llm.err.log"
        kill -0 "$PID_TTS" 2>/dev/null || die "the speaking model stopped - see $LOG_DIR/tts.err.log"
        sleep 2
    done
fi

step "Starting Reachy Mini"
info "camera, microphone, face tracking, speaking movements and the browser UI"

if port_busy "$WEB_PORT"; then
    die "port $WEB_PORT is already in use - another copy of the app may still be running"
fi

# Whose HuggingFace cache the app will actually use - root's if it stays root,
# the invoking user's if it is about to step back down to them.
APP_HOME="${HOME:-}"
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    APP_HOME="$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6)"
fi

# The first run downloads Pollen's recorded moves before the robot says
# anything, and the silence while it does is easy to read as a hang.
if [ -n "$APP_HOME" ] && [ ! -d "$APP_HOME/.cache/huggingface/hub/models--pollen-robotics--reachy-mini-emotions-library" ]; then
    info "first run: the Pollen emotions library (~172 files) downloads before the"
    info "first movement. That happens once; later starts go straight through."
fi

# The two model servers are happy as root - the app is not. PulseAudio runs in
# the user's session, so a root process finds no sound server at all: no
# microphone, no speech out. Since the usual invocation is 'sudo ./start.sh'
# (or a sudo that escalated for drop_caches), step back down to the invoking
# user for this one child, carrying the XDG_RUNTIME_DIR that points at their
# PulseAudio socket.
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    APP_UID="$(id -u "$SUDO_USER")"
    info "dropping to $SUDO_USER for the app - PulseAudio lives in the user session"
    # The app keeps the terminal: its own output is the conversation, and Ctrl+C
    # has to reach it so the robot is put to sleep before the motors are cut.
    # HOME goes with it: without it the model and emotion-library downloads land
    # in root's cache and are downloaded again the next time this runs as a
    # normal user. sudo relays SIGTERM to the command, so shutdown() still
    # reaches the app and the robot is still put to sleep.
    sudo -u "$SUDO_USER" \
        env XDG_RUNTIME_DIR="/run/user/$APP_UID" \
            HOME="${APP_HOME:-/home/$SUDO_USER}" \
            LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$PYTHON" "$REPO/run_web_vision_chat.py" --host "$WEB_HOST" --port "$WEB_PORT" &
else
    if [ "$(id -u)" -eq 0 ]; then
        warn "running as root with no SUDO_USER to step back down to. PulseAudio is"
        warn "a per-user service, so the microphone and speaker will probably be"
        warn "missing. Start this as your normal user instead: sudo -v && ./start.sh"
    fi
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    "$PYTHON" "$REPO/run_web_vision_chat.py" --host "$WEB_HOST" --port "$WEB_PORT" &
fi
PID_APP=$!
CHILDREN+=("$PID_APP")

echo
printf '  %s======================================================%s\n' "$C_OK" "$C_OFF"
printf '  %s READY%s\n' "$C_OK" "$C_OFF"
printf '  %s======================================================%s\n' "$C_OK" "$C_OFF"
echo
printf '   Both models are loaded and warm (%ss).\n' "$(( $(date +%s) - T0 ))"
echo
printf '   Watch and talk to it here:\n'
printf '       %s%s%s\n' "$C_CYAN" "$WEB_SHOWN" "$C_OFF"
echo
info "Speak to the robot - there is no wake word, the VAD decides when you started."
info "Leave this running. Ctrl+C here stops everything and puts Reachy to sleep."
info "Logs: $LOG_DIR/"
echo

# Watch the children rather than blocking on wait, so a crashed server is
# reported instead of silently leaving a half-dead assistant behind.
while true; do
    kill -0 "$PID_LLM" 2>/dev/null || die "the thinking model stopped - see $LOG_DIR/llm.err.log"
    kill -0 "$PID_TTS" 2>/dev/null || die "the speaking model stopped - see $LOG_DIR/tts.err.log"
    kill -0 "$PID_APP" 2>/dev/null || { warn "the assistant exited"; exit 0; }
    sleep 2
done
