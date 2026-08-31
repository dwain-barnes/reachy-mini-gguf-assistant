#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 the reachy-mini-gguf-assistant contributors
# SPDX-License-Identifier: Apache-2.0
#
# One-time setup for Reachy Mini on a Jetson Orin Nano Super.
#
# Gets the two llama.cpp servers - either the prebuilt JetPack 6 binaries
# (a minute) or a patched llama.cpp compiled here for CUDA compute 8.7 (30-60
# minutes) - downloads the two models, makes the Python virtualenv and writes
# config/servers.local.json.
#
# Safe to run more than once: anything already on disk is left alone.
#
#   ./setup.sh                      normal run - offers the prebuilt binaries first
#   ./setup.sh --quant Q4_K_M       pick a different LLM quant
#   ./setup.sh --build-from-source  never download, always compile
#   ./setup.sh --prebuilt           never compile, insist on the download
#   ./setup.sh --skip-build         only fetch models and make the venv
#   ./setup.sh --skip-models        only build
#   ./setup.sh --skip-venv          leave Python alone
#   ./setup.sh --no-robot           skip the Reachy Mini wiring (SDK, udev, audio)
#   ./setup.sh --force              re-resolve every path
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$REPO/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
MODELS_DIR="$REPO/models"
VENV_DIR="$REPO/venv"
CONFIG_TEMPLATE="$REPO/config/servers.json"
CONFIG_LOCAL="$REPO/config/servers.local.json"

# Pinned, and start.sh checks the binaries really are this build. Server-side
# input_audio routing - the thing that lets the microphone reach Gemma without
# a speech-to-text model - lands here, and the llama-tts-server patch is cut
# against this tree.
LLAMA_COMMIT="9f0d017"
CUDA_ARCH=87                      # Orin = Ampere, compute capability 8.7

# Prebuilt binaries, so most people never sit through the build. Built on
# JetPack 6 (L4T R36.4.7), CUDA arch 87, from llama.cpp at $LLAMA_COMMIT with
# the llama-tts-server patch applied. The tarball holds llama-server,
# llama-tts-server and the libggml*/libllama*/libmtmd shared objects.
PREBUILT_URL="https://github.com/dwain-barnes/jetson-voice-assistant/releases/download/v1.0.0/jetson-voice-assistant-orin-jp6-cuda-arm64.tar.gz"
PREBUILT_NAME="jetson-voice-assistant-orin-jp6-cuda-arm64.tar.gz"
PREBUILT_SHA256="a0cf9b0650bdaf1d72ad93c87ac8a900c9d612f2855929a050da665c6b5a0826"

# The llama-tts-server patch, only needed on the build-from-source path.
PATCH_URL="https://raw.githubusercontent.com/dwain-barnes/llama-tts-server/main/llama-tts-server.patch"
PATCH_FILE="$REPO/llama-tts-server.patch"

LLM_REPO="unsloth/gemma-4-E2B-it-GGUF"
LLM_QUANT="UD-Q2_K_XL"            # 2.24 GiB - leaves room for speech on the GPU.
                                  # UD-Q4_K_XL answers better but then speech has
                                  # to move to the CPU (~19s a sentence).
LLM_MMPROJ="mmproj-F16.gguf"      # 0.92 GiB; does both vision and audio
TTS_REPO="EryriLabs/pocket-tts-GGUF"
TTS_FILES=(pocket-tts-en.gguf mmproj-pocket-tts-en.gguf)
VOICE_REPO="kyutai/tts-voices"
VOICE_FILE="unmute-prod-website/default_voice.wav"

SKIP_BUILD=0
SKIP_MODELS=0
SKIP_VENV=0
SKIP_ROBOT=0
FORCE=0
WANT_PREBUILT=auto            # auto | always | never

# The Reachy Mini SDK, pinned to the version this fork is runtime-tested on.
REACHY_SDK_VERSION="1.3.1"
UDEV_RULES_FILE="/etc/udev/rules.d/99-reachy-mini.rules"

while [ $# -gt 0 ]; do
    case "$1" in
        --quant)             LLM_QUANT="$2"; shift 2 ;;
        --quant=*)           LLM_QUANT="${1#*=}"; shift ;;
        --build-from-source) WANT_PREBUILT=never; shift ;;
        --prebuilt)          WANT_PREBUILT=always; shift ;;
        --skip-build)        SKIP_BUILD=1; shift ;;
        --skip-models)       SKIP_MODELS=1; shift ;;
        --skip-venv)         SKIP_VENV=1; shift ;;
        --no-robot)          SKIP_ROBOT=1; shift ;;
        --force)             FORCE=1; shift ;;
        -h|--help)           sed -n '5,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)                   echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

LLM_FILE="gemma-4-E2B-it-${LLM_QUANT}.gguf"

# ------------------------------------------------------------------ output

if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'
    C_DIM=$'\033[90m'; C_HEAD=$'\033[1m'; C_OFF=$'\033[0m'
else
    C_OK=; C_WARN=; C_ERR=; C_DIM=; C_HEAD=; C_OFF=
fi
step() { printf '\n%s==> %s%s\n' "$C_HEAD" "$*" "$C_OFF"; }
ok()   { printf '    %s[ok]%s %s\n'   "$C_OK"   "$C_OFF" "$*"; }
info() { printf '    %s%s%s\n'        "$C_DIM"  "$*"     "$C_OFF"; }
warn() { printf '    %s[!]%s %s\n'    "$C_WARN" "$C_OFF" "$*"; }
die()  { printf '    %s[x]%s %s\n'    "$C_ERR"  "$C_OFF" "$*" >&2; exit 1; }

# Can we run sudo without a password prompt nobody is here to answer? A terminal
# gets the normal prompt; a headless/scripted run only qualifies if the sudo
# timestamp is already cached (sudo -v) or the rule is NOPASSWD. Same guard the
# apt step below uses - callers print the exact commands instead of letting sudo
# abort with something cryptic.
can_sudo() { [ -t 0 ] || sudo -n true 2>/dev/null; }

echo
printf '  %sReachy Mini GGUF Assistant - setup%s\n' "$C_HEAD" "$C_OFF"
info 'everything runs on this board; nothing is sent anywhere'

# ------------------------------------------------------------- is a Jetson?

step "Checking the board"

IS_JETSON=0
L4T_R36=0
if [ -f /etc/nv_tegra_release ]; then
    IS_JETSON=1
    ok "$(head -n1 /etc/nv_tegra_release)"
    # The prebuilt binaries were compiled against JetPack 6 (L4T R36). Anything
    # older links a different CUDA runtime and will not load them.
    if grep -q 'R36' /etc/nv_tegra_release; then L4T_R36=1; fi
elif [ -f /proc/device-tree/model ] && tr -d '\0' < /proc/device-tree/model | grep -qi 'jetson\|orin'; then
    IS_JETSON=1
    ok "$(tr -d '\0' < /proc/device-tree/model)"
fi

if [ "$IS_JETSON" -eq 0 ]; then
    warn "this does not look like a Jetson (no /etc/nv_tegra_release)."
    warn "the build flags below target CUDA compute 8.7 (Orin) and will not"
    warn "produce useful binaries elsewhere."
    if [ -t 0 ]; then
        read -r -p "    carry on anyway? [y/N] " reply
        case "$reply" in [Yy]*) ;; *) exit 1 ;; esac
    else
        die "refusing to run unattended on a non-Jetson; re-run from a terminal to override."
    fi
fi

ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] || warn "architecture is $ARCH, not aarch64 - expect trouble"

TOTAL_MB=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)
info "RAM (shared with the GPU): ${TOTAL_MB} MB"
if [ "$TOTAL_MB" -lt 6500 ]; then
    warn "less than ~7 GB visible - the defaults here assume the 8 GB Orin Nano."
fi

if command -v nvcc >/dev/null 2>&1; then
    ok "$(nvcc --version | tail -n1 | sed 's/^ *//')"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    ok "found nvcc at /usr/local/cuda/bin/nvcc (added to PATH for this run)"
else
    warn "nvcc is not on PATH. JetPack usually puts it in /usr/local/cuda/bin."
    warn "install it with:  sudo apt install nvidia-cuda-dev  (or reflash JetPack 6)"
fi

# Something else on the board usually owns the GPU at boot. Say so now rather
# than at the OOM three steps later.
if systemctl list-units --type=service --state=running 2>/dev/null \
        | grep -qi 'nanoowl\|nano_owl\|jetson-inference'; then
    warn "another GPU service is running on this board. Before ./start.sh:"
    info "    sudo systemctl stop <service>       # for this session only"
    info "do NOT 'disable' it - that is the user's own setup, and it comes back"
    info "on the next boot either way."
fi

# ----------------------------------------------------------- swap / memory

step "Checking swap"

SWAP_MB=$(awk '/SwapTotal/ {printf "%d", $2/1024}' /proc/meminfo)
if [ "$SWAP_MB" -ge 4096 ]; then
    ok "${SWAP_MB} MB of swap - plenty for the build"
else
    warn "only ${SWAP_MB} MB of swap. Compiling the CUDA kernels can peak well"
    warn "past 8 GB and the OOM killer will end the build hours in."
    info "Add a real swap file on the NVMe/SD card before building:"
    info "    sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile"
    info "    sudo mkswap /swapfile && sudo swapon /swapfile"
    info "It is only needed for the build; you can swapoff afterwards."
fi

if command -v nvpmodel >/dev/null 2>&1; then
    info "tip: 'sudo nvpmodel -m 2 && sudo jetson_clocks' selects the 25W"
    info "     Super profile, which roughly halves both build and inference time."
fi

# ------------------------------------------------------------ dependencies

step "Installing system dependencies"

APT_PKGS=(build-essential cmake git curl ca-certificates pkg-config
          python3 python3-pip python3-venv ffmpeg alsa-utils
          pulseaudio-utils libgomp1)
MISSING=()
for p in "${APT_PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
done

if [ ${#MISSING[@]} -eq 0 ]; then
    ok "all apt packages already installed"
else
    info "installing: ${MISSING[*]}"
    # A terminal gets the normal sudo prompt; a headless/scripted run (no tty)
    # cannot answer one, so fail with the exact command instead of a cryptic
    # sudo abort.
    if [ -t 0 ] || sudo -n true 2>/dev/null; then
        sudo apt-get update
        sudo apt-get install -y "${MISSING[@]}"
        ok "apt packages installed"
    else
        warn "cannot ask for a sudo password without a terminal."
        warn "run this once, then re-run ./setup.sh:"
        warn "    sudo apt-get install -y ${MISSING[*]}"
        exit 1
    fi
fi

if ! python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    info "installing huggingface_hub for the downloads"
    python3 -m pip install --user --upgrade huggingface_hub || \
        warn "pip install failed; setup will fall back to plain curl downloads"
else
    ok "huggingface_hub already installed"
fi

# ------------------------------------------------------------------- build

# The shared objects ship next to the binaries rather than being installed, so
# every attempt to run one needs the loader pointed at that directory.
# start.sh exports the same thing.
BIN_LD_PATH="$BUILD_DIR/bin"

binaries_present() {
    [ -x "$BUILD_DIR/bin/llama-server" ] && [ -x "$BUILD_DIR/bin/llama-tts-server" ]
}

# Actually run one of them. A file of the right name that cannot resolve its
# libraries is worse than no file at all, because start.sh would accept it.
verify_binaries() {
    LD_LIBRARY_PATH="$BIN_LD_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$BUILD_DIR/bin/llama-server" --version >/dev/null 2>&1
}

prebuilt_eligible() {
    [ "$ARCH" = "aarch64" ] || { info "not aarch64 ($ARCH) - the prebuilt binaries would not run here" >&2; return 1; }
    [ "$L4T_R36" -eq 1 ] || { info "no 'R36' in /etc/nv_tegra_release - built for JetPack 6, so not this board" >&2; return 1; }
    return 0
}

try_prebuilt() {
    local tmp tgz got moved=0 f
    tmp="$(mktemp -d)" || return 1
    tgz="$tmp/$PREBUILT_NAME"

    info "downloading $PREBUILT_NAME"
    if ! curl -fL --retry 3 --progress-bar -o "$tgz" "$PREBUILT_URL"; then
        warn "the download failed"
        rm -rf "$tmp"
        return 1
    fi

    # We are about to execute what we just downloaded - verify it first.
    got="$(sha256sum "$tgz" | awk '{print $1}')"
    if [ "$got" != "$PREBUILT_SHA256" ]; then
        warn "checksum mismatch (got $got) - refusing the download, will build instead"
        rm -rf "$tmp"
        return 1
    fi
    ok "sha256 matches"

    mkdir -p "$BUILD_DIR/bin"
    if ! tar -xzf "$tgz" -C "$tmp"; then
        warn "the tarball did not extract - it may be a truncated download"
        rm -rf "$tmp"
        return 1
    fi

    # Flatten whatever shape the archive has: binaries and .so files all have to
    # end up side by side in build/bin.
    while IFS= read -r f; do
        cp -f "$f" "$BUILD_DIR/bin/" && moved=$((moved + 1))
    done < <(find "$tmp" -type f \( -name 'llama-server' -o -name 'llama-tts-server' \
                                    -o -name '*.so' -o -name '*.so.*' \))
    rm -rf "$tmp"

    [ "$moved" -gt 0 ] || { warn "the tarball did not contain anything recognisable"; return 1; }
    info "unpacked $moved files into $BUILD_DIR/bin"

    chmod +x "$BUILD_DIR/bin/llama-server" "$BUILD_DIR/bin/llama-tts-server" 2>/dev/null || true
    binaries_present || { warn "llama-server or llama-tts-server was not in the tarball"; return 1; }
    verify_binaries || { warn "the downloaded llama-server will not run on this board"; return 1; }

    ok "$(LD_LIBRARY_PATH="$BIN_LD_PATH" "$BUILD_DIR/bin/llama-server" --version 2>&1 | head -n1)"
    return 0
}

build_from_source() {
    step "Fetching llama.cpp at $LLAMA_COMMIT"

    if [ ! -d "$LLAMA_DIR/.git" ]; then
        git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    fi
    git -C "$LLAMA_DIR" fetch --all --tags
    # Reset hard so a re-run after a half-applied patch starts clean.
    git -C "$LLAMA_DIR" checkout --force "$LLAMA_COMMIT"
    git -C "$LLAMA_DIR" reset --hard "$LLAMA_COMMIT"
    git -C "$LLAMA_DIR" clean -fd -e build
    ok "at $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"

    step "Applying the llama-tts-server patch"
    if [ ! -f "$PATCH_FILE" ]; then
        info "fetching the patch from dwain-barnes/llama-tts-server"
        curl -fL --retry 3 -o "$PATCH_FILE" "$PATCH_URL" || \
            die "could not download the patch; put it at $PATCH_FILE by hand"
    fi
    if git -C "$LLAMA_DIR" apply --check "$PATCH_FILE" 2>/dev/null; then
        git -C "$LLAMA_DIR" apply "$PATCH_FILE"
        ok "patch applied"
    elif [ -f "$LLAMA_DIR/tools/tts/tts-server.cpp" ]; then
        ok "patch already applied"
    else
        die "the patch did not apply - is llama.cpp/ modified? delete it and re-run."
    fi

    step "Building (this is the slow part)"
    warn "expect 30-60 minutes on an Orin Nano. Do not let the board sleep."

    cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF

    JOBS=$(nproc)
    # Each nvcc job can want well over 1 GB. On a 6-core 8 GB board, -j6 with
    # thin swap is how builds die at 90%; back off unless there is swap to spare.
    if [ "$SWAP_MB" -lt 4096 ] && [ "$JOBS" -gt 4 ]; then
        JOBS=4
        info "using -j4 rather than -j$(nproc) because swap is thin"
    fi
    cmake --build "$BUILD_DIR" --config Release -j"$JOBS" \
        --target llama-server llama-tts-server

    [ -x "$BUILD_DIR/bin/llama-server" ]     || die "llama-server was not produced"
    [ -x "$BUILD_DIR/bin/llama-tts-server" ] || die "llama-tts-server was not produced"
    ok "built both servers"
}

if [ "$SKIP_BUILD" -eq 1 ]; then
    step "Skipping the build (--skip-build)"
elif [ "$FORCE" -eq 0 ] && binaries_present; then
    step "Both servers are already here"
    ok "$BUILD_DIR/bin/llama-server"
    ok "$BUILD_DIR/bin/llama-tts-server"
    info "pass --force to fetch or build them again"
else
    GOT_BINARIES=0

    if [ "$WANT_PREBUILT" = "never" ]; then
        info "building from source because you asked for it (--build-from-source)"
    elif prebuilt_eligible; then
        step "Prebuilt binaries are available for this board"
        info "built on JetPack 6 (R36), CUDA arch 87, llama.cpp $LLAMA_COMMIT with"
        info "the llama-tts-server patch. Taking them saves the 30-60 minute build."
        info "Compiling yourself is the honest option if you would rather not trust"
        info "someone else's binaries; pass --build-from-source for that."

        TAKE_PREBUILT=1
        if [ "$WANT_PREBUILT" != "always" ] && [ -t 0 ]; then
            read -r -p "    download the prebuilt binaries? [Y/n] " reply
            case "$reply" in [Nn]*) TAKE_PREBUILT=0 ;; esac
        fi

        if [ "$TAKE_PREBUILT" -eq 1 ]; then
            if try_prebuilt; then
                GOT_BINARIES=1
                ok "prebuilt binaries in place - no build needed"
            else
                warn "falling back to building from source"
            fi
        else
            info "building from source instead"
        fi
    else
        info "no prebuilt binaries for this board, so building from source"
    fi

    if [ "$GOT_BINARIES" -eq 0 ]; then
        [ "$WANT_PREBUILT" = "always" ] && \
            die "--prebuilt was given but the download could not be used. See above."
        build_from_source
    fi
fi

# --------------------------------------------------------------- downloads

mkdir -p "$MODELS_DIR"

# fetch <repo> <path-in-repo> -> prints the local path
fetch() {
    local repo="$1" path="$2" leaf dest
    leaf="$(basename "$path")"
    dest="$MODELS_DIR/$leaf"

    # Everything chatty goes to stderr: stdout is the returned path.
    if [ -s "$dest" ]; then
        ok "$leaf - already here ($(du -h "$dest" | cut -f1))" >&2
        printf '%s' "$dest"
        return
    fi

    info "downloading $leaf from $repo ..." >&2
    if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
        python3 - "$repo" "$path" "$MODELS_DIR" <<'PY' >&2
import os, shutil, sys
from huggingface_hub import hf_hub_download
repo, path, dest_dir = sys.argv[1:4]
src = hf_hub_download(repo_id=repo, filename=path)
dest = os.path.join(dest_dir, os.path.basename(path))
# Copy rather than symlink: the HF cache may sit on a different filesystem
# from models/, and llama.cpp opens these by path at every start.
shutil.copyfile(src, dest)
PY
    else
        curl -fL --retry 3 --progress-bar -o "$dest.part" \
            "https://huggingface.co/$repo/resolve/main/$path?download=true" >&2
        mv "$dest.part" "$dest"
    fi
    [ -s "$dest" ] || die "download did not produce $dest"
    ok "$leaf - $(du -h "$dest" | cut -f1)" >&2
    printf '%s' "$dest"
}

if [ "$SKIP_MODELS" -eq 1 ]; then
    step "Skipping the downloads (--skip-models)"
    M_LLM="$MODELS_DIR/$LLM_FILE"
    M_LLM_MMPROJ="$MODELS_DIR/$LLM_MMPROJ"
    M_TTS="$MODELS_DIR/${TTS_FILES[0]}"
    M_TTS_MMPROJ="$MODELS_DIR/${TTS_FILES[1]}"
    M_VOICE="$MODELS_DIR/$(basename "$VOICE_FILE")"
else
    step "Fetching the models (about 3.4 GB in total)"
    info "into $MODELS_DIR"
    AVAIL_MB=$(df -Pm "$MODELS_DIR" | awk 'NR==2 {print $4}')
    info "free space there: ${AVAIL_MB} MB"
    [ "$AVAIL_MB" -gt 5000 ] || warn "that is tight - the models need about 3.4 GB"

    M_LLM="$(fetch "$LLM_REPO" "$LLM_FILE")"
    M_LLM_MMPROJ="$(fetch "$LLM_REPO" "$LLM_MMPROJ")"
    M_TTS="$(fetch "$TTS_REPO" "${TTS_FILES[0]}")"
    M_TTS_MMPROJ="$(fetch "$TTS_REPO" "${TTS_FILES[1]}")"
    M_VOICE="$(fetch "$VOICE_REPO" "$VOICE_FILE")"
fi

# ------------------------------------------------------------------ python

if [ "$SKIP_VENV" -eq 1 ]; then
    step "Skipping the virtualenv (--skip-venv)"
else
    step "Python environment"
    if [ ! -d "$VENV_DIR" ]; then
        # --system-site-packages so the JetPack-provided CUDA wheels
        # (onnxruntime-gpu, torch) stay visible inside the venv.
        python3 -m venv --system-site-packages "$VENV_DIR"
        ok "created $VENV_DIR"
    else
        ok "$VENV_DIR is already here"
    fi
    # shellcheck disable=SC1091
    "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null 2>&1 || true
    info "installing requirements.txt (a few minutes on first run)"
    "$VENV_DIR/bin/pip" install -r "$REPO/requirements.txt" \
        || die "pip install failed - see the output above"
    ok "python packages installed"
    info "the robot SDK is handled by the Reachy step below (or --no-robot to skip it)"
    info "Jetson wheels, if you have not already:"
    info "    $VENV_DIR/bin/pip install numpy==1.26.4"
    info "    $VENV_DIR/bin/pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126"
fi

# ------------------------------------------------------------------- robot
#
# Everything the robot itself needs, none of which the steps above cover: the
# SDK and scipy inside the venv, the apt packages the SDK builds and runs
# against, the udev rules that make the serial port reachable without root, and
# dialout membership. All of it verified on a Jetson Orin Nano Super driving a
# Reachy Mini Lite; the notes below are what that bring-up actually cost.
#
# Idempotent throughout: an apt package already installed, a udev file already
# byte-identical, a group already joined and a pinned SDK already present are
# all left alone.

ROBOT_TODO=0        # set when something was left for the user to do by hand

robot_seen() {
    [ -e /dev/reachy_mini ] && return 0
    command -v lsusb >/dev/null 2>&1 || return 1
    # 2e8a:000a is the RP2040 serial port upstream documents. Newer revisions
    # ship a QinHeng CH343 (1a86:55d3) instead, and 38fb:1002 is the SunplusIT
    # camera - which on those units is also what the microphone array
    # enumerates under.
    lsusb 2>/dev/null | grep -qiE '2e8a:000a|1a86:55d3|38fb:1002'
}

if [ "$SKIP_ROBOT" -eq 1 ]; then
    step "Skipping the Reachy Mini setup (--no-robot)"
else
    step "Reachy Mini"

    if robot_seen; then
        ok "a Reachy Mini looks to be plugged in"
    else
        info "no Reachy Mini seen on USB right now - setting it up anyway, so the"
        info "robot works the moment you plug it in. Pass --no-robot to skip this."
    fi

    # ---- apt: what the SDK builds against, and what the daemon needs to run.
    #
    # libportaudio2 is the one that bites. Without it the Reachy daemon does not
    # limp along without sound, it dies at startup with
    #   OSError: PortAudio library not found
    # which reads like a Python packaging problem and is not. The three -dev
    # packages are only needed while pip builds PyGObject/pycairo for the SDK.
    ROBOT_APT=(libcairo2-dev libgirepository1.0-dev pkg-config
               libportaudio2 portaudio19-dev)
    ROBOT_MISSING=()
    for p in "${ROBOT_APT[@]}"; do
        dpkg -s "$p" >/dev/null 2>&1 || ROBOT_MISSING+=("$p")
    done

    INSTALLED_PORTAUDIO=0
    if [ ${#ROBOT_MISSING[@]} -eq 0 ]; then
        ok "robot apt packages already installed"
    elif can_sudo; then
        info "installing: ${ROBOT_MISSING[*]}"
        # Not fatal if it fails: the rest of setup still has to finish and write
        # config/servers.local.json, and everything but the robot works without.
        if sudo apt-get update && sudo apt-get install -y "${ROBOT_MISSING[@]}"; then
            ok "robot apt packages installed"
            case " ${ROBOT_MISSING[*]} " in *" libportaudio2 "*) INSTALLED_PORTAUDIO=1 ;; esac
        else
            warn "apt could not install: ${ROBOT_MISSING[*]}"
            ROBOT_TODO=1
        fi
    else
        warn "cannot ask for a sudo password without a terminal - skipping the apt step."
        warn "run this once, then re-run ./setup.sh:"
        warn "    sudo apt-get install -y ${ROBOT_MISSING[*]}"
        ROBOT_TODO=1
    fi

    # A daemon that was already running when libportaudio2 arrived has already
    # failed to find it and will not look again. Kill it so the next connect
    # spawns a fresh one; the app starts it on demand.
    if [ "$INSTALLED_PORTAUDIO" -eq 1 ] && pgrep -f reachy-mini-daemon >/dev/null 2>&1; then
        pkill -f reachy-mini-daemon 2>/dev/null || sudo -n pkill -f reachy-mini-daemon 2>/dev/null || true
        info "stopped the running reachy-mini-daemon so it picks up PortAudio"
    fi

    # ---- udev: both vendors, one file.
    #
    # Upstream documents only 2e8a:000a. Newer robots present a QinHeng CH343
    # (1a86:55d3) and match nothing, so /dev/reachy_mini never appears and the
    # SDK cannot open the port without root. Ship both lines - the wrong one
    # simply never matches.
    UDEV_WANT="$(cat <<'RULES'
# Reachy Mini serial port. Two vendors on purpose: 2e8a:000a is the RP2040 that
# upstream documents, 1a86:55d3 the QinHeng CH343 on newer hardware revisions.
SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="000a", MODE="0666", SYMLINK+="reachy_mini"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", SYMLINK+="reachy_mini"
RULES
)"

    if [ -f "$UDEV_RULES_FILE" ] && [ "$(cat "$UDEV_RULES_FILE")" = "$UDEV_WANT" ]; then
        ok "$UDEV_RULES_FILE is already correct"
    elif can_sudo; then
        if printf '%s\n' "$UDEV_WANT" | sudo tee "$UDEV_RULES_FILE" >/dev/null \
           && sudo udevadm control --reload-rules && sudo udevadm trigger; then
            ok "wrote $UDEV_RULES_FILE and reloaded udev"
            if [ -e /dev/reachy_mini ]; then
                ok "/dev/reachy_mini -> $(readlink -f /dev/reachy_mini 2>/dev/null || echo '?')"
            else
                info "/dev/reachy_mini will appear when the robot is plugged in"
            fi
        else
            warn "could not write or reload $UDEV_RULES_FILE"
            ROBOT_TODO=1
        fi
    else
        warn "cannot write $UDEV_RULES_FILE without a sudo password - skipping."
        warn "run this once, then re-run ./setup.sh:"
        warn "    sudo tee $UDEV_RULES_FILE >/dev/null <<'EOF'"
        printf '%s\n' "$UDEV_WANT" | sed 's/^/        /'
        warn "    EOF"
        warn "    sudo udevadm control --reload-rules && sudo udevadm trigger"
        ROBOT_TODO=1
    fi

    # ---- dialout, so the serial port is readable as a normal user.
    ROBOT_USER="${SUDO_USER:-$(id -un)}"
    if id -nG "$ROBOT_USER" 2>/dev/null | tr ' ' '\n' | grep -qx dialout; then
        ok "$ROBOT_USER is already in the dialout group"
    elif can_sudo; then
        if sudo usermod -aG dialout "$ROBOT_USER"; then
            ok "added $ROBOT_USER to the dialout group"
            warn "group membership only applies to NEW logins - log out and back in"
            warn "(or reboot) before starting the robot."
        else
            warn "could not add $ROBOT_USER to dialout"
            ROBOT_TODO=1
        fi
    else
        warn "cannot add $ROBOT_USER to dialout without a sudo password - skipping."
        warn "run this once, then log out and back in:"
        warn "    sudo usermod -aG dialout $ROBOT_USER"
        ROBOT_TODO=1
    fi

    # ---- the SDK and scipy, inside the venv.
    if [ ! -x "$VENV_DIR/bin/pip" ]; then
        warn "no virtualenv at $VENV_DIR - re-run without --skip-venv to get the SDK"
        ROBOT_TODO=1
    else
        if "$VENV_DIR/bin/python3" - "$REACHY_SDK_VERSION" <<'PY' >/dev/null 2>&1
import importlib.metadata as md, sys
sys.exit(0 if md.version("reachy-mini") == sys.argv[1] else 1)
PY
        then
            ok "reachy-mini $REACHY_SDK_VERSION already in the venv"
        else
            info "installing reachy-mini==$REACHY_SDK_VERSION (builds PyGObject; a few minutes)"
            # Not fatal: the rest of setup (including config/servers.local.json)
            # still has to finish, and everything but the robot works without it.
            if "$VENV_DIR/bin/pip" install "reachy-mini==$REACHY_SDK_VERSION"; then
                ok "reachy-mini $REACHY_SDK_VERSION installed"
            else
                warn "the SDK would not install. If it failed building pycairo or"
                warn "PyGObject, the apt packages above are the fix; install them"
                warn "and re-run ./setup.sh."
                ROBOT_TODO=1
            fi
        fi

        # The SDK's audio helpers resample with scipy and do not declare it.
        if "$VENV_DIR/bin/python3" -c 'import scipy' >/dev/null 2>&1; then
            ok "scipy already in the venv"
        else
            info "installing scipy (the SDK's audio path needs it)"
            if "$VENV_DIR/bin/pip" install scipy; then
                ok "scipy installed"
            else
                warn "scipy would not install - the robot's audio path needs it"
                ROBOT_TODO=1
            fi
        fi

        # reachy-mini declares numpy>=2.2.5; the Jetson onnxruntime-gpu wheel
        # wants 1.x. It runs either way, but say so rather than let a later
        # import error look mysterious.
        if "$VENV_DIR/bin/python3" -c 'import onnxruntime' >/dev/null 2>&1 \
           && ! "$VENV_DIR/bin/python3" -c 'import numpy,sys; sys.exit(0 if numpy.__version__.startswith("1.") else 1)' >/dev/null 2>&1; then
            warn "numpy is 2.x now and the Jetson onnxruntime-gpu wheel expects 1.x."
            warn "if ONNX starts complaining:  $VENV_DIR/bin/pip install numpy==1.26.4"
        fi
    fi

    info "audio on newer robot revisions: the mic array enumerates under the CAMERA"
    info "name (\"Reachy Mini Camera\"), and there is no USB speaker at all - speech"
    info "comes out of the Jetson. config/settings.yaml is already set for both."
    info "if the robot shows up on USB with no audio interface, unplug and replug it."
fi

# ------------------------------------------------------------------ config

step "Writing config/servers.local.json"

python3 - "$CONFIG_TEMPLATE" "$CONFIG_LOCAL" <<PY
import json, sys
template, out = sys.argv[1:3]
with open(template, encoding="utf-8") as f:
    cfg = json.load(f)
cfg.pop("_comment", None)
cfg["quant"] = "$LLM_QUANT"
cfg["llamaCommit"] = "$LLAMA_COMMIT"
cfg["modelsDir"] = "$MODELS_DIR"
cfg["bin"] = {
    "llamaServer": "$BUILD_DIR/bin/llama-server",
    "llamaTtsServer": "$BUILD_DIR/bin/llama-tts-server",
}
cfg["models"] = {
    "llm": "$M_LLM",
    "llmMmproj": "$M_LLM_MMPROJ",
    "tts": "$M_TTS",
    "ttsMmproj": "$M_TTS_MMPROJ",
    "voice": "$M_VOICE",
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
ok "written to $CONFIG_LOCAL"

# ----------------------------------------------------------------- summary

step "Checking what is in place"

MISSING=0
for f in "$M_LLM" "$M_LLM_MMPROJ" "$M_TTS" "$M_TTS_MMPROJ" "$M_VOICE"; do
    if [ -s "$f" ]; then ok "$(basename "$f")"; else warn "missing: $f"; MISSING=1; fi
done
for b in "$BUILD_DIR/bin/llama-server" "$BUILD_DIR/bin/llama-tts-server"; do
    if [ -x "$b" ]; then ok "$(basename "$b")"; else warn "missing: $b"; MISSING=1; fi
done

if [ "$MISSING" -eq 0 ] && [ "$SKIP_BUILD" -eq 0 ]; then
    if verify_binaries; then
        ok "llama-server runs: $(LD_LIBRARY_PATH="$BIN_LD_PATH" "$BUILD_DIR/bin/llama-server" --version 2>&1 | head -n1)"
    else
        warn "llama-server will not start. If these came from the prebuilt tarball,"
        warn "re-run with --build-from-source to compile against your own CUDA."
        MISSING=1
    fi
fi

if [ "$ROBOT_TODO" -eq 1 ]; then
    warn "some of the robot setup needed a sudo password and was skipped -"
    warn "run the commands printed above, then re-run ./setup.sh."
fi

echo
if [ "$MISSING" -eq 0 ]; then
    printf '  %sSetup finished.%s Start it with:  sudo -v && ./start.sh\n' "$C_OK" "$C_OFF"
    echo
    info "The sudo is not for the app. On a Jetson the page cache has to be freed"
    info "immediately before each model load or the CUDA allocation fails with"
    info "'NvMapMemAllocInternalTagged error 12', and start.sh can only do that"
    info "if sudo does not stop to ask for a password."
else
    printf '  %sSetup is not finished%s - see the warnings above, then re-run.\n' "$C_WARN" "$C_OFF"
    exit 1
fi
echo
