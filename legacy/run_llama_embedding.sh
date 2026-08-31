#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run llama.cpp embedding server (GPU) — wrapper around run_llama_cpp.sh.
#
# Usage:
#   ./run_llama_embedding.sh ggml-org/bge-small-en-v1.5-Q8_0-GGUF:Q8_0
#   ./run_llama_embedding.sh ./models/bge-small-en-v1.5-q8_0.gguf
#
# Stop:
#   docker stop assistant-embed

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8081}" NAME="${NAME:-assistant-embed}" EMBED=1 \
    "$SCRIPT_DIR/run_llama_cpp.sh" "${1:?Usage: $0 <user/repo:quant or path/to/model.gguf>}"
