#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
MODEL_PATH="${MODEL_PATH:-}"
VOICELAB_ENABLE_TTS="${VOICELAB_ENABLE_TTS:-auto}"
XTTS_CHECKPOINT="${XTTS_CHECKPOINT:-}"
XTTS_CONFIG="${XTTS_CONFIG:-}"
XTTS_VOCAB="${XTTS_VOCAB:-}"
DEFAULT_MODEL_PATH="$ROOT_DIR/../experiments/dl1_xtts_ft/speaker2/run/training/GPT_XTTS_FT-April-19-2026_03+30AM-0000000"

if [[ "$VOICELAB_ENABLE_TTS" != "false" && -z "$MODEL_PATH" && -d "$DEFAULT_MODEL_PATH" ]]; then
  MODEL_PATH="$DEFAULT_MODEL_PATH"
fi

# You can override this if XTTS/TTS is installed in a different interpreter.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Local vendor path to bypass full /home site-packages (contains python-multipart).
export PYTHONPATH="$ROOT_DIR/.vendor:${PYTHONPATH:-}"

SHOULD_CHECK_TTS="false"
case "${VOICELAB_ENABLE_TTS,,}" in
  1|true|yes|on)
    SHOULD_CHECK_TTS="true"
    ;;
  auto)
    if [[ -n "$MODEL_PATH" || -n "$XTTS_CHECKPOINT" || -n "$XTTS_CONFIG" || -n "$XTTS_VOCAB" ]]; then
      SHOULD_CHECK_TTS="true"
    fi
    ;;
esac

if [[ "$SHOULD_CHECK_TTS" == "true" ]] && ! "$PYTHON_BIN" - <<'PY'
import importlib.util
ok = importlib.util.find_spec("TTS") is not None
raise SystemExit(0 if ok else 1)
PY
then
  echo "[ERROR] XTTS/TTS module is not importable in this interpreter: $PYTHON_BIN"
  echo "        Set PYTHON_BIN to the interpreter that has coqui-tts installed."
  exit 1
fi

cmd=("$PYTHON_BIN" main.py --host "$HOST" --port "$PORT")
if [[ -n "$MODEL_PATH" ]]; then
  cmd+=(--model-path "$MODEL_PATH")
fi
if [[ -n "$XTTS_CHECKPOINT" ]]; then
  cmd+=(--xtts-checkpoint "$XTTS_CHECKPOINT")
fi
if [[ -n "$XTTS_CONFIG" ]]; then
  cmd+=(--xtts-config "$XTTS_CONFIG")
fi
if [[ -n "$XTTS_VOCAB" ]]; then
  cmd+=(--xtts-vocab "$XTTS_VOCAB")
fi

exec "${cmd[@]}"
