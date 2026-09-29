#!/usr/bin/env bash
# Launch the FLUX.2 Klein Alpha Gradio demo.
#
#   AITK_PATH=/path/to/ai-toolkit ./run.sh           -> http://127.0.0.1:7860
#   ./run.sh --host 0.0.0.0 --port 7870               -> reachable from other machines on your network
#   PYTHON=/path/to/ai-toolkit/venv/bin/python ./run.sh
#   CUDA_VISIBLE_DEVICES=1 ./run.sh                   -> use another GPU (default: GPU 0)
#   PRELOAD_9B=0 ./run.sh                             -> load the 9B on the first generation request
#   VAE_PATH=... EXTRACT_LORA=... REMOVE_LORA=... ./run.sh   -> use local weights instead of Hugging Face
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GRADIO_ANALYTICS_ENABLED=False
PYTHON="${PYTHON:-python}"
exec "$PYTHON" app.py "$@"
