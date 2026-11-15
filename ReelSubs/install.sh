#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${HOME}/video-forensics-ai"

if ! command -v dnf >/dev/null 2>&1; then
    echo "This installer requires Fedora or another dnf-based distribution." >&2
    exit 1
fi

sudo dnf install -y \
    ffmpeg \
    python3 \
    python3-pip \
    python3-devel \
    python3-tkinter \
    gcc \
    gcc-c++ \
    cmake \
    make

mkdir -p "$PROJECT_DIR"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/app.py" "$SCRIPT_DIR/requirements.txt" "$PROJECT_DIR/"

cd "$PROJECT_DIR"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchaudio
python -m pip install -r requirements.txt

mkdir -p input output cache

cat <<EOF

Installation complete.

Project: $PROJECT_DIR
Run the app:
  source "$PROJECT_DIR/venv/bin/activate"
  python "$PROJECT_DIR/app.py"

Before using diarization, accept the model conditions at:
  https://huggingface.co/pyannote/speaker-diarization-3.1

EOF
