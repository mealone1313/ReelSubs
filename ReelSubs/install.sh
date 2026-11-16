#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${HOME}/ReelSubs"
TOKEN_DIR="${HOME}/.config/ReelSubs"
TOKEN_FILE="${TOKEN_DIR}/huggingface_token"
MODEL_ACCESS_FILE="${TOKEN_DIR}/model_access_accepted"

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
install -d -m 755 "$PROJECT_DIR/icons"
install -m 644 "$SCRIPT_DIR/icons/reelsubs.png" "$PROJECT_DIR/icons/reelsubs.png"

cd "$PROJECT_DIR"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchaudio
python -m pip install -r requirements.txt

mkdir -p input output cache

cat > "$PROJECT_DIR/launch.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_DIR"
exec "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/app.py" "\$@"
EOF
chmod 755 "$PROJECT_DIR/launch.sh"

APPLICATIONS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
install -d -m 755 "$APPLICATIONS_DIR"
python3 - "$PROJECT_DIR/launch.sh" "$PROJECT_DIR/icons/reelsubs.png" "$APPLICATIONS_DIR/reelsubs.desktop" <<'PY'
import sys
from pathlib import Path


def quote_exec_arg(argument: str) -> str:
    escaped = (
        argument.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
        .replace("%", "%%")
    )
    return f'"{escaped}"'


def desktop_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


launcher, icon, desktop_file = map(Path, sys.argv[1:])
desktop_file.write_text(
    "[Desktop Entry]\n"
    "Type=Application\n"
    "Name=ReelSubs\n"
    "Comment=Transcribe media and create speaker-labeled subtitles\n"
    f"Exec={quote_exec_arg(str(launcher))}\n"
    f"Icon={desktop_string(str(icon))}\n"
    "Terminal=false\n"
    "Categories=AudioVideo;Video;\n",
    encoding="utf-8",
)
PY

cat <<EOF
Before configuring your token, sign in to Hugging Face and accept the access
conditions for both models:
  https://huggingface.co/pyannote/speaker-diarization-3.1
  https://huggingface.co/pyannote/segmentation-3.0

The installer cannot accept these conditions for you. The token must belong to
an account that has been granted access to the model.

EOF

read -r -p "Have you accepted both model conditions on Hugging Face? [y/N] " MODEL_ACCEPTED
if [[ "$MODEL_ACCEPTED" =~ ^[Yy]$ ]]; then
    install -d -m 700 "$TOKEN_DIR"
    (umask 077; printf 'accepted\n' > "$MODEL_ACCESS_FILE")
    chmod 600 "$MODEL_ACCESS_FILE"
    MODEL_ACCESS_STATUS="Model access acceptance recorded locally."
else
    if [[ -e "$MODEL_ACCESS_FILE" ]]; then
        rm -- "$MODEL_ACCESS_FILE"
    fi
    MODEL_ACCESS_STATUS="Model access acceptance not recorded; confirm it in the app after accepting on Hugging Face."
fi

printf "Hugging Face access token (press Enter to configure it later): "
if ! read -r -s HF_TOKEN; then
    printf "\nUnable to read the Hugging Face token.\n" >&2
    exit 1
fi
printf "\n"

if [[ -n "$HF_TOKEN" ]]; then
    install -d -m 700 "$TOKEN_DIR"
    (umask 077; printf '%s\n' "$HF_TOKEN" > "$TOKEN_FILE")
    chmod 600 "$TOKEN_FILE"
    unset HF_TOKEN
    TOKEN_STATUS="Saved the token in $TOKEN_FILE with owner-only permissions."
else
    TOKEN_STATUS="No token saved; enter one in the app when you use diarization."
fi

cat <<EOF

Installation complete.

Project: $PROJECT_DIR
$MODEL_ACCESS_STATUS
$TOKEN_STATUS
Launch from the desktop applications menu or run:
  "$PROJECT_DIR/launch.sh"

EOF
