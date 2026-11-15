# ReelSubs

ReelSubs is a Linux desktop app that transcribes video and creates speaker-labeled
SRT subtitles with OpenAI Whisper and pyannote.audio.

## Features

- Choose a video and where to save the resulting SRT.
- Select a Whisper model for transcription.
- Label transcript segments with diarized speakers.
- Process in the background while keeping the desktop window responsive.

## Requirements

- Fedora or another `dnf`-based Linux distribution.
- Internet access for dependencies and first-run model downloads.
- A Hugging Face account with access to
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

## Installation

Clone the repository, then run the installer:

```bash
git clone https://github.com/YOUR-USERNAME/ReelSubs.git
cd ReelSubs
./install.sh
```

The installer uses `sudo` to install system packages and creates a virtual
environment in `~/video-forensics-ai`. Run the app with:

```bash
source ~/video-forensics-ai/venv/bin/activate
python ~/video-forensics-ai/app.py
```

## First use

1. Accept the access conditions for
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
2. Create a Hugging Face access token.
3. Open ReelSubs, choose a video, an SRT destination, and a Whisper model.
4. Enter your Hugging Face token when prompted in the app.

The token is used for the current run and is not saved. Model weights are
downloaded by their libraries the first time they are needed.

## Tests

The helper tests use Python's standard library:

```bash
python3 -m unittest -v
```

## Notes

- The included installer targets Fedora and other `dnf`-based systems.
- PyTorch and model downloads are large. GPU acceleration depends on the
  installed PyTorch build and available hardware.
- Speaker labels are assigned by the greatest time overlap between a transcript
  segment and diarization turns.
