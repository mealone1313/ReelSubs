# ReelSubs

A Linux desktop app that extracts audio from a video, transcribes it with
OpenAI Whisper, matches transcript segments to pyannote speaker turns, and saves
an SRT subtitle file.

## Features

- Browse for a video and choose an SRT output location.
- Transcribe with a selectable Whisper model.
- Add speaker labels using pyannote diarization.
- Keep processing in the background while the desktop window remains responsive.

## Requirements

- Fedora or another `dnf`-based Linux distribution.
- Internet access for installing packages and downloading model weights.
- A Hugging Face account and access to
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

## Install

From the project directory, run the included installer:

```bash
cd /path/to/ReelSubs
./install.sh
```

The installer installs system dependencies and creates
`~/ReelSubs`. Launch the app with:

```bash
source ~/ReelSubs/venv/bin/activate
python ~/ReelSubs/app.py
```

The installer uses `sudo` to install Fedora packages. It may prompt you for your
system password.



## First use

1. Sign in to Hugging Face and accept the model access conditions at
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
2. Create a Hugging Face access token.
3. Open the app, select a video and an SRT destination, choose a Whisper model,
   and enter your token.

The token is used for that run and is not saved to disk. Model weights are
downloaded by their respective libraries on first use.

## Run tests

The helper tests use only the Python standard library:

```bash
python3 -m unittest -v
```

## Notes

- PyTorch and model downloads are large. GPU acceleration depends on the
  installed PyTorch build and available hardware.
- Speaker labels are assigned to each transcript segment by maximum time
  overlap with the diarization turns.
