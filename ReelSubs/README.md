# ReelSubs

ReelSubs is a Linux desktop app that extracts audio from media files, translates
speech into English with OpenAI Whisper, matches transcript segments to pyannote
speaker turns, and saves speaker-labeled SRT subtitle files.

## Features

- Queue multiple video or audio files and process them sequentially.
- Save each SRT beside its media file or choose a shared output folder.
- Translate speech into English with a selectable Whisper model.
- Add speaker labels using pyannote diarization.
- Use a cinema-inspired burgundy, cream, and gold interface.
- Keep processing in the background while the desktop window remains responsive.
- Launch ReelSubs from your desktop applications menu with its custom icon.

## Requirements

- Fedora or another `dnf`-based Linux distribution.
- Internet access for installing packages and downloading model weights.
- Several gigabytes of free disk space for Python packages and model weights.
- Python dependencies are installed in an isolated virtual environment, with
  pyannote.audio 3.x for the supported pipeline API.
- A Hugging Face account and access to both
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

## Download and install

Download the source from GitHub using **Code → Download ZIP**, extract it, open
a terminal in the extracted folder, and run:

```bash
./install.sh
```

Alternatively, if you have a `ReelSubs-linux.tar.gz` installer bundle, extract
and install it with:

```bash
tar -xzf ReelSubs-linux.tar.gz
cd ReelSubs
./install.sh
```

The installer installs system dependencies and Python packages, creates
`~/ReelSubs`, and adds ReelSubs to your desktop applications menu. It uses
`sudo` to install Fedora packages and may prompt for your system password. You
can also start the app from a terminal:

```bash
~/ReelSubs/launch.sh
```

The download is an installer bundle, not a fully offline standalone executable:
installation needs internet access, and the large Whisper and pyannote models
are downloaded on first use.

To create a Linux installer bundle from a checkout, run `./package.sh`. It
creates `dist/ReelSubs-linux.tar.gz`; the generated `dist/` folder is not
included in Git. Attach the archive to a GitHub release if you want to offer it
as a separate download.

The installer asks whether you accepted the model conditions and records your
confirmation in `~/.config/ReelSubs/model_access_accepted`. It also offers to
save an optional Hugging Face access token in
`~/.config/ReelSubs/huggingface_token` with owner-only permissions. Press Enter
at the token prompt to skip saving a token and enter one in the app later.

## First use

1. Sign in to Hugging Face and accept the model access conditions on both the
   [pyannote/speaker-diarization-3.1 page](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and the
   [pyannote/segmentation-3.0 page](https://huggingface.co/pyannote/segmentation-3.0).
   The installer cannot accept them for you; it only records your confirmation.
2. Create a Hugging Face access token for the account that has model access.
3. Open ReelSubs, add media files, choose an output folder if desired, select a
   Whisper model, and use the saved token or enter one.

ReelSubs processes queued files one by one and reports any individual failures
after continuing with the rest of the queue. It shows the saved model access
confirmation and requires it before starting diarization. The confirmation is
stored only on this computer; Hugging Face still verifies model access using
your account and token.

Model weights are downloaded by their respective libraries on first use.
ReelSubs pins pyannote.audio to its supported 3.x pipeline API and includes a
compatibility adapter for newer torchaudio versions that removed legacy
audio-backend APIs. The adapter uses SoundFile for the PCM WAV files prepared
for processing. It also maps pyannote's legacy Hugging Face authentication
arguments to the current Hugging Face Hub `token` argument and allowlists the
pyannote checkpoint metadata types required by recent PyTorch safe loading.

## Run tests

The tests use only the Python standard library. Interface tests need an
available graphical display and are skipped if one is unavailable:

```bash
python3 -m unittest -v
```

## Notes

- PyTorch and model downloads are large. GPU acceleration depends on the
  installed PyTorch build and available hardware.
- Speaker labels are assigned to each transcript segment by maximum time
  overlap with the diarization turns.
- Whisper's `turbo` model is not offered because it does not support translation
  into English.
- Translation and speaker labels are model-generated and may contain errors.

## License

ReelSubs is available under the [MIT License](LICENSE). Dependencies and
downloaded models have their own licenses and access conditions.
