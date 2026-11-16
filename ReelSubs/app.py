#!/usr/bin/env python3
"""Desktop video transcription and speaker diarization app."""

from __future__ import annotations

import queue
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

WHISPER_MODEL_OPTIONS = ("tiny", "base", "small", "medium", "large")

COLORS = {
    "background": "#511b19",
    "panel": "#381715",
    "field": "#211512",
    "cream": "#ffe5b5",
    "muted": "#d9b795",
    "gold": "#e9a048",
    "gold_hover": "#f4b65f",
    "border": "#a46138",
    "disabled": "#9b7460",
}


def format_srt_time(seconds: float) -> str:
    """Format seconds as an SRT timestamp, rounding to the nearest millisecond."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def speaker_for_segment(
    start: float, end: float, turns: list[tuple[float, float, str]]
) -> str:
    """Return the speaker with the greatest time overlap with a transcript segment."""
    best_speaker = "UNKNOWN"
    greatest_overlap = 0.0
    for turn_start, turn_end, speaker in turns:
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap > greatest_overlap:
            greatest_overlap = overlap
            best_speaker = speaker
    return best_speaker


def build_srt(
    segments: list[dict[str, Any]], turns: list[tuple[float, float, str]]
) -> str:
    entries = []
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        text = str(segment["text"]).strip()
        speaker = speaker_for_segment(start, end, turns)
        entries.append(
            f"{len(entries) + 1}\n"
            f"{format_srt_time(start)} --> {format_srt_time(end)}\n"
            f"[{speaker}] {text}\n"
        )
    return "\n".join(entries)


def translate_to_english(model: Any, audio_path: Path) -> dict[str, Any]:
    return model.transcribe(str(audio_path), task="translate")


def subtitle_output_paths(
    videos: list[Path], output_directory: Path | None = None
) -> list[Path]:
    outputs = [
        (output_directory or video.parent) / f"{video.stem}.srt"
        for video in videos
    ]
    if len(set(outputs)) != len(outputs):
        raise ValueError(
            "The selected media files would create duplicate subtitle filenames. "
            "Choose a different output folder or remove duplicate filenames."
        )
    return outputs


def load_saved_token(token_path: Path | None = None) -> str:
    if token_path is None:
        token_path = Path.home() / ".config" / "ReelSubs" / "huggingface_token"
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def is_model_access_accepted(
    acceptance_path: Path | None = None,
) -> bool:
    if acceptance_path is None:
        acceptance_path = (
            Path.home() / ".config" / "ReelSubs" / "model_access_accepted"
        )
    try:
        return acceptance_path.read_text(encoding="utf-8").strip() == "accepted"
    except FileNotFoundError:
        return False


def save_model_access_acceptance(
    accepted: bool, acceptance_path: Path | None = None
) -> None:
    if acceptance_path is None:
        acceptance_path = (
            Path.home() / ".config" / "ReelSubs" / "model_access_accepted"
        )
    if accepted:
        acceptance_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        acceptance_path.write_text("accepted\n", encoding="utf-8")
        acceptance_path.chmod(0o600)
    elif acceptance_path.exists():
        acceptance_path.unlink()


def ensure_torchaudio_legacy_io() -> None:
    import soundfile
    import torch
    import torchaudio
    from numpy import ascontiguousarray
    from types import SimpleNamespace

    if hasattr(torchaudio, "list_audio_backends"):
        return

    def audio_info(file: Any, backend: str | None = None) -> SimpleNamespace:
        info = soundfile.info(file)
        subtype_bits = {
            "PCM_S8": 8,
            "PCM_16": 16,
            "PCM_24": 24,
            "PCM_32": 32,
            "FLOAT": 32,
            "DOUBLE": 64,
            "ULAW": 8,
            "ALAW": 8,
        }
        return SimpleNamespace(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=subtype_bits.get(info.subtype, 0),
            encoding=info.subtype,
        )

    def audio_load(
        file: Any,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: str | None = None,
        buffer_size: int = 4096,
        backend: str | None = None,
    ) -> tuple[Any, int]:
        waveform, sample_rate = soundfile.read(
            file,
            start=frame_offset,
            frames=num_frames,
            dtype="float32",
            always_2d=True,
        )
        if channels_first:
            waveform = waveform.T
        return torch.from_numpy(ascontiguousarray(waveform)), sample_rate

    torchaudio.list_audio_backends = lambda: ["soundfile"]
    torchaudio.info = audio_info
    torchaudio.load = audio_load
    torchaudio.AudioMetaData = SimpleNamespace


def ensure_huggingface_hub_auth_compatibility() -> None:
    import inspect
    import huggingface_hub

    if getattr(huggingface_hub, "_reelsubs_auth_compatibility_applied", False):
        return

    hf_hub_download = huggingface_hub.hf_hub_download
    parameters = inspect.signature(hf_hub_download).parameters
    legacy_auth_parameters = {"use_auth_token", "user_auth_token"}
    if legacy_auth_parameters.issubset(parameters):
        return
    if "token" not in parameters:
        raise RuntimeError(
            "The installed huggingface_hub version does not support the token "
            "argument required to load the pyannote model."
        )

    def compatible_hf_hub_download(*args: Any, **kwargs: Any) -> Any:
        legacy_tokens = [
            kwargs.pop(name)
            for name in legacy_auth_parameters
            if name in kwargs
        ]
        if len(legacy_tokens) > 1:
            raise TypeError("Only one legacy Hugging Face auth argument can be used.")
        if legacy_tokens:
            if "token" in kwargs:
                raise TypeError(
                    "Specify either token or a legacy Hugging Face auth argument, "
                    "not both."
                )
            kwargs["token"] = legacy_tokens[0]
        return hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = compatible_hf_hub_download
    setattr(huggingface_hub, "_reelsubs_auth_compatibility_applied", True)


def pyannote_model_access_error(error: AttributeError) -> str | None:
    if "'NoneType' object has no attribute 'eval'" not in str(error):
        return None
    return (
        "A required pyannote model could not be loaded. Accept the access "
        "conditions for pyannote/segmentation-3.0 at "
        "https://huggingface.co/pyannote/segmentation-3.0, then make sure the "
        "Hugging Face token entered in ReelSubs belongs to that account and try "
        "again. The speaker-diarization-3.1 model access must also be accepted."
    )


def ensure_pyannote_checkpoint_compatibility() -> None:
    import torch
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    torch.serialization.add_safe_globals(
        [
            torch.torch_version.TorchVersion,
            Problem,
            Resolution,
            Specifications,
        ]
    )


class ReelSubsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ReelSubs")
        self.icon = tk.PhotoImage(file=Path(__file__).parent / "icons" / "reelsubs.png")
        self.root.iconphoto(True, self.icon)
        self.root.minsize(760, 700)
        self.root.geometry("920x820")
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.video_paths: list[Path] = []
        self.output_directory = tk.StringVar()
        self.model_name = tk.StringVar(value="medium")
        self.token = tk.StringVar(value=load_saved_token())
        self.model_access_accepted = tk.BooleanVar(
            value=is_model_access_accepted()
        )
        self._build_ui()
        self.root.after(100, self._drain_events)

    def _configure_theme(self) -> None:
        self.root.configure(background=COLORS["background"])
        self.root.option_add("*TCombobox*Listbox.background", COLORS["field"])
        self.root.option_add("*TCombobox*Listbox.foreground", COLORS["cream"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", COLORS["gold"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", COLORS["field"])

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("TkDefaultFont", 10))
        style.configure("TFrame", background=COLORS["background"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["background"], foreground=COLORS["cream"])
        style.configure(
            "Title.TLabel", font=("TkDefaultFont", 25, "bold"), foreground=COLORS["cream"]
        )
        style.configure("Subtitle.TLabel", foreground=COLORS["muted"])
        style.configure(
            "Section.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["gold"],
            font=("TkDefaultFont", 11, "bold"),
        )
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["cream"])
        style.configure("Hint.TLabel", background=COLORS["panel"], foreground=COLORS["muted"])
        style.configure(
            "TButton", background=COLORS["panel"], foreground=COLORS["cream"],
            bordercolor=COLORS["border"], lightcolor=COLORS["panel"],
            darkcolor=COLORS["panel"], padding=(12, 7),
        )
        style.map(
            "TButton",
            background=[("disabled", COLORS["panel"]), ("active", COLORS["background"])],
            foreground=[("disabled", COLORS["disabled"])],
        )
        style.configure(
            "Accent.TButton", background=COLORS["gold"], foreground=COLORS["field"],
            bordercolor=COLORS["gold"], lightcolor=COLORS["gold"],
            darkcolor=COLORS["gold"], font=("TkDefaultFont", 10, "bold"),
            padding=(18, 9),
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", COLORS["border"]), ("active", COLORS["gold_hover"])],
            foreground=[("disabled", COLORS["panel"]), ("!disabled", COLORS["field"])],
        )
        style.configure(
            "TEntry", fieldbackground=COLORS["field"], foreground=COLORS["cream"],
            insertcolor=COLORS["cream"], bordercolor=COLORS["border"], padding=6,
        )
        style.configure(
            "TCombobox", fieldbackground=COLORS["field"], foreground=COLORS["cream"],
            background=COLORS["panel"], arrowcolor=COLORS["gold"],
            bordercolor=COLORS["border"], padding=5,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", COLORS["field"])],
            foreground=[("readonly", COLORS["cream"])],
            selectbackground=[("readonly", COLORS["field"])],
            selectforeground=[("readonly", COLORS["cream"])],
        )
        style.configure(
            "TCheckbutton", background=COLORS["panel"], foreground=COLORS["cream"],
            indicatorbackground=COLORS["field"], indicatorcolor=COLORS["gold"],
        )
        style.map(
            "TCheckbutton",
            background=[("active", COLORS["panel"])],
            foreground=[("disabled", COLORS["disabled"])],
            indicatorbackground=[("selected", COLORS["gold"]), ("!selected", COLORS["field"])],
            indicatorcolor=[("selected", COLORS["field"])],
        )
        style.configure(
            "Horizontal.TProgressbar", troughcolor=COLORS["field"],
            background=COLORS["gold"], bordercolor=COLORS["panel"],
            lightcolor=COLORS["gold"], darkcolor=COLORS["gold"],
        )
        style.configure(
            "Vertical.TScrollbar", background=COLORS["border"],
            troughcolor=COLORS["field"], arrowcolor=COLORS["cream"],
            bordercolor=COLORS["field"], lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
        )
        style.map(
            "Vertical.TScrollbar", background=[("active", COLORS["gold"])],
            lightcolor=[("active", COLORS["gold"])], darkcolor=[("active", COLORS["gold"])],
        )

    def _build_ui(self) -> None:
        self._configure_theme()
        frame = ttk.Frame(self.root, padding=(22, 16))
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=3)
        frame.rowconfigure(3, weight=2)

        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        self.header_icon = self.icon.subsample(15, 15)
        ttk.Label(header, image=self.header_icon).grid(row=0, column=0, rowspan=2, padx=(0, 14))
        ttk.Label(header, text="ReelSubs", style="Title.TLabel").grid(
            row=0, column=1, sticky="sw"
        )
        ttk.Label(
            header, text="SPEAKER-LABELED SUBTITLES  /  ALWAYS IN ENGLISH",
            style="Subtitle.TLabel",
        ).grid(row=1, column=1, sticky="nw", pady=(2, 0))

        queue_frame = ttk.Frame(frame, style="Panel.TFrame", padding=14)
        queue_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(1, weight=1)
        queue_header = ttk.Frame(queue_frame, style="Panel.TFrame")
        queue_header.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        queue_header.columnconfigure(0, weight=1)
        ttk.Label(queue_header, text="01  /  MEDIA QUEUE", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(queue_header, text="Add files…", command=self._choose_input).grid(
            row=0, column=1, padx=(8, 0)
        )
        ttk.Button(queue_header, text="Remove selected", command=self._remove_selected).grid(
            row=0, column=2, padx=(8, 0)
        )
        ttk.Button(queue_header, text="Clear list", command=self._clear_list).grid(
            row=0, column=3, padx=(8, 0)
        )
        queue_list = ttk.Frame(queue_frame, style="Panel.TFrame")
        queue_list.grid(row=1, column=0, sticky="nsew")
        queue_list.columnconfigure(0, weight=1)
        queue_list.rowconfigure(0, weight=1)
        self.media_list = tk.Listbox(
            queue_list, height=4, selectmode=tk.EXTENDED, exportselection=False,
            background=COLORS["field"], foreground=COLORS["cream"],
            selectbackground=COLORS["gold"], selectforeground=COLORS["field"],
            highlightthickness=1, highlightbackground=COLORS["border"],
            highlightcolor=COLORS["gold"], borderwidth=0,
            activestyle="dotbox", font=("TkDefaultFont", 10),
        )
        self.media_list.grid(row=0, column=0, sticky="nsew")
        queue_scroll = ttk.Scrollbar(queue_list, orient="vertical", command=self.media_list.yview)
        queue_scroll.grid(row=0, column=1, sticky="ns")
        self.media_list.configure(yscrollcommand=queue_scroll.set)

        settings = ttk.Frame(frame, style="Panel.TFrame", padding=14)
        settings.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="02  /  SUBTITLE SETTINGS", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        ttk.Label(settings, text="Save subtitles in", style="Panel.TLabel").grid(
            row=1, column=0, sticky="w", pady=4
        )
        ttk.Entry(settings, textvariable=self.output_directory).grid(
            row=1, column=1, sticky="ew", padx=12
        )
        ttk.Button(settings, text="Browse…", command=self._choose_output).grid(
            row=1, column=2
        )
        ttk.Label(
            settings, text="Leave blank to save each SRT beside its media file.",
            style="Hint.TLabel",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=12, pady=(0, 4))

        ttk.Label(settings, text="Whisper model", style="Panel.TLabel").grid(
            row=3, column=0, sticky="w", pady=4
        )
        ttk.Combobox(
            settings,
            textvariable=self.model_name,
            values=WHISPER_MODEL_OPTIONS,
            state="readonly",
            width=18,
        ).grid(row=3, column=1, sticky="w", padx=12)

        ttk.Label(settings, text="Hugging Face token", style="Panel.TLabel").grid(
            row=4, column=0, sticky="w", pady=4
        )
        ttk.Entry(settings, textvariable=self.token, show="•").grid(
            row=4, column=1, sticky="ew", padx=12
        )
        ttk.Checkbutton(
            settings,
            text="I have accepted the pyannote model conditions on Hugging Face",
            variable=self.model_access_accepted,
            command=self._save_model_access_acceptance,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

        activity = ttk.Frame(frame, style="Panel.TFrame", padding=14)
        activity.grid(row=3, column=0, sticky="nsew")
        activity.columnconfigure(0, weight=1)
        activity.rowconfigure(1, weight=1)
        ttk.Label(activity, text="03  /  ACTIVITY", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.log = tk.Text(
            activity, height=4, wrap="word", state="disabled",
            background=COLORS["field"], foreground=COLORS["cream"],
            insertbackground=COLORS["cream"], selectbackground=COLORS["gold"],
            selectforeground=COLORS["field"], highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["gold"],
            borderwidth=0, padx=8, pady=6, font=("TkDefaultFont", 10),
        )
        self.log.grid(row=1, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(activity, orient="vertical", command=self.log.yview)
        log_scroll.grid(row=1, column=1, sticky="ns")
        self.log.configure(yscrollcommand=log_scroll.set)

        footer = ttk.Frame(frame)
        footer.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 18))
        self.start_button = ttk.Button(
            footer, text="Process media queue", command=self._start, style="Accent.TButton"
        )
        self.start_button.grid(row=0, column=1, sticky="e")

    def _choose_input(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Add media files",
            filetypes=(
                (
                    "Media files",
                    "*.mp4 *.mkv *.mov *.avi *.webm *.m4v "
                    "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma",
                ),
                ("All files", "*"),
            ),
        )
        for path in selected:
            video = Path(path)
            if video not in self.video_paths:
                self.video_paths.append(video)
                self.media_list.insert("end", str(video))

    def _remove_selected(self) -> None:
        selected = list(self.media_list.curselection())
        for index in reversed(selected):
            self.media_list.delete(index)
            del self.video_paths[index]

    def _clear_list(self) -> None:
        self.video_paths.clear()
        self.media_list.delete(0, "end")

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose a folder for the generated SRT files",
            mustexist=True,
        )
        if selected:
            self.output_directory.set(selected)

    def _save_model_access_acceptance(self) -> None:
        try:
            save_model_access_acceptance(self.model_access_accepted.get())
        except OSError as error:
            messagebox.showerror(
                "Could not save model access status",
                f"Unable to update the local acceptance record: {error}",
            )
            self.model_access_accepted.set(
                is_model_access_accepted()
            )

    def _start(self) -> None:
        videos = [video.expanduser() for video in self.video_paths]
        output_directory_text = self.output_directory.get().strip()
        output_directory = (
            Path(output_directory_text).expanduser()
            if output_directory_text
            else None
        )
        token = self.token.get().strip()
        if not videos:
            messagebox.showerror("Add media", "Add at least one media file.")
            return
        missing_videos = [str(video) for video in videos if not video.is_file()]
        if missing_videos:
            messagebox.showerror(
                "Media files not found",
                "These files are no longer available:\n" + "\n".join(missing_videos),
            )
            return
        if output_directory is not None and not output_directory.is_dir():
            messagebox.showerror(
                "Choose an output folder",
                "The subtitle output folder must already exist.",
            )
            return
        try:
            outputs = subtitle_output_paths(videos, output_directory)
        except ValueError as error:
            messagebox.showerror("Duplicate subtitle filenames", str(error))
            return
        if not self.model_access_accepted.get():
            messagebox.showerror(
                "Model conditions not confirmed",
                "Accept the model conditions on Hugging Face, then check the "
                "confirmation box in ReelSubs.",
            )
            return
        if not token:
            messagebox.showerror(
                "Hugging Face token required",
                "Enter a Hugging Face token with access to the diarization model.",
            )
            return
        if shutil.which("ffmpeg") is None:
            messagebox.showerror(
                "FFmpeg not found",
                "Install FFmpeg and make sure the ffmpeg command is on PATH.",
            )
            return

        self.start_button.configure(state="disabled")
        self.progress.start(10)
        self._write_log(f"Starting processing for {len(videos)} media file(s)…")
        threading.Thread(
            target=self._process_video,
            args=(list(zip(videos, outputs)), token, self.model_name.get()),
            daemon=True,
        ).start()

    def _process_video(
        self,
        jobs: list[tuple[Path, Path]],
        token: str,
        model_name: str,
    ) -> None:
        try:
            import whisper
            ensure_torchaudio_legacy_io()
            ensure_huggingface_hub_auth_compatibility()
            ensure_pyannote_checkpoint_compatibility()
            from pyannote.audio import Pipeline

            self.events.put(("log", f"Loading Whisper model: {model_name}…"))
            model = whisper.load_model(model_name)
            self.events.put(("log", "Loading speaker diarization model…"))
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token,
                )
            except AttributeError as error:
                message = pyannote_model_access_error(error)
                if message is None:
                    raise
                raise RuntimeError(message) from error
            if pipeline is None:
                raise RuntimeError(
                    "Could not load the diarization model. Check your Hugging Face "
                    "token and model access."
                )

            failures: list[str] = []
            for index, (video, output) in enumerate(jobs, start=1):
                self.events.put(
                    ("log", f"Processing file {index}/{len(jobs)}: {video}")
                )
                try:
                    with tempfile.TemporaryDirectory(prefix="reelsubs-") as temp_dir:
                        audio_path = Path(temp_dir) / "audio.wav"
                        try:
                            subprocess.run(
                                [
                                    "ffmpeg",
                                    "-y",
                                    "-i",
                                    str(video),
                                    "-vn",
                                    "-acodec",
                                    "pcm_s16le",
                                    "-ar",
                                    "16000",
                                    "-ac",
                                    "1",
                                    str(audio_path),
                                ],
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                        except subprocess.CalledProcessError as error:
                            details = (
                                error.stderr.strip() if error.stderr else str(error)
                            )
                            raise RuntimeError(
                                f"FFmpeg audio extraction failed: {details}"
                            ) from error

                        self.events.put(
                            ("log", f"Translating to English: {video.name}…")
                        )
                        result = translate_to_english(model, audio_path)
                        self.events.put(("log", f"Identifying speakers in {video.name}…"))
                        diarization = pipeline(str(audio_path))
                        turns = [
                            (turn.start, turn.end, label)
                            for turn, _, label in diarization.itertracks(
                                yield_label=True
                            )
                        ]
                        output.write_text(
                            build_srt(result["segments"], turns), encoding="utf-8"
                        )
                    self.events.put(("log", f"Created subtitles: {output}"))
                except Exception as error:
                    failure = f"{video}: {type(error).__name__}: {error}"
                    failures.append(failure)
                    self.events.put(("log", f"Failed: {failure}"))

            processed = len(jobs) - len(failures)
            if failures:
                self.events.put(
                    (
                        "error",
                        f"Processed {processed} of {len(jobs)} media file(s). "
                        f"{len(failures)} failed:\n" + "\n".join(failures),
                    )
                )
            else:
                self.events.put(
                    (
                        "done",
                        f"Created subtitles for all {processed} media file(s).",
                    )
                )
        except Exception as error:
            self.events.put(("error", f"{type(error).__name__}: {error}"))

    def _write_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_events(self) -> None:
        try:
            while True:
                kind, message = self.events.get_nowait()
                self._write_log(message)
                if kind in ("done", "error"):
                    self.progress.stop()
                    self.start_button.configure(state="normal")
                    if kind == "done":
                        messagebox.showinfo("Complete", message)
                    else:
                        messagebox.showerror("Processing failed", message)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)


def main() -> None:
    root = tk.Tk()
    ReelSubsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
