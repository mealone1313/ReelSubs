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


class VideoForensicsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ReelSubs")
        self.root.minsize(620, 480)
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_name = tk.StringVar(value="medium")
        self.token = tk.StringVar()
        self._build_ui()
        self.root.after(100, self._drain_events)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text="Video file").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.input_path).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(frame, text="Browse…", command=self._choose_input).grid(
            row=0, column=2
        )

        ttk.Label(frame, text="Save SRT as").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.output_path).grid(
            row=1, column=1, sticky="ew", padx=8
        )
        ttk.Button(frame, text="Browse…", command=self._choose_output).grid(
            row=1, column=2
        )

        ttk.Label(frame, text="Whisper model").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Combobox(
            frame,
            textvariable=self.model_name,
            values=("tiny", "base", "small", "medium", "large", "turbo"),
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky="w", padx=8)

        ttk.Label(frame, text="Hugging Face token").grid(
            row=3, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.token, show="•").grid(
            row=3, column=1, sticky="ew", padx=8
        )
        ttk.Label(
            frame,
            text="Accept access to pyannote/speaker-diarization-3.1 on Hugging Face.",
            wraplength=430,
        ).grid(row=4, column=1, sticky="w", padx=8)

        self.log = tk.Text(frame, height=12, wrap="word", state="disabled")
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(14, 8))
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=4)
        self.start_button = ttk.Button(
            frame, text="Create subtitles", command=self._start
        )
        self.start_button.grid(row=7, column=0, columnspan=3, pady=(8, 0))

    def _choose_input(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=(
                ("Video files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"),
                ("All files", "*"),
            ),
        )
        if selected:
            self.input_path.set(selected)
            if not self.output_path.get():
                self.output_path.set(str(Path(selected).with_suffix(".srt")))

    def _choose_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Save subtitles",
            defaultextension=".srt",
            filetypes=(("SubRip subtitles", "*.srt"), ("All files", "*")),
        )
        if selected:
            self.output_path.set(selected)

    def _start(self) -> None:
        video = Path(self.input_path.get()).expanduser()
        output = Path(self.output_path.get()).expanduser()
        token = self.token.get().strip()
        if not video.is_file():
            messagebox.showerror("Choose a video", "Select an existing video file.")
            return
        if not output.name:
            messagebox.showerror("Choose an output", "Choose where to save the SRT.")
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
        self._write_log("Starting transcription and speaker diarization…")
        threading.Thread(
            target=self._process_video,
            args=(video, output, token, self.model_name.get()),
            daemon=True,
        ).start()

    def _process_video(
        self, video: Path, output: Path, token: str, model_name: str
    ) -> None:
        try:
            import whisper
            from pyannote.audio import Pipeline

            self.events.put(("log", "Extracting audio with FFmpeg…"))
            with tempfile.TemporaryDirectory(prefix="video-forensics-") as temp_dir:
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
                    details = error.stderr.strip() if error.stderr else str(error)
                    raise RuntimeError(f"FFmpeg audio extraction failed: {details}") from error

                self.events.put(("log", f"Loading Whisper model: {model_name}…"))
                model = whisper.load_model(model_name)
                self.events.put(("log", "Transcribing audio…"))
                result = model.transcribe(str(audio_path))

                self.events.put(("log", "Loading speaker diarization model…"))
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token,
                )
                if pipeline is None:
                    raise RuntimeError(
                        "Could not load the diarization model. Check your Hugging Face "
                        "token and model access."
                    )
                self.events.put(("log", "Identifying speakers…"))
                diarization = pipeline(str(audio_path))
                turns = [
                    (turn.start, turn.end, label)
                    for turn, _, label in diarization.itertracks(yield_label=True)
                ]

                self.events.put(("log", "Writing subtitle file…"))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    build_srt(result["segments"], turns), encoding="utf-8"
                )
            self.events.put(("done", f"Created subtitles: {output}"))
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
    VideoForensicsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
