import unittest
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import tkinter as tk
from tkinter import ttk
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from app import (
    WHISPER_MODEL_OPTIONS,
    ReelSubsApp,
    build_srt,
    ensure_huggingface_hub_auth_compatibility,
    ensure_pyannote_checkpoint_compatibility,
    ensure_torchaudio_legacy_io,
    format_srt_time,
    is_model_access_accepted,
    load_saved_token,
    pyannote_model_access_error,
    save_model_access_acceptance,
    speaker_for_segment,
    subtitle_output_paths,
    translate_to_english,
)


class SubtitleHelpersTests(unittest.TestCase):
    def test_application_sets_its_window_icon(self):
        root = Mock()
        icon = object()
        with (
            patch("app.tk.PhotoImage", return_value=icon) as photo_image,
            patch("app.tk.StringVar"),
            patch("app.tk.BooleanVar"),
            patch("app.load_saved_token", return_value=""),
            patch("app.is_model_access_accepted", return_value=False),
            patch.object(ReelSubsApp, "_build_ui"),
        ):
            application = ReelSubsApp(root)

        photo_image.assert_called_once_with(
            file=Path(__file__).parent / "icons" / "reelsubs.png"
        )
        root.iconphoto.assert_called_once_with(True, icon)
        self.assertIs(application.icon, icon)

    def test_format_srt_time_rounds_milliseconds(self):
        self.assertEqual(format_srt_time(3661.2346), "01:01:01,235")

    def test_speaker_for_segment_uses_greatest_overlap(self):
        turns = [(0.0, 1.0, "SPEAKER_00"), (0.5, 2.0, "SPEAKER_01")]
        self.assertEqual(speaker_for_segment(0.6, 1.4, turns), "SPEAKER_01")

    def test_build_srt_formats_transcript_and_unknown_speaker(self):
        result = build_srt(
            [{"start": 1.0, "end": 2.5, "text": " Hello there. "}],
            [(0.0, 0.5, "SPEAKER_00")],
        )
        self.assertEqual(
            result,
            "1\n00:00:01,000 --> 00:00:02,500\n[UNKNOWN] Hello there.\n",
        )

    def test_subtitles_are_translated_to_english(self):
        model = Mock()
        model.transcribe.return_value = {"segments": []}
        audio_path = Path("/tmp/audio.wav")

        result = translate_to_english(model, audio_path)

        self.assertEqual(result, {"segments": []})
        model.transcribe.assert_called_once_with(
            str(audio_path), task="translate"
        )

    def test_whisper_models_supporting_translation_are_selectable(self):
        self.assertEqual(
            WHISPER_MODEL_OPTIONS, ("tiny", "base", "small", "medium", "large")
        )

    def test_subtitle_output_paths_use_media_folder_by_default(self):
        videos = [Path("/media/first.mp4"), Path("/media/second.wav")]

        self.assertEqual(
            subtitle_output_paths(videos),
            [Path("/media/first.srt"), Path("/media/second.srt")],
        )

    def test_subtitle_output_paths_use_selected_folder(self):
        videos = [Path("/media/first.mp4"), Path("/media/second.wav")]

        self.assertEqual(
            subtitle_output_paths(videos, Path("/output")),
            [Path("/output/first.srt"), Path("/output/second.srt")],
        )

    def test_subtitle_output_paths_reject_duplicate_names(self):
        videos = [Path("/one/clip.mp4"), Path("/two/clip.mkv")]

        with self.assertRaisesRegex(ValueError, "duplicate subtitle filenames"):
            subtitle_output_paths(videos, Path("/output"))

    def test_torchaudio_adapter_restores_legacy_io_apis(self):
        soundfile = ModuleType("soundfile")
        soundfile.info = lambda file: SimpleNamespace(
            samplerate=16000,
            frames=4,
            channels=1,
            subtype="PCM_16",
        )

        class Waveform:
            @property
            def T(self):
                return "transposed-waveform"

        soundfile.read = lambda *args, **kwargs: (Waveform(), 16000)

        torch = ModuleType("torch")
        torch.from_numpy = lambda waveform: ("tensor", waveform)
        torchaudio = ModuleType("torchaudio")
        numpy = ModuleType("numpy")
        numpy.ascontiguousarray = lambda waveform: waveform

        with patch.dict(
            sys.modules,
            {
                "soundfile": soundfile,
                "torch": torch,
                "torchaudio": torchaudio,
                "numpy": numpy,
            },
        ):
            ensure_torchaudio_legacy_io()

        self.assertEqual(torchaudio.list_audio_backends(), ["soundfile"])
        self.assertEqual(torchaudio.info("media.wav").num_frames, 4)
        self.assertEqual(
            torchaudio.load("media.wav"),
            (("tensor", "transposed-waveform"), 16000),
        )

    def test_huggingface_hub_adapter_maps_legacy_auth_arguments(self):
        huggingface_hub = ModuleType("huggingface_hub")
        downloads = []

        def hf_hub_download(repo_id, filename, *, token=None):
            downloads.append((repo_id, filename, token))
            return "downloaded-file"

        huggingface_hub.hf_hub_download = hf_hub_download
        with patch.dict(sys.modules, {"huggingface_hub": huggingface_hub}):
            ensure_huggingface_hub_auth_compatibility()
            self.assertEqual(
                huggingface_hub.hf_hub_download(
                    "model/repo", "config.yaml", user_auth_token="user-token"
                ),
                "downloaded-file",
            )
            self.assertEqual(
                huggingface_hub.hf_hub_download(
                    "model/repo", "weights.bin", use_auth_token="use-token"
                ),
                "downloaded-file",
            )

        self.assertEqual(
            downloads,
            [
                ("model/repo", "config.yaml", "user-token"),
                ("model/repo", "weights.bin", "use-token"),
            ],
        )

    def test_load_saved_token_reads_token_file(self):
        with TemporaryDirectory() as directory:
            token_path = Path(directory) / "huggingface_token"
            token_path.write_text("hf_example_token\n", encoding="utf-8")

            self.assertEqual(load_saved_token(token_path), "hf_example_token")

    def test_load_saved_token_returns_empty_when_missing(self):
        with TemporaryDirectory() as directory:
            token_path = Path(directory) / "missing_token"

            self.assertEqual(load_saved_token(token_path), "")

    def test_model_access_acceptance_is_saved_and_cleared(self):
        with TemporaryDirectory() as directory:
            acceptance_path = Path(directory) / "config" / "model_access_accepted"

            self.assertFalse(is_model_access_accepted(acceptance_path))
            save_model_access_acceptance(True, acceptance_path)
            self.assertTrue(is_model_access_accepted(acceptance_path))
            self.assertEqual(acceptance_path.stat().st_mode & 0o777, 0o600)
            save_model_access_acceptance(False, acceptance_path)
            self.assertFalse(is_model_access_accepted(acceptance_path))

    def test_pyannote_missing_model_error_explains_required_access(self):
        error = AttributeError("'NoneType' object has no attribute 'eval'")

        message = pyannote_model_access_error(error)

        self.assertIn("pyannote/segmentation-3.0", message)
        self.assertIn("Hugging Face token", message)
        self.assertIn("speaker-diarization-3.1", message)

    def test_pyannote_other_attribute_errors_are_not_rewritten(self):
        error = AttributeError("'NoneType' object has no attribute 'to'")

        self.assertIsNone(pyannote_model_access_error(error))

    def test_pyannote_checkpoint_compatibility_allowlists_checkpoint_types(self):
        torch = ModuleType("torch")
        torch.torch_version = SimpleNamespace(TorchVersion=type("TorchVersion", (), {}))
        torch.serialization = SimpleNamespace(add_safe_globals=Mock())
        package_modules = {
            "pyannote": ModuleType("pyannote"),
            "pyannote.audio": ModuleType("pyannote.audio"),
            "pyannote.audio.core": ModuleType("pyannote.audio.core"),
            "pyannote.audio.core.task": ModuleType("pyannote.audio.core.task"),
        }
        task_types = {
            name: type(name, (), {})
            for name in ("Problem", "Resolution", "Specifications")
        }
        for name, value in task_types.items():
            setattr(package_modules["pyannote.audio.core.task"], name, value)

        with patch.dict(
            sys.modules,
            {"torch": torch, **package_modules},
        ):
            ensure_pyannote_checkpoint_compatibility()

        torch.serialization.add_safe_globals.assert_called_once_with(
            [
                torch.torch_version.TorchVersion,
                task_types["Problem"],
                task_types["Resolution"],
                task_types["Specifications"],
            ]
        )


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.addCleanup(self.root.destroy)
        with (
            patch("app.load_saved_token", return_value=""),
            patch("app.is_model_access_accepted", return_value=False),
        ):
            self.app = ReelSubsApp(self.root)
        self.root.geometry("760x700")
        self.root.update()

    def test_theme_icon_and_controls_fit_minimum_window(self):
        self.assertEqual(ttk.Style(self.root).theme_use(), "clam")
        with (Path(__file__).parent / "icons" / "reelsubs.png").open("rb") as icon:
            header = icon.read(24)
        width, height = struct.unpack(">II", header[16:24])
        self.assertEqual((self.app.icon.width(), self.app.icon.height()), (width, height))
        self.assertGreater(self.app.header_icon.width(), 50)

        for widget in (self.app.media_list, self.app.log, self.app.progress, self.app.start_button):
            with self.subTest(widget=widget):
                self.assertTrue(widget.winfo_ismapped())
                right = widget.winfo_rootx() - self.root.winfo_rootx() + widget.winfo_width()
                bottom = widget.winfo_rooty() - self.root.winfo_rooty() + widget.winfo_height()
                self.assertLessEqual(right, self.root.winfo_width())
                self.assertLessEqual(bottom, self.root.winfo_height())

    def test_queue_and_activity_controls_still_work(self):
        with patch("app.filedialog.askopenfilenames", return_value=("/one.mp4", "/two.mp4")):
            self.app._choose_input()
        self.assertEqual(self.app.media_list.size(), 2)
        self.app.media_list.selection_set(0)
        self.app._remove_selected()
        self.assertEqual(self.app.video_paths, [Path("/two.mp4")])
        self.app._clear_list()
        self.assertEqual(self.app.media_list.size(), 0)
        with patch("app.messagebox.showerror") as showerror:
            self.app.start_button.invoke()
        showerror.assert_called_once_with("Add media", "Add at least one media file.")
        self.app._write_log("Processing started")
        self.assertIn("Processing started", self.app.log.get("1.0", "end"))
        self.assertEqual(str(self.app.log.cget("state")), "disabled")


if __name__ == "__main__":
    unittest.main()
