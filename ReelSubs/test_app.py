import unittest

from app import build_srt, format_srt_time, speaker_for_segment


class SubtitleHelpersTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
