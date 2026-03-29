import unittest

from xtts_wyoming.__main__ import parse_args


class CliTests(unittest.TestCase):
    def test_parse_args_supports_runtime_options(self):
        args = parse_args(
            [
                "--uri",
                "tcp://0.0.0.0:10200",
                "--voice",
                "ania",
                "--language",
                "pl",
                "--speaker-dir",
                "/data/speakers",
                "--model-dir",
                "/data/models",
                "--device",
                "cpu",
                "--samples-per-chunk",
                "2048",
                "--no-streaming",
            ]
        )

        self.assertEqual(args.uri, "tcp://0.0.0.0:10200")
        self.assertEqual(args.voice, "ania")
        self.assertEqual(args.language, "pl")
        self.assertEqual(args.speaker_dir, "/data/speakers")
        self.assertEqual(args.model_dir, "/data/models")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.samples_per_chunk, 2048)
        self.assertTrue(args.no_streaming)


if __name__ == "__main__":
    unittest.main()
