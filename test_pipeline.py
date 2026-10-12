"""Unit tests for the pipeline and server helpers.

Run with the project venv::

    .venv/bin/python -m unittest test_pipeline -v

The pure-logic tests (word grouping, BPM fallback, session-id sanitising) have
no heavy dependencies. ``ProcessSongSessionFlowTests`` additionally exercises
the full orchestration by stubbing out torch / demucs / whisperx / librosa, so
the returned schema and stem wiring are verified without a GPU.
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

import pipeline
import server
import share_code
from server import sanitize_id


def word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end, "length": round(end - start, 3)}


class GroupWordsIntoLinesTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(pipeline.group_words_into_lines([]), [])

    def test_single_word_is_one_line(self):
        lines = pipeline.group_words_into_lines([word("hello", 0.0, 0.5)])
        self.assertEqual(len(lines), 1)
        self.assertEqual([w["word"] for w in lines[0]], ["hello"])

    def test_small_pause_stays_on_same_line(self):
        words = [word("hello", 0.0, 0.5), word("world", 0.6, 1.0)]  # 0.1s gap
        lines = pipeline.group_words_into_lines(words, silence_threshold=0.4)
        self.assertEqual(len(lines), 1)

    def test_pause_at_threshold_starts_new_line(self):
        words = [word("hello", 0.0, 0.5), word("world", 0.9, 1.2)]  # 0.4s gap
        lines = pipeline.group_words_into_lines(words, silence_threshold=0.4)
        self.assertEqual(len(lines), 2)
        self.assertEqual([w["word"] for w in lines[0]], ["hello"])
        self.assertEqual([w["word"] for w in lines[1]], ["world"])

    def test_multiple_lines(self):
        words = [
            word("a", 0.0, 0.2),
            word("b", 0.3, 0.5),
            word("c", 2.0, 2.2),  # big gap -> new line
            word("d", 2.3, 2.5),
            word("e", 5.0, 5.2),  # big gap -> new line
        ]
        lines = pipeline.group_words_into_lines(words, silence_threshold=0.4)
        self.assertEqual([[w["word"] for w in line] for line in lines], [["a", "b"], ["c", "d"], ["e"]])

    def test_custom_threshold(self):
        words = [word("a", 0.0, 0.5), word("b", 0.7, 0.9)]  # 0.2s gap
        self.assertEqual(len(pipeline.group_words_into_lines(words, 0.4)), 1)
        self.assertEqual(len(pipeline.group_words_into_lines(words, 0.1)), 2)


class EstimateLineBpmTests(unittest.TestCase):
    def test_none_slice_falls_back(self):
        self.assertEqual(pipeline.estimate_line_bpm(None, 44100), pipeline.DEFAULT_BPM)

    def test_too_short_slice_falls_back(self):
        import numpy as np

        short = np.zeros(100, dtype="float32")
        self.assertEqual(
            pipeline.estimate_line_bpm(short, 44100),
            pipeline.DEFAULT_BPM,
        )

    def test_missing_librosa_falls_back(self):
        """With librosa absent (the dev environment) the helper must not raise."""
        try:
            import librosa  # noqa: F401
        except ImportError:
            import numpy as np

            longer = np.zeros(44100, dtype="float32")
            self.assertEqual(
                pipeline.estimate_line_bpm(longer, 44100),
                pipeline.DEFAULT_BPM,
            )
        else:
            self.skipTest("librosa is installed; missing-dependency path not exercised")


class SanitizeIdTests(unittest.TestCase):
    def test_plain_name(self):
        self.assertEqual(sanitize_id("session14.m4a"), "session14.m4a")

    def test_path_traversal_is_neutralised(self):
        result = sanitize_id("../../etc/passwd")
        self.assertNotIn("/", result)
        self.assertNotIn("..", result)

    def test_weird_characters(self):
        self.assertEqual(sanitize_id("my song!! (take 2)"), "my_song_take_2")

    def test_empty_falls_back(self):
        self.assertEqual(sanitize_id("!!!./"), "session")


class ShareCodeTests(unittest.TestCase):
    """The tunnel's share code must decode with the frontend's base62 maths."""

    # The decoder the frontend uses (frontend/src/lib/base62.ts).
    FRONTEND_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def frontend_decode(self, code: str) -> str:
        value = 0
        for char in code:
            value = value * 62 + self.FRONTEND_ALPHABET.index(char)
        return format(value, "x")

    def test_known_examples(self):
        # Straight from frontend/DEVNOTES.md: hex `f03` -> `zz`, hex `56596884` -> `1a2b3c`.
        self.assertEqual(share_code.share_code("https://f03.gradio.live"), "zz")
        self.assertEqual(share_code.share_code("https://56596884.gradio.live"), "1a2b3c")

    def test_round_trips_hex_labels(self):
        for label in ("1", "f03", "56596884", "07ff8706ab", "abcdef0123456789"):
            code = share_code.share_code(f"https://{label}.gradio.live")
            # Leading zeros carry no value, exactly like the frontend decoder.
            expected = label.lower().lstrip("0") or "0"
            self.assertEqual(self.frontend_decode(code), expected, label)


# ---------------------------------------------------------------------------
# End-to-end orchestration with the heavy stack stubbed out
# ---------------------------------------------------------------------------


class _FakeTensor:
    """The small slice of the torch.Tensor API the pipeline actually uses."""

    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def dim(self):
        return self.array.ndim

    def size(self, index):
        return self.array.shape[index]

    def unsqueeze(self, index):
        return _FakeTensor(np.expand_dims(self.array, index))

    def repeat(self, *multiples):
        return _FakeTensor(np.tile(self.array, multiples))

    def __getitem__(self, index):
        return _FakeTensor(self.array[index])

    def __add__(self, other):
        return _FakeTensor(self.array + other.array)

    def mean(self, dim=0):
        return _FakeTensor(self.array.mean(axis=dim))

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def _install_fake_audio_stack():
    """Register stub modules for torch/torchaudio/demucs/whisperx/librosa."""

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    torch = module(
        "torch",
        no_grad=lambda: mock.MagicMock(
            __enter__=mock.MagicMock(return_value=None),
            __exit__=mock.MagicMock(return_value=False),
        ),
        cuda=module(
            "torch.cuda",
            is_available=lambda: False,
            empty_cache=lambda: None,
        ),
    )

    functional = module("torchaudio.functional", resample=lambda wav, a, b: wav)
    torchaudio = module(
        "torchaudio",
        functional=functional,
        load=lambda path: (_FakeTensor(np.zeros((1, 44100), dtype=np.float32)), 44100),
        save=lambda path, tensor, sr: None,
    )

    def apply_model(model, mix, **kwargs):
        # [batch][sources][channels][samples] with 4 named stems.
        return [
            [
                _FakeTensor(np.zeros((2, 44100), dtype=np.float32))  # drums
                for _ in range(3)
            ]
            + [_FakeTensor(np.ones((2, 44100), dtype=np.float32))]  # vocals
        ]

    class _FakeDemucsModel:
        samplerate = 44100

        def to(self, device):
            return self

        def eval(self):
            return self

    demucs = module("demucs")
    module("demucs.apply", apply_model=apply_model)
    module("demucs.pretrained", get_model=lambda name: _FakeDemucsModel())
    demucs.apply = sys.modules["demucs.apply"]
    demucs.pretrained = sys.modules["demucs.pretrained"]

    module(
        "whisperx",
        load_model=lambda *a, **k: mock.MagicMock(
            transcribe=lambda audio, batch_size: {
                "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            }
        ),
        load_audio=lambda path: np.zeros(16000, dtype=np.float32),
        load_align_model=lambda language_code, device: (mock.MagicMock(), {}),
        align=lambda segments, model, metadata, audio, device: {
            "segments": [
                {
                    "words": [
                        {"word": " hello", "start": 0.0, "end": 0.4},
                        {"word": " world", "start": 0.5, "end": 0.9},
                    ]
                }
            ]
        },
    )

    rhythm = module("librosa.feature.rhythm", tempo=lambda **kwargs: np.array([123.4]))
    feature = module("librosa.feature", rhythm=rhythm)
    module(
        "librosa",
        feature=feature,
        onset=module("librosa.onset", onset_strength=lambda y, sr: np.zeros(10)),
    )

    return torch, torchaudio


class ProcessSongSessionFlowTests(unittest.TestCase):
    def setUp(self):
        self._saved = dict(sys.modules)
        pipeline.clear_model_cache()
        _install_fake_audio_stack()
        self.temp_dir = tempfile.mkdtemp(prefix="catchy_test_")
        self.input_path = os.path.join(self.temp_dir, "song.m4a")
        with open(self.input_path, "wb") as handle:
            handle.write(b"fake audio bytes")

    def tearDown(self):
        pipeline.clear_model_cache()
        saved = self._saved
        for name in list(sys.modules):
            if name not in saved:
                del sys.modules[name]
        sys.modules.update(saved)

    def test_returns_spec_schema_and_wires_stems(self):
        with mock.patch.object(pipeline.subprocess, "run") as run:
            result = pipeline.process_song_session(
                self.input_path, output_dir=self.temp_dir
            )

        # Top-level keys exactly match the spec.
        self.assertEqual(set(result), {"vocals", "background", "lyrics"})
        self.assertTrue(result["vocals"].endswith("vocals.m4a"))
        self.assertTrue(result["background"].endswith("background.m4a"))

        # The pause between the two words was only 0.1s, so one line.
        self.assertEqual(len(result["lyrics"]), 1)
        line = result["lyrics"][0]
        self.assertEqual(set(line), {"bpm", "start time", "end time", "words"})
        self.assertEqual(line["bpm"], 123.4)
        self.assertEqual(line["start time"], 0.0)
        self.assertEqual(line["end time"], 0.9)
        self.assertEqual(
            line["words"],
            [{"word": "hello", "length": 0.4}, {"word": "world", "length": 0.4}],
        )

        # Two stems exported, plus scratch WAVs are cleaned up.
        self.assertEqual(run.call_count, 2)
        leftovers = [f for f in os.listdir(self.temp_dir) if f.endswith(".wav")]
        self.assertEqual(leftovers, [])

    def test_long_pause_splits_lines(self):
        def align(segments, model, metadata, audio, device):
            return {
                "segments": [
                    {
                        "words": [
                            {"word": " hello", "start": 0.0, "end": 0.4},
                            {"word": " world", "start": 2.0, "end": 2.4},
                        ]
                    }
                ]
            }

        sys.modules["whisperx"].align = align
        with mock.patch.object(pipeline.subprocess, "run"):
            result = pipeline.process_song_session(
                self.input_path, output_dir=self.temp_dir
            )

        self.assertEqual(len(result["lyrics"]), 2)
        self.assertEqual(result["lyrics"][0]["words"], [{"word": "hello", "length": 0.4}])


class ApiTests(unittest.TestCase):
    """HTTP-level tests for session/take organisation, with the pipeline stubbed."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="catchy_api_")
        self._orig_sessions = server.SESSIONS_DIR
        server.SESSIONS_DIR = Path(self.temp_dir)
        self.client = TestClient(server.app)

    def tearDown(self):
        server.SESSIONS_DIR = self._orig_sessions
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def _fake_process(holder):
        def fake(input_path, output_dir="./output", *, silence_threshold=0.4):
            Path(output_dir, "vocals.m4a").write_bytes(b"VOCALS")
            Path(output_dir, "background.m4a").write_bytes(b"BACKGROUND")
            result = {
                "vocals": str(Path(output_dir, "vocals.m4a")),
                "background": str(Path(output_dir, "background.m4a")),
                "lyrics": [
                    {
                        "bpm": 120.0,
                        "start time": 0.0,
                        "end time": 0.5,
                        "words": [{"word": "hi", "length": 0.5}],
                    }
                ],
            }
            holder.append(result)
            return result

        return fake

    def test_take_lifecycle(self):
        holder: list = []
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process(holder)
        ):
            resp = self.client.post(
                "/sessions/song1/takes/t1",
                files={"file": ("take1.m4a", b"audio-bytes", "audio/mp4")},
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.headers["x-session-id"], "song1")
        self.assertEqual(resp.headers["x-take-id"], "t1")
        self.assertEqual(
            resp.json()["lyrics"][0]["words"], [{"word": "hi", "length": 0.5}]
        )

        manifest = self.client.get("/sessions/song1").json()
        self.assertEqual(manifest["take_count"], 1)
        self.assertEqual(manifest["takes"][0]["status"], "done")
        self.assertEqual(manifest["takes"][0]["index"], 1)

        take = self.client.get("/sessions/song1/takes/t1").json()
        self.assertEqual(take["meta"]["status"], "done")
        self.assertEqual(take["result"], holder[0])

        audio = self.client.get("/sessions/song1/takes/t1/audio/vocals")
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.content, b"VOCALS")
        self.assertIn("audio/mp4", audio.headers["content-type"])

        self.assertEqual(
            self.client.get("/sessions/song1/takes/t1/audio/drums").status_code, 404
        )

    def test_index_autoincrements_and_explicit_index_wins(self):
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ):
            self.client.post(
                "/sessions/s/takes/a", files={"file": ("a.m4a", b"x", "audio/mp4")}
            )
            self.client.post(
                "/sessions/s/takes/b", files={"file": ("b.m4a", b"x", "audio/mp4")}
            )
            self.client.post(
                "/sessions/s/takes/c",
                files={"file": ("c.m4a", b"x", "audio/mp4")},
                params={"index": 1},
            )

        takes = self.client.get("/sessions/s").json()["takes"]
        by_id = {t["take_id"]: t["index"] for t in takes}
        self.assertEqual(by_id, {"a": 1, "b": 2, "c": 1})

    def test_files_persisted_and_delete_removes_take(self):
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ):
            self.client.post(
                "/sessions/s2/takes/t", files={"file": ("t.m4a", b"x", "audio/mp4")}
            )

        take_dir = Path(self.temp_dir) / "s2" / "takes" / "t"
        self.assertTrue((take_dir / "meta.json").is_file())
        self.assertTrue((take_dir / "result.json").is_file())

        deleted = self.client.delete("/sessions/s2/takes/t")
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(take_dir.exists())

    def test_error_is_recorded_on_the_take(self):
        def boom(*args, **kwargs):
            raise RuntimeError("Missing audio processing dependencies: torch")

        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=boom
        ):
            resp = self.client.post(
                "/sessions/e/takes/t", files={"file": ("t.m4a", b"x", "audio/mp4")}
            )

        self.assertEqual(resp.status_code, 503)
        meta = self.client.get("/sessions/e/takes/t").json()["meta"]
        self.assertEqual(meta["status"], "error")
        self.assertIn("torch", meta["error"])

    def test_create_session_makes_folder_and_is_idempotent(self):
        resp = self.client.post("/sessions/newsession")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["session_id"], "newsession")
        self.assertEqual(resp.json()["name"], "newsession")
        self.assertTrue(resp.json()["created"])

        session_dir = Path(self.temp_dir) / "newsession"
        self.assertTrue(session_dir.is_dir())
        self.assertTrue((session_dir / "takes").is_dir())
        listed = self.client.get("/sessions").json()["sessions"]
        self.assertIn("newsession", [s["id"] for s in listed])

        # Posting again leaves the folder alone and reports it already existed.
        again = self.client.post("/sessions/newsession")
        self.assertEqual(again.status_code, 200)
        self.assertFalse(again.json()["created"])

    def test_create_session_stores_the_display_name(self):
        resp = self.client.post(
            "/sessions/abc123", params={"name": "Lazy Horse (10/12/2026)"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["name"], "Lazy Horse (10/12/2026)")

        # The id stays a safe path segment; the pretty name lives in the manifest.
        self.assertEqual(
            self.client.get("/sessions/abc123").json()["name"],
            "Lazy Horse (10/12/2026)",
        )
        listed = self.client.get("/sessions").json()["sessions"]
        self.assertEqual(
            listed, [{"id": "abc123", "name": "Lazy Horse (10/12/2026)"}]
        )

    def test_create_session_keeps_existing_name_when_name_omitted(self):
        self.client.post("/sessions/keepme", params={"name": "First Name"})
        again = self.client.post("/sessions/keepme")
        self.assertFalse(again.json()["created"])
        self.assertEqual(again.json()["name"], "First Name")

    def test_rename_session(self):
        self.client.post("/sessions/renameme", params={"name": "Old Name"})
        resp = self.client.patch(
            "/sessions/renameme", params={"name": "New Name"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"session_id": "renameme", "name": "New Name"})
        self.assertEqual(self.client.get("/sessions/renameme").json()["name"], "New Name")

        missing = self.client.patch("/sessions/nope", params={"name": "x"})
        self.assertEqual(missing.status_code, 404)

    def test_delete_session_removes_folder_and_takes(self):
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ):
            self.client.post(
                "/sessions/gone/takes/t", files={"file": ("t.m4a", b"x", "audio/mp4")}
            )
        self.assertTrue((Path(self.temp_dir) / "gone" / "takes" / "t").is_dir())

        resp = self.client.delete("/sessions/gone")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"deleted": True, "session_id": "gone"})
        self.assertFalse((Path(self.temp_dir) / "gone").exists())
        self.assertEqual(self.client.get("/sessions").json()["sessions"], [])

        self.assertEqual(self.client.delete("/sessions/gone").status_code, 404)

    def test_create_session_sanitises_id(self):
        resp = self.client.post("/sessions/my%20song%21%21%20take%202")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["session_id"], "my_song_take_2")
        self.assertTrue((Path(self.temp_dir) / "my_song_take_2").is_dir())

    def test_process_session_compat_endpoint(self):
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ):
            resp = self.client.post(
                "/process-session", files={"file": ("mysong.m4a", b"x", "audio/mp4")}
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["x-session-id"], "mysong")
        self.assertEqual(resp.headers["x-take-id"], "take")


if __name__ == "__main__":
    unittest.main()

