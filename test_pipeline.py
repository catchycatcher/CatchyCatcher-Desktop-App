"""Unit tests for the pipeline and server helpers.

Run with the project venv::

    .venv/bin/python -m unittest test_pipeline -v

The pure-logic tests (word grouping, BPM fallback, session-id sanitising) have
no heavy dependencies. ``ProcessSongSessionFlowTests`` additionally exercises
the full orchestration by stubbing out torch / demucs / whisperx / librosa, so
the returned schema and stem wiring are verified without a GPU.
"""

import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

import phonetics
import pipeline
import scrapbook
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

    def test_sentence_end_needs_a_real_pause_to_split(self):
        # A full stop with a solid breath ends the line ...
        long_breath = [word("hello.", 0.0, 0.5), word("world", 0.8, 1.2)]  # 0.3s
        self.assertEqual(len(pipeline.group_words_into_lines(long_breath, 0.4)), 2)

        # ... but a tiny wobble after a full stop does not (that was noise).
        tiny = [word("hello.", 0.0, 0.5), word("world", 0.62, 1.0)]  # 0.12s
        self.assertEqual(len(pipeline.group_words_into_lines(tiny, 0.4)), 1)

    def test_period_without_any_pause_stays_on_one_line(self):
        # A '.' with no breath at all is not trusted (WhisperX slips them in).
        words = [word("hello.", 0.0, 0.5), word("world", 0.5, 1.0)]
        self.assertEqual(len(pipeline.group_words_into_lines(words, 0.4)), 1)

    def test_timing_jitter_alone_does_not_split(self):
        # Gaps wobbling between 0.02s and 0.2s with no punctuation is ordinary
        # timing noise, not phrase boundaries: it must stay on one line.
        gaps = [0.02, 0.18, 0.05, 0.12, 0.2, 0.03, 0.16, 0.08]
        words = []
        cursor = 0.0
        for index, gap in enumerate(gaps):
            words.append(word(f"w{index}", round(cursor, 3), round(cursor + 0.25, 3)))
            cursor += 0.25 + gap
        self.assertEqual(len(pipeline.group_words_into_lines(words, silence_threshold=0.4)), 1)

    def test_modest_gap_variation_stays_on_one_line(self):
        # Gaps that keep jumping to roughly twice their neighbours, yet stay
        # well under the threshold, are still just the singer breathing between
        # words — a scale-free "this pause is long relative to the others" rule
        # must not turn that ordinary variation into a line break.
        gaps = [0.1, 0.22, 0.1, 0.1, 0.22, 0.1, 0.1]
        words = []
        cursor = 0.0
        for index, gap in enumerate(gaps):
            words.append(word(f"w{index}", round(cursor, 3), round(cursor + 0.25, 3)))
            cursor += 0.25 + gap
        self.assertEqual(len(pipeline.group_words_into_lines(words, silence_threshold=0.4)), 1)

    def test_comma_tips_a_moderate_pause_over_the_line(self):
        # A 0.25s gap with gaps of ~0.15s around it sits just under the break
        # score; the comma's weight pushes it over. Remove the comma and it
        # stays on one line.
        base = [
            word("a", 0.0, 0.5),
            word("hello,", 0.65, 1.15),
            word("b", 1.40, 1.90),
            word("c", 2.05, 2.55),
            word("d", 2.70, 3.20),
        ]
        comma = pipeline.group_words_into_lines(base, silence_threshold=0.4)
        self.assertEqual(len(comma), 2)

        no_comma = [dict(w, word=w["word"].rstrip(",")) for w in base]
        self.assertEqual(len(pipeline.group_words_into_lines(no_comma, silence_threshold=0.4)), 1)

    def test_nearby_line_length_splits_a_long_unpunctuated_run(self):
        # Two short (~0.85s) lines then a long run with only tiny gaps and no
        # punctuation. The run is far longer than the lines around it, so the
        # line-length signal should break it up even with no pause to do it.
        words = [
            word("a0", 0.00, 0.25),
            word("a1", 0.30, 0.55),
            word("a2", 0.60, 0.85),
            word("b0", 1.45, 1.70),
            word("b1", 1.75, 2.00),
            word("b2", 2.05, 2.30),
            word("c0", 2.90, 3.15),
            word("c1", 3.20, 3.45),
            word("c2", 3.50, 3.75),
            word("c3", 3.80, 4.05),
            word("c4", 4.10, 4.35),
            word("c5", 4.40, 4.65),
            word("c6", 4.70, 4.95),
            word("c7", 5.00, 5.25),
            word("c8", 5.30, 5.55),
        ]
        lines = pipeline.group_words_into_lines(words, silence_threshold=0.4)
        texts = [[w["word"] for w in line] for line in lines]

        # a0..a2 and b0..b2 are separate lines; the c-run is not one big line.
        self.assertEqual(texts[:2], [["a0", "a1", "a2"], ["b0", "b1", "b2"]])
        self.assertGreater(len(lines), 3)
        c_lines = [line for line in texts if any(text.startswith("c") for text in line)]
        self.assertGreater(len(c_lines), 1)

    def test_relative_pause_splits_below_the_absolute_threshold(self):
        # One 0.3s gap among 0.05s gaps is a phrase boundary even though it is
        # under the 0.4s threshold.
        words = [
            word("a", 0.0, 0.2),
            word("b", 0.25, 0.45),
            word("c", 0.5, 0.7),
            word("d", 1.0, 1.2),
            word("e", 1.25, 1.45),
            word("f", 1.5, 1.7),
        ]
        lines = pipeline.group_words_into_lines(words, silence_threshold=0.4)
        self.assertEqual([[w["word"] for w in line] for line in lines], [["a", "b", "c"], ["d", "e", "f"]])

    def test_uniform_small_gaps_stay_on_one_line(self):
        words = [word(f"w{i}", i * 0.2, i * 0.2 + 0.15) for i in range(6)]  # 0.05s gaps
        self.assertEqual(len(pipeline.group_words_into_lines(words, 0.4)), 1)

    def test_line_cap_splits_a_long_run(self):
        # No natural break, but the line would be 7 words: cap it at 3.
        words = [word(f"w{i}", i * 0.2, i * 0.2 + 0.15) for i in range(7)]
        lines = pipeline.group_words_into_lines(words, 0.4, max_line_words=3)
        self.assertTrue(all(len(line) <= 3 for line in lines))
        self.assertEqual([w["word"] for line in lines for w in line], [w["word"] for w in words])


class BreakModelTests(unittest.TestCase):
    """The stages of the break model, exercised on their own."""

    def test_bigger_pause_scores_higher(self):
        words = [word("a", 0.0, 0.3), word("b", 0.4, 0.7), word("c", 1.0, 1.3)]
        gaps = pipeline._word_gaps(words)  # 0.1 then 0.3
        probs = pipeline._seed_break_probabilities(words, gaps, 0.4)
        self.assertLess(probs[0], probs[1])
        self.assertGreater(probs[1], 0.5)

    def test_pause_at_threshold_is_above_half(self):
        words = [word("a", 0.0, 0.5), word("b", 0.9, 1.3)]  # 0.4s gap
        gaps = pipeline._word_gaps(words)
        probs = pipeline._seed_break_probabilities(words, gaps, 0.4)
        self.assertGreater(probs[0], 0.5)

    def test_sentence_punctuation_raises_the_seed(self):
        plain = [word("a", 0.0, 0.3), word("b", 0.45, 0.75)]
        stopped = [word("a.", 0.0, 0.3), word("b", 0.45, 0.75)]
        plain_prob = pipeline._seed_break_probabilities(plain, pipeline._word_gaps(plain), 0.4)[0]
        stop_prob = pipeline._seed_break_probabilities(stopped, pipeline._word_gaps(stopped), 0.4)[0]
        self.assertGreater(stop_prob, plain_prob)

    @unittest.skipUnless(phonetics.available(), "pronunciation dictionary not installed")
    def test_near_rhyme_similarity(self):
        self.assertEqual(pipeline._near_rhyme_similarity("trying", "lying"), 1.0)
        self.assertEqual(pipeline._near_rhyme_similarity("die", "lie"), 1.0)
        self.assertGreaterEqual(pipeline._near_rhyme_similarity("pain.", "insane."), 0.8)
        self.assertEqual(pipeline._near_rhyme_similarity("cat", "dog"), 0.0)

    @unittest.skipUnless(phonetics.available(), "pronunciation dictionary not installed")
    def test_phonetic_rime_groups_sounding_ends(self):
        # CMU dictionary rhymes, so the spelling differences converge.
        self.assertEqual(phonetics.rime("pain."), ("EY", ("N",)))
        self.assertEqual(phonetics.rime("insane."), ("EY", ("N",)))
        self.assertEqual(phonetics.rime("masquerade")[0], "EY")

    @unittest.skipUnless(phonetics.available(), "pronunciation dictionary not installed")
    def test_sound_alike_tolerates_a_mis_heard_word(self):
        self.assertTrue(phonetics.sound_alike("smile", "smiles"))
        self.assertTrue(phonetics.sound_alike("die", "dye"))
        self.assertFalse(phonetics.sound_alike("cat", "dog"))

    def test_rhyme_scheme_reinforces_rhyming_endings(self):
        words = [
            word("trying", 0.0, 0.4),
            word("around", 0.85, 1.2),  # long gap -> candidate ending "trying"
            word("lying", 1.25, 1.65),
            word("again", 2.1, 2.5),  # long gap -> candidate ending "lying"
        ]
        gaps = pipeline._word_gaps(words)
        seeded = pipeline._seed_break_probabilities(words, gaps, 0.4)
        boosted = pipeline._apply_rhyme_scheme(words, seeded)
        # Both candidate endings are now certain breaks.
        self.assertGreater(boosted[0], seeded[0])
        self.assertGreater(boosted[2], seeded[2])
        self.assertAlmostEqual(boosted[0], 1.0)
        self.assertAlmostEqual(boosted[2], 1.0)

    def test_rhyme_scheme_ignores_non_rhymes(self):
        words = [
            word("cat", 0.0, 0.4),
            word("around", 0.85, 1.2),
            word("dog", 1.25, 1.65),
            word("again", 2.1, 2.5),
        ]
        gaps = pipeline._word_gaps(words)
        seeded = pipeline._seed_break_probabilities(words, gaps, 0.4)
        self.assertEqual(pipeline._apply_rhyme_scheme(words, seeded), seeded)

    def test_length_pass_does_not_invent_a_weak_break(self):
        # A long line whose only pause is small: the length pass may not lift a
        # sub-threshold seam into a line on its own.
        words = [word(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(6)]
        gaps = pipeline._word_gaps(words)  # all 0.05s
        seeded = pipeline._seed_break_probabilities(words, gaps, 0.4)
        adjusted = pipeline._apply_line_length_consistency(
            words, gaps, seeded, target_seconds=0.5
        )
        self.assertTrue(all(value < 0.5 for value in adjusted))

    def test_rolling_threshold_drifts_with_recent_lines(self):
        # Long recent lines push the threshold down (break more often)...
        self.assertLess(
            pipeline._rolling_threshold([3.0, 3.0], 2.0),
            pipeline.BASE_BREAK_THRESHOLD,
        )
        # ... short recent lines push it up (break less often) ...
        self.assertGreater(
            pipeline._rolling_threshold([0.5, 0.5], 2.0),
            pipeline.BASE_BREAK_THRESHOLD,
        )
        # ... and it never leaves its bounds.
        self.assertGreaterEqual(
            pipeline._rolling_threshold([100.0], 1.0), pipeline.THRESHOLD_MIN
        )
        self.assertLessEqual(
            pipeline._rolling_threshold([0.01], 1.0), pipeline.THRESHOLD_MAX
        )
        self.assertEqual(
            pipeline._rolling_threshold([], None), pipeline.BASE_BREAK_THRESHOLD
        )


EXAMPLE_WORDS_PATH = (
    Path(__file__).resolve().parent / "test_fixtures" / "example_words.json"
)


class ExampleSongLineGroupingTests(unittest.TestCase):
    """Run the real word timeline of ``example.m4a`` through the grouping.

    ``test_fixtures/example_words.json`` is the WhisperX word alignment of the
    committed example take, captured once. It is a frozen regression case for
    the whole break model: pauses, punctuation, rhymes and all.
    """

    @classmethod
    def setUpClass(cls):
        with open(EXAMPLE_WORDS_PATH, encoding="utf-8") as handle:
            cls.words = json.load(handle)
        cls.lines = pipeline.group_words_into_lines(cls.words, 0.4)

    def _texts(self):
        return [" ".join(w["word"] for w in line) for line in self.lines]

    def test_every_word_is_kept_in_order(self):
        flattened = [w["word"] for line in self.lines for w in line]
        self.assertEqual(flattened, [w["word"] for w in self.words])

    def test_no_line_exceeds_the_caps(self):
        for line in self.lines:
            self.assertLessEqual(len(line), pipeline.MAX_LINE_WORDS)
            self.assertLessEqual(
                line[-1]["end"] - line[0]["start"], pipeline.MAX_LINE_SECONDS
            )

    def test_break_count_stays_in_a_sane_band(self):
        # Loose enough to survive small tuning, tight enough to catch the song
        # collapsing into one run or shattering into fragments.
        self.assertGreater(len(self.lines), 80)
        self.assertLess(len(self.lines), 110)
        self.assertLess(max(len(line) for line in self.lines), 11)

    def test_the_opening_verses_split_on_the_real_phrases(self):
        # These match the handwritten ground truth line-for-line.
        texts = self._texts()
        self.assertEqual(texts[0], "You have that hole still,")
        self.assertEqual(texts[1], "the one you can't fill")
        self.assertEqual(texts[2], "But I know you're trying")
        self.assertEqual(texts[3], "Find a lover,")
        self.assertEqual(texts[4], "tell her you love her")
        self.assertEqual(texts[5], "But you're just lying")

    def test_a_long_unpunctuated_sentence_is_not_chopped(self):
        # No comma or stop to lean on, so a pause sized phrase stays whole.
        self.assertIn("Hide your hurt and feed your burdens.", self._texts())

    def test_a_comma_ends_a_line_even_without_a_pause(self):
        # "still," and "you," end lines although the singer runs straight on.
        texts = self._texts()
        self.assertEqual(texts[0], "You have that hole still,")
        self.assertIn("You think no one knows you,", texts)

    def test_repeated_hook_lines_stay_separate(self):
        texts = self._texts()
        self.assertEqual(texts[6:9], ["You smile,", "you smile,", "you smile,"])
        self.assertEqual(texts[10:12], ["You sit here,", "you sit here,"])

    def test_line_lengths_are_centred(self):
        durations = [line[-1]["end"] - line[0]["start"] for line in self.lines]
        self.assertGreater(statistics.median(durations), 1.0)
        self.assertLess(statistics.median(durations), 2.0)


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

    def test_format_share_code_groups_into_fours(self):
        # length -> expected grouping (the tail is never a lone 1 or 2)
        cases = {
            4: "abcd",
            5: "abcde",
            6: "abc-def",
            7: "abcd-efg",
            8: "abcd-efgh",
            9: "abcd-efghi",
            10: "abcd-efg-hij",
            11: "abcd-efgh-ijk",
            12: "abcd-efgh-ijkl",
            13: "abcd-efgh-ijklm",
        }
        for length, expected in cases.items():
            self.assertEqual(share_code.format_share_code("abcdefghijklm"[:length]), expected)

    def test_format_share_code_strips_separators_and_is_idempotent(self):
        self.assertEqual(share_code.format_share_code("ab-cd ef"), "abc-def")
        once = share_code.format_share_code("abcdefghij")
        self.assertEqual(share_code.format_share_code(once), once)
        self.assertEqual(once, "abcd-efg-hij")


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

        # Top-level keys match the spec (plus the whole-take tempo).
        self.assertEqual(set(result), {"vocals", "background", "bpm", "lyrics"})
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
            [
                {"word": "hello", "start": 0.0, "end": 0.4, "length": 0.4},
                {"word": "world", "start": 0.5, "end": 0.9, "length": 0.4},
            ],
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
        self.assertEqual(
            result["lyrics"][0]["words"],
            [{"word": "hello", "start": 0.0, "end": 0.4, "length": 0.4}],
        )


class ScrapbookTimelineTests(unittest.TestCase):
    """The pure per-line tempo/gap maths behind a rendered scrapbook mix."""

    def test_atempo_is_split_into_supported_factors(self):
        self.assertEqual(scrapbook.atempo_factors(1.0), [1.0])
        self.assertEqual(scrapbook.atempo_factors(3.0), [2.0, 1.5])
        self.assertEqual(scrapbook.atempo_factors(0.25), [0.5, 0.5])
        product = 1.0
        for factor in scrapbook.atempo_factors(1.7):
            product *= factor
        self.assertAlmostEqual(product, 1.7, places=5)

    def test_matching_journals_need_no_stretch(self):
        entries = [
            {"bpm": 100.0, "start": 0.0, "end": 1.0, "words": [], "next_bpm": 120.0},
            {"bpm": 120.0, "start": 0.0, "end": 1.0, "words": [], "prev_bpm": 100.0},
        ]
        timeline, gaps = scrapbook.build_timeline(entries)
        # A's after-ratio and B's before-ratio both say 1.2, matching B's
        # natural ratio, so neither line is stretched.
        self.assertAlmostEqual(timeline[0]["speed"], 1.0, places=3)
        self.assertAlmostEqual(timeline[1]["speed"], 1.0, places=3)
        self.assertAlmostEqual(timeline[0]["end time"], 1.0, places=3)
        self.assertAlmostEqual(gaps[0], scrapbook.DEFAULT_GAP_SECONDS, places=3)
        self.assertAlmostEqual(timeline[1]["start time"], 1.0 + gaps[0], places=3)

    def test_chooses_the_ratio_closest_to_natural(self):
        entries = [
            {"bpm": 100.0, "start": 0.0, "end": 1.0, "words": [], "next_bpm": 150.0},
            {"bpm": 120.0, "start": 0.0, "end": 1.0, "words": [], "prev_bpm": 110.0},
        ]
        # natural = 120/100 = 1.2; candidates are A.after=1.5 and B.before=1.0909.
        # 1.0909 is closer to 1.2, so B slows slightly: raw speed 0.909.
        timeline, _gaps = scrapbook.build_timeline(entries)
        # Median log-speed is normalised to zero, so the two speeds mirror.
        self.assertAlmostEqual(timeline[0]["speed"], 1.049, places=2)
        self.assertAlmostEqual(timeline[1]["speed"], 0.953, places=2)

    def test_median_log_speed_is_zero(self):
        entries = [
            {"bpm": 100.0, "start": 0.0, "end": 1.0, "words": [], "next_bpm": 130.0},
            {"bpm": 130.0, "start": 0.0, "end": 1.0, "words": [], "prev_bpm": 100.0, "next_bpm": 80.0},
            {"bpm": 80.0, "start": 0.0, "end": 1.0, "words": [], "prev_bpm": 130.0},
        ]
        timeline, _gaps = scrapbook.build_timeline(entries)
        logs = sorted(math.log(line["speed"]) for line in timeline)
        self.assertAlmostEqual(logs[len(logs) // 2], 0.0, places=3)

    def test_same_bpm_is_always_an_option(self):
        # Two takes at very different tempos, no neighbour info: the take-ratio
        # baseline is 1.6, but same-BPM (1.0) is still a candidate and wins as
        # the only option, so B is played at A's tempo.
        entries = [
            {"bpm": 100.0, "take_bpm": 100.0, "start": 0.0, "end": 1.0, "words": []},
            {"bpm": 160.0, "take_bpm": 160.0, "start": 0.0, "end": 1.0, "words": []},
        ]
        timeline, _gaps = scrapbook.build_timeline(entries)
        self.assertAlmostEqual(timeline[0]["speed"], 1.265, places=2)
        self.assertAlmostEqual(timeline[1]["speed"], 0.790, places=2)

    def test_baseline_prefers_the_source_take_ratio(self):
        # A's take is 100, B's take is 160 -> baseline 1.6. A's after-ratio of
        # 2.0 is closer to 1.6 than same-BPM (1.0), so the expectation is used.
        entries = [
            {"bpm": 100.0, "take_bpm": 100.0, "next_bpm": 200.0, "start": 0.0, "end": 1.0, "words": []},
            {"bpm": 160.0, "take_bpm": 160.0, "start": 0.0, "end": 1.0, "words": []},
        ]
        timeline, _gaps = scrapbook.build_timeline(entries)
        self.assertAlmostEqual(timeline[0]["speed"], 0.89, places=2)
        self.assertAlmostEqual(timeline[1]["speed"], 1.12, places=2)

    def test_single_line_plays_at_its_tempo(self):
        entries = [{"bpm": 90.0, "start": 1.0, "end": 2.0, "words": []}]
        timeline, gaps = scrapbook.build_timeline(entries)
        self.assertEqual(gaps, [])
        self.assertAlmostEqual(timeline[0]["speed"], 1.0, places=3)
        self.assertAlmostEqual(timeline[0]["start time"], 0.0, places=3)
        self.assertAlmostEqual(timeline[0]["end time"], 1.0, places=3)

    def test_timeline_tags_each_line_with_its_source(self):
        entries = [
            {"bpm": 100.0, "start": 0.0, "end": 1.0, "words": [], "source": {"session_id": "s", "take_id": "t", "line_index": 0}},
            {"bpm": 100.0, "start": 2.0, "end": 3.0, "words": [], "source": {"session_id": "s", "take_id": "t", "line_index": 1}},
        ]
        timeline, _gaps = scrapbook.build_timeline(entries)
        self.assertEqual(timeline[0]["source"]["line_index"], 0)
        self.assertEqual(timeline[1]["source"]["line_index"], 1)

    def test_gap_prefers_the_earlier_pause_when_they_diverge(self):
        entries = [
            {"bpm": 100.0, "take_bpm": 100.0, "start": 0.0, "end": 1.0, "words": [], "gap_after": 0.5, "next_bpm": 100.0},
            {"bpm": 100.0, "take_bpm": 100.0, "start": 0.0, "end": 1.0, "words": [], "gap_before": 1.2, "prev_bpm": 100.0},
        ]
        _timeline, gaps = scrapbook.build_timeline(entries)
        # 1.2 / 0.5 = 2.4 >= 1.5, so the earlier clip's 0.5s pause wins.
        self.assertAlmostEqual(gaps[0], 0.5, places=3)

    def test_gap_blends_when_the_pauses_are_close(self):
        entries = [
            {"bpm": 100.0, "take_bpm": 100.0, "start": 0.0, "end": 1.0, "words": [], "gap_after": 0.5, "next_bpm": 100.0},
            {"bpm": 100.0, "take_bpm": 100.0, "start": 0.0, "end": 1.0, "words": [], "gap_before": 0.7, "prev_bpm": 100.0},
        ]
        _timeline, gaps = scrapbook.build_timeline(entries)
        # 0.7 / 0.5 = 1.4 < 1.5, so the two are averaged.
        self.assertAlmostEqual(gaps[0], 0.6, places=3)

    def test_gap_scales_with_the_new_speed(self):
        entries = [
            {"bpm": 100.0, "start": 0.0, "end": 1.0, "words": [], "gap_after": 0.5, "next_bpm": 200.0},
            {"bpm": 200.0, "start": 0.0, "end": 1.0, "words": [], "gap_before": 0.5, "prev_bpm": 100.0},
        ]
        timeline, gaps = scrapbook.build_timeline(entries)
        # B is nudged faster, so the blended gap shrinks (each half divided by
        # that line's speed); it must stay positive and below the raw 0.5s.
        self.assertGreater(gaps[0], 0.0)
        self.assertLessEqual(gaps[0], 0.5)
        self.assertAlmostEqual(timeline[1]["start time"], timeline[0]["end time"] + gaps[0], places=3)

    def test_extract_command_applies_tempo(self):
        command = scrapbook.extract_command("vocals.m4a", 1.5, 2.0, 1.5, Path("/tmp/out.wav"))
        self.assertIn("atempo=1.5", command)
        self.assertIn("vocals.m4a", command)
        self.assertIn("-ss", command)


class ApiTests(unittest.TestCase):
    """HTTP-level tests for session/take organisation, with the pipeline stubbed."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="catchy_api_")
        self.scrapbook_dir = tempfile.mkdtemp(prefix="catchy_sb_")
        self._orig_sessions = server.SESSIONS_DIR
        self._orig_scrapbooks = server.SCRAPBOOKS_DIR
        server.SESSIONS_DIR = Path(self.temp_dir)
        server.SCRAPBOOKS_DIR = Path(self.scrapbook_dir)
        self.client = TestClient(server.app)

    def tearDown(self):
        server.SESSIONS_DIR = self._orig_sessions
        server.SCRAPBOOKS_DIR = self._orig_scrapbooks
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.scrapbook_dir, ignore_errors=True)

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

    def test_requests_are_logged(self):
        with self.assertLogs("catchycatcher.requests", level="INFO") as captured:
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            any("GET /health" in line and "200" in line for line in captured.output),
            captured.output,
        )

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
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], "abc123")
        self.assertEqual(listed[0]["name"], "Lazy Horse (10/12/2026)")
        self.assertTrue(listed[0]["updated_at"])

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
        self.assertEqual(resp.json()["session_id"], "renameme")
        self.assertEqual(resp.json()["name"], "New Name")
        self.assertEqual(self.client.get("/sessions/renameme").json()["name"], "New Name")

        missing = self.client.patch("/sessions/nope", params={"name": "x"})
        self.assertEqual(missing.status_code, 404)

    def test_updated_at_is_reported_and_tracks_new_takes(self):
        self.client.post("/sessions/active", params={"name": "Active"})
        before = self.client.get("/sessions").json()["sessions"][0]
        self.assertTrue(before["updated_at"])

        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ):
            self.client.post(
                "/sessions/active/takes/t",
                files={"file": ("t.m4a", b"x", "audio/mp4")},
            )

        after = self.client.get("/sessions").json()["sessions"][0]
        self.assertIn("updated_at", after)
        # ISO-8601 timestamps sort chronologically, even at second resolution.
        self.assertGreaterEqual(after["updated_at"], before["updated_at"])

    def test_delete_take_touches_the_session(self):
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ):
            self.client.post(
                "/sessions/touch/takes/t", files={"file": ("t.m4a", b"x", "audio/mp4")}
            )
        self.assertIn("updated_at", self.client.get("/sessions/touch").json())
        self.assertEqual(self.client.delete("/sessions/touch/takes/t").status_code, 200)

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

    def test_store_without_processing_then_process_later(self):
        stored = self.client.post(
            "/sessions/journal/takes/entry1",
            params={"process": "false", "index": 1},
            files={"file": ("entry1.webm", b"raw-audio-bytes", "audio/webm")},
        )
        self.assertEqual(stored.status_code, 200, stored.text)
        meta = stored.json()
        self.assertEqual(meta["status"], "recorded")
        self.assertFalse(meta["processed"])
        self.assertEqual(meta["index"], 1)

        # The manifest keeps the entry around, unprocessed.
        manifest = self.client.get("/sessions/journal").json()
        self.assertEqual(manifest["take_count"], 1)
        self.assertEqual(manifest["takes"][0]["status"], "recorded")

        # The stored audio can be streamed back as-is.
        original = self.client.get("/sessions/journal/takes/entry1/original")
        self.assertEqual(original.status_code, 200)
        self.assertEqual(original.content, b"raw-audio-bytes")

        # ... and processed on demand.
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ):
            processed = self.client.post("/sessions/journal/takes/entry1/process")
        self.assertEqual(processed.status_code, 200, processed.text)
        self.assertEqual(
            processed.json()["lyrics"][0]["words"], [{"word": "hi", "length": 0.5}]
        )
        self.assertEqual(
            self.client.get("/sessions/journal/takes/entry1").json()["meta"]["status"],
            "done",
        )

    def test_upload_is_processed_by_default(self):
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ) as process:
            resp = self.client.post(
                "/sessions/default/takes/t",
                files={"file": ("t.m4a", b"x", "audio/mp4")},
            )
        self.assertEqual(resp.status_code, 200)
        process.assert_called_once()

    def test_take_names_default_and_rename(self):
        named = self.client.post(
            "/sessions/named/takes/e1",
            params={"process": "false", "name": "Lazy Horse"},
            files={"file": ("e1.webm", b"x", "audio/webm")},
        )
        self.assertEqual(named.json()["name"], "Lazy Horse")

        default = self.client.post(
            "/sessions/named/takes/e2",
            params={"process": "false"},
            files={"file": ("e2.webm", b"x", "audio/webm")},
        )
        self.assertEqual(default.json()["name"], "e2")

        renamed = self.client.patch(
            "/sessions/named/takes/e1", params={"name": "Brave Comet"}
        )
        self.assertEqual(renamed.status_code, 200, renamed.text)
        self.assertEqual(renamed.json()["name"], "Brave Comet")

        manifest = self.client.get("/sessions/named").json()
        names = {take["take_id"]: take["name"] for take in manifest["takes"]}
        self.assertEqual(names, {"e1": "Brave Comet", "e2": "e2"})

        missing = self.client.patch("/sessions/named/takes/nope", params={"name": "x"})
        self.assertEqual(missing.status_code, 404)

    def test_bookmark_take(self):
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ):
            self.client.post(
                "/sessions/bm/takes/t", files={"file": ("t.m4a", b"x", "audio/mp4")}
            )
        resp = self.client.patch("/sessions/bm/takes/t", params={"bookmarked": "true"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["bookmarked"])
        self.assertTrue(self.client.get("/sessions/bm").json()["takes"][0]["bookmarked"])

        off = self.client.patch("/sessions/bm/takes/t", params={"bookmarked": "false"})
        self.assertFalse(off.json()["bookmarked"])

    def test_scrapbook_lifecycle_and_render(self):
        with mock.patch.object(
            server.pipeline, "process_song_session", side_effect=self._fake_process([])
        ):
            self.client.post(
                "/sessions/src/takes/t1", files={"file": ("t1.m4a", b"x", "audio/mp4")}
            )

        created = self.client.post(
            "/scrapbooks",
            json={
                "name": "My Mix",
                "lines": [
                    {"session_id": "src", "take_id": "t1", "line_index": 0},
                    {"session_id": "src", "take_id": "t1", "line_index": 99},  # bad, dropped
                    "junk",
                ],
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        sid = created.json()["scrapbook_id"]
        self.assertEqual(created.json()["line_count"], 1)

        listed = self.client.get("/scrapbooks").json()["scrapbooks"]
        self.assertIn(sid, [item["scrapbook_id"] for item in listed])

        def fake_render(entries, output_dir):
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            mix = path / "mix.m4a"
            mix.write_bytes(b"MIX")
            timeline = [{"bpm": 120.0, "start time": 0.0, "end time": 1.0, "words": []}]
            return timeline, [], mix

        with mock.patch.object(server.scrapbook, "render", side_effect=fake_render) as render:
            rendered = self.client.post(f"/scrapbooks/{sid}/render")
        self.assertEqual(rendered.status_code, 200, rendered.text)
        self.assertEqual(rendered.json()["audio"], f"/scrapbooks/{sid}/audio")
        self.assertEqual(len(rendered.json()["timeline"]), 1)
        self.assertEqual(render.call_count, 1)

        # A second render is served from cache (the signature is unchanged).
        with mock.patch.object(server.scrapbook, "render", side_effect=fake_render) as render2:
            again = self.client.post(f"/scrapbooks/{sid}/render")
        self.assertEqual(again.status_code, 200)
        render2.assert_not_called()

        audio = self.client.get(f"/scrapbooks/{sid}/audio")
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.content, b"MIX")

        updated = self.client.patch(f"/scrapbooks/{sid}", json={"lines": []})
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["line_count"], 0)

        # Empty scrapbooks cannot render.
        empty = self.client.post(f"/scrapbooks/{sid}/render")
        self.assertEqual(empty.status_code, 400)

        self.assertEqual(self.client.delete(f"/scrapbooks/{sid}").status_code, 200)
        self.assertEqual(self.client.get(f"/scrapbooks/{sid}").status_code, 404)


if __name__ == "__main__":
    unittest.main()

