import os
import tempfile
import unittest

from structai.utils import (
    extract_within_tags,
    filter_excessive_repeats,
    get_all_file_paths,
    parse_think_answer,
    remove_tag,
    run_with_timeout,
    sanitize_text,
)


class UtilsTests(unittest.TestCase):
    def test_parse_think_answer_with_explicit_answer_tag(self):
        self.assertEqual(
            parse_think_answer("<think>reasoning</think><answer>final</answer>"),
            ("reasoning", "final"),
        )

    def test_parse_think_answer_raises_on_missing_tags(self):
        with self.assertRaises(ValueError):
            parse_think_answer("final only")

    def test_extract_within_tags_uses_last_opening_tag(self):
        self.assertEqual(
            extract_within_tags("<answer>old</answer> text <answer>new</answer>"),
            "new",
        )

    def test_remove_tag_replaces_and_trims(self):
        self.assertEqual(remove_tag("<think>a</think><answer>b</answer>", r=" "), "a  b")

    def test_sanitize_text_removes_non_ascii_and_control_characters(self):
        cleaned = sanitize_text("abc中文\x00\n")

        self.assertEqual(cleaned, "abc\n")

    def test_filter_excessive_repeats(self):
        self.assertEqual(filter_excessive_repeats("aaab", threshold=3), "b")
        self.assertEqual(filter_excessive_repeats("ababab!", threshold=3), "!")

    def test_get_all_file_paths_relative_sorted_and_filtered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "nested"))
            for relpath in ("b.txt", "a.md", "nested/c.txt"):
                with open(os.path.join(tmpdir, relpath), "w", encoding="utf-8") as f:
                    f.write("x")

            paths = get_all_file_paths(tmpdir, suffix=".txt", absolute=False)

            self.assertEqual(paths, ["b.txt", os.path.join("nested", "c.txt")])

    def test_run_with_timeout_returns_value_and_propagates_exception(self):
        self.assertEqual(run_with_timeout(lambda x: x + 1, args=(1,), timeout=None), 2)

        with self.assertRaises(RuntimeError):
            run_with_timeout(lambda: (_ for _ in ()).throw(RuntimeError("boom")), timeout=None)
