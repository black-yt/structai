import json
import os
import tempfile
import unittest

from structai.io import load_file, save_file


class IoRoundTripTests(unittest.TestCase):
    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "data.json")
            save_file({"a": 1, "b": [2]}, path)

            self.assertEqual(load_file(path), {"a": 1, "b": [2]})

    def test_jsonl_loads_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "records.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"a": 1}) + "\n")
                f.write(json.dumps({"a": 2}) + "\n")

            self.assertEqual(load_file(path), [{"a": 1}, {"a": 2}])

    def test_text_like_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for filename in ("note.txt", "README.md", "script.py"):
                path = os.path.join(tmpdir, filename)
                save_file("hello", path)
                self.assertEqual(load_file(path), "hello")

    def test_unsupported_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.unsupported")
            with self.assertRaises(ValueError):
                load_file(path)
