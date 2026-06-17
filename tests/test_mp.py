import unittest
from contextlib import redirect_stdout
from io import StringIO

from structai.mp import multi_thread


def _add(a, b):
    return a + b


def _fail_on_negative(x):
    if x < 0:
        raise ValueError("negative")
    return x


class MultiThreadTests(unittest.TestCase):
    def test_multi_thread_preserves_input_order(self):
        items = [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]

        self.assertEqual(multi_thread(items, _add, max_workers=2, use_tqdm=False), [11, 22, 33])

    def test_multi_thread_keeps_none_for_failed_items(self):
        items = [{"x": 1}, {"x": -1}, {"x": 2}]

        with redirect_stdout(StringIO()):
            result = multi_thread(items, _fail_on_negative, max_workers=2, use_tqdm=False)

        self.assertEqual(result, [1, None, 2])
