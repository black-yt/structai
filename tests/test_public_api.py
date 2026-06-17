import unittest

import structai


class PublicApiTests(unittest.TestCase):
    def test_public_package_import_and_version(self):
        self.assertEqual(structai.__version__, "0.1.23")

    def test_core_symbols_are_exported(self):
        for name in (
            "LLMAgent",
            "Judge",
            "load_file",
            "save_file",
            "multi_thread",
            "parse_think_answer",
            "read_pdf",
        ):
            self.assertTrue(hasattr(structai, name), name)
