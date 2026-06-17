import unittest

from structai.judge import Judge


class JudgeNoNetworkTests(unittest.TestCase):
    def test_exact_match_path_without_optional_judges(self):
        judge = Judge(use_math_verify=False, use_llm_judge=False)
        item = {
            "question": "What is the answer?",
            "answer": "42",
            "model_answer": "41<answer_split>42",
        }

        result = judge(item)

        self.assertEqual(result["exact_match_list"], [0, 1])
        self.assertEqual(result["exact_match"], 1)
        self.assertEqual(result["exact_match_pass@k"], 1)
        self.assertEqual(result["exact_match_passall@k"], 0)
        self.assertNotIn("math_verify", result)
        self.assertNotIn("llm_judge", result)

    def test_parse_short_answer_with_think_answer_tags(self):
        judge = Judge(use_math_verify=False, use_llm_judge=False)

        self.assertEqual(
            judge.parse_short_answer("<think>because</think><answer>42</answer>"),
            "42",
        )
