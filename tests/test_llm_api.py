import unittest
import base64
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from structai.llm_api import (
    LLMAgent,
    _completion_token_limit_kwargs,
    extract_text_outputs,
    messages_to_responses_input,
    str2dict,
    str2list,
)


class _FakeChatCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content="ok")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self):
        self.chat_completions = _FakeChatCompletions()
        self.chat = SimpleNamespace(completions=self.chat_completions)


class CompletionTokenLimitTests(unittest.TestCase):
    def test_gpt5_uses_max_completion_tokens(self):
        self.assertEqual(
            _completion_token_limit_kwargs("gpt-5.1", 500),
            {"max_completion_tokens": 500},
        )

    def test_prefixed_gpt5_uses_max_completion_tokens(self):
        self.assertEqual(
            _completion_token_limit_kwargs("openai/gpt-5.1", 500),
            {"max_completion_tokens": 500},
        )

    def test_legacy_chat_models_keep_max_tokens(self):
        self.assertEqual(
            _completion_token_limit_kwargs("gpt-4.1-mini", 500),
            {"max_tokens": 500},
        )

    def test_none_omits_token_limit(self):
        self.assertEqual(_completion_token_limit_kwargs("gpt-5.1", None), {})

    def test_chat_completion_request_uses_selected_token_argument(self):
        agent = LLMAgent(api_key="test-key", model_version="openai/gpt-5.1", max_tokens=500)
        fake_client = _FakeClient()
        agent.client = fake_client

        self.assertEqual(agent._llm_api_impl("hello"), ["ok"])

        sent_kwargs = fake_client.chat_completions.last_kwargs
        self.assertEqual(sent_kwargs["max_completion_tokens"], 500)
        self.assertNotIn("max_tokens", sent_kwargs)


class ImagePayloadTests(unittest.TestCase):
    def test_image_payload_declares_png_matching_encoded_bytes(self):
        agent = LLMAgent(api_key="test-key", model_version="gpt-4.1-mini")
        fake_client = _FakeClient()
        agent.client = fake_client

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "input.jpg"
            Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path, format="JPEG")

            self.assertEqual(agent._llm_api_impl("describe", image_paths=[str(image_path)]), ["ok"])

        sent_kwargs = fake_client.chat_completions.last_kwargs
        content = sent_kwargs["messages"][-1]["content"]
        image_url = content[1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        encoded = image_url.split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded)[:8], b"\x89PNG\r\n\x1a\n")


class ParsingAndResponseShapeTests(unittest.TestCase):
    def test_str2dict_extracts_noisy_dict(self):
        self.assertEqual(str2dict("prefix {'answer': 42,} suffix"), {"answer": 42})

    def test_str2list_extracts_noisy_list(self):
        self.assertEqual(str2list("prefix [1, 2, 3] suffix"), [1, 2, 3])

    def test_messages_to_responses_input_splits_system_prompt(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        system_prompt, input_blocks = messages_to_responses_input(messages)

        self.assertEqual(system_prompt, "Be concise.")
        self.assertEqual(input_blocks[0]["role"], "user")
        self.assertEqual(input_blocks[0]["content"][0], {"type": "input_text", "text": "Hello"})
        self.assertEqual(input_blocks[1]["role"], "assistant")
        self.assertEqual(input_blocks[1]["content"][0], {"type": "output_text", "text": "Hi"})

    def test_extract_text_outputs_supports_chat_completion_shape(self):
        message = SimpleNamespace(content="final", reasoning_content="reasoning")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

        self.assertEqual(extract_text_outputs(response), ["<think>reasoning</think>final"])

    def test_extract_text_outputs_supports_responses_shortcut(self):
        response = SimpleNamespace(output_text="final")

        self.assertEqual(extract_text_outputs(response), ["final"])


if __name__ == "__main__":
    unittest.main()
