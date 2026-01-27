from openai import OpenAI
from typing import Union
import Levenshtein
import time
from json_repair import repair_json
from PIL import Image
import io
import base64
import os
import ipaddress
import ast
import json
from urllib.parse import urlparse
from .io import load_file
from .utils import run_with_timeout, sanitize_text
from .mp import multi_thread


def str2dict(s: str) -> dict:
    """
    Robustly converts a string representation of a dictionary to a Python `dict`.
    It handles common formatting errors and uses `json_repair` as a fallback.

    Args:
        s (str): The string representation of a dictionary.

    Returns:
        dict: The parsed dictionary.
    """
    start_index = s.find('{')
    if start_index != -1:
        end_index = s.rfind('}') + 1
        s = s[start_index:end_index]
    try:
        d = ast.literal_eval(s)
    except:
        try:
            d = json.loads(repair_json(s))
        except:
            d = json.loads(repair_json(sanitize_text(s)))
    return d


def str2list(s: str) -> list:
    """
    Robustly converts a string representation of a list to a Python `list`.

    Args:
        s (str): The string representation of a list.

    Returns:
        list: The parsed list.
    """
    start_index = s.find('[')
    if start_index != -1:
        end_index = s.rfind(']') + 1
        s = s[start_index:end_index]
    try:
        l = ast.literal_eval(s)
    except:
        try:
            l = json.loads(repair_json(s))
        except:
            l = json.loads(repair_json(sanitize_text(s)))
    return l


def encode_image(image_obj: Image.Image) -> str:
    """
    Encodes a PIL Image object into a base64 string.

    Args:
        image_obj (PIL.Image.Image): The image object to encode.

    Returns:
        str: The base64 encoded string.
    """
    buffered = io.BytesIO()
    image_obj.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def add_no_proxy_if_private(url: str):
    """
    Checks if the hostname in the URL is a private IP address.
    If so, it adds it to the `no_proxy` environment variable to bypass proxies.

    Args:
        url (str): The URL to check.
    """
    if not url:
        return
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return

    # Only handle IP
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return

    # Not effective for public IP
    if ip.is_global:
        return

    for key in ("no_proxy", "NO_PROXY"):
        old = os.environ.get(key, "")
        entries = [x.strip() for x in old.split(",") if x.strip()]

        if host not in entries:
            entries.append(host)
            os.environ[key] = ",".join(entries)
            print(f"[no_proxy] added {host} to {key}")


def messages_to_responses_input(messages):
    """
    Converts standard Chat Completions `messages` format (list of dicts) to the input format required by the Responses API.

    Args:
        messages (list[dict]): List of message dictionaries with 'role' and 'content'.

    Returns:
        tuple: A tuple containing `(system_prompt_content, input_blocks)`.
    """
    system_prompt_content = None
    input_blocks = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            # Responses API uses top-level system parameter
            # If there are multiple system messages, concatenate them
            if system_prompt_content is None:
                system_prompt_content = content
            else:
                system_prompt_content += "\n" + content
        else:
            # Handle content which can be str or list (multimodal)
            # Determine text type based on role: user -> input_text, assistant -> output_text
            text_type = "input_text" if role == "user" else "output_text"
            
            if isinstance(content, str):
                input_blocks.append({
                    "role": role,
                    "content": [
                        {"type": text_type, "text": content}
                    ]
                })
            elif isinstance(content, list):
                # Convert Chat Completion content list to Responses API content list
                new_content_list = []
                for item in content:
                    if item["type"] == "text":
                        new_content_list.append({"type": text_type, "text": item["text"]})
                    elif item["type"] == "image_url":
                        # Images are typically inputs
                        new_content_list.append({"type": "input_image", "image_url": item["image_url"]["url"]})
                    # Add other types if necessary
                
                input_blocks.append({
                    "role": role,
                    "content": new_content_list
                })
    
    return system_prompt_content, input_blocks


def extract_text_outputs(result) -> list[str]:
    """
    Extracts the text content from an LLM API response object (supports both Chat Completions and Responses API formats).

    Args:
        result (object): The response object from the LLM API.

    Returns:
        list[str]: A list of extracted text outputs.
    """

    # ---------- Chat Completions ----------
    # response.choices[i].message.content
    if hasattr(result, "choices"):
        outputs = []
        for choice in result.choices:
            msg = getattr(choice, "message", None)
            if msg and msg.content:
                if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    content = f"<think>{msg.reasoning_content}</think>{msg.content}"
                else:
                    content = msg.content
                outputs.append(content)
        return outputs

    # ---------- Responses API ----------
    # response.output_text (recommended shortcut)
    if hasattr(result, "output_text") and result.output_text:
        return [result.output_text]

    # ---------- Responses API (manual fallback) ----------
    # Traverse response.output blocks
    outputs = []
    if hasattr(result, "output"):
        current = []

        for item in result.output:
            # item might be an object or dict depending on SDK version
            # Assuming object access based on previous code, but let's be safe with getattr/get
            item_type = getattr(item, "type", None)
            if not item_type and isinstance(item, dict):
                item_type = item.get("type")
            
            if item_type != "message":
                continue

            content = getattr(item, "content", [])
            if not content and isinstance(item, dict):
                content = item.get("content", [])

            for block in content:
                block_type = getattr(block, "type", None)
                if not block_type and isinstance(block, dict):
                    block_type = block.get("type")
                
                if block_type == "output_text":
                    text = getattr(block, "text", "")
                    if not text and isinstance(block, dict):
                        text = block.get("text", "")
                    current.append(text)

            if current:
                outputs.append("".join(current))
                current = []

    return outputs


def print_messages(messages, user_color="cyan", ai_color="yellow", label_text_color="grey"):
    """
    Print chat messages with colored labels and text.

    Args:
        messages (list): List of message dictionaries with `role` and `content`.
        user_color (str, optional): Color for the user's message text and label background. Default is `cyan`.
        ai_color (str, optional): Color for the assistant's message text and label background. Default is `yellow`.
        label_text_color (str, optional): Color for the label text (User and Assistant). Default is `grey`.
    """
    try:
        from termcolor import colored
    except:
        print("Please install termcolor: pip install termcolor")
        return
    
    user_label = colored("[User]", label_text_color, f"on_{user_color}", attrs=["bold"])
    ai_label = colored("[Assistant]", label_text_color, f"on_{ai_color}", attrs=["bold"])

    for idx, item in enumerate(messages):
        role = item["role"]
        content = item["content"]
        newline = "\n" if idx > 0 else ""
        if role == "user":
            print(f"{newline}{user_label}:\n{colored(content, user_color)}")
        if role == "assistant":
            print(f"{newline}{ai_label}:\n{colored(content, ai_color)}")


class LLMAgent:
    """
    A powerful wrapper class for interacting with OpenAI-compatible LLM APIs.
    It handles retries, timeouts, and structured output validation.
    """
    def __init__(self,
                api_key = None,
                api_base = None,
                model_version = 'gpt-4.1-mini',
                system_prompt = 'You are a helpful assistant.',
                max_tokens = None,
                temperature = 0,
                http_client = None,
                headers = None,
                time_limit = 5*60,
                max_try = 1,
                use_responses_api = False
        ):
        """
        Initialize the LLMAgent.

        Args:
            api_key (str, optional): API Key. Defaults to `os.environ["LLM_API_KEY"]`.
            api_base (str, optional): Base URL. Defaults to `os.environ["LLM_BASE_URL"]`.
            model_version (str, optional): Model identifier. Default `'gpt-4.1-mini'`.
            system_prompt (str, optional): Default system prompt. Default `'You are a helpful assistant.'`.
            max_tokens (int, optional): Maximum tokens for generation. Default `None`.
            temperature (float, optional): Sampling temperature. Default `0`.
            http_client (httpx.Client, optional): Optional custom httpx client.
            headers (dict, optional): Optional custom headers.
            time_limit (int, optional): Timeout in seconds. Default `300` (5 minutes).
            max_try (int, optional): Default number of retries. Default `1`.
            use_responses_api (bool, optional): Whether to use the Responses API format. Default `False`.
        """
        
        # Load from environment if not provided
        if api_key is None:
            api_key = os.environ.get("LLM_API_KEY")
        if api_base is None:
            api_base = os.environ.get("LLM_BASE_URL")
            
        add_no_proxy_if_private(api_base)
        self.api_key = api_key
        self.api_base = api_base
        self.model_version = model_version
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.time_limit = time_limit
        self.max_try = max_try
        self.use_responses_api = use_responses_api
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base, http_client=http_client, default_headers=headers)


    def _llm_api_impl(self, query, system_prompt=None, **kwargs):
        image_paths = kwargs.get('image_paths', None)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        history = kwargs.get('history', None)
        n = kwargs.get('n', 1)
        if system_prompt is None:
            system_prompt = self.system_prompt
        
        if image_paths is None: # without image
            content = query
        else: # with image
            content = [
                {"type": "text", "text": query}
            ]

            for image_path in image_paths:
                try:
                    img = load_file(image_path)
                    ima_str = encode_image(img)
                except:
                    continue

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ima_str}",
                    }
                })
        
        if isinstance(history, list) and len(history) > 0:
            messages=[{"role": "system", "content": system_prompt}]+\
                history+\
                [{"role": "user", "content": content}]
        else:
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        
        use_responses_api = kwargs.get('use_responses_api', self.use_responses_api)
        if use_responses_api:
            system_prompt_content, input_blocks = messages_to_responses_input(messages)

            # Prepare arguments for responses.create
            create_kwargs = {
                "model": self.model_version,
                "input": input_blocks,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            
            if system_prompt_content:
                create_kwargs["instructions"] = system_prompt_content

            # Handle n > 1 manually for Responses API
            if n > 1:
                # Use multi_thread for parallel execution
                inp_list = [create_kwargs] * n
                responses = multi_thread(inp_list, self.client.responses.create, max_workers=min(n, 20), use_tqdm=False)
                
                assistant_responses = []
                for response in responses:
                    if response:
                        assistant_responses.extend(extract_text_outputs(response))
            else:
                response = self.client.responses.create(**create_kwargs)
                assistant_responses = extract_text_outputs(response)
        else:
            response = self.client.chat.completions.create(
                model=self.model_version,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
            )
            assistant_responses = extract_text_outputs(response)
        
        return assistant_responses

    def llm_api(self, query, system_prompt=None, **kwargs):
        """
        Executes the LLM API call with a timeout.

        Args:
            query (str): The input query.
            system_prompt (str, optional): The system prompt.
            **kwargs: Additional arguments for the API call.

        Returns:
            list[str]: A list of response strings.
        """
        return run_with_timeout(
            self._llm_api_impl, 
            args=(query, system_prompt), 
            kwargs=kwargs, 
            timeout=self.time_limit
        )

    def safe_api(self, query, system_prompt=None, return_example: Union[list, dict, str]=None, max_try=None, wait_time=0.0, **kwargs):
        """
        Sends a query to the LLM with built-in validation, parsing, and retry logic.

        Args:
            query (str): The main input text or prompt to be sent to the LLM.
            system_prompt (str, optional): The system instruction. Overrides the default if provided.
            return_example (str | list | dict, optional): A template defining the expected structure and type of the response.
                - `None` or `str` (default): Returns raw response string.
                - `list`: Expects a JSON list string. Validates element types if example elements are provided.
                - `dict`: Expects a JSON object string. Validates keys (supports fuzzy matching).
            max_try (int, optional): Max attempts. Defaults to instance's `max_try`.
            wait_time (float, optional): Time in seconds to wait between retries. Default `0.0`.
            **kwargs: Additional arguments:
                - n (int, optional): Number of completion choices. Default `1`.
                - max_tokens (int, optional): Overrides instance's `max_tokens`.
                - temperature (float, optional): Overrides instance's `temperature`.
                - image_paths (list[str], optional): List of local image paths for multimodal models.
                - history (list[dict], optional): Conversation history `[{"role": "user", "content": "..."}, ...]`.
                - use_responses_api (bool, optional): Overrides instance setting.
                - list_len (int, optional): *Validation* - Enforces exact list length.
                - list_min (int | float, optional): *Validation* - Enforces minimum value for list elements.
                - list_max (int | float, optional): *Validation* - Enforces maximum value for list elements.
                - check_keys (bool, optional): *Validation* - Whether to validate dict keys. Default `True`.

        Returns:
            str | list | dict: The parsed response from the LLM.
                - If `n > 1`, returns a list of results.
                - Returns `None` if all retries fail.
        """
        if max_try is None:
            max_try = self.max_try
            
        if return_example is not None:
            assert isinstance(return_example, Union[list, dict, str]), f"[===ERROR===][structai][llm_api.py][LLMAgent.safe_api] return_example should be list, dict or str: {type(return_example)}"

        n = kwargs.get('n', 1)
        response_list = []
        for try_idx in range(max_try):
            try:
                responses = self.llm_api(query, system_prompt, **kwargs)

                # str
                if return_example is None or isinstance(return_example, str):
                    response_list = response_list + responses
                
                # list
                elif isinstance(return_example, list):
                    for response in responses:
                        result_list = str2list(response)

                        list_len = kwargs.get('list_len', None)
                        if list_len is not None:
                            assert len(result_list) == list_len, f"[===ERROR===][structai][llm_api.py][LLMAgent.safe_api] LLM response does not match the required length: {len(result_list)} != {list_len}\nResponse: {result_list}"

                        # type check
                        if len(return_example) > 0:
                            for result_item in result_list:
                                if isinstance(return_example[0], Union[float, int]):
                                    assert isinstance(result_item, Union[float, int]), f"[===ERROR===][structai][llm_api.py][LLMAgent.safe_api] LLM response does not match the example list type: {type(result_item)} != {type(return_example[0])}\nItem: {result_item}"
                                else:
                                    assert type(result_item) == type(return_example[0]), f"[===ERROR===][structai][llm_api.py][LLMAgent.safe_api] LLM response does not match the example list type: {type(result_item)} != {type(return_example[0])}\nItem: {result_item}"

                        # range check
                        list_min = kwargs.get('list_min', None)
                        list_max = kwargs.get('list_max', None)
                        if list_min is not None or list_max is not None:
                            for result_item in result_list:
                                if list_min is not None:
                                    assert result_item >= list_min, f"[===ERROR===][structai][llm_api.py][LLMAgent.safe_api] LLM response {result_item} < list_min {list_min}"
                                if list_max is not None:
                                    assert result_item <= list_max, f"[===ERROR===][structai][llm_api.py][LLMAgent.safe_api] LLM response {result_item} > list_max {list_max}"

                        response_list.append(result_list)
                
                # dict
                elif isinstance(return_example, dict):
                    for response in responses:
                        result_dict = str2dict(response)

                        if kwargs.get('check_keys', True):
                            result_dict_correct = {}
                            for k in return_example.keys():
                                if k in result_dict:
                                    result_dict_correct[k] = result_dict[k]
                                else:
                                    for out_k in result_dict.keys():
                                        if len(k) > 5 and Levenshtein.distance(out_k.lower(), k.lower()) <= 2:
                                            result_dict_correct[k] = result_dict[out_k]
                                            break

                                assert k in result_dict_correct, f"[===ERROR===][structai][llm_api.py][LLMAgent.safe_api] LLM response does not match the example dict: missing key {k}\nResponse: {result_dict}\n"
                        else:
                            result_dict_correct = result_dict

                        response_list.append(result_dict_correct)
                
                if len(response_list) >= n:
                    response_list = response_list[:n]
                    break
            
            except Exception as e:
                print(f'[===ERROR===][safe_api][{e}]')
                if try_idx < max_try - 1:
                    time.sleep(wait_time)
        
        if len(response_list) == 0:
            return None
        
        if n == 1:
            return response_list[0]
        else:
            return response_list
        
    
    def __call__(self, query, *args, **kwargs):
        """
        Sends a query to the LLM with built-in validation, parsing, and retry logic.

        Args:
            query (str): The main input text or prompt to be sent to the LLM.
            system_prompt (str, optional): The system instruction. Overrides the default if provided.
            return_example (str | list | dict, optional): A template defining the expected structure and type of the response.
                - `None` or `str` (default): Returns raw response string.
                - `list`: Expects a JSON list string. Validates element types if example elements are provided.
                - `dict`: Expects a JSON object string. Validates keys (supports fuzzy matching).
            max_try (int, optional): Max attempts. Defaults to instance's `max_try`.
            wait_time (float, optional): Time in seconds to wait between retries. Default `0.0`.
            **kwargs: Additional arguments:
                - n (int, optional): Number of completion choices. Default `1`.
                - max_tokens (int, optional): Overrides instance's `max_tokens`.
                - temperature (float, optional): Overrides instance's `temperature`.
                - image_paths (list[str], optional): List of local image paths for multimodal models.
                - history (list[dict], optional): Conversation history `[{"role": "user", "content": "..."}, ...]`.
                - use_responses_api (bool, optional): Overrides instance setting.
                - list_len (int, optional): *Validation* - Enforces exact list length.
                - list_min (int | float, optional): *Validation* - Enforces minimum value for list elements.
                - list_max (int | float, optional): *Validation* - Enforces maximum value for list elements.
                - check_keys (bool, optional): *Validation* - Whether to validate dict keys. Default `True`.

        Returns:
            str | list | dict: The parsed response from the LLM.
                - If `n > 1`, returns a list of results.
                - Returns `None` if all retries fail.
        """
        return self.safe_api(query, *args, **kwargs)
    

    def __str__(self) -> str:
        return self.model_version.replace("/", "_")


if __name__ == '__main__':
    # python -m structai.llm_api
    print("Testing llm_api.py...")

    # Test str2dict
    print("Testing str2dict...")
    assert str2dict('{"a": 1}') == {"a": 1}, f"[===ERROR===][structai][llm_api.py][main] str2dict failed"
    assert str2dict('  {"a": 1}  ') == {"a": 1}, f"[===ERROR===][structai][llm_api.py][main] str2dict failed"
    assert str2dict("some text {'a': 1} more text") == {"a": 1}, f"[===ERROR===][structai][llm_api.py][main] str2dict failed"
    assert str2dict("{'a': 1,}") == {"a": 1}, f"[===ERROR===][structai][llm_api.py][main] str2dict failed"
    print("str2dict passed")

    # Test str2list
    print("Testing str2list...")
    assert str2list('[1, 2, 3]') == [1, 2, 3], f"[===ERROR===][structai][llm_api.py][main] str2list failed"
    assert str2list('  [1, 2, 3]  ') == [1, 2, 3], f"[===ERROR===][structai][llm_api.py][main] str2list failed"
    assert str2list("text [1, 2] text") == [1, 2], f"[===ERROR===][structai][llm_api.py][main] str2list failed"
    print("str2list passed")

    # Test print_messages
    print("Testing print_messages...")
    pmessages = [
        {"role": "user", "content": "My name is Bob."},
        {"role": "assistant", "content": "Hello Bob."}
    ]
    print_messages(pmessages)
    print("print_messages passed")

    # Test LLMAgent
    print("\nTesting LLMAgent...")
    if os.environ.get("LLM_API_KEY"):
        
        def run_test(name, func, **kwargs):
            print(f"\n[{name}]")
            # Test with use_responses_api=False
            print("  - use_responses_api=False:")
            try:
                func(use_responses_api=False, **kwargs)
            except Exception as e:
                print(f"    Error: {e}")
            
            # Test with use_responses_api=True
            print("  - use_responses_api=True:")
            try:
                func(use_responses_api=True, **kwargs)
            except Exception as e:
                print(f"    Error: {e}")

        # 1. Test gpt-4.1-mini
        def test_gpt_4_1_mini(use_responses_api):
            agent = LLMAgent(model_version='gpt-4.1-mini', max_try=1, use_responses_api=use_responses_api)
            res = agent("Say 'hello'", max_tokens=20)
            print(f"    Result: {res}")
        run_test("Test 1: gpt-4.1-mini", test_gpt_4_1_mini)

        # 2. Test gpt-5.2-pro
        def test_gpt_5_2_pro(use_responses_api):
            agent = LLMAgent(model_version='gpt-5.2-pro', max_try=1, temperature=None, use_responses_api=use_responses_api)
            res = agent("Say 'hello'", max_tokens=20)
            print(f"    Result: {res}")
        run_test("Test 2: gpt-5.2-pro", test_gpt_5_2_pro)

        # 3. Test time_limit=1, max_try=3
        def test_retry(use_responses_api):
            start_time = time.time()
            agent = LLMAgent(time_limit=1, max_try=3, use_responses_api=use_responses_api)
            res = agent("Write a 500 word essay about AI.", max_tokens=500)
            print(f"    Result: {res}")
            print(f"    Time taken: {time.time() - start_time:.2f}s")
        run_test("Test 3: time_limit=1, max_try=3 (expect retries)", test_retry)

        # 4. Test image_paths
        def test_image(use_responses_api):
            # Create dummy image
            img = Image.new('RGB', (100, 100), color='red')
            img_path = "test_image.png"
            img.save(img_path)
            
            try:
                agent = LLMAgent(model_version='gpt-4o', max_try=1, use_responses_api=use_responses_api) # Assuming gpt-4o for vision
                res = agent("What color is this image?", image_paths=[img_path], max_tokens=20)
                print(f"    Result: {res}")
            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)
        run_test("Test 4: image_paths", test_image)

        # 5. Test history
        def test_history(use_responses_api):
            agent = LLMAgent(max_try=1, use_responses_api=use_responses_api)
            history = [
                {"role": "user", "content": "My name is Bob."},
                {"role": "assistant", "content": "Hello Bob."}
            ]
            res = agent("What is my name?", history=history, max_tokens=20)
            print(f"    Result: {res}")
        run_test("Test 5: history", test_history)

        # 6. Test n=3
        def test_n_3(use_responses_api):
            agent = LLMAgent(max_try=1, use_responses_api=use_responses_api)
            res = agent("Generate a random number.", n=3, max_tokens=20)
            print(f"    Result (len={len(res) if isinstance(res, list) else 'N/A'}): {res}")
        run_test("Test 6: n=3", test_n_3)

        # 7. Test return_example (list and dict)
        def test_return_example(use_responses_api):
            agent = LLMAgent(max_try=1, use_responses_api=use_responses_api)
            # List
            print("    - List:")
            res = agent("Return the list [1, 2, 3].", return_example=[1])
            print(f"      Result: {res}")
            # Dict
            print("    - Dict:")
            res = agent("Return JSON {'a': 1}.", return_example={'a': 0})
            print(f"      Result: {res}")
        run_test("Test 7: return_example", test_return_example)

        # 8. Test list_min
        def test_list_min(use_responses_api):
            agent = LLMAgent(max_try=1, use_responses_api=use_responses_api)
            res = agent("Return the list [10, 11, 12].", return_example=[1], list_min=5, list_max=20)
            print(f"    Result: {res}")
            
            print("    - Testing failure case (list_min=20)...")
            res = agent("Return the list [10, 11, 12].", return_example=[1], list_min=20, max_try=2)
            print(f"      Result (should be None): {res}")
        run_test("Test 8: list_min", test_list_min)

    else:
        print("Skipping LLMAgent test: LLM_API_KEY not set")

    # 9. Test Validation Logic with Mock
    print("\nTesting Validation Logic with MockLLMAgent...")
    
    class MockLLMAgent(LLMAgent):
        def __init__(self, responses_map, **kwargs):
            # responses_map: query -> response string or list of strings
            super().__init__(api_key="dummy", **kwargs)
            self.responses_map = responses_map

        def llm_api(self, query, system_prompt=None, **kwargs):
            # Simple mock: return predefined response based on query
            res = self.responses_map.get(query, "{}")
            if isinstance(res, str):
                return [res]
            return res

    # Define test cases
    mock_responses = {
        "dict_exact": '{"long_name": "Alice", "years_old": 25}',
        "dict_fuzzy": '{"long_nmae": "Alice", "years_oldd": 25}', # long_nmae->long_name (dist 1), years_oldd->years_old (dist 1)
        "dict_bad": '{"wrong_name": "Alice", "wrong_age": 25}',
        "dict_missing": '{"long_name": "Alice"}',
        "list_int": '[10, 20]',
        "list_str": '["10", "20"]',
        "list_len_3": '[1, 2, 3]',
        "list_len_2": '[1, 2]',
        "list_range_ok": '[1, 5, 9]',
        "list_range_bad_min": '[-1, 5]',
        "list_range_bad_max": '[1, 11]',
    }
    
    # Use max_try=1 so that failures return None immediately
    agent = MockLLMAgent(mock_responses, max_try=1)

    # 9.1 Dict Key Tests
    print("  - Dict Key Tests:")
    # Use keys with len > 5 to trigger Levenshtein check
    example_dict = {'long_name': 'John', 'years_old': 30}
    
    # Exact match
    res = agent("dict_exact", return_example=example_dict)
    assert res == {"long_name": "Alice", "years_old": 25}, f"[===ERROR===][structai][llm_api.py][main] Dict exact match failed: {res}"
    print("    [Pass] Exact match")

    # Fuzzy match (Levenshtein)
    res = agent("dict_fuzzy", return_example=example_dict)
    # Should be corrected to match keys
    assert res == {"long_name": "Alice", "years_old": 25}, f"[===ERROR===][structai][llm_api.py][main] Dict fuzzy match failed: {res}"
    print("    [Pass] Fuzzy match (Levenshtein)")

    # Bad keys
    res = agent("dict_bad", return_example=example_dict)
    assert res is None, f"[===ERROR===][structai][llm_api.py][main] Dict bad keys should fail: {res}"
    print("    [Pass] Bad keys")

    # Missing keys
    res = agent("dict_missing", return_example=example_dict)
    assert res is None, f"[===ERROR===][structai][llm_api.py][main] Dict missing keys should fail: {res}"
    print("    [Pass] Missing keys")

    # 9.2 List Type Tests
    print("  - List Type Tests:")
    example_list_int = [1]
    
    # Correct type
    res = agent("list_int", return_example=example_list_int)
    assert res == [10, 20], f"[===ERROR===][structai][llm_api.py][main] List type ok failed: {res}"
    print("    [Pass] Correct type")

    # Incorrect type
    res = agent("list_str", return_example=example_list_int)
    assert res is None, f"[===ERROR===][structai][llm_api.py][main] List type bad should fail: {res}"
    print("    [Pass] Incorrect type")

    # 9.3 List Length Tests
    print("  - List Length Tests:")
    
    # Correct length
    res = agent("list_len_3", return_example=[1], list_len=3)
    assert res == [1, 2, 3], f"[===ERROR===][structai][llm_api.py][main] List len ok failed: {res}"
    print("    [Pass] Correct length")

    # Incorrect length
    res = agent("list_len_2", return_example=[1], list_len=3)
    assert res is None, f"[===ERROR===][structai][llm_api.py][main] List len bad should fail: {res}"
    print("    [Pass] Incorrect length")

    # 9.4 List Range Tests
    print("  - List Range Tests:")
    
    # In range
    res = agent("list_range_ok", return_example=[1], list_min=0, list_max=10)
    assert res == [1, 5, 9], f"[===ERROR===][structai][llm_api.py][main] List range ok failed: {res}"
    print("    [Pass] In range")

    # Bad min
    res = agent("list_range_bad_min", return_example=[1], list_min=0)
    assert res is None, f"[===ERROR===][structai][llm_api.py][main] List range bad min should fail: {res}"
    print("    [Pass] Bad min")

    # Bad max
    res = agent("list_range_bad_max", return_example=[1], list_max=10)
    assert res is None, f"[===ERROR===][structai][llm_api.py][main] List range bad max should fail: {res}"
    print("    [Pass] Bad max")
    
    print("llm_api.py tests completed.")
    print("--------------------------------------------")
