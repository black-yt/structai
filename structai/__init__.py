from .io import load_file, save_file, print_once, make_print_once
from .llm_api import LLMAgent, sanitize_text, filter_excessive_repeats, str2dict, str2list, add_no_proxy_if_private, read_image, encode_image, messages_to_responses_input, extract_text_outputs
from .mp import multi_thread, multi_process
from .openai_server import run_server
from .utils import timeout_limit, run_with_timeout, parse_think_answer, extract_within_tags, get_all_file_paths, remove_tag

def structai_skill():
    return """## StructAI Library Documentation

StructAI is a comprehensive utility package for AI development, offering a robust set of tools for file operations, LLM interactions, parallel processing, and general programming tasks.

### `structai_skill`

Returns a comprehensive documentation string for the StructAI library in Markdown format. This is useful for providing context to LLMs about the available tools in this library.

*   **Args**:
    *   None
*   **Returns**:
    *   (str): The documentation string.

*   **Example**:
```python
from structai import structai_skill

docs = structai_skill()
print(docs)
```


### `load_file`
Automatically reads a file based on its extension.

*   **Args**:
    *   `path` (str): The path to the file to be read.
*   **Returns**:
    *   (Any): The content of the file, parsed into an appropriate Python object.
        *   `.json` -> `dict` or `list`
        *   `.jsonl` -> `list` of dicts
        *   `.csv`, `.parquet`, `.xlsx` -> `pandas.DataFrame`
        *   `.txt`, `.md`, `.py` -> `str`
        *   `.pkl` -> unpickled object
        *   `.npy` -> `numpy.ndarray`
        *   `.pt` -> `torch` object
        *   `.png`, `.jpg`, `.jpeg` -> `PIL.Image.Image`

*   **Example**:
```python
from structai import load_file

# Load a JSON file
data = load_file("config.json")

# Load a CSV file as a pandas DataFrame
df = load_file("data.csv")

# Load an image
image = load_file("photo.jpg")
```

### `save_file`
Automatically saves data to a file based on the extension. Creates necessary directories if they don't exist.

*   **Args**:
    *   `data` (Any): The data object to save.
    *   `path` (str): The destination file path.
*   **Returns**:
    *   None

*   **Example**:
```python
from structai import save_file

data = {"key": "value"}

# Save as JSON
save_file(data, "output.json")

# Save as Pickle
save_file(data, "backup.pkl")
```

### `print_once`
Prints a message to stdout only once during the entire program execution. Useful for logging warnings or info inside loops.

*   **Args**:
    *   `msg` (str): The message to print.
*   **Returns**:
    *   None

*   **Example**:
```python
from structai import print_once

for i in range(10):
    print_once("Starting processing...") # print only once
```

### `make_print_once`
Creates and returns a local function that prints a message only once. This is useful if you need a "print once" behavior scoped to a specific function or instance rather than globally.

*   **Args**:
    *   None
*   **Returns**:
    *   (callable): A function `inner(msg)` that behaves like `print_once`.

*   **Example**:
```python
from structai import make_print_once

logger1 = make_print_once()
logger2 = make_print_once()

logger1("Hello") # Prints "Hello"
logger1("Hello") # Does nothing

logger2("World") # Prints "World"
logger2("World") # Does nothing
```

### `LLMAgent` Class

A powerful wrapper class for interacting with OpenAI-compatible LLM APIs. It handles retries, timeouts, and structured output validation.

#### `initialization`

*   **Args**:
    *   `api_key` (str, optional): API Key. Defaults to `os.environ["LLM_API_KEY"]`.
    *   `api_base` (str, optional): Base URL. Defaults to `os.environ["LLM_BASE_URL"]`.
    *   `model_version` (str, optional): Model identifier. Default `'gpt-4.1-mini'`.
    *   `system_prompt` (str, optional): Default system prompt. Default `'You are a helpful assistant.'`.
    *   `max_tokens` (int, optional): Maximum tokens for generation. Default `None`.
    *   `temperature` (float, optional): Sampling temperature. Default `0`.
    *   `http_client` (httpx.Client, optional): Optional custom httpx client.
    *   `headers` (dict, optional): Optional custom headers.
    *   `time_limit` (int, optional): Timeout in seconds. Default `300` (5 minutes).
    *   `max_try` (int, optional): Default number of retries. Default `1`.
    *   `use_responses_api` (bool, optional): Whether to use the Responses API format. Default `False`.

*   **Returns**:
    *   (LLMAgent): LLMAgent instance.

*   **Example**:
```python
from structai import LLMAgent

agent = LLMAgent()
```

#### `__call__`
Sends a query to the LLM with built-in validation, parsing, and retry logic.


*   **Args**:
    *   `query` (str): The main input text or prompt to be sent to the LLM.
    *   `system_prompt` (str, optional): The system instruction. Overrides the default if provided.
    *   `return_example` (str | list | dict, optional): A template defining the expected structure and type of the response.
        *   `None` or `str` (default): Returns raw response string.
        *   `list`: Expects a JSON list string. Validates element types if example elements are provided.
        *   `dict`: Expects a JSON object string. Validates keys (supports fuzzy matching).
    *   `max_try` (int, optional): Max attempts. Defaults to instance's `max_try`.
    *   `wait_time` (float, optional): Time in seconds to wait between retries. Default `0.0`.
    *   `n` (int, optional): Number of completion choices. Default `1`.
    *   `max_tokens` (int, optional): Overrides instance's `max_tokens`.
    *   `temperature` (float, optional): Overrides instance's `temperature`.
    *   `image_paths` (list[str], optional): List of local image paths for multimodal models.
    *   `history` (list[dict], optional): Conversation history `[{"role": "user", "content": "..."}, ...]`.
    *   `use_responses_api` (bool, optional): Overrides instance setting.
    *   `list_len` (int, optional): *Validation* - Enforces exact list length.
    *   `list_min` (int | float, optional): *Validation* - Enforces minimum value for list elements.
    *   `list_max` (int | float, optional): *Validation* - Enforces maximum value for list elements.
    *   `check_keys` (bool, optional): *Validation* - Whether to validate dict keys. Default `True`.

*   **Returns**:
    *   (str | list | dict): The parsed response from the LLM.
        *   If `n > 1`, returns a list of results.
        *   Returns `None` if all retries fail.

*   **Example**:
```python
# Basic usage
response = agent("Generate a random number.", n=3, temperature=1)
# Output: ["Sure! Here's a random number for you: 738", "Sure! Here's a random number: 7382", "Sure! Here's a random number: 487."]

# Enforce the output format (List, Dict, or specific types) using `return_example`. Note that the output format needs to be explicitly specified in the prompt.
numbers = agent(
    "Generate 3 random numbers, for example, [1, 2, 3].", 
    return_example=[1], 
    list_len=3
)
# Output: [10, 42, 7]

profile = agent(
    "Create a user profile for Alice, for example, {'name': Alice, 'age': 1, 'city': 'shanghai'}.", 
    return_example={"name": "str", "age": 1, "city": "str"}
)
# Output: {'name': 'Alice', 'age': 25, 'city': 'New York'}

# Multimodal input for vision models
description = agent(
    "Describe these images", 
    image_paths=["path/to/image_1.jpg", "path/to/image_2.jpg"]
)

# Memory context
history = [
    {"role": "user", "content": "My name is Bob."},
    {"role": "assistant", "content": "Hello Bob."}
]
answer = agent(
    "What is my name?", 
    history=history, 
)
# Output: 'Your name is Bob.'
```

### `sanitize_text`

Sanitizes text by keeping only ASCII English characters, digits, and common punctuation. Removes control characters and ANSI codes.

*   **Args**:
    *   `text` (str): The text to sanitize.
*   **Returns**:
    *   (str): The sanitized text.

*   **Example**:
```python
from structai import sanitize_text

clean = sanitize_text("Hello \x1b[31mWorld\x1b[0m!")
print(clean) # 'Hello [31mWorld[0m!'
```

### `filter_excessive_repeats`

Identifies sequences where a single character repeats more than the specified threshold and removes them entirely from the string.

*   **Args**:
    *   `text` (str): The input string.
    *   `threshold` (int, optional): The maximum allowed consecutive repetitions. Default `5`.
*   **Returns**:
    *   (str): The processed string with excessive repetitions removed.

*   **Example**:
```python
from structai import filter_excessive_repeats

clean = filter_excessive_repeats("Helloooooo World", threshold=5)
print(clean) # "Hell World"
```

### `str2dict`

Robustly converts a string representation of a dictionary to a Python `dict`. It handles common formatting errors and uses `json_repair` as a fallback.

*   **Args**:
    *   `s` (str): The string representation of a dictionary.
*   **Returns**:
    *   (dict): The parsed dictionary.

*   **Example**:
```python
from structai import str2dict

d = str2dict("{'a': 1, 'b': 2}")
print(d['a']) # 1
```

### `str2list`

Robustly converts a string representation of a list to a Python `list`.

*   **Args**:
    *   `s` (str): The string representation of a list.
*   **Returns**:
    *   (list): The parsed list.

*   **Example**:
```python
from structai import str2list

l = str2list("[1, 2, 3]")
print(len(l)) # 3
```

### `add_no_proxy_if_private`

Checks if the hostname in the URL is a private IP address. If so, it adds it to the `no_proxy` environment variable to bypass proxies.

*   **Args**:
    *   `url` (str): The URL to check.
*   **Returns**:
    *   None

*   **Example**:
```python
from structai import add_no_proxy_if_private

add_no_proxy_if_private("http://192.168.1.100:8080/v1")
```

### `read_image`

Reads an image from a path and returns a PIL Image object.

*   **Args**:
    *   `image_path` (str): The path to the image file.
*   **Returns**:
    *   (PIL.Image.Image): The loaded image object.

*   **Example**:
```python
from structai import read_image

img = read_image("photo.jpg")
```

### `encode_image`

Encodes a PIL Image object into a base64 string.

*   **Args**:
    *   `image_obj` (PIL.Image.Image): The image object to encode.
*   **Returns**:
    *   (str): The base64 encoded string.

*   **Example**:
```python
from structai import encode_image

b64_str = encode_image(img)
```

### `messages_to_responses_input`

Converts standard Chat Completions `messages` format (list of dicts) to the input format required by the Responses API.

*   **Args**:
    *   `messages` (list[dict]): List of message dictionaries with 'role' and 'content'.
*   **Returns**:
    *   (tuple): A tuple containing `(system_prompt_content, input_blocks)`.

*   **Example**:
```python
from structai import messages_to_responses_input

messages = [{"role": "user", "content": "Hello"}]
system_prompt, input_blocks = messages_to_responses_input(messages)
```

### `extract_text_outputs`

Extracts the text content from an LLM API response object (supports both Chat Completions and Responses API formats).

*   **Args**:
    *   `result` (object): The response object from the LLM API.
*   **Returns**:
    *   (list[str]): A list of extracted text outputs.

*   **Example**:
```python
from structai import extract_text_outputs

# Assuming 'response' is the object returned by the OpenAI client
texts = extract_text_outputs(response)
print(texts[0])
```

### `multi_thread`

Executes a function concurrently for each item in `inp_list` using a thread pool.

*   **Args**:
    *   `inp_list` (list[dict]): A list of dictionaries, where each dictionary contains keyword arguments for `function`.
    *   `function` (callable): The function to execute.
    *   `max_workers` (int, optional): The maximum number of threads. Default `40`.
    *   `use_tqdm` (bool, optional): Whether to show a progress bar. Default `True`.
*   **Returns**:
    *   (list): A list of results corresponding to the input list order.

*   **Example**:
```python
from structai import multi_thread
import time

def square(x):
    return x * x

inputs = [{"x": i} for i in range(10)]
results = multi_thread(inputs, square, max_workers=4)
print(results) # [0, 1, 4, 9, ...]
```

### `multi_process`

Executes a function concurrently for each item in `inp_list` using a process pool. Ideal for CPU-bound tasks.

*   **Args**:
    *   `inp_list` (list[dict]): A list of dictionaries, where each dictionary contains keyword arguments for `function`.
    *   `function` (callable): The function to execute.
    *   `max_workers` (int, optional): The maximum number of processes. Default `40`.
    *   `use_tqdm` (bool, optional): Whether to show a progress bar. Default `True`.
*   **Returns**:
    *   (list): A list of results corresponding to the input list order.

*   **Example**:
```python
from structai import multi_process

# 'heavy_computation' must be defined at the top level for multiprocessing pickling.
def heavy_computation(n):
    return sum(range(n))

inputs = [{"n": 1000} for _ in range(5)]
results = multi_process(inputs, heavy_computation)
```

### `run_server`

Starts a FastAPI server that acts as a proxy to an OpenAI-compatible LLM provider using LLM_BASE_URL and LLM_API_KEY in environment variables.

*   **Args**:
    *   `host` (str, optional): The host to bind to. Default `"0.0.0.0"`.
    *   `port` (int, optional): The port to bind to. Default `8001`.
*   **Returns**:
    *   None (Runs indefinitely until stopped).

*   **Example**:
```python
from structai import run_server

if __name__ == "__main__":
    run_server()
```

### `timeout_limit`

A decorator that enforces a maximum execution time on a function. Raises `TimeoutError` if the limit is exceeded.

*   **Args**:
    *   `timeout` (float | None): Maximum allowed execution time in seconds.
*   **Returns**:
    *   (decorator): A decorator function that wraps the target function.

*   **Example**:
```python
from structai import timeout_limit
import time

@timeout_limit(timeout=2.0)
def task():
    time.sleep(5)

# This will raise TimeoutError
task()
```

### `run_with_timeout`

Runs a function with a specified timeout without using a decorator.

*   **Args**:
    *   `func` (callable): The function to run.
    *   `args` (tuple, optional): Positional arguments for the function. Default `()`.
    *   `kwargs` (dict, optional): Keyword arguments for the function. Default `None`.
    *   `timeout` (float | None): Maximum allowed execution time in seconds.
*   **Returns**:
    *   (Any): The return value of the function.

*   **Example**:
```python
from structai import run_with_timeout

def task(x):
    return x * 2

result = run_with_timeout(task, args=(10,), timeout=1.0)
```

### `remove_tag`

Removes specified tags from a string, replacing them with a separator (default newline).

*   **Args**:
    *   `s` (str): The input string.
    *   `tags` (list[str], optional): A list of tags to remove. Default `["<think>", "</think>", "<answer>", "</answer>"]`.
    *   `r` (str, optional): The replacement string. Default `"\n"`.
*   **Returns**:
    *   (str): The cleaned string.

*   **Example**:
```python
from structai import remove_tag

clean_text = remove_tag("<think>...</think> Answer")
# Output: "...\n Answer"
```

### `parse_think_answer`

Parses a string containing Chain-of-Thought tags (`<think>...</think>` and `<answer>...</answer>`) and returns the content of both.

*   **Args**:
    *   `text` (str): The input text containing the tags.
*   **Returns**:
    *   (tuple): A tuple `(think_content, answer_content)`.

*   **Example**:
```python
from structai import parse_think_answer

raw_text = "<think>Step 1...</think><answer>42</answer>"
think, answer = parse_think_answer(raw_text)
print(f"Reasoning: {think}") # Reasoning: Step 1...
print(f"Result: {answer}") # Result: 42
```

### `extract_within_tags`

Extracts the substring found between two specific tags.

*   **Args**:
    *   `content` (str): The text to search within.
    *   `start_tag` (str, optional): The opening tag. Default `'<answer>'`.
    *   `end_tag` (str, optional): The closing tag. Default `'</answer>'`.
    *   `default_return` (Any, optional): The value to return if tags are not found. Default `None`.
*   **Returns**:
    *   (str | Any): The extracted content string, or `default_return` if not found.

*   **Example**:
```python
from structai import extract_within_tags

text = "Result: <json>{...}</json>"
json_str = extract_within_tags(text, "<json>", "</json>")
# Output: "{...}"
```

### `get_all_file_paths`

Recursively retrieves all file paths in a directory that match a given suffix.

*   **Args**:
    *   `directory` (str): The root directory to search.
    *   `suffix` (str, optional): The file suffix to filter by (e.g., '.py'). Default `''` (matches all files).
*   **Returns**:
    *   (list[str]): A list of matching file paths.

*   **Example**:
```python
from structai import get_all_file_paths

# Get all Python files in the current directory
py_files = get_all_file_paths(".", suffix=".py")
print(py_files)
```"""

__all__ = [
    "structai_skill",
    "load_file",
    "save_file",
    "print_once",
    "make_print_once",
    "LLMAgent",
    "filter_excessive_repeats",
    "sanitize_text",
    "str2dict",
    "str2list",
    "add_no_proxy_if_private",
    "read_image",
    "encode_image",
    "messages_to_responses_input",
    "extract_text_outputs",
    "multi_thread",
    "multi_process",
    "run_server",
    "timeout_limit",
    "run_with_timeout",
    "parse_think_answer",
    "extract_within_tags",
    "get_all_file_paths",
    "remove_tag",
]
