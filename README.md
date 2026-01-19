# StructAI

StructAI is a comprehensive utility package for AI development, offering a robust set of tools for file operations, LLM interactions, parallel processing, and general programming tasks.

## Installation

> **Recommended for most users.** Installs the latest stable release from PyPI.
```bash
pip install structai
```

> **For development.** Installs StructAI in editable mode from source, enabling live code changes.

```bash
git clone https://github.com/black-yt/structai.git
cd structai
pip install -e .
```

## Usage

> **Note:** Before using LLM-related features, please ensure you have set the necessary environment variables:
> ```bash
> export LLM_API_KEY="your-api-key"
> export LLM_BASE_URL="your-api-base-url"
> ```

#### `load_file(path)`

Automatically reads a file based on its extension.

**Supported formats:** `.json`, `.jsonl`, `.csv`, `.txt`, `.md`, `.pkl`, `.parquet`, `.xlsx`, `.py`, `.npy`, `.pt`, `.png`, `.jpg`, `.jpeg`.

```python
from structai import load_file

# Load a JSON file
data = load_file("config.json")

# Load a CSV file as a pandas DataFrame
df = load_file("data.csv")

# Load an image
image = load_file("photo.jpg")
```

#### `save_file(data, path)`

Automatically saves data to a file based on the extension.

```python
from structai import save_file

data = {"key": "value"}

# Save as JSON
save_file(data, "output.json")

# Save as Pickle
save_file(data, "backup.pkl")
```

#### `print_once(msg)`

Prints a message to stdout only the first time it is called. Useful for logging inside loops.

```python
from structai import print_once

for i in range(10):
    print_once("Starting processing...") # Prints only once
    # process(i)
```

#### `make_print_once()`

Returns a new function that prints a message only once. This allows for creating local "print once" scopes.

```python
from structai import make_print_once

logger1 = make_print_once()
logger2 = make_print_once()

logger1("Hello") # Prints "Hello"
logger1("Hello") # Does nothing

logger2("World") # Prints "World"
```

#### `LLMAgent`

A powerful wrapper class for interacting with OpenAI-compatible LLM APIs. It handles retries, timeouts, and structured output validation.

**Initialization:**

```python
from structai import LLMAgent

agent = LLMAgent(
    api_key="sk-...",              # Optional if LLM_API_KEY env var is set
    api_base="https://...",        # Optional if LLM_BASE_URL env var is set
    model_version='gpt-4.1-mini',  # Default model
    system_prompt='You are a helpful assistant.',
    temperature=0,
    time_limit=300,                # Timeout in seconds
    max_try=1                      # Number of retries
)
```

**Basic Usage (`__call__` or `safe_api`):**

```python
response = agent("Generate a random number.", n=3, temperature=1)
print(response)
# Output: ["Sure! Here's a random number for you: 738", "Sure! Here's a random number: 7382", "Sure! Here's a random number: 487."]
```

**Structured Output Validation:**

You can enforce the output format (List, Dict, or specific types) using `return_example`.

```python
# Enforce a list of integers
numbers = agent(
    "Generate 3 random numbers, for example, [1, 2, 3].", 
    return_example=[1], 
    list_len=3
)
# Output: [10, 42, 7]

# Enforce a dictionary with specific keys
profile = agent(
    "Create a user profile for Alice, for example, {'name': Alice, 'age': 1, 'city': 'shanghai'}.", 
    return_example={"name": "str", "age": 1, "city": "str"}
)
# Output: {'name': 'Alice', 'age': 25, 'city': 'New York'}
```

**Multimodal Input:**

```python
# Pass image paths for vision models
description = agent(
    "Describe this image", 
    image_paths=["image.jpg"]
)
```

**Memory Context:**

```python
history = [
    {"role": "user", "content": "My name is Bob."},
    {"role": "assistant", "content": "Hello Bob."}
]
answer = agent(
    "What is my name?", 
    history=history, 
)
```

#### `sanitize_text(text)`

Sanitizes text by keeping only ASCII English characters, digits, and common punctuation. Removes control characters and ANSI codes.

```python
from structai import sanitize_text

clean = sanitize_text("Hello \x1b[31mWorld\x1b[0m!")
print(clean) # "Hello World!"
```

#### `str2dict(s)`

Robustly converts a string representation of a dictionary to a Python `dict`. It handles common formatting errors and uses `json_repair` as a fallback.

```python
from structai import str2dict

d = str2dict("{'a': 1, 'b': 2}")
print(d['a']) # 1
```

#### `str2list(s)`

Robustly converts a string representation of a list to a Python `list`.

```python
from structai import str2list

l = str2list("[1, 2, 3]")
print(len(l)) # 3
```

#### `add_no_proxy_if_private(url)`

Checks if the hostname in the URL is a private IP address. If so, it adds it to the `no_proxy` environment variable to bypass proxies.

```python
from structai import add_no_proxy_if_private

add_no_proxy_if_private("http://192.168.1.100:8080/v1")
```

#### `read_image(image_path)`

Reads an image from a path and returns a PIL Image object.

```python
from structai import read_image

img = read_image("photo.jpg")
```

#### `encode_image(image_obj)`

Encodes a PIL Image object into a base64 string.

```python
from structai import encode_image

b64_str = encode_image(img)
```

#### `messages_to_responses_input(messages)`

Converts standard Chat Completions `messages` format (list of dicts) to the input format required by the Responses API.

```python
from structai import messages_to_responses_input

messages = [{"role": "user", "content": "Hello"}]
system_prompt, input_blocks = messages_to_responses_input(messages)
```

#### `extract_text_outputs(result)`

Extracts the text content from an LLM API response object (supports both Chat Completions and Responses API formats).

```python
from structai import extract_text_outputs

# Assuming 'response' is the object returned by the OpenAI client
texts = extract_text_outputs(response)
print(texts[0])
```

#### `multi_thread(inp_list, function, max_workers=40, use_tqdm=True)`

Executes a function concurrently for each item in `inp_list` using a thread pool.

```python
from structai import multi_thread
import time

def square(x):
    return x * x

inputs = [{"x": i} for i in range(10)]
results = multi_thread(inputs, square, max_workers=4)
print(results) # [0, 1, 4, 9, ...]
```

#### `multi_process(inp_list, function, max_workers=40, use_tqdm=True)`

Executes a function concurrently for each item in `inp_list` using a process pool. Ideal for CPU-bound tasks.

```python
from structai import multi_process

def heavy_computation(n):
    return sum(range(n))

inputs = [{"n": 1000000} for _ in range(5)]
results = multi_process(inputs, heavy_computation)
```

#### `run_server(host="0.0.0.0", port=8001)`

Starts a FastAPI server that acts as a proxy to an OpenAI-compatible LLM provider.

```python
from structai import run_server

if __name__ == "__main__":
    run_server()
```

#### `timeout_limit(timeout=None)`

A decorator that enforces a maximum execution time on a function. Raises `TimeoutError` if the limit is exceeded.

```python
from structai import timeout_limit
import time

@timeout_limit(timeout=2.0)
def task():
    time.sleep(5)

# This will raise TimeoutError
task()
```

#### `run_with_timeout(func, args=(), kwargs=None, timeout=None)`

Runs a function with a specified timeout without using a decorator.

```python
from structai import run_with_timeout

def task(x):
    return x * 2

result = run_with_timeout(task, args=(10,), timeout=1.0)
```

#### `parse_think_answer(text)`

Parses a string containing Chain-of-Thought tags (`<think>...</think>` and `<answer>...</answer>`) and returns the content of both.

```python
from structai import parse_think_answer

raw_text = "<think>Step 1...</think><answer>42</answer>"
think, answer = parse_think_answer(raw_text)
print(f"Reasoning: {think}")
print(f"Result: {answer}")
```

#### `extract_within_tags(content, start_tag='<answer>', end_tag='</answer>', default_return=None)`

Extracts the substring found between two specific tags.

```python
from structai import extract_within_tags

text = "Result: <json>{...}</json>"
json_str = extract_within_tags(text, "<json>", "</json>")
```

#### `get_all_file_paths(directory, suffix='')`

Recursively retrieves all file paths in a directory that match a given suffix.

```python
from structai import get_all_file_paths

# Get all Python files in the current directory
py_files = get_all_file_paths(".", suffix=".py")
print(py_files)
```

#### `remove_tag(s, tags=["<think>", "</think>", "<answer>", "</answer>"], r="\n")`

Removes specified tags from a string, replacing them with a separator (default newline).

```python
from structai import remove_tag

clean_text = remove_tag("<think>...</think> Answer")
```

## License

[MIT License](LICENSE)