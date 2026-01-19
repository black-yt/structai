# StructAI

StructAI is a utility package for AI development, providing a collection of helper functions and classes for file I/O, LLM interactions, multiprocessing, and more.

## Installation

You can install StructAI directly from the source code:

```bash
git clone https://github.com/black-yt/structai.git
cd structai
pip install -e .
```

## Usage

### File I/O (`structai.io`)

Helper functions for reading and writing various file formats.

-   **`load_file(path)`**
    Automatically reads files based on their extension. Supported formats: `.json`, `.jsonl`, `.csv`, `.txt`, `.md`, `.pkl`, `.parquet`, `.xlsx`, `.py`, `.npy`, `.pt`, `.png`, `.jpg`, `.jpeg`.

-   **`save_file(data, path)`**
    Automatically saves data to a file based on the extension. Supported formats: same as `load_file`.

-   **`print_once(msg)`**
    Prints a message only once during the execution of the program.

-   **`make_print_once()`**
    Returns a function that prints a message only once. Useful for creating local "print once" scopes.

### LLM API (`structai.llm_api`)

Utilities for interacting with Large Language Models (LLMs), specifically OpenAI-compatible APIs.

-   **`LLMAgent` (Class)**
    A wrapper around the OpenAI client for easier interaction with LLMs.
    -   `__init__(api_key=None, api_base=None, model_version='gpt-4.1-mini', system_prompt='...', ...)`
    -   `__call__(query, ...)`: Alias for `safe_api`.
    -   `llm_api(query, ...)`: Makes a request to the LLM with timeout handling.
    -   `safe_api(query, return_example=None, max_try=1, ...)`: Makes a request with retries and output validation (type checking, range checking, etc.) based on `return_example`.

-   **`sanitize_text(text)`**
    Sanitizes text by keeping only allowed characters (ASCII English, digits, punctuation).

-   **`str2dict(s)`**
    Converts a string representation of a dictionary to a Python dict, handling potential formatting issues.

-   **`str2list(s)`**
    Converts a string representation of a list to a Python list.

-   **`add_no_proxy_if_private(url)`**
    Adds the host of a URL to the `no_proxy` environment variable if it's a private IP.

-   **`read_image(image_path)`**
    Reads an image using PIL.

-   **`encode_image(image_obj)`**
    Encodes a PIL image to a base64 string.

-   **`messages_to_responses_input(messages)`**
    Converts Chat Completions messages format to Responses API input format.

-   **`extract_text_outputs(result)`**
    Extracts text outputs from Chat Completions or Responses API results.

### Multiprocessing (`structai.mp`)

Utilities for parallel execution.

-   **`multi_thread(inp_list, function, max_workers=40, use_tqdm=True)`**
    Executes a function concurrently using threads. Returns a list of results in the same order as `inp_list`.

-   **`multi_process(inp_list, function, max_workers=40, use_tqdm=True)`**
    Executes a function concurrently using processes. Returns a list of results in the same order as `inp_list`.

### OpenAI Server (`structai.openai_server`)

-   **`run_server(host="0.0.0.0", port=8001)`**
    Runs a FastAPI-based OpenAI-compatible proxy server.

### General Utilities (`structai.utils`)

-   **`run_with_timeout(func, args=(), kwargs=None, timeout=None)`**
    Runs a function with a time limit. Raises `TimeoutError` if it exceeds the limit.

-   **`timeout_limit(timeout=None)`**
    A decorator to enforce a time limit on a function.

-   **`remove_tag(s, tags=["<think>", "</think>", "<answer>", "</answer>"], r="\n")`**
    Removes specified tags from a string.

-   **`parse_think_answer(text)`**
    Parses a string containing `<think>` and `<answer>` tags, returning the content of both.

-   **`extract_within_tags(content, start_tag='<answer>', end_tag='</answer>', default_return=None)`**
    Extracts content between two tags.

-   **`get_all_file_paths(directory, suffix='')`**
    Recursively finds all files with a specific suffix in a directory.

## License

MIT License
