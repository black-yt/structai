from functools import wraps
import threading
import os


def run_with_timeout(func, args=(), kwargs=None, timeout=None):
    """
    Run a function with a timeout limit.
    
    Args:
        func (callable): The function to run.
        args (tuple): Positional arguments for the function.
        kwargs (dict): Keyword arguments for the function.
        timeout (float or None): Maximum allowed execution time in seconds.
    
    Returns:
        The function's return value.
    
    Raises:
        TimeoutError: If execution exceeds the allowed time.
        Exception: Any exception raised by the function.
    """
    if kwargs is None:
        kwargs = {}
        
    if timeout is None:
        return func(*args, **kwargs)

    result = [None]
    exc = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exc[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"[{str(func.__name__)} timeout after {timeout}s]")

    if exc[0] is not None:
        raise exc[0]

    return result[0]


def timeout_limit(timeout=None):
    """
    A decorator for enforcing function execution time limits.

    Args:
        timeout (float or None): Maximum allowed execution time in seconds.
                                If None, no time limit is applied.

    Returns:
        The function's return value if completed in time.
        Raises TimeoutError if execution exceeds the allowed time.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return run_with_timeout(func, args, kwargs, timeout)
        return wrapper

    return decorator


def remove_tag(s: str, tags: list[str] = ["<think>", "</think>", "<answer>", "</answer>"], r: str = "\n"):
    for tag in tags:
        s = s.replace(tag, r)
    return s.strip(r).strip()


def parse_think_answer(text):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    
    text = text.strip()
    
    tag_think_start = "<think>"
    tag_think_end = "</think>"
    tag_answer_start = "<answer>"
    tag_answer_end = "</answer>"
    
    idx_answer_start = text.find(tag_answer_start)
    idx_think_end = text.find(tag_think_end)
    
    think_raw = ""
    answer_raw = ""
    
    if idx_answer_start != -1:
        think_raw = text[:idx_answer_start]
        answer_raw = text[idx_answer_start + len(tag_answer_start):]
        
        idx_answer_end = answer_raw.find(tag_answer_end)
        if idx_answer_end != -1:
            answer_raw = answer_raw[:idx_answer_end]
            
    elif idx_think_end != -1:
        think_raw = text[:idx_think_end]
        answer_raw = text[idx_think_end + len(tag_think_end):]
        
        idx_answer_end = answer_raw.find(tag_answer_end)
        if idx_answer_end != -1:
            answer_raw = answer_raw[:idx_answer_end]
            
    else:
        raise ValueError("Could not parse response: missing <answer> or </think> tags.")
        
    idx_think_start = think_raw.find(tag_think_start)
    if idx_think_start != -1:
        think_raw = think_raw[idx_think_start + len(tag_think_start):]

    think = remove_tag(think_raw, [tag_think_start, tag_think_end, tag_answer_start, tag_answer_end], "")
    answer = remove_tag(answer_raw, [tag_think_start, tag_think_end, tag_answer_start, tag_answer_end], "")
    
    if not think:
        raise ValueError("Parsed think content is empty")
    if not answer:
        raise ValueError("Parsed answer content is empty")
        
    return think, answer


def extract_within_tags(content: str, start_tag='<answer>', end_tag='</answer>', default_return=None):
    content = str(content)
    start_index = content.rfind(start_tag)
    if start_index != -1:
        end_index = content.find(end_tag, start_index)
        if end_index != -1:
            return content[start_index + len(start_tag):end_index].strip()
    return default_return


def get_all_file_paths(directory, suffix=''):
    """
    Get all file paths with the specified suffix in the directory.
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                file_paths.append(os.path.join(root, file))
    return file_paths


if __name__ == "__main__":
    # python -m structai.utils
    print("Testing utils.py...")
    import time
    
    # Test run_with_timeout
    print("Testing run_with_timeout...")
    def slow_func(seconds):
        time.sleep(seconds)
        return "done"
        
    # Should pass
    res = run_with_timeout(slow_func, args=(0.1,), timeout=1.0)
    assert res == "done", f"[===ERROR===][structai][utils.py][main] run_with_timeout failed: {res} != done"
    
    # Should fail
    try:
        run_with_timeout(slow_func, args=(0.5,), timeout=0.1)
        print("run_with_timeout failed to timeout (unexpected)")
    except TimeoutError:
        print("run_with_timeout correctly timed out")
        
    # Test timeout_limit decorator
    print("Testing timeout_limit decorator...")
    @timeout_limit(timeout=0.2)
    def decorated_slow_func(seconds):
        time.sleep(seconds)
        return "done"
        
    try:
        decorated_slow_func(0.5)
        print("timeout_limit failed to timeout (unexpected)")
    except TimeoutError:
        print("timeout_limit correctly timed out")
        
    # Test remove_tag
    print("Testing remove_tag...")
    text = "<think>thinking...</think> <answer>42</answer>"
    cleaned = remove_tag(text)
    assert "thinking..." in cleaned and "42" in cleaned, f"[===ERROR===][structai][utils.py][main] remove_tag failed: {cleaned}"
    assert "<think>" not in cleaned, f"[===ERROR===][structai][utils.py][main] remove_tag failed: {cleaned}"
    print("remove_tag passed")
    
    # Test parse_think_answer
    print("Testing parse_think_answer...")
    text = "<think>This is thinking</think><answer>This is answer</answer>"
    think, answer = parse_think_answer(text)
    assert think == "This is thinking", f"[===ERROR===][structai][utils.py][main] parse_think_answer failed: {think} != This is thinking"
    assert answer == "This is answer", f"[===ERROR===][structai][utils.py][main] parse_think_answer failed: {answer} != This is answer"
    print("parse_think_answer passed")
    
    # Test extract_within_tags
    print("Testing extract_within_tags...")
    text = "prefix <tag>content</tag> suffix"
    extracted = extract_within_tags(text, "<tag>", "</tag>")
    assert extracted == "content", f"[===ERROR===][structai][utils.py][main] extract_within_tags failed: {extracted} != content"
    print("extract_within_tags passed")
    
    print("utils.py tests completed.")
