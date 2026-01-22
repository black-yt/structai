import os
import json
import pickle


def load_file(path):
    """
    Automatically reads a file based on its extension.

    Supported formats:
    - .json: Returns dict or list.
    - .jsonl: Returns list of dicts.
    - .csv: Returns pandas.DataFrame.
    - .txt, .md, .py: Returns str.
    - .pkl: Returns unpickled object.
    - .parquet: Returns pandas.DataFrame.
    - .xlsx: Returns pandas.DataFrame.
    - .npy: Returns numpy.ndarray.
    - .pt: Returns torch object.
    - .png, .jpg, .jpeg: Returns PIL.Image.Image.

    Args:
        path (str): The path to the file to be read.

    Returns:
        Any: The content of the file, parsed into an appropriate Python object.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)

    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [json.loads(line) for line in f]

    elif ext == ".csv":
        import pandas as pd
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")

    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    elif ext == ".md":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    elif ext == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)

    elif ext == ".parquet":
        import pandas as pd
        return pd.read_parquet(path)

    elif ext == ".xlsx":
        import pandas as pd
        return pd.read_excel(path)

    elif ext == ".py":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    elif ext == ".npy":
        import numpy as np
        return np.load(path)

    elif ext == ".pt":
        import torch
        return torch.load(path)

    elif ext in [".png", ".jpg", ".jpeg"]:
        from PIL import Image
        return Image.open(path)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_file(data, path):
    """
    Automatically saves data to a file based on the extension. Creates necessary directories if they don't exist.

    Supported formats:
    - .json: Saves dict or list.
    - .jsonl: Saves list of dicts.
    - .csv: Saves pandas.DataFrame.
    - .txt, .md, .py: Saves string representation.
    - .pkl: Saves object using pickle.
    - .parquet: Saves pandas.DataFrame.
    - .xlsx: Saves pandas.DataFrame.
    - .npy: Saves numpy array.
    - .pt: Saves torch object.
    - .png, .jpg, .jpeg: Saves PIL.Image or numpy array.

    Args:
        data (Any): The data object to save.
        path (str): The destination file path.

    Raises:
        ValueError: If the file extension is not supported or data type doesn't match extension requirements.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    ext = os.path.splitext(path)[1].lower()

    # JSON
    if ext == ".json":
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # JSON Lines
    elif ext == ".jsonl":
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # CSV
    elif ext == ".csv":
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False, encoding="utf-8", errors="replace")
        else:
            raise ValueError("Saving CSV requires a pandas DataFrame.")

    # TXT
    elif ext == ".txt":
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(str(data))

    # Markdown
    elif ext == ".md":
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(str(data))

    # Pickle
    elif ext == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # Parquet
    elif ext == ".parquet":
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            raise ValueError("Saving parquet requires a pandas DataFrame.")

    # Excel
    elif ext == ".xlsx":
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_excel(path, index=False)
        else:
            raise ValueError("Saving Excel requires a pandas DataFrame.")

    # Python script
    elif ext == ".py":
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(str(data))

    # Numpy array
    elif ext == ".npy":
        import numpy as np
        np.save(path, data)

    # PyTorch tensor/model
    elif ext == ".pt":
        import torch
        torch.save(data, path)

    # Image
    elif ext in [".png", ".jpg", ".jpeg"]:
        import numpy as np
        from PIL import Image
        if isinstance(data, np.ndarray):
            Image.fromarray(data).save(path)
        else:
            data.save(path)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def print_once(msg):
    """
    Prints a message to stdout only once during the entire program execution.
    Useful for logging warnings or info inside loops.

    Args:
        msg (str): The message to print.
    """
    if not hasattr(print_once, "_printed"):
        print(msg)
        print_once._printed = True


def make_print_once():
    """
    Creates and returns a local function that prints a message only once.
    This is useful if you need a "print once" behavior scoped to a specific function or instance rather than globally.

    Returns:
        callable: A function `inner(msg)` that behaves like `print_once`.
    """
    printed = False
    
    def inner(msg):
        nonlocal printed
        if not printed:
            print(msg)
            printed = True
    return inner


if __name__ == "__main__":
    # python -m structai.io
    print("Testing io.py...")
    
    # Test data
    test_dict = {"key": "value 🌍", "num": 123}
    test_list = [{"a": 1}, {"b": 2}]
    test_str = "Hello World"
    
    # Define paths
    base_path = "test_io_temp"
    json_path = f"{base_path}.json"
    jsonl_path = f"{base_path}.jsonl"
    txt_path = f"{base_path}.txt"
    pkl_path = f"{base_path}.pkl"
    
    try:
        # Test JSON
        save_file(test_dict, json_path)
        loaded_dict = load_file(json_path)
        assert loaded_dict == test_dict, f"[===ERROR===][structai][io.py][main] JSON mismatch: {loaded_dict} != {test_dict}"
        print("JSON test passed")
        
        # Test JSONL
        save_file(test_list, jsonl_path)
        loaded_list = load_file(jsonl_path)
        assert loaded_list == test_list, f"[===ERROR===][structai][io.py][main] JSONL mismatch: {loaded_list} != {test_list}"
        print("JSONL test passed")
        
        # Test TXT
        save_file(test_str, txt_path)
        loaded_str = load_file(txt_path)
        assert loaded_str == test_str, f"[===ERROR===][structai][io.py][main] TXT mismatch: {loaded_str} != {test_str}"
        print("TXT test passed")
        
        # Test Pickle
        save_file(test_dict, pkl_path)
        loaded_pkl = load_file(pkl_path)
        assert loaded_pkl == test_dict, f"[===ERROR===][structai][io.py][main] Pickle mismatch: {loaded_pkl} != {test_dict}"
        print("Pickle test passed")
        
        # Test print_once
        print("Testing print_once (should see 'Hello Once' only once):")
        print_once("Hello Once")
        print_once("Hello Once")
        
        po = make_print_once()
        print("Testing make_print_once (should see 'Hello Again' only once):")
        po("Hello Again")
        po("Hello Again")
        
    finally:
        # Cleanup
        for p in [json_path, jsonl_path, txt_path, pkl_path]:
            if os.path.exists(p):
                os.remove(p)
    
    print("io.py tests completed.")
