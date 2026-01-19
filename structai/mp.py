import concurrent.futures
from tqdm import tqdm

def multi_thread(inp_list, function, max_workers=40, use_tqdm=True):
    results = [None] * len(inp_list)  # Initialize results list
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(function, **item): index  # Unpack item
            for index, item in enumerate(inp_list)
        }
        
        futures_iterator = concurrent.futures.as_completed(future_to_index)
        if use_tqdm:
            futures_iterator = tqdm(futures_iterator, total=len(future_to_index))

        for future in futures_iterator:
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result  # Place result in correct position
            except Exception as e:
                print(f"Error processing item {inp_list[index]}: {str(e)}")
    
    return results

def multi_process(inp_list, function, max_workers=40, use_tqdm=True):
    results = [None] * len(inp_list)  # Initialize results list
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(function, **item): index  # Unpack item
            for index, item in enumerate(inp_list)
        }

        futures_iterator = concurrent.futures.as_completed(future_to_index)
        if use_tqdm:
            futures_iterator = tqdm(futures_iterator, total=len(future_to_index))

        for future in futures_iterator:
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result  # Place result in correct position
            except Exception as e:
                print(f"Error processing item {inp_list[index]}: {str(e)}")
    
    return results

if __name__ == '__main__':
    # python -m structai.mp
    print("Testing mp.py...")
    import time
    
    def process(a, b):
        time.sleep(0.1)
        return a + b
    
    # Test data
    inp = []
    for a in range(10):
        inp.append({'a': a, 'b': 100})
    
    # Test multi_thread
    print("Testing multi_thread...")
    results = multi_thread(inp, process, max_workers=5)
    expected = [a + 100 for a in range(10)]
    assert results == expected, f"[===ERROR===][structai][mp.py][main] multi_thread failed: {results} != {expected}"
    print("multi_thread passed")
    
    # Test multi_process
    print("Testing multi_process...")
    results = multi_process(inp, process, max_workers=5)
    assert results == expected, f"[===ERROR===][structai][mp.py][main] multi_process failed: {results} != {expected}"
    print("multi_process passed")

    # Test with torch if available
    try:
        import torch
        if torch.cuda.is_available():
            print("Testing with CUDA tensors...")
            inp_cuda = []
            for a in range(5):
                inp_cuda.append({'a': torch.tensor(a).cuda(), 'b': torch.tensor(100).cuda()})
            
            # Note: multi_process with CUDA tensors might be tricky due to pickling/context
            # We'll just test multi_thread here as it's safer for shared CUDA tensors in simple scripts
            results = multi_thread(inp_cuda, process, max_workers=2)
            results_cpu = [r.item() for r in results]
            expected_cpu = [a + 100 for a in range(5)]
            assert results_cpu == expected_cpu, f"[===ERROR===][structai][mp.py][main] CUDA multi_thread failed: {results_cpu} != {expected_cpu}"
            print("CUDA multi_thread passed")
        else:
            print("Skipping CUDA tests: CUDA not available")
    except ImportError:
        print("Skipping torch tests: torch not installed")
        
    print("mp.py tests completed.")
