import os
import time
import zipfile
import re
import requests
from structai.io import load_file

def get_headers():
    """
    Retrieves the necessary headers for making requests to the MinerU API.

    This function constructs the headers dictionary, including the 'Content-Type'
    and the 'Authorization' token retrieved from the environment variables.

    Returns:
        dict: A dictionary containing the headers:
            - "Content-Type": "application/json"
            - "Authorization": "Bearer <token>"

    Raises:
        ValueError: If the 'MINERU_TOKEN' environment variable is not set.
    """
    token = os.environ.get("MINERU_TOKEN")
    if not token:
        raise ValueError("MINERU_TOKEN not found. Please register a free account at https://mineru.net/ and set the environment variable.")
    
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

def upload_pdf_to_mineru(pdfs: list[str]):
    """
    Uploads a list of PDF files to the MinerU service for parsing.

    This function initiates a batch upload process, sends the file metadata,
    and then uploads the actual file content for each PDF.

    Args:
        pdfs (list[str]): A list of file paths to the PDF files to be uploaded.

    Returns:
        str: The batch ID assigned by MinerU for this set of uploaded files.
             This ID is used to track the processing status and download results.

    Raises:
        Exception: If the initial batch creation request fails or if any file upload fails.
    """
    url = "https://mineru.net/api/v4/file-urls/batch"
    files = [{"name": pdf, "is_ocr": True, "data_id": f'{i:03}'} for i, pdf in enumerate(pdfs)]

    data = {
        "enable_formula": True,
        "language": "en",
        "enable_table": True,
        "files": files
    }

    headers = get_headers()

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if result["code"] == 0:
                urls = result["data"]["file_urls"]
                for i in range(len(urls)):
                    with open(pdfs[i], 'rb') as f:
                        requests.put(urls[i], data=f)
            else:
                raise Exception(f"Apply upload url failed: {result.get('msg')}")
        else:
            raise Exception(f"Response not success: {response.status_code}")
    except Exception as err:
        print(f"[ERROR] Upload failed: {err}")
        raise err

    return result['data']['batch_id']


def download_and_unzip(url, output_folder):
    """
    Downloads a ZIP file from a given URL and extracts its contents to a specified folder.

    This function includes retry logic for the download process to handle transient network issues.
    After extraction, it performs cleanup by removing the temporary ZIP file and any PDF files
    that might have been included in the archive (keeping only the parsed results).

    Args:
        url (str): The direct URL to the ZIP file to be downloaded.
        output_folder (str): The local directory path where the files should be extracted.
                             The directory will be created if it does not exist.
    """
    os.makedirs(output_folder, exist_ok=True)
    zip_filename = os.path.join(output_folder, "temp.zip")

    try:
        # Download with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with requests.get(url, stream=True, timeout=(10, 120)) as response:
                    response.raise_for_status()
                    with open(zip_filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                break # Success
            except requests.exceptions.RequestException:
                if attempt == max_retries - 1:
                    raise
                time.sleep(3)
        
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
            
    except Exception as e:
        print(f"[ERROR] Download/Unzip failed: {e}")
    finally:
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        # Cleanup PDF if exists
        for f in os.listdir(output_folder):
            if f.endswith('.pdf'):
                os.remove(os.path.join(output_folder, f))


def download_pdf_from_mineru(batch_id):
    """
    Polls the MinerU API for the status of a parsing batch and downloads the results when ready.

    This function continuously checks the status of each file in the batch. Once a file is
    successfully processed ('done'), its result (a ZIP file) is downloaded and extracted.
    If a file fails processing, it is marked as failed. The function returns when all files
    in the batch have been processed (either success or failure).

    Args:
        batch_id (str): The unique identifier for the batch of files to track, returned by `upload_pdf_to_mineru`.
    """
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    headers = get_headers()
    
    file_tag = {}
    
    while True:
        try:
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                time.sleep(5)
                continue
                
            extract_result = res.json().get("data", {}).get("extract_result", [])
            
            # Initialize file_tag if empty
            if not file_tag:
                for result in extract_result:
                    file_tag[result['file_name']] = 0
            
            all_done = True
            for result in extract_result:
                file_path = result['file_name']
                if file_tag.get(file_path) == 0:
                    if result['state'] == 'done':
                        download_and_unzip(result['full_zip_url'], file_path.replace(".pdf", ""))
                        file_tag[file_path] = 1
                    elif result['state'] == 'failed':
                        file_tag[file_path] = -1
                    else:
                        all_done = False
            
            if all_done:
                break
                
        except Exception as e:
            print(f"[WARNING] Polling error: {e}")
            
        time.sleep(5)


def extract_markdown_images(text):
    """
    Parses Markdown text to extract paths of embedded images.

    This function looks for standard Markdown image syntax `![alt](path "title")` and extracts the path component.
    It handles optional titles and ensures that paths containing parentheses are correctly parsed.

    Args:
        text (str): The Markdown content string to analyze.

    Returns:
        list[str]: A list of image file paths extracted from the Markdown text.
    """
    pattern = r'!\[[^\]]*\]\(([^\s]+)(?:\s+"[^"]*")?\)'

    results = re.findall(pattern, text)

    final_images = []
    for path in results:
        if path.endswith(')'):
            if path.count(')') > path.count('('):
                path = path[:-1]
        
        final_images.append(path)
        
    return final_images


def read_pdf(path: str | list[str]):
    """
    Processes PDF file(s) by uploading them to MinerU for parsing, downloading the results,
    and loading the extracted content (text and images) into memory.

    This function handles the entire pipeline:
    1. Checks if the PDF has already been processed locally.
    2. If not, uploads the PDF to MinerU and waits for processing to complete.
    3. Downloads and extracts the result (Markdown and images).
    4. Reads the Markdown content and loads referenced images.

    Args:
        path (str | list[str]): A single file path (str) or a list of file paths (list[str]) pointing to the PDF files to be processed.

    Returns:
        dict | list[dict | None] | None:
            - If `path` is a single string, returns a dictionary containing the parsed data, or None if processing failed.
            - If `path` is a list, returns a list where each element is either a dictionary (success) or None (failure).

            The result dictionary has the following structure:
            {
                "path": str,        # The original path of the PDF file.
                "text": str,        # The full extracted text content in Markdown format.
                "img_paths": list[str], # A list of absolute file paths to the extracted images.
                "imgs": list[PIL.Image.Image] # A list of PIL Image objects corresponding to the images in `img_paths`.
            }
    """
    if isinstance(path, str):
        paths = [path]
        is_single = True
    else:
        paths = path
        is_single = False

    # Filter paths that need processing
    paths_to_process = []
    for p in paths:
        if not p.endswith(".pdf") or not os.path.exists(p):
            continue
        md_dir = p[:-4]
        if not os.path.exists(os.path.join(md_dir, "full.md")):
            paths_to_process.append(p)
    
    if paths_to_process:
        try:
            batch_id = upload_pdf_to_mineru(paths_to_process)
            download_pdf_from_mineru(batch_id)
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            if is_single: return None

    results = []
    for p in paths:
        md_dir = p[:-4] if p.endswith(".pdf") else p
        md_path = os.path.join(md_dir, "full.md")
        
        if not os.path.exists(md_path):
            results.append(None)
            continue

        try:
            md_content = load_file(md_path)
            img_names = extract_markdown_images(md_content)
            
            img_paths = []
            imgs = []
            for img_name in img_names:
                full_path = os.path.join(md_dir, img_name)
                if os.path.exists(full_path):
                    img_paths.append(full_path)
                    imgs.append(load_file(full_path))
                else:
                    # Try images dir fallback
                    alt_path = os.path.join(md_dir, "images", os.path.basename(img_name))
                    if os.path.exists(alt_path):
                        img_paths.append(alt_path)
                        imgs.append(load_file(alt_path))
            
            results.append({
                "path": p,
                "text": md_content,
                "img_paths": img_paths,
                "imgs": imgs
            })
        except Exception:
            results.append(None)

    if is_single:
        return results[0] if results else None
    return results


if __name__ == "__main__":
    # python -m structai.pdf
    print("Testing pdf.py...")
    
    # Test extract_markdown_images
    print("Testing extract_markdown_images...")
    md_text = """
    Here is an image: ![alt text](images/img1.jpg)
    And another: ![alt](images/img2.png)
    And a link [link](http://example.com)
    """
    images = extract_markdown_images(md_text)
    assert "images/img1.jpg" in images, f"[===ERROR===][structai][pdf.py][main] Failed to extract img1: {images}"
    assert "images/img2.png" in images, f"[===ERROR===][structai][pdf.py][main] Failed to extract img2: {images}"
    assert len(images) == 2, f"[===ERROR===][structai][pdf.py][main] Expected 2 images, got {len(images)}"
    print("extract_markdown_images passed")

    # Test get_headers error message
    original_token = os.environ.get("MINERU_TOKEN")
    if "MINERU_TOKEN" in os.environ:
        del os.environ["MINERU_TOKEN"]
        
    try:
        get_headers()
        print("[ERROR] Should have raised ValueError")
    except ValueError as e:
        print(f"Caught expected error: {e}")
        assert "https://mineru.net/" in str(e), f"[===ERROR===][structai][pdf.py][main] Error message mismatch: {e}"
        
    # Restore token
    if original_token:
        os.environ["MINERU_TOKEN"] = original_token

    # Test with actual file if token exists
    test_pdf_path_0 = "test_files/paper_0.pdf"
    test_pdf_path_1 = "test_files/paper_1.pdf"
    
    if os.environ.get("MINERU_TOKEN") and os.path.exists(test_pdf_path_0) and os.path.exists(test_pdf_path_1):
        print(f"Testing read_pdf with multiple files: {[test_pdf_path_0, test_pdf_path_1]}...")
        try:
            results = read_pdf([test_pdf_path_0, test_pdf_path_1])
            if results and isinstance(results, list) and len(results) == 2:
                print("Success! Processed 2 files.")
                for i, res in enumerate(results):
                    if res:
                        print(f"File {i}: Text length: {len(res.get('text', ''))}, Images: {len(res.get('img_paths', []))}")
                    else:
                        print(f"File {i}: Failed")
            else:
                print(f"Failed to read multiple PDFs. Result type: {type(results)}")
        except Exception as e:
            print(f"Multiple files test failed: {e}")
        
        print(f"Testing read_pdf with single file: {test_pdf_path_0}...")
        try:
            result = read_pdf(test_pdf_path_0)
            if result:
                print(f"Success! Text length: {len(result.get('text', ''))}")
                print(f"Images: {len(result.get('img_paths', []))}")
            else:
                print("Failed to read PDF")
        except Exception as e:
            print(f"Single file test failed: {e}")
    else:
        print(f"Skipping actual test. File exists: {os.path.exists(test_pdf_path_0)}, Token exists: {bool(os.environ.get('MINERU_TOKEN'))}")
