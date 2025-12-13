import asyncio
import time
import re
import os
import json
import tempfile
import base64
import sys
import io
import traceback
import math
import random
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta

import httpx
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Try to import PIL for sandbox environment (some runners may not have it)
try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

# ==========================================
# 1. CONFIGURATION
# ==========================================

LOG_PREFIX = "[SOLVER]"
MAX_TASK_DURATION = 165  # Max time per question in seconds
AUDIO_MODEL_SIZE = "tiny"
MAX_NESTED_DEPTH = 3

AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.ogg', '.opus', '.m4a', '.aac', '.flac',
    '.wma', '.oga', '.webm', '.mp4', '.m4b', '.3gp'
}

# Add image extensions so they're detected and downloaded
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp']

# LLM Config
LLM_API_URL = "https://aipipe.org/openrouter/v1/chat/completions"
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = "gpt-4o"
LLM_TIMEOUT = 120.0

def log(*args):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{LOG_PREFIX} [{timestamp}]", *args)

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def safe_json_parse(text):
    if not text: return None
    try:
        return json.loads(text)
    except:
        patterns = [
            r"```(?:json)?\s*(\{.*?\})\s*```",
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
    return None

def is_audio_file(filename_or_ext):
    if not filename_or_ext:
        return False
    ext = os.path.splitext(filename_or_ext.lower())[1]
    if not ext:
        ext = filename_or_ext.lower()
    return ext in AUDIO_EXTENSIONS

def is_image_file(filename_or_ext):
    if not filename_or_ext:
        return False
    ext = os.path.splitext(filename_or_ext.lower())[1]
    if not ext:
        ext = filename_or_ext.lower()
    return ext in IMAGE_EXTENSIONS

# ==========================================
# 2. LLM CLIENT (returns parsed + raw)
# ==========================================

async def ask_brain(system_prompt, user_prompt, retries=3):
    """Return tuple (parsed_json_or_None, raw_content_or_None)"""
    if not LLM_API_KEY:
        log("CRITICAL: No API Key found.")
        return None, None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 2000
    }

    last_raw = None

    for attempt in range(retries):
        try:
            log(f"Calling LLM (attempt {attempt + 1}/{retries})...")
            async with httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT, connect=30.0)) as client:
                resp = await client.post(LLM_API_URL, headers=headers, json=payload)
                log(f"LLM Response Status: {resp.status_code}")

                if resp.status_code == 429:
                    wait = (2 ** attempt) + random.uniform(1, 3)
                    log(f"Rate limit. Sleeping {wait:.2f}s...")
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    log(f"Server error {resp.status_code}. Retrying...")
                    await asyncio.sleep(2 ** attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()
                if 'choices' not in data or not data['choices']:
                    log(f"Invalid response structure: {data}")
                    await asyncio.sleep(1)
                    continue

                content = data['choices'][0]['message']['content']
                last_raw = content
                log(f"LLM returned content (length: {len(content)})")
                parsed = safe_json_parse(content)
                if parsed:
                    return parsed, content
                else:
                    # If we couldn't parse JSON, still return raw so caller can decide
                    log(f"Failed to parse JSON from LLM content (will return raw): {content[:200]}")
                    return None, content

        except httpx.TimeoutException as e:
            log(f"LLM Timeout (Attempt {attempt+1}): {e}")
            await asyncio.sleep(2)
        except httpx.HTTPStatusError as e:
            log(f"LLM HTTP Error (Attempt {attempt+1}): {e.response.status_code} - {e.response.text[:200]}")
            await asyncio.sleep(2)
        except Exception as e:
            log(f"LLM Error (Attempt {attempt+1}): {type(e).__name__} - {str(e)[:200]}")
            await asyncio.sleep(2)

    log("All LLM attempts failed")
    return None, last_raw

# ==========================================
# 2b. NEW: FORMAT ANSWER VIA LLM
# ==========================================
async def format_answer_via_llm(raw_answer, context_description="", retries=2):
    """
    Ask the LLM to format/escape a raw string answer for safe submission.
    Expects LLM to return a JSON object: {"formatted_answer": "<string>", "notes": "<optional>"}
    Returns the formatted string or original raw_answer if LLM fails.
    """
    if raw_answer is None:
        return raw_answer

    system_prompt = """You are a careful formatter. Receive an arbitrary answer value and return JSON with a single key "formatted_answer"
that is safe to send in an HTTP JSON payload under the "answer" field. The output must be valid JSON only.

Rules:
- If the input is a string, ensure it's escaped/quoted so when submitted as a JSON value it preserves its characters.
- If the input is a numeric string but the server expects a number, you may keep it numeric. (But do not guess server expectations — default to string with correct escaping.)
- Replace newlines with literal "\n" unless explicitly told otherwise.
- Escape existing double quotes and backslashes properly.
- Trim leading/trailing whitespace unless it's significant (assume it's not).
- Return only JSON: {"formatted_answer": "...", "notes": "..."} (notes optional).
- If input looks like structured JSON and needs minimal changes, preserve structure but ensure valid quoting.

Return exactly one JSON object. Do not include any explanation outside the JSON object.
"""

    # Put the raw answer in the user prompt, include short context for guidance
    # Keep answer snippet reasonably sized
    preview = str(raw_answer)
    if len(preview) > 5000:
        preview = preview[:5000] + "...(truncated)"

    user_prompt = f"""CONTEXT:
{context_description or "(none provided)"}

INPUT_ANSWER (raw):
{preview}

TASK:
Produce a JSON object: {{ "formatted_answer": "<value>", "notes": "<optional notes>" }}
Make sure formatted_answer is a string literal properly escaped so when the runner inserts it into JSON it will parse exactly as intended.
"""

    parsed, raw = await ask_brain(system_prompt, user_prompt, retries=retries)
    if parsed and isinstance(parsed, dict) and parsed.get("formatted_answer") is not None:
        return parsed["formatted_answer"]
    else:
        log("Answer formatting via LLM failed or returned non-JSON. Falling back to safe local escaping.")
        # Local fallback: simple JSON-safe encoding
        try:
            # Use json.dumps to get a JSON-safe string, then strip outer quotes to return raw string representation if needed.
            dum = json.dumps(str(raw_answer))
            # If server expects the string value (not JSON literal), send unquoted string but escaped; we'll return the raw JSON literal for safety.
            # It's safer to return the JSON literal (with quotes) as the final answer so the server receives exact payload.
            return json.loads(dum)  # this returns a Python string with escapes applied
        except Exception as e:
            log("Local fallback escaping failed:", e)
            return str(raw_answer)

# ==========================================
# 3. AUDIO TRANSCRIPTION
# (unchanged)
# ==========================================

_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            log(f"Loading Whisper ({AUDIO_MODEL_SIZE})...")
            _whisper_model = WhisperModel(
                AUDIO_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                cpu_threads=4
            )
        except Exception as e:
            log(f"WARNING: faster-whisper not available: {e}")
            _whisper_model = False
    return _whisper_model

def transcribe_audio(path):
    model = get_whisper()
    if not model:
        return "[Audio transcription not available]"
    try:
        log(f"Transcribing audio file: {path}")
        segments, _ = model.transcribe(
            path,
            beam_size=1,
            language="en",
            vad_filter=True
        )
        transcript = " ".join([s.text for s in segments])
        log(f"Transcript: {transcript[:200]}...")
        return transcript
    except Exception as e:
        return f"[Transcription Failed: {e}]"

# ==========================================
# 4. PYTHON SANDBOX
# (unchanged except exposes 'files' etc.)
# ==========================================

def run_python(code, file_map, nested_data=None):
    log("Executing Python code...")
    log(f"Code to execute:\n{code[:500]}...")
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    # Provide PIL Image as Image if available
    env = {
        "pd": pd,
        "json": json,
        "re": re,
        "math": math,
        "plt": plt,
        "files": file_map,            # mapping filename -> local path
        "pdfplumber": pdfplumber,
        "base64": base64,
        "nested_pages": nested_data or {},
        # Useful utilities LLM code might expect:
        "os": os,
        "open": open,
        "print": print,
        "PILImage": PILImage,
        "zipfile": __import__('zipfile'),  # if LLM imports from PIL, this may help
    }
    # Also expose Image name commonly used
    if PILImage is not None:
        env["Image"] = PILImage

    result = {"output": "", "image": None, "error": None}
    try:
        exec(code, env, env)
        result["output"] = buffer.getvalue().strip()
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            result["image"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close('all')
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
    log(f"Execution result - Output: {result['output']}, Error: {result['error'] if result['error'] else None}")
    return result

# ==========================================
# 5. NESTED LINK SCRAPER
# ==========================================

async def scrape_nested_links(page, base_url, soup, depth=0):
    if depth >= MAX_NESTED_DEPTH:
        return {}
    nested_data = {}
    internal_links = []
    base_domain = urlparse(base_url).netloc
    skip_extensions = ['.pdf', '.csv', '.xlsx', '.xls', '.json', '.txt', '.zip'] + list(AUDIO_EXTENSIONS)
    for link_tag in soup.find_all('a', href=True):
        href = link_tag.get('href')
        if not href:
            continue
        if any(ext in href.lower() for ext in skip_extensions):
            continue
        absolute_url = urljoin(base_url, href)
        link_domain = urlparse(absolute_url).netloc
        if link_domain == base_domain and absolute_url != base_url:
            internal_links.append(absolute_url)
    # Also consider image tags and direct image links that may not be <a>
    for img_tag in soup.find_all('img', src=True):
        src = img_tag.get('src')
        if src:
            absolute_url = urljoin(base_url, src)
            if urlparse(absolute_url).netloc == base_domain:
                internal_links.append(absolute_url)
    internal_links = list(set(internal_links))
    if internal_links:
        log(f"Found {len(internal_links)} nested links to scrape at depth {depth}")
    for nested_url in internal_links[:5]:
        try:
            log(f"  Scraping nested: {nested_url}")
            await page.goto(nested_url, wait_until="networkidle", timeout=15000)
            await asyncio.sleep(0.5)
            nested_html = await page.content()
            nested_soup = BeautifulSoup(nested_html, 'html.parser')
            hidden_content = []
            for match in re.findall(r'atob\s*\(\s*["\']([^"\']+)["\']\s*\)', nested_html):
                try:
                    decoded = base64.b64decode(match).decode('utf-8')
                    hidden_content.append(decoded)
                except:
                    pass
            nested_text = nested_soup.get_text(separator="\n", strip=True)
            nested_data[nested_url] = {
                "url": nested_url,
                "text": nested_text,
                "hidden_content": "\n---\n".join(hidden_content) if hidden_content else "",
                "html": nested_html[:2000]
            }
            if depth < MAX_NESTED_DEPTH - 1:
                deeper_data = await scrape_nested_links(page, nested_url, nested_soup, depth + 1)
                nested_data.update(deeper_data)
        except Exception as e:
            log(f"  Error scraping nested link {nested_url}: {e}")
    return nested_data

# ==========================================
# 6. COMPREHENSIVE PAGE SCRAPER
# ==========================================

async def scrape_page_content(page, url):
    log("Scraping page content...")
    content = await page.content()
    soup = BeautifulSoup(content, 'html.parser')
    tables_text = ""
    for idx, table in enumerate(soup.find_all("table")):
        try:
            df = pd.read_html(str(table))[0]
            tables_text += f"\n\n[TABLE {idx+1}]:\n{df.to_markdown(index=False)}\n"
            table.decompose()
        except:
            pass
    form_values = []
    for inp in soup.find_all("input"):
        val = inp.get("value")
        name = inp.get("name")
        if val and name:
            form_values.append(f"{name}: {val}")
    form_text = ("\n\n[FORM VALUES]:\n" + "\n".join(form_values)) if form_values else ""
    hidden_content = []
    for match in re.findall(r'atob\s*\(\s*["\']([^"\']+)["\']\s*\)', content):
        try:
            decoded = base64.b64decode(match).decode('utf-8')
            hidden_content.append(decoded)
        except:
            pass
    hidden_text = ("\n\n[DECODED BASE64 CONTENT]:\n" + "\n---\n".join(hidden_content)) if hidden_content else ""
    visible_text = soup.get_text(separator="\n", strip=True)
    log("Checking for nested links...")
    nested_data = await scrape_nested_links(page, url, soup)
    nested_text = ""
    if nested_data:
        log(f"Scraped {len(nested_data)} nested pages")
        nested_text = "\n\n[NESTED PAGES CONTENT]:\n"
        for nested_url, nested_info in nested_data.items():
            nested_text += f"\n--- URL: {nested_url} ---\n"
            nested_text += nested_info['text'][:1000] + "\n"
            if nested_info['hidden_content']:
                nested_text += f"[Hidden Content]: {nested_info['hidden_content']}\n"
    file_links = []
    # include image extensions in file detection
    all_file_extensions = ['.csv', '.pdf', '.xlsx', '.xls', '.json', '.txt', '.zip', 'data:'] + list(AUDIO_EXTENSIONS) + IMAGE_EXTENSIONS
    # check tags including <img>
    for tag in soup.find_all(['a', 'audio', 'source', 'video', 'img']):
        link = tag.get('href') or tag.get('src')
        if link:
            lower = link.lower()
            if any(ext in lower for ext in all_file_extensions):
                file_links.append(link)
    # Also add nested_data URLs that look like files (images etc.)
    for nested_url in nested_data.keys():
        lower = nested_url.lower()
        if any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS + ['.pdf', '.csv', '.json', '.txt', '.xlsx', '.xls', '.zip']):
            file_links.append(nested_url)
    # dedupe file_links and normalize
    file_links = list(dict.fromkeys(file_links))
    submit_url = None
    next_url = None
    full_text = visible_text + hidden_text + nested_text
    next_patterns = [
        r'next.*?question.*?(https?://[^\s<>"]+)',
        r'(https?://[^\s<>"]+/question/\d+)',
        r'move.*?to.*?(https?://[^\s<>"]+)',
    ]
    for pattern in next_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            next_url = match.group(1)
            break
    submit_patterns = [
        r'POST.*?to\s+(https?://[^\s<>"]+/submit[^\s<>"]*)',
        r'Post your answer to\s+(https?://[^\s<>"]+)',
        r'submit.*?to\s+(https?://[^\s<>"]+)',
        r'(https?://[^\s<>"]+/submit[^\s<>"]*)',
        r'answer.*?endpoint.*?(https?://[^\s<>"]+)',
    ]
    for pattern in submit_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            submit_url = match.group(1)
            break
    if not submit_url:
        parsed = urlparse(url)
        submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
        log(f"No explicit submit URL found, using: {submit_url}")
    return {
        "visible_text": visible_text,
        "tables": tables_text,
        "form_values": form_text,
        "hidden_content": hidden_text,
        "nested_content": nested_text,
        "nested_data": nested_data,
        "file_links": file_links,
        "submit_url": submit_url,
        "next_url": next_url,
        "full_context": visible_text + tables_text + form_text + hidden_text + nested_text
    }

# ==========================================
# 7. FILE DOWNLOADER AND PROCESSOR
# ==========================================

async def download_and_process_files(file_links, base_url):
    """
    Deduplicate links, download them, infer types and return:
      - files_data: list of dicts with name, ext, content, path
      - file_map: dict mapping name -> absolute local path (this mapping is given to LLM prompt)
    """
    log(f"Processing {len(file_links)} file links...")
    files_data = []
    file_map = {}
    processed_targets = set()
    # Make absolute, dedupe by normalized target URL
    normalized_links = []
    for link in file_links:
        target = urljoin(base_url, link)
        if target not in processed_targets:
            normalized_links.append(target)
            processed_targets.add(target)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for target in normalized_links:
            try:
                file_info = await download_file(client, target, base_url, already_absolute=True)
                if file_info:
                    content = extract_file_content(file_info)
                    # Ensure unique filename in file_map: if duplicate name exists, append an index
                    name = file_info["name"]
                    base_name, ext = os.path.splitext(name)
                    idx = 1
                    while name in file_map:
                        name = f"{base_name}_{idx}{ext}"
                        idx += 1
                    files_data.append({
                        "name": name,
                        "type": file_info["ext"],
                        "content": content,
                        "path": file_info["path"]
                    })
                    file_map[name] = file_info["path"]
                    log(f"  Processed: {name} ({file_info['ext']}) -> {file_info['path']}")
            except Exception as e:
                log(f"Error processing file {target}: {e}")
    # helpful debug: list available files with local paths
    if file_map:
        for n, p in file_map.items():
            log(f"Available file -> {n} : {p}")
    else:
        log("No files downloaded for this page.")
    return files_data, file_map

async def download_file(client, url, base_url, already_absolute=False):
    """
    Download a file. If already_absolute=True, `url` is the full target.
    Returns {"name": ..., "path": ..., "ext": ...} or None.
    """
    target = url if already_absolute else urljoin(base_url, url)
    try:
        # handle data: URIs
        if target.startswith("data:"):
            header, b64 = target.split(",", 1)
            ext = ".bin"
            if "pdf" in header:
                ext = ".pdf"
            elif "csv" in header:
                ext = ".csv"
            elif "json" in header:
                ext = ".json"
            elif "audio" in header:
                if "mpeg" in header or "mp3" in header:
                    ext = ".mp3"
                elif "wav" in header:
                    ext = ".wav"
                elif "ogg" in header:
                    ext = ".ogg"
                elif "opus" in header:
                    ext = ".opus"
                else:
                    ext = ".mp3"
            data = base64.b64decode(b64)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tf.write(data)
            tf.close()
            return {"name": f"data{ext}", "path": tf.name, "ext": ext}

        resp = await client.get(target, follow_redirects=True)
        if resp.status_code != 200:
            log(f"Download failed {target} -> status {resp.status_code}")
            return None

        # Try to infer a name from the URL path
        url_path_name = os.path.basename(urlparse(target).path) or ""
        ext = os.path.splitext(url_path_name)[1].lower() or ""
        name = url_path_name or "file"

        # Infer extension from content-type if missing
        if not ext:
            content_type = resp.headers.get("content-type", "").lower()
            if "image" in content_type:
                if "png" in content_type:
                    ext = ".png"
                elif "jpeg" in content_type or "jpg" in content_type:
                    ext = ".jpg"
                elif "gif" in content_type:
                    ext = ".gif"
                elif "bmp" in content_type:
                    ext = ".bmp"
                elif "webp" in content_type:
                    ext = ".webp"
                else:
                    ext = ".png"
            elif "audio" in content_type:
                if "mpeg" in content_type or "mp3" in content_type:
                    ext = ".mp3"
                elif "wav" in content_type:
                    ext = ".wav"
                elif "ogg" in content_type:
                    ext = ".ogg"
                elif "opus" in content_type:
                    ext = ".opus"
                else:
                    ext = ".mp3"
            elif "pdf" in content_type:
                ext = ".pdf"
            elif "csv" in content_type:
                ext = ".csv"
            elif "json" in content_type:
                ext = ".json"
            elif "zip" in content_type:
                ext = ".zip"
            else:
                ext = os.path.splitext(name)[1] or ".bin"

        # Ensure name ends with ext
        if not name.lower().endswith(ext.lower()):
            name = os.path.splitext(name)[0] + ext

        tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tf.write(resp.content)
        tf.close()
        return {"name": name, "path": tf.name, "ext": ext}
    except Exception as e:
        log(f"Error downloading {target}: {e}")
        return None

def extract_file_content(file_info):
    try:
        ext = file_info["ext"]
        path = file_info["path"]
        if ext == ".pdf":
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages[:10]):
                    text = page.extract_text() or ""
                    if text:
                        text_parts.append(f"[Page {i+1}]\n{text}")
            return "\n\n".join(text_parts)[:5000]
        elif ext in [".csv", ".txt"]:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read(3000)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
            return f"Shape: {df.shape}\n\n{df.head(20).to_markdown()}"
        elif ext == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)[:3000]
        elif ext == ".zip":
            import zipfile
            try:
                with zipfile.ZipFile(path, 'r') as z:
                    files_list = z.namelist()
                    return f"[ZIP ARCHIVE] Contains {len(files_list)} files:\n" + "\n".join(files_list[:50])
            except Exception as e:
                return f"[ZIP ARCHIVE - Error reading: {e}]"
        elif is_audio_file(ext):
            transcript = transcribe_audio(path)
            return f"[AUDIO TRANSCRIPT - THIS MAY CONTAIN THE ACTUAL QUESTION]:\n{transcript}"
        elif is_image_file(ext):
            # Provide a short image summary: size and mode
            try:
                if PILImage is not None:
                    with PILImage.open(path) as im:
                        w, h = im.size
                        mode = im.mode
                        return f"[IMAGE - {os.path.basename(path)}] size={w}x{h}, mode={mode}"
                else:
                    return "[IMAGE - PIL not available for preview]"
            except Exception as e:
                return f"[Error opening image for preview: {e}]"
        return "[Binary file - no preview available]"
    except Exception as e:
        return f"[Error extracting content: {e}]"

# ==========================================
# 8. SOLUTION GENERATOR (returns parsed + raw)
#    NOTE: now includes explicit FILES mapping (name -> local path) in prompt
# ==========================================

def needs_computation(scraped_data, files_data):
    full_text = scraped_data.get('full_context', '').lower()
    compute_keywords = [
        'calculate', 'sum', 'count', 'average', 'mean', 'median',
        'maximum', 'minimum', 'filter', 'find all', 'how many',
        'total', 'aggregate', 'statistics', 'analysis', 'heatmap',
        'chart', 'graph', 'plot', 'data', 'csv', 'excel', 'logs'
    ]
    has_compute_keyword = any(keyword in full_text for keyword in compute_keywords)
    has_data_files = any(f['type'] in ['.csv', '.xlsx', '.xls', '.json', '.zip'] for f in files_data)
    has_image_task = 'heatmap' in full_text or 'image' in full_text or 'png' in full_text
    return has_compute_keyword or has_data_files or has_image_task

async def generate_solution(scraped_data, files_data, file_map=None, feedback="", force_python=False):
    """
    Pass file_map (name -> local path) into the LLM prompt so the model can
    use exact local paths in its generated python_code.
    Returns (parsed_json, raw_text)
    """
    log("Generating solution strategy...")
    files_summary = ""
    has_audio_question = False

    # Build files summary and include name->path mapping
    files_mapping_text = ""
    if file_map:
        files_mapping_text += "FILES (name -> local_path):\n"
        for n, p in file_map.items():
            files_mapping_text += f"{n} -> {p}\n"

    for f in files_data:
        files_summary += f"\n\n{'='*50}\n"
        files_summary += f"FILE: {f['name']} (Type: {f['type']})\n"
        files_summary += f"{'='*50}\n"
        if is_audio_file(f['type']):
            files_summary += "⚠️ AUDIO FILE - MAY CONTAIN THE ACTUAL QUESTION ⚠️\n"
            has_audio_question = True
        files_summary += f"{f['content']}\n"

    nested_summary = ""
    if scraped_data.get('nested_data'):
        nested_summary = "\n\nNESTED PAGES (already scraped):\n"
        for url, data in scraped_data['nested_data'].items():
            nested_summary += f"\nURL: {url}\n"
            nested_summary += f"Content: {data['text'][:500]}...\n"
            if data['hidden_content']:
                nested_summary += f"Hidden: {data['hidden_content'][:200]}\n"

    requires_computation = force_python or needs_computation(scraped_data, files_data)
    if requires_computation:
        log("⚙️ Detected computational task - will prioritize Python strategy")

    main_context = scraped_data['full_context'][:2000]
    files_context = files_summary[:3000]
    nested_context = nested_summary[:1000]

    system_prompt = """You are an expert quiz solver. Analyze the content and provide a solution.

CRITICAL: Your response MUST be valid JSON with this exact structure:
{
    "question_location": "where you found the question",
    "question_understanding": "what is being asked",
    "reasoning": "how to solve it",
    "strategy": "direct_answer" or "python_code",
    "answer": "the answer if direct_answer, else null",
    "code": "python code if python_code, else null"
}

INSTRUCTIONS:
1. If there's an AUDIO TRANSCRIPT, it likely contains the ACTUAL QUESTION
2. Look for keywords: sum, count, filter, calculate, find
3. For direct_answer: provide the answer value directly
4. For python_code: write code that prints the answer

PYTHON CODE RULES:
- Use files['filename.ext'] to access local file **paths** (the actual mapping is provided below).
- Example: path = files['heatmap.png']  # this is a local filesystem path you can open with PIL
- Use pandas: df = pd.read_csv(files['data.csv'])
- MUST print() the final answer
- Keep code simple and focused

CRITICAL FILE RULES:
- NEVER assume a file exists.
- Only use filenames explicitly listed in "FILES (name -> local_path)".
- If FILES list is empty, you MUST parse the logs or data from the scraped page text only.
- Do NOT reference files['logs.zip'] unless it actually appears in the mapping.
- If no files exist, construct the dataset by parsing scraped_data['full_context'].

CRITICAL PANDAS RULES:
- NEVER use .str accessor on non-string columns (causes AttributeError)
- Check column types first: df.dtypes or df.info()
- If you need to strip whitespace from numeric column: df['col'] = pd.to_numeric(df['col'], errors='coerce')
- If column might be string OR numeric: df['col'] = df['col'].astype(str).str.strip() then convert
- For dates: use pd.to_datetime() with appropriate format parameter to avoid warnings
- Example safe approach:
  df['Value'] = pd.to_numeric(df['Value'], errors='coerce')  # handles mixed types
  df['Joined'] = pd.to_datetime(df['Joined'], format='%Y-%m-%d', errors='coerce')
  - For dates in messy CSVs, ALWAYS parse using:
  pd.to_datetime(df['Joined'], dayfirst=True, format='mixed', errors='coerce')
  This ensures '02-01-2024' becomes '2024-01-02' (correct), not '2024-02-01' (wrong).
- After parsing, ALWAYS format as:
  df['Joined'] = df['Joined'].dt.strftime('%Y-%m-%d')

PDF INVOICE RULE:
- Always use pdfplumber for reading PDFs.
- Never assume fixed column order.
- Never use fixed indices like parts[1].
- Extract ALL numeric values using regex.
- For each line item, use the LAST TWO numeric tokens as (quantity, unit price).
- Multiply them and add to total.
- Round final result to 2 decimals.

RATE-LIMIT RULES (MANDATORY):
- ALWAYS compute both request interpretations:
    A) requests = pages
    B) requests = pages * items_per_page
  Pick the interpretation that makes sense based on context.

- For the chosen interpretation, compute:
    t_min = requests / per_minute  (in minutes)
    t_hour = requests / (per_hour / 60)  (in minutes)
  BASE_TIME = ceil(max(t_min, t_hour))

- If retry_every and retry_after_seconds exist:
    total_requests = requests
    retry_count = floor(total_requests / retry_every)
    retry_delay_minutes = retry_count * (retry_after_seconds / 60)
    BASE_TIME = ceil(BASE_TIME + retry_delay_minutes)

- OFFSET CALCULATION:
    email = '24f2005894@ds.study.iitm.ac.in'
    offset = len(email) % 3
    FINAL_TIME = BASE_TIME + offset

- CRITICAL: Output ONLY the integer FINAL_TIME.
- NEVER skip the offset step.
- The offset is added AFTER all other calculations.

IMPORTANT: Return ONLY valid JSON, no markdown, no extra text, use this email(24f2005894@ds.study.iitm.ac.in) for any email related calculation.
"""

    # Include explicit files mapping in prompt (name -> local path)
    user_prompt = f"""MAIN PAGE:
{main_context}
EMAIL: (24f2005894@ds.study.iitm.ac.in)
FILES (summaries):
{files_context}

{files_mapping_text}

{nested_context}

FEEDBACK FROM PREVIOUS ATTEMPT:
{feedback if feedback else "None - this is the first attempt"}

{'⚠️ AUDIO FILE DETECTED - READ THE TRANSCRIPT CAREFULLY! ⚠️' if has_audio_question else ''}
{'⚠️ COMPUTATIONAL TASK DETECTED - USE PYTHON CODE! ⚠️' if requires_computation else ''}

IMPORTANT: Use the mapping shown in "FILES (name -> local_path)" to open local files. Provide your solution as JSON."""

    parsed, raw = await ask_brain(system_prompt, user_prompt)
    if not parsed and raw:
        return None, raw
    if not parsed:
        log("LLM failed - generating fallback python code")
        fallback = {
            "question_location": "Page content",
            "question_understanding": "Analysis task requiring computation",
            "reasoning": "Fallback python code",
            "strategy": "python_code",
            "answer": None,
            "code": generate_fallback_code(scraped_data, files_data)
        }
        return fallback, "(fallback)"
    return parsed, raw


def generate_fallback_code(scraped_data, files_data):
    code_parts = ["# Fallback analysis code", ""]
    has_csv = any(f['type'] == '.csv' for f in files_data)
    has_excel = any(f['type'] in ['.xlsx', '.xls'] for f in files_data)
    has_json = any(f['type'] == '.json' for f in files_data)
    full_text = scraped_data.get('full_context', '').lower()
    if has_csv:
        csv_file = next(f for f in files_data if f['type'] == '.csv')
        code_parts.append("# Loading CSV file")
        code_parts.append(f"df = pd.read_csv(files['{csv_file['name']}'])")
        code_parts.append("print('Data shape:', df.shape)")
        code_parts.append("print('\\nFirst few rows:')")
        code_parts.append("print(df.head())")
        if 'sum' in full_text or 'total' in full_text:
            code_parts.append("# Calculate sum")
            code_parts.append("print('\\nSum of numeric columns:')")
            code_parts.append("print(df.select_dtypes(include=['number']).sum())")
        elif 'count' in full_text or 'how many' in full_text:
            code_parts.append("# Count rows")
            code_parts.append("print('\\nTotal rows:', len(df))")
    elif has_excel:
        excel_file = next(f for f in files_data if f['type'] in ['.xlsx', '.xls'])
        code_parts.append("# Loading Excel file")
        code_parts.append(f"df = pd.read_excel(files['{excel_file['name']}'])")
        code_parts.append("print('Data shape:', df.shape)")
        code_parts.append("print(df.head())")
    elif has_json:
        json_file = next(f for f in files_data if f['type'] == '.json')
        code_parts.append("# Loading JSON file")
        code_parts.append(f"with open(files['{json_file['name']}'], 'r') as f:")
        code_parts.append("    data = json.load(f)")
        code_parts.append("print(data)")
    else:
        code_parts.append("# Analyzing page content")
        code_parts.append("print('No data files found - check page content')")
    return "\n".join(code_parts)


async def fix_rejected_answer(question_context, attempted_answer, payload_sent, server_error, attempt_history=None):
    """
    Ask LLM to fix a rejected answer by showing it:
    - The original question
    - What we tried to submit
    - The exact payload sent (after JSON serialization)
    - Server's error message
    - History of previous attempts (if any)
    
    Returns: {
        "corrected_answer": <value>,
        "send_as_type": "string"|"number"|"array"|"object",
        "explanation": "why this format"
    }
    """
    log("🔧 Asking LLM to fix rejected answer...")
    
    # Build attempt history summary
    history_text = ""
    if attempt_history and len(attempt_history) > 0:
        history_text = "\n\nPREVIOUS ATTEMPTS THAT FAILED:\n"
        for i, attempt in enumerate(attempt_history):
            history_text += f"\nAttempt {i+1}:\n"
            history_text += f"  Answer tried: {str(attempt['answer'])[:200]}\n"
            history_text += f"  Type sent: {attempt['type']}\n"
            history_text += f"  Server said: {attempt['error'][:200]}\n"
    
    system_prompt = """You are a debugging expert. An answer was rejected by the server. Your job is to fix it.

CRITICAL: You must analyze what went wrong and provide the EXACT answer in the EXACT format the server expects.

Return ONLY valid JSON with this structure:
{
    "corrected_answer": <the exact value to send - could be string, number, array, or object>,
    "send_as_type": "string" or "number" or "array" or "object",
    "explanation": "brief explanation of what was wrong and why this fixes it"
}

RULES FOR send_as_type:
- "string": Send as JSON string (with quotes in the JSON payload)
- "number": Send as JSON number (no quotes)
- "array": Send as JSON array [...]
- "object": Send as JSON object {...}

IMPORTANT:
- If the answer should be a JSON-formatted string (like "[{...}]"), use send_as_type: "string" and put the entire JSON as the corrected_answer value
- If server expects structured data directly, use "array" or "object"
- Look at the server error message for hints about expected format
- Consider what worked in previous attempts (if any)
"""

    user_prompt = f"""QUESTION CONTEXT:
{question_context[:3000]}

OUR ATTEMPTED ANSWER:
{str(attempted_answer)[:1000]}

EXACT PAYLOAD WE SENT TO SERVER:
{json.dumps(payload_sent, indent=2)}

SERVER REJECTION:
{server_error[:2000]}

{history_text}

TASK: Analyze what went wrong and provide the corrected answer in the exact format the server expects.
Return your response as JSON."""

    parsed, raw = await ask_brain(system_prompt, user_prompt, retries=2)
    
    if parsed and isinstance(parsed, dict):
        if 'corrected_answer' in parsed and 'send_as_type' in parsed:
            log(f"✓ LLM provided correction: type={parsed['send_as_type']}")
            log(f"  Explanation: {parsed.get('explanation', 'N/A')[:150]}")
            return parsed
        else:
            log("⚠️ LLM response missing required fields")
            return None
    else:
        log("⚠️ LLM failed to provide valid correction JSON")
        return None

# ==========================================
# HELPER: Inspect Payload
# ==========================================

def inspect_payload(payload):
    """Returns the exact JSON string that will be sent over HTTP"""
    try:
        json_str = json.dumps(payload, indent=2)
        return json_str
    except Exception as e:
        return f"[Error serializing payload: {e}]"

def get_answer_type(value):
    """Determine the JSON type of a value"""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int) or isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "object"
    else:
        return "unknown"

# ==========================================
# 9. MAIN SOLVER — keep trying, chain server feedback + last LLM raw
# ==========================================
async def solve_quiz_task(email, secret, start_url):
    if not secret or " " in secret:
        log("WARNING: Secret contains whitespace or is empty.")

    question_number = 1
    total_start_time = time.time()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        current_url = start_url

        while current_url:
            print("\n" + "="*80)
            print(f"🔷 QUESTION {question_number}")
            print("="*80)
            log(f"URL: {current_url}")

            question_start_time = time.time()
            solved = False
            next_url_from_page = None

            async def print_elapsed():
                while not solved:
                    elapsed = time.time() - question_start_time
                    print(f"\r⏱️  Time elapsed: {format_time(elapsed)}", end="", flush=True)
                    await asyncio.sleep(1)

            timer_task = asyncio.create_task(print_elapsed())

            try:
                # Navigate & scrape
                await page.goto(current_url, wait_until="networkidle", timeout=20000)
                await asyncio.sleep(1)
                scraped_data = await scrape_page_content(page, current_url)
                log(f"Submit URL: {scraped_data['submit_url']}")
                next_url_from_page = scraped_data.get('next_url')

                # Process files
                files_data, file_map = await download_and_process_files(
                    scraped_data['file_links'],
                    current_url
                )
                log(f"Processed {len(files_data)} files")

                audio_files = [f for f in files_data if is_audio_file(f['type'])]
                if audio_files:
                    log(f"⚠️  Found {len(audio_files)} audio file(s)")
                    for af in audio_files:
                        log(f"   - {af['name']} ({af['type']})")

                # Initial probe
                initial_server_feedback = ""
                submit_url = scraped_data.get('submit_url')
                if submit_url:
                    try:
                        log(f"Sending initial probe to {submit_url}")
                        async with httpx.AsyncClient(timeout=15.0) as client:
                            probe_payload = {
                                "email": email,
                                "secret": secret.strip(),
                                "url": current_url,
                                "answer": ""
                            }
                            resp = await client.post(submit_url, json=probe_payload)
                            text = resp.text if hasattr(resp, "text") else str(resp)
                            initial_server_feedback = f"[Probe] status={resp.status_code}\n\n{text[:5000]}"
                            log("Captured probe feedback")
                    except Exception as e:
                        initial_server_feedback = f"[Probe Error] {e}"

                # IMPROVED RETRY LOOP WITH CODE ERROR LIMIT
                last_llm_raw = None
                feedback = initial_server_feedback or ""
                attempt_history = []
                is_correction_mode = False
                changes_tried = set()
                correction_attempts = 0
                code_error_count = 0  # NEW: Track code execution errors
                MAX_CODE_ERRORS = 3   # NEW: Limit code errors
                
                while True:
                    elapsed = time.time() - question_start_time
                    if elapsed > MAX_TASK_DURATION:
                        print(f"\n⏰ Time limit exceeded for Question {question_number}")
                        if next_url_from_page:
                            log(f"Moving to next question: {next_url_from_page}")
                            current_url = next_url_from_page
                            solved = True
                            break
                        else:
                            log("No next URL - stopping")
                            timer_task.cancel()
                            await browser.close()
                            return

                    log(f"\n--- Attempt #{len(attempt_history)+1} (elapsed {format_time(elapsed)}) ---")
                    
                    final_answer = None
                    answer_type_hint = None
                    
                    # DECISION: Correction mode or regenerate?
                    if is_correction_mode and len(attempt_history) > 0:
                        if correction_attempts >= 2:
                            log("⚠️ Too many correction attempts, regenerating from scratch")
                            is_correction_mode = False
                            correction_attempts = 0
                            changes_tried.clear()
                            feedback = f"Multiple corrections failed. Regenerate with completely different approach.\nPrevious attempts: {len(attempt_history)}"
                            continue
                        
                        log("🔧 Correction mode: asking LLM to fix rejected answer")
                        
                        last_attempt = attempt_history[-1]
                        question_context = scraped_data['full_context'][:3000]
                        
                        correction = await fix_rejected_answer(
                            question_context=question_context,
                            attempted_answer=last_attempt['answer'],
                            payload_sent=last_attempt['payload'],
                            server_error=last_attempt['error'],
                            attempt_history=attempt_history[:-1] if len(attempt_history) > 1 else None
                        )
                        
                        if correction:
                            final_answer = correction['corrected_answer']
                            answer_type_hint = correction['send_as_type']
                            
                            answer_signature = f"{answer_type_hint}:{str(final_answer)[:100]}"
                            if answer_signature in changes_tried:
                                log("⚠️ Already tried this answer variant, regenerating")
                                is_correction_mode = False
                                correction_attempts = 0
                                changes_tried.clear()
                                feedback = "Repeated same answer. Generate completely new approach."
                                continue
                            
                            changes_tried.add(answer_signature)
                            correction_attempts += 1
                            log(f"📝 Using corrected answer (type: {answer_type_hint}, attempt {correction_attempts}/2)")
                        else:
                            log("❌ Correction failed, falling back to regeneration")
                            is_correction_mode = False
                            correction_attempts = 0
                            feedback = "LLM correction failed. Regenerate answer from scratch."
                            continue
                    else:
                        # Normal mode - generate fresh answer
                        correction_attempts = 0
                        composite_feedback = ""
                        if feedback:
                            composite_feedback += f"SERVER FEEDBACK:\n{feedback}\n\n"
                        if last_llm_raw:
                            composite_feedback += f"LAST_LLM_OUTPUT:\n{last_llm_raw[:2000]}\n\n"
                        if len(attempt_history) > 0:
                            composite_feedback += f"PREVIOUS ATTEMPTS: {len(attempt_history)}\n"

                        solution_parsed, llm_raw = await generate_solution(
                            scraped_data, files_data, 
                            file_map=file_map, 
                            feedback=composite_feedback, 
                            force_python=False
                        )
                        
                        if isinstance(solution_parsed, dict):
                            solution = solution_parsed
                        else:
                            solution = None

                        last_llm_raw = llm_raw

                        if not solution:
                            log("LLM didn't return parsed JSON")
                            if llm_raw:
                                feedback = f"LLM returned non-JSON: {llm_raw[:1500]}"
                            else:
                                feedback = "LLM produced no output"
                            await asyncio.sleep(1.5)
                            continue

                        log(f"Question: {solution.get('question_understanding','N/A')[:100]}")
                        log(f"Strategy: {solution.get('strategy')}")

                        # Execute strategy
                        if solution.get('strategy') == 'python_code':
                            code = solution.get('code', '') or ""
                            if not code.strip():
                                feedback = "No code provided"
                                await asyncio.sleep(1)
                                continue

                            result = run_python(code, file_map, scraped_data.get('nested_data', {}))
                            
                            # NEW: Enhanced error handling with code error limit
                            if result['error']:
                                code_error_count += 1
                                log(f"❌ Code error #{code_error_count}/{MAX_CODE_ERRORS}")
                                
                                # Check if we've hit the limit
                                if code_error_count >= MAX_CODE_ERRORS:
                                    print(f"\n⛔ Max code errors reached ({MAX_CODE_ERRORS}), giving up on this question")
                                    
                                    if next_url_from_page:
                                        log(f"Moving to next question: {next_url_from_page}")
                                        current_url = next_url_from_page
                                        question_number += 1
                                        solved = True
                                        break
                                    else:
                                        log("No next URL - stopping solver")
                                        timer_task.cancel()
                                        await browser.close()
                                        return
                                
                                # Enhanced feedback with debugging hints
                                error_msg = result['error']
                                feedback = f"""Python execution error (attempt {code_error_count}/{MAX_CODE_ERRORS}):
{error_msg[:2000]}

DEBUGGING HINTS:
- If error says "Can only use .str accessor with string values", the column is numeric, not string
  FIX: Use pd.to_numeric(df['column'], errors='coerce') instead of .str operations
  OR: Convert first: df['column'].astype(str).str.strip()
  
- If error mentions "AttributeError" with .str, check df.dtypes before using string methods

- For date parsing warnings, specify format explicitly:
  df['date_col'] = pd.to_datetime(df['date_col'], format='%Y-%m-%d', errors='coerce')

- Common pattern for messy CSV:
  df['numeric_col'] = pd.to_numeric(df['numeric_col'].astype(str).str.strip(), errors='coerce')

Try a different approach that handles data type issues properly.
"""
                                log("Code errored - providing enhanced feedback")
                                await asyncio.sleep(1)
                                continue
                            
                            final_answer = result['image'] if result['image'] else result['output']
                            
                            if not final_answer:
                                feedback = "Code produced no output"
                                await asyncio.sleep(1)
                                continue
                        else:
                            final_answer = solution.get('answer')
                            if final_answer is None:
                                feedback = "No answer provided"
                                await asyncio.sleep(1)
                                continue

                    # Prepare answer for submission
                    if answer_type_hint:
                        # Correction mode - use LLM's type hint
                        formatted_candidate = final_answer
                        log(f"✓ Using LLM-specified type: {answer_type_hint}")
                        
                        # Ensure correct type
                        if answer_type_hint == "number":
                            try:
                                if isinstance(formatted_candidate, str):
                                    formatted_candidate = float(formatted_candidate) if '.' in formatted_candidate else int(formatted_candidate)
                            except:
                                log("⚠️ Could not convert to number, keeping as-is")
                        elif answer_type_hint == "string":
                            # Keep as string - this is the default behavior
                            formatted_candidate = str(formatted_candidate)
                        elif answer_type_hint == "array":
                            # If it's a JSON string, parse it to array
                            if isinstance(formatted_candidate, str):
                                try:
                                    parsed = json.loads(formatted_candidate)
                                    if isinstance(parsed, list):
                                        formatted_candidate = parsed
                                except:
                                    pass
                        elif answer_type_hint == "object":
                            # If it's a JSON string, parse it to object
                            if isinstance(formatted_candidate, str):
                                try:
                                    parsed = json.loads(formatted_candidate)
                                    if isinstance(parsed, dict):
                                        formatted_candidate = parsed
                                except:
                                    pass
                    else:
                        # First attempt - KEEP AS STRING if it looks like JSON
                        formatted_candidate = final_answer
                        
                        # Basic type conversion for simple cases
                        if isinstance(final_answer, str):
                            stripped = final_answer.strip()
                            # Check if it looks like JSON - KEEP IT AS STRING
                            if stripped.startswith('[') or stripped.startswith('{'):
                                # Validate it's valid JSON, but keep as string
                                try:
                                    json.loads(stripped)  # Just validate
                                    formatted_candidate = stripped  # Keep as string
                                    log(f"✓ Keeping JSON-formatted string as-is (will be sent with escape sequences)")
                                except:
                                    log("⚠️ Looks like JSON but invalid, keeping as string anyway")
                                    formatted_candidate = stripped
                            # Simple number conversion (only for non-JSON strings)
                            elif stripped.isdigit():
                                formatted_candidate = int(stripped)
                            elif re.match(r'^-?\d+\.?\d*$', stripped):
                                try:
                                    formatted_candidate = float(stripped)
                                except:
                                    pass

                    if not submit_url:
                        feedback = "No submit URL"
                        log("ERROR: No submit URL")
                        await asyncio.sleep(1)
                        continue

                    # Build payload
                    payload = {
                        "email": email,
                        "secret": secret.strip(),
                        "url": current_url,
                        "answer": formatted_candidate
                    }

                    # Log submission details
                    log(f"📤 Submitting to: {submit_url}")
                    log(f"📋 Answer type: {get_answer_type(formatted_candidate)}")
                    log(f"📋 Answer preview: {str(formatted_candidate)[:200]}")

                    # Submit
                    try:
                        async with httpx.AsyncClient(timeout=15.0) as client:
                            resp = await client.post(submit_url, json=payload)
                            
                            try:
                                result_json = resp.json()
                            except:
                                result_json = None
                            
                            server_text = resp.text if hasattr(resp, "text") else str(resp)

                            # SUCCESS
                            if resp.status_code == 200 and result_json and result_json.get('correct'):
                                elapsed_time = time.time() - question_start_time
                                print(f"\n✅ CORRECT! Question {question_number} solved in {format_time(elapsed_time)}")
                                print(f"   Total attempts: {len(attempt_history) + 1}")
                                
                                current_url = result_json.get('url') or next_url_from_page
                                
                                if current_url:
                                    log("Moving to next question...")
                                    question_number += 1
                                else:
                                    print("\n🎉 ALL QUESTIONS COMPLETED!")
                                    total_time = time.time() - total_start_time
                                    print(f"📊 Total time: {format_time(total_time)}")
                                    print(f"📊 Questions solved: {question_number}")
                                
                                solved = True
                                break
                            
                            # REJECTION
                            else:
                                if result_json and 'reason' in result_json:
                                    reason = result_json.get('reason')
                                else:
                                    reason = server_text[:2000]
                                
                                # Check for critical errors
                                if "Secret mismatch" in str(reason):
                                    log("CRITICAL: Secret mismatch - stopping")
                                    timer_task.cancel()
                                    await browser.close()
                                    return
                                
                                # Record this attempt
                                attempt_history.append({
                                    'answer': formatted_candidate,
                                    'type': get_answer_type(formatted_candidate),
                                    'payload': payload.copy(),
                                    'error': reason
                                })
                                
                                log(f"❌ Rejected: {str(reason)[:150]}")
                                log(f"📊 Total attempts: {len(attempt_history)}")
                                
                                # Switch to correction mode after first failure
                                if not is_correction_mode and len(attempt_history) == 1:
                                    log("🔄 Switching to correction mode")
                                    is_correction_mode = True
                                
                                feedback = f"Answer rejected: {reason}"
                                await asyncio.sleep(1.0)
                                continue

                    except Exception as e:
                        feedback = f"Submission error: {e}"
                        log(f"Submit error: {e}")
                        await asyncio.sleep(1.0)
                        continue

                timer_task.cancel()

                if not solved:
                    elapsed_time = time.time() - question_start_time
                    print(f"\n❌ Failed Question {question_number} after {format_time(elapsed_time)}")
                    print(f"   Attempts made: {len(attempt_history)}")
                    
                    if next_url_from_page:
                        log(f"Moving to next question: {next_url_from_page}")
                        current_url = next_url_from_page
                        question_number += 1
                        continue
                    else:
                        break

            except Exception as e:
                timer_task.cancel()
                elapsed_time = time.time() - question_start_time
                log(f"Critical error after {format_time(elapsed_time)}: {e}")
                traceback.print_exc()
                
                if next_url_from_page:
                    log(f"Moving to next question: {next_url_from_page}")
                    current_url = next_url_from_page
                    question_number += 1
                    continue
                else:
                    break

        await browser.close()

        total_time = time.time() - total_start_time
        print("\n" + "="*80)
        print("🏁 QUIZ SOLVING COMPLETE")
        print("="*80)
        print(f"📊 Total questions attempted: {question_number}")
        print(f"📊 Total time elapsed: {format_time(total_time)}")
        print(f"📊 Average time per question: {format_time(total_time / question_number)}")
        print("="*80)
        # Example usage (uncomment & replace)
# if __name__ == "__main__":
#     email = "your_email@example.com"
#     secret = "your_secret"
#     start_url = "https://example.com/first-question"
#     asyncio.run(solve_quiz_task(email, secret, start_url))