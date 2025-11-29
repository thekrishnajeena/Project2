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

import httpx
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# ==========================================
# 1. CONFIGURATION
# ==========================================

LOG_PREFIX = "[SOLVER]"
MAX_TASK_DURATION = 165
AUDIO_MODEL_SIZE = "tiny"
MAX_NESTED_DEPTH = 3

# Comprehensive list of audio file extensions
AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.ogg', '.opus', '.m4a', '.aac', '.flac', 
    '.wma', '.oga', '.webm', '.mp4', '.m4b', '.3gp'
}

# LLM Config
LLM_API_URL = "https://aipipe.org/openrouter/v1/chat/completions"
LLM_API_KEY = os.getenv("LLM_API_KEY") or "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDU4OTRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.11Pa3UNcezX27plvgMxnAvONVP2IsVa6ynIUVYuZuoY"
LLM_MODEL = "gpt-5-nano"
LLM_TIMEOUT = 120.0  # Increased timeout

def log(*args):
    print(LOG_PREFIX, *args)

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
    """Check if a file is an audio file based on extension or MIME type"""
    if not filename_or_ext:
        return False
    ext = os.path.splitext(filename_or_ext.lower())[1]
    if not ext:
        ext = filename_or_ext.lower()
    return ext in AUDIO_EXTENSIONS

# ==========================================
# 2. LLM CLIENT (IMPROVED)
# ==========================================

async def ask_brain(system_prompt, user_prompt, retries=3):
    if not LLM_API_KEY:
        log("CRITICAL: No API Key found.")
        return None

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
        "max_tokens": 4000
    }

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
                log(f"LLM returned content (length: {len(content)})")
                
                parsed = safe_json_parse(content)
                if parsed:
                    return parsed
                else:
                    log(f"Failed to parse JSON from: {content[:200]}")
                    await asyncio.sleep(1)
                    continue

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
    return None

# ==========================================
# 3. AUDIO TRANSCRIPTION
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
# ==========================================

def run_python(code, file_map, nested_data=None):
    log("Executing Python code...")
    log(f"Code to execute:\n{code[:500]}...")
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    env = {
        "pd": pd, 
        "json": json, 
        "re": re, 
        "math": math, 
        "plt": plt,
        "files": file_map,
        "pdfplumber": pdfplumber,
        "base64": base64,
        "nested_pages": nested_data or {}
    }
    
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
    
    log(f"Execution result - Output: {result['output'][:200]}, Error: {result['error'][:200] if result['error'] else None}")
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
    
    # 1. Extract tables
    tables_text = ""
    for idx, table in enumerate(soup.find_all("table")):
        try:
            df = pd.read_html(str(table))[0]
            tables_text += f"\n\n[TABLE {idx+1}]:\n{df.to_markdown(index=False)}\n"
            table.decompose()
        except Exception as e:
            pass
    
    # 2. Extract form values
    form_values = []
    for inp in soup.find_all("input"):
        val = inp.get("value")
        name = inp.get("name")
        if val and name:
            form_values.append(f"{name}: {val}")
    
    form_text = ""
    if form_values:
        form_text = "\n\n[FORM VALUES]:\n" + "\n".join(form_values)
    
    # 3. Decode base64
    hidden_content = []
    for match in re.findall(r'atob\s*\(\s*["\']([^"\']+)["\']\s*\)', content):
        try:
            decoded = base64.b64decode(match).decode('utf-8')
            hidden_content.append(decoded)
        except:
            pass
    
    hidden_text = ""
    if hidden_content:
        hidden_text = "\n\n[DECODED BASE64 CONTENT]:\n" + "\n---\n".join(hidden_content)
    
    # 4. Get visible text
    visible_text = soup.get_text(separator="\n", strip=True)
    
    # 5. Scrape nested links
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
    
    # 6. Extract file links
    file_links = []
    all_file_extensions = ['.csv', '.pdf', '.xlsx', '.xls', '.json', '.txt', 'data:'] + list(AUDIO_EXTENSIONS)
    
    for tag in soup.find_all(['a', 'audio', 'source', 'video']):
        link = tag.get('href') or tag.get('src')
        if link and any(ext in link.lower() for ext in all_file_extensions):
            file_links.append(link)
    
    # 7. Extract submit URL
    submit_url = None
    full_text = visible_text + hidden_text + nested_text
    
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
        "full_context": visible_text + tables_text + form_text + hidden_text + nested_text
    }

# ==========================================
# 7. FILE DOWNLOADER AND PROCESSOR
# ==========================================

async def download_and_process_files(file_links, base_url):
    log(f"Processing {len(file_links)} file links...")
    
    files_data = []
    file_map = {}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for link in file_links:
            try:
                file_info = await download_file(client, link, base_url)
                if file_info:
                    content = extract_file_content(file_info)
                    files_data.append({
                        "name": file_info["name"],
                        "type": file_info["ext"],
                        "content": content,
                        "path": file_info["path"]
                    })
                    file_map[file_info["name"]] = file_info["path"]
                    log(f"  Processed: {file_info['name']} ({file_info['ext']})")
            except Exception as e:
                log(f"Error processing file {link}: {e}")
    
    return files_data, file_map

async def download_file(client, url, base_url):
    target = urljoin(base_url, url)
    
    if target.startswith("data:"):
        try:
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
        except Exception as e:
            log(f"Error processing data URI: {e}")
            return None
    
    try:
        resp = await client.get(target, follow_redirects=True)
        if resp.status_code == 200:
            name = os.path.basename(urlparse(target).path) or "file.bin"
            ext = os.path.splitext(name)[1].lower() or ".bin"
            
            if ext == ".bin" and "content-type" in resp.headers:
                content_type = resp.headers["content-type"].lower()
                if "audio" in content_type:
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
                    name = f"audio{ext}"
            
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
        
        elif is_audio_file(ext):
            transcript = transcribe_audio(path)
            return f"[AUDIO TRANSCRIPT - THIS MAY CONTAIN THE ACTUAL QUESTION]:\n{transcript}"
        
        return "[Binary file - no preview available]"
        
    except Exception as e:
        return f"[Error extracting content: {e}]"

# ==========================================
# 8. SOLUTION GENERATOR (IMPROVED)
# ==========================================

async def generate_solution(scraped_data, files_data, feedback=""):
    log("Generating solution strategy...")
    
    # Build comprehensive context
    files_summary = ""
    has_audio_question = False
    
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
    
    # Truncate context to avoid token limits
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
- Use files['filename.ext'] to access file paths
- Use pandas: df = pd.read_csv(files['data.csv'])
- MUST print() the final answer
- Keep code simple and focused

IMPORTANT: Return ONLY valid JSON, no markdown, no extra text."""

    user_prompt = f"""MAIN PAGE:
{main_context}

FILES:
{files_context}

{nested_context}

FEEDBACK FROM PREVIOUS ATTEMPT:
{feedback if feedback else "None - this is the first attempt"}

{'⚠️ AUDIO FILE DETECTED - READ THE TRANSCRIPT CAREFULLY! ⚠️' if has_audio_question else ''}

Provide your solution as JSON."""

    return await ask_brain(system_prompt, user_prompt)

# ==========================================
# 9. MAIN SOLVER
# ==========================================

async def solve_quiz_task(email, secret, start_url):
    if not secret or " " in secret:
        log("WARNING: Secret contains whitespace or is empty.")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        current_url = start_url
        
        while current_url:
            log(f"\n{'='*60}")
            log(f"SOLVING: {current_url}")
            log(f"{'='*60}")
            
            start_time = time.time()
            solved = False
            
            try:
                # STEP 1: Navigate
                await page.goto(current_url, wait_until="networkidle", timeout=20000)
                await asyncio.sleep(1)
                
                # STEP 2: Scrape content
                scraped_data = await scrape_page_content(page, current_url)
                log(f"Submit URL: {scraped_data['submit_url']}")
                
                # STEP 3: Process files
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
                
                # STEP 4: Solve with retries
                feedback = ""
                max_attempts = 3
                
                for attempt in range(max_attempts):
                    if time.time() - start_time > MAX_TASK_DURATION:
                        log("Time limit exceeded")
                        break
                    
                    log(f"\n=== Attempt {attempt + 1}/{max_attempts} ===")
                    
                    solution = await generate_solution(scraped_data, files_data, feedback)
                    
                    if not solution:
                        feedback = "LLM failed to respond. Try again with simpler approach."
                        log("LLM returned None - retrying...")
                        await asyncio.sleep(3)
                        continue
                    
                    log(f"Question location: {solution.get('question_location', 'N/A')}")
                    log(f"Understanding: {solution.get('question_understanding', 'N/A')[:150]}")
                    log(f"Strategy: {solution.get('strategy')}")
                    
                    final_answer = None
                    
                    if solution.get('strategy') == 'python_code':
                        code = solution.get('code', '')
                        if not code:
                            feedback = "No Python code provided"
                            continue
                        
                        result = run_python(code, file_map, scraped_data.get('nested_data', {}))
                        
                        if result['error']:
                            feedback = f"Python error: {result['error'][:300]}\nFix the code."
                            log(f"Code error: {result['error'][:200]}")
                            continue
                        
                        final_answer = result['image'] if result['image'] else result['output']
                        
                        if not final_answer:
                            feedback = "Code produced no output. Make sure to print() the answer."
                            continue
                        
                    else:
                        final_answer = solution.get('answer')
                        if final_answer is None:
                            feedback = "No answer provided"
                            continue
                    
                    # Type conversion
                    try:
                        if isinstance(final_answer, str) and final_answer.strip().isdigit():
                            final_answer = int(final_answer.strip())
                        elif isinstance(final_answer, str) and re.match(r'^-?\d+\.?\d*$', final_answer.strip()):
                            final_answer = float(final_answer.strip())
                    except:
                        pass
                    
                    # STEP 5: Submit
                    submit_url = scraped_data['submit_url']
                    
                    if not submit_url:
                        log("ERROR: No submit URL")
                        feedback = "No submit URL found"
                        continue
                    
                    log(f"Submitting to: {submit_url}")
                    log(f"Answer (type={type(final_answer).__name__}): {str(final_answer)[:100]}")
                    
                    payload = {
                        "email": email,
                        "secret": secret.strip(),
                        "url": current_url,
                        "answer": final_answer
                    }
                    
                    try:
                        async with httpx.AsyncClient(timeout=15.0) as client:
                            resp = await client.post(submit_url, json=payload)
                            
                            try:
                                result_json = resp.json()
                            except:
                                feedback = f"Server returned non-JSON: {resp.text[:200]}"
                                log(feedback)
                                continue
                            
                            if resp.status_code == 200 and result_json.get('correct'):
                                log("✓ CORRECT! Moving to next level...")
                                current_url = result_json.get('url')
                                solved = True
                                break
                            else:
                                reason = result_json.get('reason', 'Unknown')
                                
                                if "Secret mismatch" in str(reason):
                                    log("CRITICAL: Secret mismatch - stopping")
                                    await browser.close()
                                    return
                                
                                feedback = f"Wrong answer. Server: {reason}\n\nReview:\n- Did you understand the question?\n- Did you use the right data?\n- Check audio transcript if present"
                                log(f"✗ Wrong: {reason}")
                                
                    except Exception as e:
                        feedback = f"Submit error: {str(e)}"
                        log(f"Submit error: {e}")
                        continue
                
                if not solved:
                    log("Failed to solve after all attempts")
                    break
                    
            except Exception as e:
                log(f"Critical error: {e}")
                traceback.print_exc()
                break
        
        await browser.close()
        log("\nQuiz solving complete!")