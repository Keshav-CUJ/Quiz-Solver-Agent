import csv
import os
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse
import mimetypes
import io
from contextlib import redirect_stdout
from typing import Any, Dict        
import requests
import base64
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import torch
import whisper
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import sys
import subprocess
from contextlib import redirect_stdout
from typing import List, Dict
import pkg_resources 


ASSET_ROOT = Path(tempfile.gettempdir()) / "task_assets"
ASSET_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_filename(url: str) -> str:
    name = os.path.basename(urlparse(url).path) or "file"
    return name.replace("/", "_").replace("\\", "_")


def fetch_page(url: str) -> dict:
    """
    Use Playwright to load a JS-rendered page.

    Returns:
      {
        "html": full_rendered_html,
        "visible_text": extracted_text,
        "asset_urls": [list of discovered asset URLs],
        "final_url": the final URL after redirects (if any)
      }
    """
    # 1. Use Playwright to render JS
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # You can tweak timeout / wait strategy if needed
        page.goto(url, wait_until="networkidle", timeout=30_000)

        final_url = page.url
        html = page.content()

        browser.close()

    # 2. Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles/noscript to get cleaner visible text
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    visible_text = " ".join(soup.stripped_strings)
    visible_text = visible_text[:20000]  # hard cap to avoid huge contexts

    # 3. Collect asset URLs (anchors, images, audio, video sources, etc.)
    asset_urls: list[str] = []

    # <a href="...">
    for a in soup.find_all("a", href=True):
        asset_urls.append(urljoin(final_url, a["href"]))

    # <img src="...">
    for img in soup.find_all("img", src=True):
        asset_urls.append(urljoin(final_url, img["src"]))

    # <audio src="...">
    for audio in soup.find_all("audio", src=True):
        asset_urls.append(urljoin(final_url, audio["src"]))

    # <source src="..."> (for audio/video)
    for source in soup.find_all("source", src=True):
        asset_urls.append(urljoin(final_url, source["src"]))

    # You can add more tags if needed (e.g., <link rel="stylesheet"> if tasks hide data there)

    # Deduplicate while preserving order
    asset_urls = list(dict.fromkeys(asset_urls))

    return {
        "html": html,
        "visible_text": visible_text,
        "asset_urls": asset_urls,
        "final_url": final_url,
    }



def _guess_kind(url: str, content_type: str | None) -> str:
    ct = (content_type or "").lower()
    path = urlparse(url).path.lower()

    if "audio" in ct or path.endswith((".mp3", ".wav", ".m4a", ".ogg", ".opus")):
        return "audio"
    if "image" in ct or path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return "image"
    if "csv" in ct or path.endswith(".csv"):
        return "csv"
    if "pdf" in ct or path.endswith(".pdf"):
        return "pdf"
    return "other"


def fetch_asset(url: str) -> dict:
    """
    Download a single asset and return an asset_id and type.
    LLM will use asset_id in later calls.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type")
    kind = _guess_kind(url, ct)

    filename = _safe_filename(url)
    # guess extension if missing
    if not os.path.splitext(filename)[1]:
        ext = mimetypes.guess_extension(ct or "") or ""
        filename += ext

    asset_path = ASSET_ROOT / filename
    with open(asset_path, "wb") as f:
        f.write(resp.content)

    asset_id = asset_path.name  # simple id = filename

    return {
        "asset_id": asset_id,
        "kind": kind,
        "content_type": ct,
        "path": str(asset_path),
    }



_whisper_model = None

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _whisper_model = whisper.load_model("base").to(device)
    return _whisper_model



def transcribe_audio(asset_id: str) -> dict:
    """
    Transcribe audio using openai-whisper.
    asset_id must exist in ASSET_ROOT.
    """
    path = ASSET_ROOT / asset_id
    if not path.exists():
        return {"error": f"File not found: {asset_id}"}

    try:
        model = _load_whisper()
        result = model.transcribe(str(path))
        transcript = result.get("text", "").strip()
        return {
            "asset_id": asset_id,
            "transcript": transcript
        }
    except Exception as e:
        return {
            "asset_id": asset_id,
            "error": f"Whisper transcription failed: {e}"
        }



def ocr_image(asset_id: str) -> dict:
    """
    Run OCR on an image asset using Tesseract (via pytesseract).

    Input:
      asset_id: filename in ASSET_ROOT (created by fetch_asset)

    Output:
      {
        "asset_id": "...",
        "text": "recognized text...",
      }

      or on error:
      {
        "asset_id": "...",
        "error": "message"
      }
    """
    path = ASSET_ROOT / asset_id
    if not path.exists():
        return {"asset_id": asset_id, "error": f"File not found: {asset_id}"}

    try:
        # 1. Load image
        img = Image.open(path)

        # 2. Basic pre-processing to help OCR
        #    - convert to grayscale
        #    - auto-contrast
        #    - (optional) slight sharpen
        img = img.convert("L")  # grayscale
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)

        # 3. Run Tesseract
        text = pytesseract.image_to_string(img, lang="eng")
        text = text.strip()

        return {
            "asset_id": asset_id,
            "text": text,
        }

    except Exception as e:
        return {
            "asset_id": asset_id,
            "error": f"OCR failed: {e}",
        }


def summarize_csv(asset_id: str, max_rows: int = 5) -> dict:
    """
    Lightweight CSV summary for task identification.
    Shows:
      - columns
      - number of rows (approx if huge)
      - first few rows
      - inferred column types
    """
    path = ASSET_ROOT / asset_id
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= max_rows + 5:  # read a few extra rows for type inference
                break

    if not rows:
        return {"asset_id": asset_id, "summary": "Empty CSV."}

    header = rows[0]
    samples = rows[1:]

    # type inference
    inferred_types = []
    for col_index, col_name in enumerate(header):
        col_values = [r[col_index] for r in samples if len(r) > col_index]
        inferred_types.append(_infer_type(col_values))

    # build summary
    lines = []
    lines.append("Columns:")
    for name, t in zip(header, inferred_types):
        lines.append(f"  - {name} ({t})")

    lines.append("\nSample rows:")
    for row in samples[:max_rows]:
        lines.append("  - " + ", ".join(row))

    summary = "\n".join(lines)
    return {"asset_id": asset_id, "summary": summary}


def _infer_type(values: list[str]) -> str:
    """
    Infer simple type of column.
    """
    num = 0
    for v in values:
        try:
            float(v)
            num += 1
        except:
            pass

    if num >= len(values) * 0.8:
        return "numeric"
    return "string"

def get_installed_packages() -> dict:
    """
    Return currently installed packages as {name: version}.
    """
    pkgs = {dist.project_name: dist.version for dist in pkg_resources.working_set}
    return {"packages": pkgs}


def pip_install_packages(packages: List[str]) -> dict:
    """
    Install one or more packages into the *current env* using pip.
    Returns a status and simplified log.

    Example input: ["pandas", "numpy==1.26.4"]
    """
    if not packages:
        return {"status": "no_packages", "log": "No packages requested."}

    # Build command: python -m pip install ...
    cmd = [sys.executable, "-m", "pip", "install"] + list(packages)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )
        ok = proc.returncode == 0
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        # Truncate huge logs
        log = log[-4000:]
        return {
            "status": "ok" if ok else "error",
            "returncode": proc.returncode,
            "log": log,
        }
    except Exception as e:
        return {
            "status": "exception",
            "error": str(e),
        }






def run_python_code(code: str, context: Dict[str, Any] | None = None) -> dict:
    """
    Execute arbitrary Python code in the current environment.

    - Full imports allowed (pandas, numpy, etc.)
    - Full builtins
    - context: optional dict of initial local variables

    LLM is instructed to put the final result in a variable named `answer`.
    """
    # Full builtins, unrestricted
    global_ns: Dict[str, Any] = {
        "__builtins__": __builtins__,
    }

    # Local namespace gets any context we pass in (e.g. asset paths)
    local_ns: Dict[str, Any] = {}
    if context:
        local_ns.update(context)

    stdout_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf):
            exec(code, global_ns, local_ns)

        out = stdout_buf.getvalue()
        # We expect LLM to set a variable `answer`
        answer = local_ns.get("answer", None)

        return {
            "status": "ok",
            "stdout": out[-4000:],  # truncated
            "answer": repr(answer),
            "locals": {k: repr(v) for k, v in local_ns.items()},
        }
    except Exception as e:
        out = stdout_buf.getvalue()
        return {
            "status": "error",
            "error": str(e),
            "stdout": out[-4000:],
        }
        


def http_request(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    params: dict | None = None,
    json_body: dict | None = None,
    text_body: str | None = None,
    timeout: int = 30,
) -> dict:
    """
    Perform an HTTP request with any method.

    Args:
        url: URL string
        method: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
        headers: dict of request headers
        params: dict of URL query params
        json_body: JSON body (if provided, overrides text_body)
        text_body: raw text body (sent as-is)
        timeout: seconds to wait

    Returns:
        {
            "status": <int>,
            "ok": <bool>,
            "headers": { ... },
            "text": "response text (if utf8)",
            "json": { ... } or None,
            "binary_base64": "data:;base64,...." (if binary),
        }
    """
    # Normalize headers: support dict OR list[{"key","value"}]
    if isinstance(headers, list):
        headers = {h.get("key", ""): h.get("value", "") for h in headers if "key" in h}
    headers = headers or {}

    # Normalize params: support dict OR list[{"key","value"}]
    if isinstance(params, list):
        params = {p.get("key", ""): p.get("value", "") for p in params if "key" in p}
    params = params or {}

    # Decide request body
    data = None
    json_payload = None
    if json_body is not None:
        json_payload = json_body
    elif text_body is not None:
        data = text_body.encode("utf-8")

    try:
        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_payload,
            data=data,
            timeout=timeout,
        )
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

    # Build response dict
    out = {
        "status": resp.status_code,
        "ok": resp.ok,
        "headers": dict(resp.headers),
    }

    content_type = resp.headers.get("content-type", "").lower()

    # Try JSON response
    if "application/json" in content_type:
        try:
            out["json"] = resp.json()
        except Exception:
            out["json"] = None
        out["text"] = resp.text[:20000]  # truncate for safety
        return out

    # Try UTF-8 text
    try:
        text = resp.text
        out["text"] = text[:20000]
        out["json"] = None
    except Exception:
        out["text"] = None
        out["json"] = None

    # If binary (image, pdf, csv, zip, octet-stream, audio, etc.)
    if (
        "application/" in content_type
        or "image/" in content_type
        or "audio/" in content_type
        or "video/" in content_type
        or "octet-stream" in content_type
    ):
        b64 = base64.b64encode(resp.content).decode("ascii")
        out["binary_base64"] = f"data:{content_type};base64,{b64}"

    return out


