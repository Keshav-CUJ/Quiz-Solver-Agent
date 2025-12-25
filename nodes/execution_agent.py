import json
import os
import re
from typing import Any
from dotenv import load_dotenv
from state import AppState
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from tools import (
    fetch_page,
    fetch_asset,
    summarize_csv,
    transcribe_audio,
    ocr_image,
    get_installed_packages,
    pip_install_packages,
    run_python_code,
    http_request,
)


MAX_JSON_BYTES = 1 * 1024 * 1024  # 3 MB

load_dotenv()
Gemini_key = os.getenv("API_GEMINI_KEY")
Posting_URL = os.getenv("POST_URL")
GEMINI_API_KEYS = [
     Gemini_key
]
RETRYABLE_STATUS = {403, 429, 500, 503}  # includes leaked / blocked / rate limit


def generate_with_keys(**kwargs):
    """
    Try all Gemini API keys one by one.

    Retryable:
      - 403: key blocked / leaked / insufficient permissions
      - 429: rate limit / quota exceeded
      - 500: backend error
      - 503: service unavailable

    Non-retryable errors immediately raise.
    """
    last_exc = None

    for api_key in GEMINI_API_KEYS:
        client = genai.Client(api_key=api_key)

        try:
            return client.models.generate_content(**kwargs)

        except ClientError as e:
            status = e.code

            if status in RETRYABLE_STATUS:
                print(f"[Gemini] Key {api_key[:10]}... failed with {status}, trying next key...")
                last_exc = e
                continue

            # Not retryable → re-raise
            raise

        except Exception as e:
            # Unknown error → try next key
            print(f"[Gemini] Unexpected error for key {api_key[:10]}...: {e}")
            last_exc = e
            continue

    # All keys failed
    if last_exc:
        raise last_exc
    else:
        raise RuntimeError("No API keys configured.")

def _clean_json_text(text: str) -> str:
    """Helper to strip Markdown code fences (```json ... ```) from LLM output."""
    if not text:
        return ""
    # Remove ```json ... ``` or just ``` ... ```
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    return text.strip()


def _log_and_print(log_list: list[str], message: str):
    """Appends to the log list and prints to stdout for 'live' view."""
    log_list.append(message)
    # Print the message for 'live' output
    print(f"[EXEC_AGENT LOG] {message}")

def execution_agent_node(state: AppState) -> AppState:
    task = state.get("prep_task_description") or {}
    url = state.get("url")
    email = state.get("email")
    secret = state.get("secret")

    # ---------- Tools ----------

    tool_impls = {
        "fetch_page": fetch_page,
        "fetch_asset": fetch_asset,
        "summarize_csv": summarize_csv,
        "transcribe_audio": transcribe_audio,
        "ocr_image": ocr_image,
        "get_installed_packages": get_installed_packages,
        "pip_install_packages": pip_install_packages,
        "run_python_code": run_python_code,
        "http_request": http_request,
    }

    fetch_page_decl = {
        "name": "fetch_page",
        "description": "Fetch an HTML page and return html, visible_text, asset_urls and final_url.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    }

    fetch_asset_decl = {
        "name": "fetch_asset",
        "description": "Download a single asset (image, audio, csv, pdf, etc.) and return asset_id and metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    }

    summarize_csv_decl = {
        "name": "summarize_csv",
        "description": "Summarize a CSV file: columns and sample rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "max_rows": {"type": "integer", "default": 5},
            },
            "required": ["asset_id"],
        },
    }

    transcribe_audio_decl = {
        "name": "transcribe_audio",
        "description": "Transcribe an audio asset (e.g. mp3, opus, wav) by asset_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
            },
            "required": ["asset_id"],
        },
    }

    ocr_image_decl = {
        "name": "ocr_image",
        "description": "Run OCR on an image asset by asset_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
            },
            "required": ["asset_id"],
        },
    }

    get_installed_decl = {
        "name": "get_installed_packages",
        "description": "Return currently installed Python packages as {name: version}.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    }

    pip_install_decl = {
        "name": "pip_install_packages",
        "description": "Install additional Python packages into the current environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of packages, e.g. ['pandas', 'numpy==1.26.4']",
                },
            },
            "required": ["packages"],
        },
    }

    run_python_decl = {
        "name": "run_python_code",
        "description": (
            "Execute Python code in the current environment. "
            "You MUST put your final answer into a variable named `answer`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "context": {
                    "type": "object",
                    "description": "Optional local variables, e.g. paths to assets.",
                },
            },
            "required": ["code"],
        },
    }
    http_request_decl = {
    "name": "http_request",
    "description": "Perform an HTTP request with any method, headers, query params, and JSON or text body.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string", "default": "GET"},

            # FIXED: use array of {key, value}
            "headers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["key", "value"]
                }
            },

            "params": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["key", "value"]
                }
            },

            "json_body": {"type": "object"},
            "text_body": {"type": "string"},
            "timeout": {"type": "integer", "default": 30},
        },
        "required": ["url"],
    },
}


    tools_list = [
        types.Tool(
            function_declarations=[
                fetch_page_decl,
                fetch_asset_decl,
                summarize_csv_decl,
                transcribe_audio_decl,
                ocr_image_decl,
                get_installed_decl,
                pip_install_decl,
                run_python_decl,
                http_request_decl,
            ]
        )
    ]

    # ---------- System Prompt ----------

    system_instructions = """
You are an execution agent for data-related tasks.

You are given a structured task description (from the prep agent) plus email, secret, and url.

Your job:
- If you can solve the task just with HTTP / scraping / crawling and normal given tools, do that via tools and return the final JSON. Only if you need more complex data processing (e.g., CSV aggregation, stats, charting), call run_python_code.  
- Use tools to fetch pages, download assets, inspect CSVs, transcribe audio, and OCR images.
- Whenever you need to run the code, inspect the current environment with get_installed_packages first. 
- Required assets and code file will be saved in the created asset directory so handle paths very carefully inside the code.
- If required, install additional Python libraries with pip_install_packages (never install tensorflow use pytorch instead).
- If computation / aggregation / transformation is needed, write Python code and execute it with run_python_code.
- In your Python code, ALWAYS put the final result into a variable named `answer`.

Your final output must be ONLY valid JSON which must be in the format of: as given in the structured task description (from the prep agent):

Post JSON to {Posting_URL}(may be changed scrap from page)

for example: (Maybe mostly similar as below)
{ "POST url" : "Posting_URL"
  "data": {
            "email": "<same email I give you>",
            "secret": "<same secret I give you>",
            "url": "<same url I give you>",
            "answer": <the final answer>
  }
}

The `answer` field can be:
- a number
- a string
- a boolean
- an object/dict
- or a base64 URI of a file attachment.
- a JSON object with a combination of these.

Do NOT return explanations. Do NOT wrap JSON in code fences. Only return the JSON.
"""

    # ---------- Initial user content ----------

    task_payload = {
        "task_description": task,
        "input_email": email,
        "input_secret": secret,
        "input_url": url,
    }

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=(
                        "Here is the task and inputs:\n\n"
                        + json.dumps(task_payload, indent=2)
                    )
                )
            ],
        )
    ]

    config = types.GenerateContentConfig(
        tools=tools_list,
        system_instruction=system_instructions,
        temperature=0.1,
    )

    log: list[str] = []
    max_steps = 16

    for step in range(max_steps):
        response = generate_with_keys(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
                                        )
        
        # ----- SAFETY: handle empty candidates or content -----
        if not response.candidates:
            _log_and_print(
                log,
                "Model returned no candidates, asking it to respond again.",
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "You returned no candidates. Please respond again, "
                                "using tools if needed, and eventually output ONLY the final JSON."
                            )
                        )
                    ],
                )
            )
            continue


        candidate = response.candidates[0]
        mc = candidate.content

        if mc is None or not getattr(mc, "parts", None):
            _log_and_print(
                log,
                "Model returned empty content or no parts, asking it to respond again.",
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "You returned an empty message. Please respond again, "
                                "using tools if needed, and then output ONLY the final JSON answer."
                            )
                        )
                    ],
                )
            )
            continue

        # from here on, mc.parts is safe to use
        parts = mc.parts

        # ---- CASE 1: Tool calls ----
        tool_calls = [part.function_call for part in parts if part.function_call]

        if tool_calls:
            contents.append(mc)
            response_parts = []

            for fc in tool_calls:
                name = fc.name
                args = dict(fc.args or {})
                fn = tool_impls.get(name)

                _log_and_print(log, f"STEP {step}: Calling {name} with {args}")

                if fn is None:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = {"error": str(e)}

                # Only show a snippet of the result in the live log
                result_snippet = str(result)[:200].replace("\n", " ")
                _log_and_print(log, f"   -> Result Snippet: {result_snippet}...")

                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response={"result": result},
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=response_parts))
            continue

        # ---- CASE 2: No tool calls => model is returning final JSON ----
        text_parts = [p.text for p in parts if p.text]
        final_text = "".join(text_parts).strip()

        if not final_text:
            log.append("Empty response text, asking model to output JSON again.")
            contents.append(mc)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "You returned no text. Please respond ONLY with the final JSON object as specified."
                            )
                        )
                    ],
                )
            )
            continue

        # 1) Try to parse as JSON
        try:
            cleaned_json = _clean_json_text(final_text)
            final_json = json.loads(cleaned_json)
        except json.JSONDecodeError:
            log.append(f"Invalid JSON from model: {final_text[:400]}")
            contents.append(mc)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Your last response was not valid JSON. "
                                "Respond again with ONLY the JSON object in the required form, no explanations."
                            )
                        )
                    ],
                )
            )
            continue

        # 2) Compact-serialize and check size (in bytes)
        compact_bytes = json.dumps(final_json, separators=(",", ":")).encode("utf-8")
        size_bytes = len(compact_bytes)

        if size_bytes > MAX_JSON_BYTES:
            log.append(
                f"JSON too large: {size_bytes} bytes (> {MAX_JSON_BYTES}). Asking model to shrink."
            )
            contents.append(mc)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                f"Your last JSON response was too large "
                                f"({size_bytes} bytes). It must be under {MAX_JSON_BYTES} bytes (~3MB) "
                                "when JSON-encoded. Please compress or shorten it:\n"
                                "- If you included large arrays, summarize them.\n"
                                "- If you included a base64 file, ensure it's small enough.\n"
                                "Respond again with ONLY the smaller JSON object."
                            )
                        )
                    ],
                )
            )
            continue

        # 3) All good: small enough and valid
        
        safe_url_part = re.sub(r'[^a-zA-Z0-9]', '_', url.replace('http://', '').replace('https://', '').strip('/'))
        log_filename = f"execution_agent_log_{safe_url_part[:50]}.txt"
        
        try:
            with open(log_filename, "w") as f:
                    f.write("\n".join(log))
            _log_and_print(log, f"Full log successfully saved to file: {log_filename}")
        except Exception as e:
                _log_and_print(log, f"Warning: Could not save log file {log_filename}. Error: {e}")
                
        _log_and_print(log, "Agent successfully completed task preparation.")
        
        
        
        
        return {
            **state,
            "final_answer": final_json,
            "exec_log": log,
        }
