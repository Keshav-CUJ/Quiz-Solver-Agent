import os
import json
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
    transcribe_audio,
    ocr_image,
    summarize_csv,
    http_request,
)

load_dotenv()
Gemini_key = os.getenv("API_GEMINI_KEY")
API_SECRET = os.getenv("API_SECRET")
Email= os.getenv("EMAIl")
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
    print(f"[PREP_AGENT LOG] {message}")

def prep_agent_node(state: AppState) -> AppState:
    url = state["url"]

    # ---------- 1. Define Tools ----------
    # (Keeping your declarations as they were correct)
    fetch_page_decl = {
        "name": "fetch_page",
        "description": "Fetch an HTML page and return html, visible_text, asset_urls and url of page itself.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The page URL to fetch."},
            },
            "required": ["url"],
        },
    }

    fetch_asset_decl = {
        "name": "fetch_asset",
        "description": "Download a single asset (image, audio, csv, pdf, etc.) and return an asset_id and metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The asset URL to fetch."},
            },
            "required": ["url"],
        },
    }

    transcribe_audio_decl = {
        "name": "transcribe_audio",
        "description": "Transcribe an audio asset by asset_id.",
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

    summarize_csv_decl = {
        "name": "summarize_csv",
        "description": "Summarize a CSV asset: columns and first few rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "max_rows": {"type": "integer", "default": 5},
            },
            "required": ["asset_id"],
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
                transcribe_audio_decl,
                ocr_image_decl,
                summarize_csv_decl,
                http_request_decl,
            ]
        )
    ]

    # Map tool names to actual Python functions
    tool_impls = {
        "fetch_page": fetch_page,
        "fetch_asset": fetch_asset,
        "transcribe_audio": transcribe_audio,
        "ocr_image": ocr_image,
        "summarize_csv": summarize_csv,
        "http_request": http_request, 
    }

    # ---------- 2. Build Initial Prompt ----------
    system_instructions = """
You are a task-preparation agent for data-related quiz pages.

You are given the main page URL of a task.

YOUR GOAL in this phase:
- Use the provided tools to inspect the page and any relevant files (audio, CSV, images, PDFs, etc.).
- Understand clearly what the human is being asked to do.
- Do NOT actually solve the task.
- Just output a structured JSON description of the task.

When you are done, respond with ONLY valid JSON following this schema:

{
  "task_summary": "One or two sentences describing the task.",
  "expected_answer_type": "number|string|boolean|object|file_base64",
  "expected_answer_description": "What the `answer` field should contain.",
  "important_urls": ["...", "..."],               // endpoints or relative URLs the solver must use
  "files_used": ["asset_id1", "asset_id2"],       // any csv/pdf/audio/image you inspected
  "steps_to_solve": [
    "step 1...",
    "step 2...",
    "..."
  ]
}

Rules:
- Always call fetch_page(url) FIRST on the main page URL I give you.
- Only download assets that seem relevant to understanding the task.
- For audio, call transcribe_audio(asset_id).
- For images that might contain instructions or codes, call ocr_image(asset_id).
- For CSVs, call summarize_csv(asset_id) to see columns and sample rows and also provide columns name(if available) and types in task discription.
- If any api calling required for completion of task so you can scrap the header, url, cookies, token and related information for next node (Dont make any mistake in it).
- Use these results to infer what the human must do.
- Email: {Email} and Secret: {API_SECRET} (Use where needed)
- After the task complitioin answer will be posted via a JSON whose structure would be defined clearly so, describe that properly for next node.
- Do NOT compute the final answer; only describe the task.
"""

    prompt = f"Main task URL: {url}"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=prompt)],
        )
    ]
    
    # Configure tools for the session
    config = types.GenerateContentConfig(
        tools=tools_list,
        system_instruction=system_instructions,
        temperature=0.1, # Low temp for deterministic tool usage
    )

    log: list[str] = []
    max_steps = 10  # Prevent infinite loops
    step = 0
    _log_and_print(log, f"Starting preparation agent for URL: {url}")
    # ---------- 3. Main Loop (Multi-Turn) ----------
    while step < max_steps:
        step += 1
        
        # Call model
        response = generate_with_keys(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
                                        )


        candidate = response.candidates[0]
        model_content = candidate.content
        
        # Check for tool calls
        tool_calls = []
        if model_content.parts:
            for part in model_content.parts:
                if part.function_call:
                    tool_calls.append(part.function_call)

        # --- CASE A: Model wants to call tools ---
        if tool_calls:
            # Append model's request to history
            contents.append(model_content)
            
            # Execute tools
            response_parts = []
            for fc in tool_calls:
                name = fc.name
                args = dict(fc.args or {})
                fn = tool_impls.get(name)

                _log_and_print(log, f"STEP {step}: Calling {name} with {args}")
                
                if fn:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {name}"}

                # Only show a snippet of the result in the live log
                result_snippet = str(result)[:100].replace('\n', ' ')
                _log_and_print(log, f"   -> Result Snippet: {result_snippet}...")

                # Build function response part explicitly
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response={"result": result}, # Wrapping in a dict is safer
                        )
                    )
                )

            # Append tool outputs to history
            contents.append(types.Content(role="user", parts=response_parts))
            continue # Loop again to let model see results

        # --- CASE B: Model returned text (Final Answer) ---
        text_parts = [p.text for p in model_content.parts if p.text]
        final_text = "".join(text_parts).strip()
        
        cleaned_json = _clean_json_text(final_text)
        
        try:
            task_desc = json.loads(cleaned_json)
            # --- Save Log to File ---
            # Create a simple, safe filename based on the URL
            safe_url_part = re.sub(r'[^a-zA-Z0-9]', '_', url.replace('http://', '').replace('https://', '').strip('/'))
            log_filename = f"prep_agent_log_{safe_url_part[:50]}.txt"
            
            try:
                with open(log_filename, "w") as f:
                    f.write("\n".join(log))
                _log_and_print(log, f"Full log successfully saved to file: {log_filename}")
            except Exception as e:
                _log_and_print(log, f"Warning: Could not save log file {log_filename}. Error: {e}")
                
            _log_and_print(log, "Agent successfully completed task preparation.")
            
            return {
                **state,
                "prep_task_description": task_desc,
                "prep_raw_log": log,
            }
        except json.JSONDecodeError:
            # If model just chatted without JSON, force it to format
            _log_and_print(log, "Error: Model returned invalid JSON. Sending instruction to retry.")
            contents.append(model_content)
            contents.append(types.Content(role="user", parts=[
                types.Part(text="Invalid JSON. Please output ONLY the JSON object matching the schema.")
            ]))
            continue
        
    _log_and_print(log, "Max steps reached without valid JSON output. Exiting.")
    return {**state, "error": "Max steps reached without valid JSON"}