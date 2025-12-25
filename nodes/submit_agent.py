# nodes/submit_agent.py
import json
from state import AppState
from tools import http_request

def submit_agent_node(state: AppState) -> AppState:
    final_answer = state.get("final_answer")
    if not final_answer:
        return {**state, "error": "final_answer missing for submission"}

    post_url = final_answer.get("POST url")
    data = final_answer.get("data")

    if not post_url or not data:
        return {
            **state,
            "error": f"Invalid final_answer structure: {final_answer}"
        }

    resp = http_request(
        url=post_url,
        method="POST",
        headers={"Content-Type": "application/json"},
        json_body=data,
        timeout=50,
    )

    # Try to get JSON from http_request output
    sub_json = resp.get("json")
    if sub_json is None and resp.get("text"):
        try:
            sub_json = json.loads(resp["text"])
        except Exception:
            sub_json = None

    return {
        **state,
        "submission_response": resp,   # raw HTTP info
        "submission_json": sub_json,   # parsed server JSON {correct, url, reason, ...}
    }
