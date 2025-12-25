# state.py
from typing import TypedDict, Optional, Any


# state.py
from typing import TypedDict, Any

class AppState(TypedDict, total=False):
    # Input fields
    url: str
    email: str
    secret: str

    # Prep node output
    prep_task_description: dict
    prep_raw_log: list[str]

    # Execution node output
    final_answer: dict
    exec_log: list[str]
    
    submission_response: dict
    submission_json: dict
    
    # Optional generic error field
    error: str
