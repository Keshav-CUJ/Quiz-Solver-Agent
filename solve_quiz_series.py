import json
import os
from typing import Optional
from dotenv import load_dotenv
from app_agent import build_app
from state import AppState

def solve_quiz_series(
    start_url: str,
    email: str,
    secret: str,
    max_rounds: int = 20,
) -> None:
    app = build_app()
    current_url: Optional[str] = start_url
    round_no = 0

    while current_url and round_no < max_rounds:
        round_no += 1
        print(f"\n=== ROUND {round_no}: solving {current_url} ===")

        # Build initial state for this round
        initial: AppState = {
            "url": current_url,
            "email": email,
            "secret": secret,
        }

        state_out = app.invoke(initial)

        # Optional: debugging
        if "error" in state_out:
            print("ERROR in pipeline:", state_out["error"])
            break

        # Show final_answer for debugging
        print("\nFinal answer for this URL:")
        print(json.dumps(state_out.get("final_answer", {}), indent=2))

        # Read server response
        sub_json = state_out.get("submission_json")
        if not sub_json:
            print("No JSON submission response from server, stopping.")
            break

        correct = sub_json.get("correct")
        next_url = sub_json.get("url")
        reason = sub_json.get("reason")

        print("\nServer response:")
        print(json.dumps(sub_json, indent=2))

        if correct:
            print("Correct for this quiz.")
        else:
            print("Incorrect for this quiz.")
            if reason:
                print("Reason from server:", reason)

        # Decide next step
        if next_url:
            # Server is giving a new URL (either next quiz or skip)
            print(f"Next URL from server: {next_url}")
            current_url = next_url
            continue
        else:
            # No new URL => quiz is over
            print("No next URL provided. Quiz is over.")
            break

    print("\n=== DONE ===")

load_dotenv()
API_SECRET = os.getenv("API_SECRET")
Email= os.getenv("EMAIl")

if __name__ == "__main__":
    # Fill with your real values:
    START_URL = "<quiz-page url>"
    EMAIL = Email
    SECRET = API_SECRET

    solve_quiz_series(START_URL, EMAIL, SECRET)
