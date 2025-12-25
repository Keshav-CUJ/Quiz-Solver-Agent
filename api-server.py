import os
from dotenv import load_dotenv
from solve_quiz_series import solve_quiz_series
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from solve_quiz_series import solve_quiz_series  # import your function

# Load environment variables from .env
load_dotenv()

API_SECRET = os.getenv("API_SECRET")
if not API_SECRET:
    raise RuntimeError("API_SECRET not found in .env")

app = FastAPI(title="Quiz Runner API", version="1.0")


@app.post("/run-quiz")
async def run_quiz(request: Request, background_tasks: BackgroundTasks):
    """
    Receive JSON:
    {
      "email": "your email",
      "secret": "your secret",
      "url": "<quiz page-url>"
    }

    - Verify `secret` against API_SECRET
    - Start solve_quiz_series(url, email, secret) as a background task
    - Return 200 immediately
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Validate required fields
    for field in ["email", "secret", "url"]:
        if field not in data:
            raise HTTPException(status_code=400, detail=f"Missing field: {field}")

    email = data["email"]
    secret = data["secret"]
    start_url = data["url"]

    # Verify API secret
    if secret != API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized: invalid secret")

    # Schedule background task
    background_tasks.add_task(run_quiz_task, start_url, email, secret)

    # Immediate response to caller
    response_payload = {
        "email": email,
        "url": start_url,
        "status": "Quiz solving started",
    }
    return JSONResponse(content=response_payload, status_code=200)


def run_quiz_task(start_url: str, email: str, secret: str):
    """
    Background task wrapper. Runs the full quiz solver.
    Any exceptions are logged but do not affect the HTTP response.
    """
    try:
        print(f"Starting quiz solving for {email} at {start_url}")
        solve_quiz_series(start_url, email, secret)
        print(f" Finished quiz solving for {email} at {start_url}")
    except Exception as e:
        # You can swap this for proper logging
        print(f" Error in solve_quiz_series: {e}")


@app.get("/")
def home():
    return {"message": "Quiz Runner API is running!"}
