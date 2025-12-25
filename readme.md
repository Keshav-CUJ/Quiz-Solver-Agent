
# **Autonomous Quiz Solver (Gemini + LangGraph + Playwright)**

This project is a fully automated agent system that **opens a quiz URL**,
**scrapes the page**, **downloads assets**, **transcribes audio**, **OCRs images**,
**analyzes CSV**, **generates Python code to compute the answer**,
**submits the answer**, and **follows the next quiz URL**—until all quizzes are solved.

It is fully autonomous and powered by:

* **Google Gemini API**
* **LangGraph multi-agent pipeline**
* **Playwright (JS-rendered scraping)**
* **OpenAI Whisper** for audio transcription
* **Pytesseract** for image OCR
* **FastAPI** for external webhook-triggered execution
* **Dynamic pip installation** for adding Python dependencies on-demand

---

# 📂 **Folder Structure**

```
project/
│
├── api-server.py                # FastAPI server (accepts /run-quiz)  :contentReference[oaicite:0]{index=0}
├── solve_quiz_series.py         # Main multi-round solver              :contentReference[oaicite:1]{index=1}
├── app_agent.py                 # LangGraph pipeline builder           :contentReference[oaicite:2]{index=2}
│
├── nodes/
│   ├── prep_agent.py            # Fetch webpage + assets, extract task description
│   ├── execution_agent.py       # Generate & run Python code to compute answer
│   └── submit_agent.py          # POST final answer to quiz API        :contentReference[oaicite:3]{index=3}
│
├── tools.py                     # Scraping, OCR, Whisper, http, code execution  :contentReference[oaicite:4]{index=4}
├── requirements.txt             # All project dependencies             :contentReference[oaicite:5]{index=5}
│
└── README.md                    # (this file)
```

---

# ⚙️ **Installation Guide**

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

---

## 2️⃣ Install Python Dependencies

Use the provided requirements file:

```bash
pip install -r requirements.txt
```

Dependencies include:

* **Playwright**
* **google-genai**
* **openai-whisper**
* **torch/torchaudio**
* **pytesseract**
* **BeautifulSoup4**
* **FastAPI + Uvicorn**

---

## 3️⃣ Install Playwright Chromium

(Needed for JS-rendered pages)

```bash
playwright install chromium
```

---

## 4️⃣ Install **FFmpeg** (Required for Whisper)

🔗 Download FFmpeg (Windows builds):
[https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)

### Steps:

1. Download *ffmpeg-release-full.zip*
2. Extract to: `C:\ffmpeg`
3. Add to PATH:
   `C:\ffmpeg\bin`

Verify:

```bash
ffmpeg -version
```

---

## 5️⃣ Install **Tesseract OCR**

Required for OCR on image-based tasks.

🔗 Windows Installer:
[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

Download: *tesseract-ocr-w64-setup-XXXX.exe*

Then add to PATH:

```
C:\Program Files\Tesseract-OCR\
```

Verify:

```bash
tesseract --version
```

---

## 6️⃣ Configure Environment Variables

Create a `.env` file (Or just rename .env.example to .env):

```
API_SECRET=your_api_secret_here
EMAIL=your_email_here
```

---

# 🚀 **How to Run the API Server**

Start FastAPI:

```bash
uvicorn api-server:app --host 0.0.0.0 --port 8000
```

Server is now live at:

```
http://localhost:8000
```

---

# 🔌 **API Endpoint Usage**

Your server exposes:

## **POST /run-quiz**

Payload:

```json
{
  "email": "your email",
  "secret": "your secret",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}
```

### Example cURL:

```bash
curl -X POST http://localhost:8000/run-quiz \
  -H "Content-Type: application/json" \
  -d '{
        "email": "23f3003494@example.com",
        "secret": "your_api_secret",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
      }'
```

### Server Response (immediate):

```json
{
  "email": "23f3003494@example.com",
  "url": "https://tds-llm-analysis.s-anand.net/demo",
  "status": "Quiz solving started"
}
```

The quiz runs **asynchronously in the background**.

---

# 🧠 **How the Autonomous Agent Works**

From your `solve_quiz_series.py` logic: 

### **Round Loop**:

1. Visit quiz URL

2. Run **Prep Agent**

   * Scrapes page using Playwright
   * Downloads assets
   * Runs OCR, Whisper, CSV summary
   * Produces structured *task description*

3. Run **Execution Agent**

   * Generates Python code
   * Installs missing dependencies
   * Executes code
   * Produces the final JSON answer payload

4. Run **Submit Agent**

   * POSTs the JSON to quiz server
   * Reads `{ correct, url, reason }`

5. Move to next URL (if provided)

---

# 🧪 **Running Locally Without API**

Edit bottom of `solve_quiz_series.py`:


```python
START_URL = "https://tds-llm-analysis.s-anand.net/demo"
EMAIL = Email
SECRET = API_SECRET

solve_quiz_series(START_URL, EMAIL, SECRET)
```

Run:

```bash
python solve_quiz_series.py
```

---

# 🐞 Troubleshooting

### ❌ Whisper errors

* FFmpeg missing → install FFmpeg and add to PATH.
* Large audio → Whisper may take time; GPU recommended.

### ❌ OCR errors

* Install Tesseract
* Add `C:\Program Files\Tesseract-OCR` to PATH.

### ❌ Playwright errors

Run:

```
playwright install chromium
```

### ❌ 403 / 429 Gemini API errors

The code automatically rotates keys (multi-key failover).

---
