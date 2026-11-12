# 07 · Running the Project, Step by Step

Every command is run **from the `Project` folder**. On Windows use `python`; on macOS/Linux use `python3`.

## Step 0 · Install

```bash
cd Project
python -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
```
The first install downloads PyTorch (CPU) and other packages, so it takes a few minutes.

## Step 0b · Add your Gemini key

Open `Project/.env` (copy `.env.example` if it's missing) and set:
```
GEMINI_API_KEY=your-real-key
GEMINI_MODEL=your-model-name        # e.g. a Gemini Flash-Lite model name from Google AI Studio
```
The key is read at the moment of each call (`load_dotenv()`), so it is never written into code. `.env` is git-ignored.

## Step 1 · Create the database and accounts
```bash
python run_database_setup.py
```
Expect: `database: database/breakcheck.db`, the six tables, 14 "account created" lines, and `accounts created: 14, already present: 0`. Running it again creates nothing (`0 created, 14 already present`).

## Step 2 · Clean the raw data
```bash
python run_data_pipeline.py
```
Expect:
```
campaign_submissions_train    30706 received ->  29697 kept  (1009 removed)
campaign_submissions_eval      8514 received ->   8224 kept  (290 removed)
programme_audiences_train     30877 received ->  29817 kept  (1060 removed)
programme_audiences_eval       7004 received ->   6770 kept  (234 removed)
clearance_notes               34727 received ->  33045 kept  (1682 removed)
quality report written to output/data_quality_report.json
```

## Step 3 · Train the models
```bash
python run_model_training.py
```
Expect (about 15 seconds):
```
amendment held-out ROC AUC 0.8571 (train 0.8639, gap +0.0069)
exposure  held-out R2 0.9213 MAE 9.46 thousand (train 0.9502, gap +0.0289)
issue     held-out macro F1 0.8624 (train 0.8768, gap +0.0144)
every model reaches the minimum config sets for it
```

## Step 4 · Build the knowledge base
```bash
python run_knowledge_indexing.py
```
The first run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB; needs internet once). Expect `documents read: 16`, `sections found: 150`, `chunks stored: 182`, then each category's five rules read back from its own code. (A Windows "symlinks" warning from Hugging Face is harmless.)

## Step 5 · Start the REST service (terminal 1, leave it running)
```bash
python run_api_service.py
```
Expect: `Starting BreakCheck AI API on port 8081. Interactive documentation: http://localhost:8081/docs` and `Uvicorn running on http://0.0.0.0:8081`. Start-up takes 20–40 seconds while it checks all 240 campaigns.

If something is missing, it names each missing path and what produces it, for example:
`output/models/amendment_model.pkl is missing; it comes from run_model_training.py`

## Step 6 · Start the dashboard (terminal 2)
```bash
python -m streamlit run run_dashboard.py
```
Open **http://localhost:8080**.

## Running the tests
The marking suite is `pytest/tests.py`; the runner copies it into the project:
```bash
cp ../pytest/tests.py tests.py        # Windows PowerShell: Copy-Item ..\pytest\tests.py tests.py
python -m pytest tests.py -v
```
Notes:
- Run Steps 1–4 first, because the tests read the cleaned files, models, store and database.
- Some tests make **real Gemini calls** (one assessment, one decision, one chat question), so a valid key is needed.
- Each test run assesses one fresh campaign. To start again from an empty state, delete `database/breakcheck.db` and `database/checkpoint.db`, then re-run Step 1.

---

## Demo script (about 10 minutes)

**A. Swagger UI (the API)**: http://localhost:8081/docs
1. `POST /auth/signin` → *Try it out* → `{"username": "tom", "password": "22222222"}` → copy `session_token`.
2. `GET /queue` → put the token in the **X-Session-Token** field, `product_category = All` → the ranked queue. CLR-SLV-0042499 is at the top.
3. `GET /queue/CLR-QSC-0042117` → the worked example: 10 spots, 6 breaching, first spot Saturday 18:20.
4. Show security: `GET /queue` with **no** token → 401. Sign in as `lucy` / `33333333` and call `GET /queue` with her token → **403** "A Agency Liaison may not use the intake area".

**B. Dashboard**: http://localhost:8080
1. Sign in as **tom** (use the *Sample credentials* drop-down). Open **Clearance queue**, explain the ranking and exposure, filter by *Betting and gaming*, select **CLR-QSC-0042117**, and open it.
2. **Campaign detail**: the *Broadcast days* chart (spot 10 shown at Saturday 29:40); *Spots and rules* (the five rules read from the PDF beside the code's own sentence); *Copy and notes*.
3. **Assess** tab: type a request → *Start the assessment* (about 20–60 s, 5 LLM calls) → *"ASM-… is waiting for a Clearance Controller, recommending CLEAR WITH RESTRICTIONS."*
4. Sign out. Sign in as **priya** (controller) → **Decision queue** → choose *CLEAR WITH RESTRICTIONS*, write a reason → *Record the decision* → "24 spots cleared and 3 withdrawn"-style result.
   - Optional: sign in as **owen**, start another assessment, then try to decide it as owen. The form is replaced by the "may not decide" message.
5. **Audit trail**: select the campaign to show the decision, who and why, **7 model calls vs 7 expected**, latency per call, and the notice and certificate.
6. **Clearance chat**: ask *"How many spots of one alcohol campaign may go out on one channel in one broadcast day?"* → CODE RULES answer with its sources. Then *"And how far apart?"* (memory). Then *"What is the capital of France?"* → refused.
7. Sign in as **lucy**: only the chat page. Ask about `CLR-QSC-0042117` (her agency QSC) → status only. Ask about a campaign from another agency → "cannot be found".

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard says the service is not answering | Start `python run_api_service.py` first and wait for "Application startup complete". |
| 502 "… failed: Set GEMINI_API_KEY and GEMINI_MODEL…" | Fill in `.env`, then restart the service. |
| 502 mentioning the model reply | The model replied in an unexpected format (the run is marked FAILED). Retry; for a failed decision, send the **same** decision and reason again. |
| `Campaign … has already been assessed` (409) | A campaign is assessed once; pick another campaign. |
| Port 8080 or 8081 already in use | Stop the other program using it. |
| You want a clean slate | Delete `database/breakcheck.db` and `database/checkpoint.db`, then run `run_database_setup.py`. |
