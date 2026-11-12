# 01 · Project Overview: BreakCheck AI

> **Read this first.** It explains the problem, what was built, how the pieces fit together, and what every file does. The other documents go deeper into each part.

| # | Document | What it covers |
|---|----------|----------------|
| 01 | **Project Overview** (this file) | Problem, requirements, architecture, every file's purpose |
| 02 | [LangGraph Workflow](02_LANGGRAPH_WORKFLOW.md) | The 10 nodes, the state, the human-in-the-loop gate and the checkpointer |
| 03 | [RAG and Knowledge Base](03_RAG_AND_KNOWLEDGE_BASE.md) | PDFs → chunks → vectors, rule reading, casebook filtering, reranking |
| 04 | [Machine Learning Models](04_MACHINE_LEARNING_MODELS.md) | The 3 models, why each algorithm, evaluation metrics |
| 05 | [Data Analysis](05_DATA_ANALYSIS.md) | Every dataset, every defect, how it was cleaned |
| 06 | [Streamlit, Sessions and Data Flow](06_STREAMLIT_SESSIONS_AND_DATA_FLOW.md) | What happens on every click, how sessions and shared data work |
| 07 | [Running the Project](07_RUNNING_THE_PROJECT.md) | Every command, from install to demo |
| 08 | [Concepts and Interview Prep](08_CONCEPTS_AND_INTERVIEW_PREP.md) | Glossary of concepts and likely review questions with answers |

---

## 1. The problem in plain words

**Harbourlight Media** runs four UK TV channels: *Harbourlight One, Sport, Drama* and *Life*. Advertising agencies book thousands of advert slots (**spots**) every week.

Five product categories are **restricted**: Alcohol, Betting and gaming, Less healthy food, Consumer credit, and Medicines and health. Before a campaign in one of these categories can go on air, the **clearance desk** must check two things:

1. **The copy** (the advert itself) is acceptable. For example, a health claim needs evidence, and an offer must show its terms.
2. **Every booked spot** sits where that category's **code** (rulebook) allows it. A betting advert must not air during a children's programme, for example.

Two things make this hard for the desk:

- **The queue is bigger than the desk.** A dangerous campaign, say a betting advert in front of 200,000 young viewers, can sit behind 20 harmless ones.
- **A spot's legality depends on three things held in three places:**
  - the category's **code**, a PDF document;
  - the **programme** around the break (its forecast under-18 audience index, whether it's live sport, whether it's made for children);
  - the **broadcast day**. TV days run from 06:00 to 05:59 the next morning, and one of the schedulers (TXLine) writes 1:10 am on Sunday as **"Saturday 25:10"**, while the others write "Sunday 01:10".

Read a spot against the wrong code, or on the wrong day, and you either clear an illegal spot or withdraw one the advertiser paid for.

## 2. What we built

**BreakCheck AI** is an end-to-end clearance platform that:

| Requirement | How it's met |
|---|---|
| Clean 5 dirty data extracts and prove each needed cleaning | `data_pipeline/` + `output/data_quality_report.json` |
| Train 2 tabular ML models + 1 NLP model | `modeling/`: amendment risk (classification), under-18 audience (regression), note issue (text classification) |
| Build a searchable knowledge base from 16 PDFs | `knowledge/`: Chroma vector store of 182 passages |
| Read every rule **from the code PDFs**, never hard-coded | `knowledge/code_reading.py`: regex patterns from config read the figures at run time |
| Place every spot on the broadcast day/clock and test all 5 rules | `tools/schedule_tools.py` (pure, deterministic code) |
| Multi-agent workflow where each agent is one LLM call that **decides nothing** | `agentic/`: LangGraph with 7 agents |
| Stop before anything is cleared; a **different** Clearance Controller decides; survive a restart | LangGraph `interrupt()` + SQLite **checkpointer** |
| Record every run, assessment, decision, LLM call and chat question | `persistence/`: 6 SQLAlchemy tables |
| REST service + dashboard; the dashboard holds no business rule | `api/` (FastAPI + Swagger UI) and `dashboard/` (Streamlit) |
| Sign-in with 3 roles, enforced **by the service** | `access/`: PBKDF2 passwords + per-route area guard |
| Chat answers only from documents, 1 LLM call, refuses rather than guesses, ownership line for liaisons | `chat/`: RAG assistant + campaign status answers |

**The golden rule of the design:** *models and code establish facts; LLM agents only explain and write; a human decides.* No spot is cleared and nothing is sent to an agency on the word of a model.

## 3. Who uses it: the three roles

| Role | Sample login | Can reach | Cannot reach |
|---|---|---|---|
| **Clearance Controller** | `priya` / `11111111` | Everything, and is the **only** role that can decide (but never on an assessment they opened themselves) | — |
| **Clearance Executive** | `tom` / `22222222` | Queue, campaign detail, assessments, audit trail, chat | The decision area |
| **Agency Liaison** | `lucy` / `33333333` | Chat only, and only hears about their own 4 agencies' campaigns | Everything else |

## 4. Architecture

```
                 ┌──────────────────────── OFFLINE BUILD (run once, in order) ────────────────────────┐
                 │ 1 run_database_setup.py   → database/breakcheck.db (6 tables + 14 accounts)       │
                 │ 2 run_data_pipeline.py    → processed/*.csv + output/data_quality_report.json     │
                 │ 3 run_model_training.py   → output/models/*.pkl + schemas + model_metrics.json    │
                 │ 4 run_knowledge_indexing.py → database/chroma_store (16 PDFs → 182 passages)      │
                 └─────────────────────────────────────────────────────────────────────────────────────┘

 ┌────────────────────┐   HTTP + X-Session-Token   ┌──────────────────────────────────────────────┐
 │  Streamlit         │ ─────────────────────────▶ │  FastAPI service  (api/routes.py, port 8081) │
 │  dashboard :8080   │ ◀───────────────────────── │  ├─ authorize_request (role → area guard)    │
 │  (no business      │          JSON              │  ├─ start-up: every campaign checked once    │
 │   rules, no files) │                            │  │   with the SAME tools as the workflow     │
 └────────────────────┘                            │  ├─ LangGraph workflow  (agentic/)           │
         ▲  Swagger UI at /docs also talks to it   │  │    tools → agents (Gemini) → gate → docs  │
                                                   │  ├─ Chat RAG assistant (chat/)               │
                                                   │  └─ Repository (persistence/) ─▶ SQLite DB   │
                                                   └──────────────────────────────────────────────┘
                     reads: models (.pkl), live queue CSVs, Chroma store; writes: breakcheck.db, checkpoint.db
```

**Layers (each only talks to the one below it):**

1. **Dashboard** (Streamlit) draws pages and calls the API through one function, `call_service`.
2. **API** (FastAPI) does validation, authentication and authorisation, and orchestrates.
3. **Workflow and chat** (LangGraph + Gemini) are the agents. They interpret facts.
4. **Tools** are deterministic functions that establish facts (placing spots, testing rules, exposure).
5. **Models and knowledge** are the trained ML models and the vector store.
6. **Persistence** is the repository, the only place that builds database queries.
7. **Data pipeline** cleans the raw files offline.

## 5. The end-to-end story (one campaign)

1. At **start-up**, the API loads the models and the live queue. For **all 240 campaigns** it places every spot on its broadcast day, tests the five rules of its category's code, forecasts the under-18 audience of every programme, computes each campaign's **exposure**, and ranks the queue (largest exposure first).
2. An **executive** opens a campaign on the *Campaign detail* page and clicks **Start the assessment**. This calls `POST /assessments`.
3. The **LangGraph workflow** runs: `load_campaign_context` (tools, no LLM) → `plan_assessment` → `assess_copy_risk` → `identify_issues` → `check_spot_schedule` → `compose_clearance_assessment`. That's **5 LLM calls**. The assessment is saved with status `PENDING_DECISION`.
4. The workflow reaches `request_controller_decision` and **pauses** (`interrupt`). Its state is written to `database/checkpoint.db`. You could switch the machine off here.
5. A **different Controller** opens the *Decision queue*, reads the AI recommendation, and chooses **CLEAR**, **CLEAR WITH RESTRICTIONS** or **REJECT** with a reason. This calls `POST /clearances/{id}/decision`.
6. The workflow **resumes from the checkpoint**. Code works out which spots are cleared or withdrawn, then the documents are drafted: restriction notice and/or clearance certificate (0–2 more LLM calls). Finally `record_clearance_outcome` writes everything in one transaction.
7. The **audit trail** shows the whole record: every LLM call with its latency, the decision, who made it and why, and the documents.
8. Anyone can use the **chat**. A liaison asking "Where has CLR-QSC-0042117 got to?" gets a status answer only if that campaign belongs to one of their agencies.

### The published worked example (reproduced exactly by our code)

`CLR-QSC-0042117`, Spinwell Bet's "Acca Boost" (Betting and gaming), has 10 spots. **4 comply and 6 breach**: WINDOW 3, INDEX 1, CHILDREN 0, FREQUENCY 1, SEPARATION 1. Its first spot is Saturday 18:20. Spot 10 was booked for "Sunday 05:40" on a calendar-clock system, so it is **Saturday 29:40** on the broadcast clock. Under CLEAR WITH RESTRICTIONS, spots 03, 04, 07 and 09 are cleared.

## 6. Project structure and the purpose of every file

```
Project/
├── .env / .env.example     Gemini key + model name (real .env is git-ignored)
├── .streamlit/config.toml  Theme and port 8080 for the dashboard (pre-loaded)
├── config/config.py        EVERY constant, pattern, path, message and mapping (pre-loaded, not edited)
├── requirements.txt        Python dependencies
├── data/                   Raw extracts, the live queue (data/input) and the 16 PDFs (data/docs)
├── docs/                   These documentation files
```

### Entry points (run in this order)

| File | Purpose |
|---|---|
| `run_database_setup.py` | Creates `database/`, the 6 tables and the 14 seed accounts (salted PBKDF2 hashes). Safe to re-run. |
| `run_data_pipeline.py` | Cleans the 5 raw extracts (`DATASET_JOBS`), profiles each as received and as cleaned, writes the quality report. |
| `run_model_training.py` | Trains, scores and saves the 3 models, their feature schemas and `model_metrics.json`. |
| `run_knowledge_indexing.py` | Reads the 16 PDFs, splits them into sections and passages, builds the Chroma store, prints each category's rules read back. |
| `run_api_service.py` | Checks everything needed exists (`REQUIRED_BEFORE_STARTING`), then starts FastAPI on port 8081 (Swagger at `/docs`). |
| `run_dashboard.py` | The Streamlit app: sign-in, sample-credentials drop-down, sidebar and role-based navigation (`PAGE_RENDERERS`). |

### `data_pipeline/`: cleaning

| File | Purpose |
|---|---|
| `dataset_loader.py` | The only module that reads/writes CSVs. Raw files are read as text, with null tokens turned into NaN. |
| `common_cleaning.py` | Shared repairs: dedupe, fold spellings, read numbers/lengths/audiences/times, move to the **broadcast clock**, strip markup, blank impossible values. Unknown notations raise `ValueError`. |
| `submission_cleaning.py` | Cleans campaign submissions. The clock number's length wins over the written length; contradictions are blanked. |
| `audience_cleaning.py` | Cleans programme audiences and puts every time on its broadcast day. |
| `note_cleaning.py` | Cleans clearance notes: plain text, lower-cases shouted notes, fixes timestamps, drops notes under 6 words. |
| `feature_builder.py` | Adds slot minutes, selects and types model inputs, and builds the sklearn `Pipeline` (impute → scale / one-hot → estimator). |
| `quality_profiler.py` | Measures every defect before and after cleaning and writes the JSON quality report. |

### `modeling/`: machine learning

| File | Purpose |
|---|---|
| `model_registry.py` | Saves/loads models (joblib), feature schemas and metrics (JSON). |
| `model_evaluation.py` | Scores a fitted model: ROC AUC / R² / macro-F1, train vs held-out, generalisation gap. |
| `amendment_model.py` | Logistic regression: will the copy need changes? |
| `exposure_model.py` | Gradient boosting (Poisson loss): how many under-18 viewers will a programme have? |
| `issue_model.py` | TF-IDF + logistic regression: which of 8 copy issues does a note describe? |

### `knowledge/`: documents and RAG

| File | Purpose |
|---|---|
| `pdf_extraction.py` | Extracts text from each PDF and refuses a document with a blank page. |
| `chunk_splitting.py` | Splits documents into numbered sections, then into ~700-character passages that never cut a sentence, with overlap. |
| `vector_index.py` | Builds and opens the Chroma store. The embedding model is loaded **once** via `@lru_cache(maxsize=1)`. Provides search; the casebook is filtered out of chat retrieval. |
| `code_reading.py` | Reads a category's five rules from its own code with regex, and refuses contradictions. |

### `tools/`: deterministic facts (no LLM)

| File | Purpose |
|---|---|
| `campaign_data_tools.py` | Fetches one campaign's profile, its spots placed on the broadcast clock, its schedules and its notes in order. |
| `schedule_tools.py` | Places a spot in or adjacent to programmes, tests the 5 rules, summarises, works out decision outcomes, ranks the queue. |
| `prediction_tools.py` | Amendment score + band, weighted note vote, programme audience forecasts, campaign exposure. |
| `retrieval_tools.py` | Reads the one passage of one document an agent needs (executive checks, re-booking steps, standing restrictions). |

### `agentic/`: the LangGraph workflow

| File | Purpose |
|---|---|
| `state.py` | `ClearanceAssessmentState`, the typed dictionary passed between nodes (`llm_call_count` adds up). |
| `agents.py` | Gemini client (with timeout and retries), the labelled-reply parser, and the 7 agent prompts. |
| `nodes.py` | The 10 workflow steps and the router. Loads models, live data and the vector store once at import. |
| `graph.py` | Wires the nodes, the conditional branch after the gate, and the SQLite checkpointer. |

### `persistence/`: the database

| File | Purpose |
|---|---|
| `database.py` | `database_engine` and `session_factory`, created once and shared. |
| `models.py` | The 6 tables: `user_accounts`, `agent_runs`, `clearance_assessments`, `controller_decisions`, `llm_calls`, `chat_turns`. |
| `repository.py` | Every query. Writes and flushes but never commits (the caller commits, so writes can share one transaction). |

### `access/`: security

| File | Purpose |
|---|---|
| `password_security.py` | PBKDF2-SHA256 with 200,000 rounds and a per-account salt. |
| `permissions.py` | `authorize_request` checks the session token (401), maps the route to an area, and checks role access (403). |

### `chat/`: the clearance assistant

| File | Purpose |
|---|---|
| `conversation_memory.py` | Recalls a user's last 3 exchanges (never anyone else's) and builds the search text for follow-ups. |
| `clearance_assistant.py` | RAG: retrieve → filter casebook → RRF rerank → one Gemini call → validate → refuse if not grounded. |
| `campaign_status_answer.py` | Answers "where is campaign X?" from the database, respecting the ownership line. No LLM. |

### `api/`: the REST service

| File | Purpose |
|---|---|
| `schemas.py` | Pydantic request/response shapes. Invalid input gets 422 before any work is done. |
| `routes.py` | The 11 endpoints, start-up precomputation, locks, and the failure handler (502 that also closes failed runs). |

### `dashboard/`: Streamlit pages

| File | Purpose |
|---|---|
| `api_client.py` | `call_service`, the only door to the API: token header, spinner with elapsed time, 401 → back to sign-in, errors shown with a "back" link. |
| `ui_components.py` | Shared fonts (Fraunces/Manrope), badges, time format, the workflow step guide, and plain-words explainers. |
| `clearance_queue.py` | Ranked queue, metrics, category filter, pick a campaign. |
| `campaign_detail.py` | Badges, metrics, broadcast-day chart (Altair), rules table, spots, copy and notes, assessment form. |
| `decision_queue.py` | Pending assessments in full and the decision form (refused for your own assessment). |
| `audit_trail.py` | Assessments by status, full record, LLM calls vs expected, documents, chat log. |
| `clearance_chat.py` | Your conversation, question types, memory note, liaison agencies. |

## 7. Generated folders

| Path | Made by | Contents |
|---|---|---|
| `processed/` | data pipeline | 5 cleaned CSVs |
| `output/data_quality_report.json` | data pipeline | Defect counts before and after cleaning |
| `output/models/` | training | 3 `.pkl` models + 2 feature schemas |
| `output/model_metrics.json` | training | Every score of every model |
| `database/breakcheck.db` | setup | The 6 application tables |
| `database/chroma_store/` | indexing | The vector store |
| `database/checkpoint.db` | API service | Every run's workflow state, saved after each node |
