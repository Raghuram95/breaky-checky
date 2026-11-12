# BreakCheck AI

An advertising clearance and spot-compliance platform for Harbourlight Media. It cleans five dirty data extracts, trains three models, reads every scheduling rule from the company's own code PDFs, checks every booked spot on the broadcast clock, and runs a LangGraph multi-agent assessment that **pauses for a human Clearance Controller** before anything is cleared. Everything is served through FastAPI (Swagger UI at `/docs`) and a Streamlit dashboard, behind role-based sign-in.

## Quick start (from this folder)
```bash
pip install -r requirements.txt
# put your GEMINI_API_KEY and GEMINI_MODEL in .env (see .env.example)
python run_database_setup.py
python run_data_pipeline.py
python run_model_training.py
python run_knowledge_indexing.py
python run_api_service.py                  # terminal 1 → http://localhost:8081/docs
python -m streamlit run run_dashboard.py   # terminal 2 → http://localhost:8080
```
Sample logins: `priya`/`11111111` (Controller), `tom`/`22222222` (Executive), `lucy`/`33333333` (Agency Liaison). The sign-in page has a drop-down with these.

## Documentation
| | |
|---|---|
| [01 Project overview](docs/01_PROJECT_OVERVIEW.md) | Problem, requirements, architecture, every file |
| [02 LangGraph workflow](docs/02_LANGGRAPH_WORKFLOW.md) | Nodes, state, the human-in-the-loop gate, the checkpointer |
| [03 RAG and knowledge base](docs/03_RAG_AND_KNOWLEDGE_BASE.md) | Chunking, embeddings, rule reading, casebook filter, RRF reranking |
| [04 Machine learning models](docs/04_MACHINE_LEARNING_MODELS.md) | Algorithm choices, metrics, results |
| [05 Data analysis](docs/05_DATA_ANALYSIS.md) | Datasets, defects, cleaning, quality report |
| [06 Streamlit, sessions and data flow](docs/06_STREAMLIT_SESSIONS_AND_DATA_FLOW.md) | What happens on every click, how data is shared |
| [07 Running the project](docs/07_RUNNING_THE_PROJECT.md) | Commands, tests, demo script, troubleshooting |
| [08 Concepts and interview prep](docs/08_CONCEPTS_AND_INTERVIEW_PREP.md) | Glossary and review questions with answers |
