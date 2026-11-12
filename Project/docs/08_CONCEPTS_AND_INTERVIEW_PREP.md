# 08 · Concepts Used, and Review-Interview Preparation

## Part A · Every concept, in plain words

### Domain
| Term | Meaning |
|---|---|
| **Clearance** | The broadcaster's check, before airing, that the copy is acceptable and every spot sits where its code allows |
| **Copy / spot / break** | The advert / one transmission of it / the pause in the schedule it airs in |
| **Centre break vs end break** | Inside a programme (the spot is **in** it) vs between two programmes (**adjacent** to both) |
| **Broadcast day / clock** | 06:00 → 05:59 next morning; after midnight written 24:00–29:59 |
| **The five rules** | WINDOW (restricted hours, maybe a live sport exception), INDEX (under-18 audience index threshold), CHILDREN (made-for-children programmes), FREQUENCY (daily cap per channel), SEPARATION (minimum minutes between spots) |
| **Exposure** | Forecast under-18 viewers the breaching spots would reach. It ranks the queue. |
| **Decisions open** | CLEAR (no breach) · CLEAR WITH RESTRICTIONS (some breach, some comply) · REJECT (always) |

### Machine learning
| Concept | Meaning |
|---|---|
| **Train / held-out split** | Fit on one set, score on rows the model never saw. That's how we measure generalisation. |
| **Overfitting / generalisation gap** | Train score − held-out score; big means memorising |
| **Data leakage** | Letting information the model wouldn't have at prediction time into training, e.g. `audience_all`, identifiers, or fitting the scaler on all data. We prevent it with a Pipeline and config-declared inputs. |
| **sklearn Pipeline / ColumnTransformer** | Chains preparation and model into one object, fitted on training data only and saved together |
| **Imputation** | Filling gaps: the median for numbers, "Not recorded" for categories; `add_indicator` flags filled values |
| **One-hot encoding** | Turns a category into 0/1 columns; `handle_unknown="ignore"` survives new values |
| **Standard scaling** | Centres numbers so logistic regression's coefficients are comparable |
| **Logistic regression** | A linear model that outputs a probability through the sigmoid; interpretable and calibrated |
| **Gradient boosting (HistGradientBoosting)** | Many small decision trees, each fixing the last one's errors; captures non-linearity and interactions |
| **Poisson loss** | For non-negative, skewed count-like targets; predicts through `exp`, so always positive |
| **TF-IDF** | Word weight = frequency in this text × rarity across all texts |
| **n-grams** | Single words (1) and word pairs (2) as features |
| **Class weights (balanced)** | Up-weight rare classes so they matter as much as common ones |
| **ROC AUC** | Probability a random positive ranks above a random negative; 0.5 = guessing, 1 = perfect |
| **R²** | Share of variance explained; 1 = perfect, 0 = no better than the mean |
| **MAE** | Average absolute miss, in the target's units (thousands of viewers) |
| **Precision / recall / F1** | Correct among predicted positives / found among actual positives / their harmonic mean |
| **Macro vs weighted F1** | Plain average over classes vs average weighted by class size |
| **Calibration** | Predicted probabilities match observed rates (mean 0.2907 vs actual 0.2917) |
| **Stratified split** | The split keeps each class's share the same in train and test |

### RAG and LLMs
| Concept | Meaning |
|---|---|
| **Embedding** | A vector representing meaning; similar texts have nearby vectors |
| **Vector store (Chroma)** | A database that finds the nearest vectors quickly (HNSW index) |
| **Cosine distance** | 1 − cosine similarity; 0 = same direction (meaning) |
| **Chunking with overlap** | Splitting documents into passages that share a little text at the boundaries |
| **RAG** | Retrieve relevant passages, then generate an answer from them |
| **Reranking / RRF** | Re-ordering retrieved results; Reciprocal Rank Fusion merges several rankings by Σ 1/(k+rank) |
| **Grounding / refusal threshold** | Answer only when the evidence is close enough (distance ≤ 0.65); otherwise refuse |
| **Prompt injection** | Text inside data trying to give the model orders; our prompts say documents and notes are material, not instructions |
| **Temperature** | Randomness of generation; 0.2 is fairly deterministic |
| **Structured (labelled) output** | Asking for `LABEL: value` fields and validating them, so model output is treated as untrusted input |
| **LRU cache** | `functools.lru_cache` remembers a function's result; `maxsize=1` keeps one entry, i.e. one loaded embedding model per process |

### Agents and workflow
| Concept | Meaning |
|---|---|
| **LangGraph StateGraph** | Nodes (functions) + edges + a shared typed state |
| **Reducer** | How a state key combines updates; `operator.add` sums `llm_call_count` |
| **Conditional edge / router** | Picks the next node from the state (the decision) |
| **`interrupt()` / `Command(resume=…)`** | Pause for outside input / continue with it |
| **Checkpointer (SqliteSaver)** | Saves the state after every node under a thread id, so a run can resume after a restart |
| **Human in the loop** | A person must approve before any consequential action |
| **Single-responsibility agents** | Each agent does one job in one call and decides nothing |

### Web, security and data
| Concept | Meaning |
|---|---|
| **REST / FastAPI / Pydantic** | HTTP endpoints / the Python framework / request-response validation (422 on bad input) |
| **Swagger UI (OpenAPI)** | Auto-generated interactive API docs at `/docs` |
| **Dependency injection (`Depends`)** | FastAPI runs `authorize_request` before guarded endpoints |
| **401 vs 403 vs 404 vs 409 vs 502** | Not signed in / not allowed / doesn't exist / conflicts with the current state / an upstream (model) failure |
| **PBKDF2 + salt** | A slow, salted password hash; the same password gives different hashes per account |
| **Constant-time comparison** | `hmac.compare_digest`, which doesn't leak through timing |
| **Session token** | A random bearer token stored per account; signing in again replaces it |
| **SQLAlchemy ORM / repository pattern** | Python classes for tables / all queries in one module |
| **Transaction** | Several writes that succeed or fail together (`session.commit()` by the caller) |
| **Threading lock** | Stops two simultaneous requests doing the same thing twice |
| **Streamlit rerun model / `st.session_state`** | The script re-runs on every interaction; session state persists per tab |

---

## Part B · Likely review questions, with answers

**Q1. Why not let the LLM decide which spots may air?**
Because it's a consequential, rule-bound decision. Code reads the figures from the code PDFs with exact regexes and tests each spot deterministically, so the result is repeatable and auditable. LLMs explain and write; a Controller decides. The spec says "decides no spot of its own."

**Q2. How do you guarantee no rule is hard-coded?**
`read_category_code` reads every figure at run time from the category's own code passages. The tests build a store with *invented* figures and a decoy document, and our code reads the invented figures correctly and ignores the decoy.

**Q3. What happens if a code contradicts itself?**
It's refused with a `ValueError` naming the document: two different values, both or neither of hours/any-time, an exception without hours, start = end, zero or two children's rules, or a cap of 0.

**Q4. Explain the broadcast clock with an example.**
Airtime Planner writes "Monday 01:10". That's before 06:00 on a calendar-clock system, so it belongs to Sunday's broadcast day, 24 hours later: Sunday 25:10. TXLine already writes 25:10 on Sunday, so nothing changes.

**Q5. Why logistic regression for amendments if boosting is "more powerful"?**
On our held-out data it scored *higher* AUC (0.857 vs 0.846), barely overfits (gap 0.007 vs 0.056), is calibrated (which matters for the fixed bands), and is explainable.

**Q6. Why Poisson loss?**
The target is non-negative and right-skewed. Poisson predicts through `exp`, so forecasts are always positive (the squared-error model predicted −4.18), and R² was also higher (0.921).

**Q7. Why is `audience_all` excluded?**
It's measured after the programme airs, so it can't be known for next week. Using it would be target leakage.

**Q8. Why macro F1 for the issue model?**
The classes are imbalanced (4% to 23%). Macro F1 weighs each issue equally, so a model can't hide poor performance on rare issues.

**Q9. How does the weighted note vote work, and why?**
Each readable note votes for its most probable issue; a legal review counts twice because legal advisers are the most authoritative source. A tie goes to the highest summed probability (never note order). Agreement = winner's share of the votes.

**Q10. How does the workflow survive a restart?**
The SQLite checkpointer saves the state after every node under the run's thread id. The thread id is stored in `agent_runs`, so the service rebuilds the graph on the same file and resumes with `Command(resume=decision)`. No agent re-runs; the tests verify the call list is unchanged.

**Q11. How do you stop the person who opened an assessment from deciding it?**
The service compares `executive_id` (the opener) with the signed-in username and answers 403 before resuming. The dashboard also hides the form, but the server is the real guard.

**Q12. Why enforce roles in the API and not the dashboard?**
Anyone can call the API directly (Swagger, curl). Server-side checks in `authorize_request` (401/403 by route area) can't be bypassed.

**Q13. Explain RAG in your chat and the reranking.**
Two searches (question; previous question + question) → drop casebook passages → fuse with Reciprocal Rank Fusion → give the model the five codes whole plus these passages → one call → validate the labelled reply → refuse if the nearest passage is farther than 0.65 or the type is out of scope.

**Q14. Why exclude the casebook, and how?**
It describes real past campaigns and decisions, so letting it into answers would leak other agencies' information and bias answers. We over-fetch by the number of casebook passages, filter them out after retrieval, and keep the top 4. So it never reaches the LLM, yet stays indexed.

**Q15. Why the LRU cache on the embedding model?**
Loading the model takes seconds and memory. `@lru_cache(maxsize=1)` loads it once per process and reuses it for indexing, opening the store, the workflow and the chat.

**Q16. How do you count and audit LLM calls?**
Every agent call is logged in `llm_calls` (agent, model, prompt and response characters, latency). `llm_call_count` sums through the reducer. `GET /clearances/{id}/calls` compares calls with the expected 5 / 6 / 7 / 5.

**Q17. What if the LLM returns garbage?**
The parser and validators refuse it (`ValueError`), the API answers 502 with a plain message, and the run is marked FAILED. Nothing invalid is stored. Transient network errors are retried with exponential backoff; unusable replies aren't, so the audited counts stay exact.

**Q18. How do liaisons get only their own campaigns?**
`answer_campaign_status` checks the campaign's agency against the liaison's assigned agencies. Another agency's campaign gives the identical "cannot be found" message as a non-existent one, and no LLM is involved.

**Q19. How is memory kept private and safe?**
It's recalled by username only (last 3 turns), and campaign status answers are replaced by a placeholder, so record details never reach a prompt.

**Q20. How did you validate the data cleaning?**
The quality report counts every defect before and after cleaning and accounts for every removed row (duplicates + no answer + impossible answer / too short). The tests re-run cleaning on crafted rows (every clock notation, unit, contradiction) and check refusals.

**Q21. What's stored where?**
`breakcheck.db` holds accounts, runs, assessments, decisions, LLM calls and chat. `checkpoint.db` holds workflow states. `chroma_store` holds passages and vectors. `output/` holds models and metrics. `processed/` holds the cleaned data.

**Q22. Which best practices did you apply?**
Config-driven (no secrets or hard-coded values), `.env` + `.env.example`, input validation (Pydantic), output validation (labelled-reply parsing), narrow exceptions, timeouts and retries with backoff, layered architecture, a repository pattern, type hints, PEP 8 (Black, isort, pyflakes, pycodestyle clean), human oversight, audit logging, prompt-injection guards, and least-privilege roles.

**Q23. Limitations and what you'd do next?**
SQLite suits a single server; production would use Postgres. A queue (e.g. Celery) could run long assessments in the background. We'd add rate limits and token budgets per user, a fallback model, an evaluation set for chat answers (groundedness), and remove the sample-credentials helper from the login page.
