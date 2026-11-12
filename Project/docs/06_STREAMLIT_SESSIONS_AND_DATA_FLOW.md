# 06 · Streamlit, Sessions and Data Flow: What Happens When You Click

> Files: `run_dashboard.py`, `dashboard/*.py`, `access/permissions.py`, `api/routes.py`, `persistence/*`

## 1. The one rule that explains everything

**The dashboard stores nothing important.** Every fact it shows comes from the **REST service**, and every change it makes is a **request to the service**, which writes it to the **database** (`database/breakcheck.db`). That's why one user's work is visible to the next user: they all read the same database through the same service.

```
 Browser tab (Streamlit)          FastAPI service                 SQLite files
 ───────────────────────          ───────────────                 ────────────
 st.session_state                 checks token + role             breakcheck.db
  ├ signed_in_user  ── token ──▶  runs tools / workflow   ──────▶  user_accounts, agent_runs,
  ├ selected_campaign             returns JSON            ◀──────  clearance_assessments, ...
  └ pages_by_name                                                  checkpoint.db (paused runs)
   (per tab, temporary)           (shared, permanent)              (shared, permanent)
```

## 2. Two kinds of "session"; don't mix them up

| | **Streamlit session** (`st.session_state`) | **Service session** (the token) |
|---|---|---|
| Lives in | the browser tab's memory on the Streamlit server | the `session_token` column of `user_accounts` |
| Holds | the sign-in reply (name, role, agencies, pages, token), the chosen campaign, the page objects | one random token per account |
| Lifetime | until sign-out or the tab is closed or reloaded | until sign-out, or until that account signs in somewhere else |
| Shared? | **No**, one per tab | **Yes**, it's the database |

**One session per account.** Signing in stores a **fresh** token and **replaces** the old one. So if Priya signs in on a second browser, her first browser's next request gets **401**, and that dashboard shows *"Your session ended: this account has signed in somewhere else or signed out."*

## 3. Click by click

### Opening the dashboard
1. `streamlit run run_dashboard.py` runs `main()` **top to bottom on every interaction**. That's how Streamlit works: each click re-runs the script, and `st.session_state` carries values between runs.
2. `st.set_page_config` sets the title and wide layout; `apply_fonts()` loads Fraunces (titles) and Manrope (text).
3. **Health check:** `GET /openapi.json`. If nothing answers on port 8081, you see *"The BreakCheck AI service is not answering at http://localhost:8081. Start it with `python run_api_service.py`…"*, and the page stops.

### Signing in
1. No `signed_in_user` in session state, so only the **sign-in form** is drawn (title, subtitle, form, and the **sample credentials** drop-down: pick a role and copy the username and password from the code boxes).
2. You click **Sign in** → `call_service("POST", "/auth/signin", …)`.
3. The service looks up the username, hashes the typed password with **that account's salt** (PBKDF2, 200,000 rounds), and compares in **constant time**. An unknown username and a wrong password get the **same** 401 message, so a stranger can't learn which usernames exist.
4. On success it stores a new random token and returns `{username, display_name, role, assigned_agencies, session_token, pages, areas}`.
5. The dashboard saves that reply in `st.session_state["signed_in_user"]` and calls `st.rerun()`.

### After sign-in: navigation
- `main()` builds one `st.Page` per page **the service listed for this role** (`pages`) and passes them to `st.navigation`. A liaison only *sees* the Clearance chat.
- But hiding pages is only a convenience. **Every** API request carries `X-Session-Token`, and `authorize_request` checks it on the server:
  - no or unknown token → **401** *Sign in first*;
  - the route's area (e.g. `/queue…` → intake, `POST …/decision` → decision) isn't granted to the role → **403** *"A Agency Liaison may not use the intake area"*. This happens **before** the endpoint runs, whether or not the thing asked for exists.
- The sidebar shows your name, role and agencies, and a **Sign out** button.

### Clearance queue page
1. `GET /queue?product_category=All` returns 240 ranked rows (computed **once at service start-up**). The metrics along the top are counted from these rows.
2. Clicking a **category pill** sends `GET /queue?product_category=Alcohol`. The service filters; positions stay the same.
3. **Selecting a row** stores `st.session_state["selected_campaign"]`, and a button switches to Campaign detail.

### Campaign detail page
1. The drop-down opens on the remembered campaign.
2. `GET /queue/{ref}` returns the profile, notes, amendment and issue facts, code rules, schedules, spot results, summary and exposure.
3. `GET /clearances?status_value=…` finds out whether it was already assessed. If so, you see who started it, when, and its status instead of the form.
4. **Start the assessment** (executives and controllers) → `POST /assessments`. A spinner shows *"Five agents are assessing the campaign…"* with the elapsed time. The service runs the LangGraph workflow until the gate, **writes the assessment and run to the database**, and returns it (201). You see the recommendation.

### Decision queue page (Controllers)
1. `GET /clearances?status_value=PENDING_DECISION`.
2. If **you** opened the assessment, the form is replaced by the refusal message (and the service would refuse with 403 anyway).
3. The form offers **only the decisions open**, each with its extra LLM-call cost. **Record the decision** → `POST /clearances/{id}/decision` → the workflow **resumes from `checkpoint.db`**, drafts the documents, and writes the decision and outcome. You see the new status and the spots cleared and withdrawn.

### Audit trail page
- Status pills → `GET /clearances?status_value=…` (counts per status).
- Selecting a row → `GET /clearances/{id}/calls` (every LLM call, the count expected, total latency), plus the assessment, notice and certificate texts.
- Chat log tab → `GET /audit/chat-turns?turn_limit=200` (everyone's latest questions).

### Clearance chat page
- `GET /chat/turns?turn_limit=20` returns **your** turns, shown oldest at the top.
- Typing a question (at least 3 characters) → `POST /chat/ask` → one LLM call → the turn is stored in `chat_turns` → `st.rerun()` redraws the conversation.

### Signing out
`POST /auth/signout` clears your token in the database (it can never be used again), then the dashboard clears its session state and reruns, back to the sign-in form.

### Errors
`call_service` handles them all in one place:
- **401 while signed in**: session ended, back to sign-in with the message.
- **Any other 4xx/5xx**: the service's own message (e.g. *"Campaign CLR-… has already been assessed…"*) plus a **"Back to …"** link to your first page, then the page stops.
- **Service unreachable**: a clear "could not be reached, check that run_api_service.py is running" message.

## 4. How changes carry over between users

**Scenario:** Tom (executive) assesses a campaign, signs out; Priya (controller) signs in.

1. Tom clicks *Start the assessment*. The service writes rows to `agent_runs` (status AWAITING_DECISION), `llm_calls` (5 rows) and `clearance_assessments` (status PENDING_DECISION), and saves the paused workflow to `checkpoint.db` under the run's thread id.
2. Tom signs out. His **token** is cleared and his **tab's session state** is wiped. **Nothing he created is lost**, because it lives in the database files, not in his tab.
3. Priya signs in (a new token, a new tab session). Her Decision queue calls `GET /clearances?status_value=PENDING_DECISION`, which **reads the same database**, so Tom's assessment is there.
4. Priya decides. The service loads the **checkpoint by thread id** and resumes the workflow where it paused (even if the service was restarted in between), then writes the decision.
5. Lucy (the liaison for that agency) asks the chat *"Where has CLR-… got to?"*. The service reads the same assessment row and tells her only the status, the decision and the spot counts (never names, scores or notes).
6. Grace opens the Audit trail and sees all of it: Tom started it, Priya decided it, why, the 7 LLM calls, and the documents.

**What is per-user, and what is shared:**

| Per user (private) | Shared by everyone |
|---|---|
| Your token; your tab's session state | Assessments, decisions, runs, LLM calls (the audit trail) |
| Your chat **memory** (last 3 turns, by username) | Chat turns are visible to the desk's **audit** page |
| The pages your role offers | The ranked queue (computed once at service start) |

**The ownership line (liaisons):** a liaison asking about another agency's campaign gets exactly the same *"That campaign cannot be found"* message as for a reference that doesn't exist, so they can't even learn that it exists.

## 5. Why build it this way?
- **Security:** roles are enforced by the server, so a modified dashboard or a direct API call (e.g. from Swagger) can't bypass them.
- **One source of truth:** the dashboard has no business rules and reads no files, so the API, Swagger and the dashboard always agree.
- **Durability:** the database and checkpoint files survive restarts, while browser tabs are temporary.
- **Auditability:** every consequential action is a row with who, what, when and why.
