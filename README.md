# AI Email Assistant

Enterprise-style AI assistant that **classifies** your Gmail, **drafts replies**, and **schedules meetings**
with Google Calendar - always behind a **human-in-the-loop** gate.

* **Python 3.11+, asyncio, Pydantic v2**, SQLAlchemy 2 async (SQLite), structlog
* **Gmail API + Google Calendar API** via the official Google client libraries (OAuth2, auto-refresh)
* **Groq** LLM (free tier) with forced function-calling → validated structured output
* **MCP pattern**: a typed, risk-classified tool registry, plus an optional stdio MCP server
* **Streamlit web UI**: connect accounts, process the inbox, approve/reject, browse history, customize everything
* **100 % free to run**: Groq free tier, Google APIs (free quotas), SQLite, open-source libraries

> **Safe by default.** The assistant never sends mail or invites on its own. It creates Gmail **drafts**
> and queues *approvals*; you approve them in the UI (or CLI). There is no delete capability at all.

## Quick start with the web UI

```bash
pip install -r requirements.txt
pip install -e .
python -m email_assistant ui          # or double-click run_ui.bat (Windows) / ./run_ui.sh
```
Open the **Settings → Connections** tab and: paste your Groq key, click *Fetch available models* and *Test both
models*, upload the Google `client_secret.json` (OAuth client type **Desktop app**), click *Connect Google account*.
Then use **Dashboard → Process inbox now** and review results under **Approvals**. No terminal commands needed.

| Page | What it does |
|---|---|
| Dashboard | Setup checklist, counters, *Process inbox now*, recent activity |
| Approvals | One card per email: reply preview + calendar invite. *Approve all* runs the **invite first, then the reply**; if the invite fails the reply is withheld. *Reject* (with optional reason, remembered for future drafts). *Approve anyway* if the calendar changed |
| History | Filter/search every processed email, see why it was classified that way, action items, audit trail, reprocess |
| Settings | Connections · Profile & style · Scheduling · Automation & safety · Memory · Advanced (export/import/reset) |

**What you can customize (no file editing):** name, signature, timezone, default tone, standing instructions for
replies (e.g. "never discuss pricing"), Groq models, working hours/days, meeting length, buffer, minimum notice,
look-ahead window, calendar, Meet links, which emails to process (Gmail search presets or custom), batch size,
confidence threshold, ignore list, VIP list, approval expiry, auto-send of plain replies (off by default), and
free-form memories (global or per sender). Customizations are saved to `data/user_settings.json`; secrets stay in `.env`.

---

## 1. Project layout

```
ai-email-assistant/
├── pyproject.toml · requirements.txt · requirements-dev.txt · .env.example · Makefile
├── docs/ARCHITECTURE.md              # diagrams, safety model, failure handling
├── credentials/                      # put client_secret.json here (git-ignored); token.json is created here
├── data/                             # SQLite database (git-ignored)
├── src/email_assistant/
│   ├── cli.py · config.py · container.py · exceptions.py · logging_config.py
│   ├── core/            pii.py · retry.py · rate_limit.py
│   ├── auth/            scopes.py · token_store.py · google_auth.py
│   ├── schemas/         enums · parsed_email · llm (tool schemas) · scheduling · tools
│   ├── preprocessing/   cleaner.py (HTML/quotes/signature) · mime_parser.py
│   ├── integrations/    base.py · gmail_client.py · calendar_client.py · google_errors.py
│   ├── llm/             base.py · groq_client.py · prompts.py · tooling.py
│   ├── scheduling/      slots.py (timezones, busy intervals, free-slot search)
│   ├── db/              models.py · session.py · repositories.py
│   ├── services/        fetcher · classification · scheduling · drafting · hitl · memory · inbox
│   ├── agent/           orchestrator.py (EmailAgent)
│   ├── ui/              app.py (Streamlit) · logic.py (testable helpers) · runtime.py (async bridge)
│   └── mcp_tools/       registry.py · tools.py · server.py
└── tests/               pytest suite with fake Gmail / Calendar / LLM
```

## 2. Setup (step by step, all free)

### 2.1 Install
```bash
cd ai-email-assistant
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .            # makes `python -m email_assistant` importable (or prefix commands with PYTHONPATH=src)
cp .env.example .env
```

### 2.2 Groq API key (free)
1. Sign in at <https://console.groq.com/keys> and create a key.
2. Put it in `.env` as `GROQ_API_KEY=...`.

Free-tier limits are tight and change over time (check *console.groq.com → Settings → Limits*). The defaults
use a smaller model for classification and the larger one for writing/meeting extraction
(`openai/gpt-oss-20b` / `openai/gpt-oss-120b`), a heuristic pre-filter that skips the LLM for bulk mail, and a
client-side rate limiter (`GROQ_REQUESTS_PER_MINUTE`). **Model names are controlled by Groq and do change** - if
calls start failing with `model_not_found`, open *Settings → Connections → Fetch available models* and pick current
ones. If you hit daily token limits, lower the batch size.

### 2.3 Google Cloud OAuth client (free)
1. <https://console.cloud.google.com> → create a project.
2. **APIs & Services → Library**: enable **Gmail API** and **Google Calendar API**.
3. **OAuth consent screen**: user type *External*, add yourself as a **test user**.
4. **Credentials → Create credentials → OAuth client ID → Desktop app**. Download the JSON and save it as
   `credentials/client_secret.json`.
5. Authenticate once:
   ```bash
   python -m email_assistant auth
   ```
   A browser opens; approve the 4 requested scopes (see `auth/scopes.py`). The token is stored in
   `credentials/token.json` (mode 0600).

> **Important - 7-day token expiry:** while the consent screen is in *Testing* status, Google expires refresh
> tokens after 7 days. For personal use you can click **Publish app** (it stays "unverified" with a warning
> screen, limited to 100 users, which is fine for yourself) so the token lasts. If the token dies the assistant
> stops with a clear message; just re-run `auth`.

Optional: encrypt the token at rest - generate a key
(`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`) and set `TOKEN_ENCRYPTION_KEY`.

### 2.4 Configure yourself
Edit `.env`: `USER_NAME`, `USER_TIMEZONE` (IANA name), `USER_SIGNATURE`, working hours, etc.

### 2.5 Verify
```bash
python -m email_assistant init-db
python -m email_assistant whoami        # prints your mailbox address
```

## 3. Daily use

```bash
python -m email_assistant run --once    # classify + draft once (safe: drafts only)
python -m email_assistant run           # keep polling every POLL_INTERVAL_SECONDS
python -m email_assistant pending       # what is waiting for your approval
python -m email_assistant approve --email 16   # invite + reply of one email together (invite first)
python -m email_assistant approve 7 8   # approve individual actions by approval id
python -m email_assistant reject 9 -r "too formal for this sender"   # reason is remembered for future drafts
python -m email_assistant remember "Always reply in a friendly tone to the design team"
python -m email_assistant remember "Prefers mornings" --sender jane@acme.com
```
You can also open the Gmail **Drafts** folder, edit the draft, and *then* approve - the approval sends whatever
the draft currently contains.

### Approval safety details
* **Invite before reply.** `approve --email N` (and *Approve all* in the UI) creates the invite first; if it fails
  the reply is **not** sent, so you never promise an invite that does not exist.
* **Calendar re-check at approval time.** Availability is checked again right before the invite is created. If the
  slot was taken meanwhile the action fails with `CONFLICT`; reject it or use *Approve anyway* /
  `--ignore-conflicts` to double-book deliberately.
* The reply says "a calendar invitation will follow" on purpose: Google Calendar emails the invite (with the Meet
  link) itself when the event is created.

### What happens to each email
| Category | Result |
|---|---|
| `spam_marketing` | Ignored (bulk mail is detected without spending LLM quota) |
| `information_only` | Marked complete, no reply |
| `action_required` / `priority_escalation` | Gmail draft + pending *send* approval; action items stored |
| `meeting_request` | Calendar checked (timezones, working hours, buffers). Free → draft "accepting" + pending *invite*. Busy → draft with 3 alternatives |
| low confidence / LLM outage | `needs_review`, nothing created |

## 4. MCP server (optional)
```bash
python -m email_assistant mcp-server
```
Exposes only **read / draft** tools over stdio (search, get message, availability, create draft). Sending and
calendar invites are intentionally *not* exposed. Example client config (Claude Desktop-style):
```json
{ "mcpServers": { "email-assistant": {
    "command": "/abs/path/.venv/bin/python",
    "args": ["-m", "email_assistant", "mcp-server"],
    "cwd": "/abs/path/ai-email-assistant",
    "env": { "PYTHONPATH": "src" } } } }
```
(The `.env` file in `cwd` is loaded automatically.)

## 5. Tests
```bash
pytest                                   # all tests, no network, no credentials
pytest --cov=email_assistant --cov-report=term-missing
```
Also covered: grouped approvals (invite-then-reply), approval-time conflict re-check, ignore/VIP lists, standing
instructions reaching the prompt, preferences persistence (whitelisted, no secrets), UI validation helpers.
Covered: PII masking, cleaning/MIME parsing, retry & rate limiting, slot/timezone math, Groq client
(validation-repair, 429 handling), OAuth refresh/revocation/scopes, Gmail client (threading, header injection,
retries), classification, guardrails, scheduling (conflicts, tz, prompt-injection attendee filter), tool risk
gating, approvals (double-approve, reject, expiry, failure+retry) and full end-to-end flows with fake Google/LLM.

## 6. Security & privacy notes
* No secrets in code; `.env` and `credentials/` are git-ignored; `GROQ_API_KEY` is a `SecretStr`.
* Least privilege: `gmail.readonly`, `gmail.compose`, `calendar.freebusy`, `calendar.events` - no delete, no full-mailbox scope.
* Email content leaves your machine **only** to Groq, after PII masking (emails, phones, cards, SSN, Aadhaar, PAN, IPs).
  Names and message text remain; review Groq's data policy for sensitive mailboxes.
* Logs go to stderr, PII-redacted, never contain bodies. The DB stores redacted subjects and model summaries,
  not bodies; approval rows for sends/invites hold the data needed to execute them - protect `data/` (disk encryption).
* `AUTO_APPROVE_HIGH_RISK=true` lets *plain* replies without warnings send automatically. Leave it `false`
  unless you accept that risk. Calendar invites and any reply with scheduling facts, commitments, warnings or
  escalation **always** require approval.

## 7. Known limitations
* Schema is created with `create_all()`; add Alembic for production migrations.
* Gmail labels are not applied (needs the broader `gmail.modify` scope); state lives in the DB.
* Reprocessing an email that failed *after* its draft was created can create a second draft.
* Free-tier LLM limits cap throughput (roughly tens of emails per day on the larger model).
* The UI processes on demand. For hands-free operation run `python -m email_assistant run` as a background service.
* The Google sign-in window opens on the machine running the UI; for a remote server authenticate locally and copy
  `credentials/token.json`.
* Rejecting leaves the Gmail draft in place (no delete permission is requested); delete it in Gmail.
