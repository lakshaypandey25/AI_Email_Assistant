"""AI Email Assistant - Streamlit UI.

Start with:   python -m email_assistant ui        (or run_ui.bat / run_ui.sh)

Pages: Dashboard (process inbox) · Approvals (human-in-the-loop) · History · Settings.
Secrets (Groq key) are stored in ``.env``; every other customization in ``data/user_settings.json``.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import time
import zoneinfo
from datetime import datetime, timedelta
from pathlib import Path


# --- make relative paths (.env, data/, credentials/) work no matter where Streamlit was launched ----------
_ROOT = Path(os.environ.get("EMAIL_ASSISTANT_HOME") or Path(__file__).resolve().parents[3])
if _ROOT.is_dir():
    os.chdir(_ROOT)
    if (_ROOT / "src").is_dir() and str(_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from email_assistant.config import (  # noqa: E402
    EDITABLE_FIELDS,
    PREFERENCES_FILE,
    Settings,
    get_settings,
    load_preferences,
    reload_settings,
    save_preferences,
    set_env_value,
)
from email_assistant.scheduling.slots import DAY_CODES, Availability, Interval, parse_time_off, parse_windows  # noqa: E402
from email_assistant.exceptions import AssistantError, AuthenticationRequiredError, ConfigurationError  # noqa: E402
from email_assistant.services.memory import GLOBAL_SCOPE, sender_scope  # noqa: E402
from email_assistant.ui.logic import (  # noqa: E402
    CUSTOM_QUERY,
    DAYS,
    GMAIL_DRAFTS_URL,
    GMAIL_PRESETS,
    csv_to_lines,
    filter_chat_models,
    lines_to_csv,
    signature_for_editing,
    signature_for_storage,
    status_label,
    validate_client_secret,
    validate_prefs,
    event_label,
    CATEGORY_LABELS,
    REPLY_CATEGORY_OPTIONS,
    MODE_LABELS,
    REASON_LABELS, 
    num, 
    pct,
    reason_text,
)
from email_assistant.evaluation.store import load_runs, save_run  # noqa: E402
from email_assistant.ui.runtime import (  # noqa: E402
    connect_google,
    disconnect_google,
    google_mailbox,
    list_groq_models,
    run_app,
    test_groq_model,
    run_benchmark_sync,
)

PAGES = ["🏠 Dashboard", "✅ Approvals", "📜 History", "📊 Evaluation", "⚙️ Settings"]


# ------------------------------------------------------------------------------------ small helpers
def actor() -> str:
    return f"ui:{getpass.getuser()}"


def flash(kind: str, message: str) -> None:
    """Queue a message that survives the next ``st.rerun()``."""
    st.session_state.setdefault("_flash", []).append((kind, message))


def show_flash() -> None:
    for kind, message in st.session_state.pop("_flash", []):
        getattr(st, kind)(message)


def explain(exc: Exception) -> None:
    """Friendly error display."""
    if isinstance(exc, AuthenticationRequiredError):
        st.error(f"Google sign-in needed: {exc}  \nGo to **Settings → Connections** and click *Connect Google account*.")
    elif isinstance(exc, ConfigurationError):
        st.error(f"Configuration problem: {exc}")
    elif isinstance(exc, AssistantError):
        st.error(f"{type(exc).__name__}: {exc}")
    else:
        st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            st.exception(exc)

def local_time(value: datetime | None, settings: Settings, fmt: str = "%d %b %H:%M") -> str:
    """Timestamps are stored in UTC; show them in the user's timezone."""
    return value.astimezone(settings.tzinfo).strftime(fmt) if value else ""


def sync_gmail(settings: Settings, force: bool = False) -> None:
    """Pick up what the user did directly in Gmail (sent a draft / deleted it) so History stays truthful."""
    if not force and time.time() - st.session_state.get("_last_sync", 0.0) < 30:
        return
    st.session_state["_last_sync"] = time.time()
    try:
        result = run_app(settings, lambda app: app.hitl.reconcile_drafts())
    except Exception:  # noqa: BLE001 - Gmail not connected / offline: nothing to sync
        return
    if result["sent"] or result["deleted"]:
        flash("info", f"Synced with Gmail: {result['sent']} reply(ies) you sent directly, {result['deleted']} draft(s) you deleted.")

def save_prefs(updates: dict) -> bool:
    """Validate, persist and reload customizations. Returns True on success."""
    merged = {**load_preferences(), **updates}
    error = validate_prefs(merged)
    if error:
        st.error(f"Not saved - {error}")
        return False
    save_preferences(merged)
    reload_settings()
    flash("success", "Settings saved.")
    return True


def setup_checklist(settings: Settings) -> list[tuple[bool, str]]:
    return [
        (bool(settings.groq_api_key.get_secret_value()), "Groq API key saved"),
        (settings.google_client_secrets_file.is_file(), "Google OAuth client file uploaded"),
        (settings.google_token_file.is_file(), "Google account connected"),
        (settings.user_name != "Assistant User", "Your name and signature set"),
    ]


# ------------------------------------------------------------------------------------ Dashboard
def page_dashboard(settings: Settings) -> None:
    st.header("Dashboard")
    checklist = setup_checklist(settings)
    ready = all(ok for ok, _ in checklist[:3])
    if not all(ok for ok, _ in checklist):
        with st.container(border=True):
            st.subheader("Finish setting up")
            for ok, label in checklist:
                st.markdown(f"{'✅' if ok else '⬜'} {label}")
            st.caption("Open **Settings → Connections** to complete the missing steps.")

    async def _load(app):  # noqa: ANN001
        return await app.inbox.status_counts(), await app.hitl.list_groups()

    try:
        counts, groups = run_app(settings, _load)
    except Exception as exc:  # noqa: BLE001
        explain(exc)
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Waiting for your approval", len(groups))
    c2.metric("Completed", counts.get("completed", 0))
    c3.metric("Ignored (spam / ignore list)", counts.get("ignored", 0))
    c4.metric("Needs review / failed", counts.get("needs_review", 0) + counts.get("failed", 0))
    c5.metric("Deferred (auto-retry)", counts.get("deferred", 0), help="Parked because the AI or Google was busy or rate-limited. Retried automatically on the next run.")
    if counts.get("deferred", 0):
        st.info("Some emails were parked because the AI service was busy. They are retried automatically on the next run - nothing is lost.")
    if groups:
        st.info(f"{len(groups)} email(s) have replies or invites waiting - open **Approvals**.")

    st.subheader("Process your inbox")
    st.caption(
        "Creates reply **drafts** in Gmail and queues approvals. Nothing is sent or invited until you approve it."
    )
    with st.form("process"):
        limit = st.slider("Maximum emails to process", 1, 50, min(settings.gmail_max_per_run, 50))
        go = st.form_submit_button("▶ Process inbox now", type="primary", disabled=not ready)
    if not ready:
        st.caption("Complete the setup checklist above to enable processing.")
    if go:
        with st.status("Reading and analysing your inbox…", expanded=True) as status:
            try:
                summary = run_app(settings, lambda app: app.agent.run_once(limit), need_llm=True)
                status.update(label="Finished", state="complete")
                st.session_state["last_run"] = (datetime.now(), dict(summary.counts))
            except Exception as exc:  # noqa: BLE001
                status.update(label="Failed", state="error")
                explain(exc)
    if "last_run" in st.session_state:
        when, result = st.session_state["last_run"]
        st.success(f"Last run {when:%H:%M:%S}: " + (", ".join(f"{v} {k.replace('_', ' ')}" for k, v in result.items()) or "no new emails"))

    st.subheader("Recent activity")
    rows = run_app(settings, lambda app: app.inbox.recent(limit=10))
    if rows:
        st.dataframe(
            [
                {"From": r.sender, "Subject": r.subject, "Status": status_label(r.status), "Reply": r.reply, "Updated": local_time(r.updated_at, settings), "Summary": r.summary or ""}
                for r in rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Nothing processed yet.")
    st.caption("For hands-free operation run `python -m email_assistant run` in a terminal or as a service - it polls continuously.")


# ------------------------------------------------------------------------------------ Approvals
def report_group(result) -> None:  # noqa: ANN001
    for r in result.results:
        if r.executed:
            flash("success", f"Action {r.approval_id} done.")
        else:
            flash("error", f"Action {r.approval_id} not completed: {r.error}")
    for skipped in result.skipped:
        flash("warning", f"Reply {skipped} was NOT sent because its calendar invite failed.")


def page_approvals(settings: Settings) -> None:
    st.header("Approvals")
    sync_gmail(settings)
    if st.button("🔄 Sync with Gmail", help="Checks whether you already sent or deleted a draft directly in Gmail."):
        sync_gmail(settings, force=True)
        st.rerun()
    st.caption("Review what the assistant prepared. Replies are saved as Gmail drafts - edit them there first if you like; approving sends the draft as it is at that moment.")
    try:
        groups = run_app(settings, lambda app: app.hitl.list_groups())
    except Exception as exc:  # noqa: BLE001
        explain(exc)
        return
    if not groups:
        st.success("Nothing is waiting for approval. 🎉")
        return

    for g in groups:
        with st.container(border=True):
            st.markdown(f"**{g.subject or '(no subject)'}**  \nFrom `{g.sender}` · {g.category or 'unclassified'} · priority {g.priority or '-'}/5")
            if g.summary:
                st.caption(g.summary)
            if g.oldest:
                waiting_h = (datetime.now(g.oldest.tzinfo) - g.oldest).total_seconds() / 3600
                if waiting_h > settings.sla_hours:
                    st.markdown(f":red[⏰ Overdue - waiting {waiting_h:.1f} h (target {settings.sla_hours} h)]")
                else:
                    st.caption(f"⏱ Waiting {waiting_h:.1f} h (target {settings.sla_hours} h)")
            conflict = False
            for item in g.items:
                if item.action_type == "send_email":
                    st.markdown("**✉️ Reply**")
                    lines = item.preview.splitlines()
                    for note in (line[6:] for line in lines if line.startswith("NOTE: ")):
                        st.warning(note)
                    st.text("\n".join(line for line in lines if not line.startswith("NOTE: ")))
                elif item.action_type == "create_calendar_event":
                    st.markdown("**📅 Calendar invite**")
                    st.info(item.preview)
                else:
                    st.text(item.preview)
                if item.error:
                    st.error(item.error)
                    conflict = conflict or item.error.startswith("CONFLICT")
            for warning in g.warnings:
                st.warning(warning)

            eid = g.email_id
            label = "✅ Approve all" if len(g.items) > 1 else "✅ Approve"
            b1, b2, b3 = st.columns([1, 1, 2])
            if b1.button(label, key=f"ok-{eid}", type="primary"):
                with st.spinner("Executing…"):
                    result = run_app(settings, lambda app, e=eid: app.hitl.approve_email(e, actor()))
                report_group(result)
                st.rerun()
            if conflict and b2.button("⚠️ Approve anyway", key=f"force-{eid}", help="Create the invite even though the slot is busy"):
                with st.spinner("Executing…"):
                    result = run_app(settings, lambda app, e=eid: app.hitl.approve_email(e, actor(), ignore_conflicts=True))
                report_group(result)
                st.rerun()
            b3.link_button("Open Gmail Drafts", GMAIL_DRAFTS_URL)
            with st.expander("Reject…"):
                reason = st.text_input("Why? (optional - remembered to improve future drafts)", key=f"why-{eid}")
                if st.button("Reject", key=f"rej-{eid}"):
                    run_app(settings, lambda app, e=eid, r=reason: app.hitl.reject_email(e, actor(), r or None))
                    if reason.strip():
                        run_app(settings, lambda app, r=reason: app.memory.add(GLOBAL_SCOPE, "feedback", f"Rejected a proposed action: {r.strip()}"))
                    flash("info", "Rejected. The Gmail draft (if any) is still in your Drafts folder - delete it there if you don't need it.")
                    st.rerun()


# ------------------------------------------------------------------------------------ History
def page_history(settings: Settings) -> None:
    st.header("History")
    sync_gmail(settings)
    f1, f2, f3 = st.columns([2, 2, 3])
    statuses = f1.multiselect("Status", ["awaiting_approval", "needs_review", "completed", "ignored", "failed", "deferred"], format_func=status_label)
    categories = f2.multiselect("Category", ["action_required", "meeting_request", "information_only", "spam_marketing", "priority_escalation"])
    search = f3.text_input("Search sender / subject / summary")
    rows = run_app(settings, lambda app: app.inbox.recent(300, statuses or None, categories or None, search))
    if not rows:
        st.info("No emails match.")
        return
    st.dataframe(
        [
            {
                "ID": r.id, "From": r.sender, "Subject": r.subject, "Status": status_label(r.status),
                "Reply": r.reply, "Invite": r.invite, "Category": r.category or "-", "Why": reason_text(r.reason),
                "Received": local_time(r.received_at, settings), "Processed": local_time(r.processed_at, settings),
                "Last update": local_time(r.updated_at, settings, "%d %b %H:%M:%S"),
                "Confidence": round(r.confidence, 2) if r.confidence is not None else None,
                "Priority": r.priority, "Summary": r.summary or "",
            }
            for r in rows
        ],
        use_container_width=True,
        hide_index=True,
    )
    chosen = st.selectbox("Inspect an email", [r.id for r in rows], format_func=lambda i: next(f"#{r.id} · {r.sender} · {r.subject[:60]}" for r in rows if r.id == i))
    row = next(r for r in rows if r.id == chosen)
    details = run_app(settings, lambda app: app.inbox.details(chosen))
    with st.container(border=True):
        st.markdown(f"**{row.subject}** - {status_label(row.status)}")
        if row.error:
            st.error(row.error)
        if row.category:
            verdict = run_app(settings, lambda app: app.inbox.feedback_for(chosen))
            st.markdown(f"**Was the category right?** (AI said *{CATEGORY_LABELS.get(row.category, row.category)}*)" + (f" - your verdict: *{CATEGORY_LABELS.get(verdict, verdict)}*" if verdict else ""))
            f1, f2, f3 = st.columns([1, 3, 1])
            if f1.button("👍 Correct", key=f"fy-{chosen}"):
                run_app(settings, lambda app: app.inbox.record_feedback(chosen, True))
                flash("success", "Thanks - recorded. This feeds the accuracy metrics on the Evaluation page.")
                st.rerun()
            right = f2.selectbox("Or the right category is", list(CATEGORY_LABELS), format_func=CATEGORY_LABELS.get, key=f"fc-{chosen}", label_visibility="collapsed")
            if f3.button("👎 Wrong", key=f"fn-{chosen}"):
                run_app(settings, lambda app: app.inbox.record_feedback(chosen, False, right))
                flash("success", "Thanks - recorded.")
                st.rerun()
        if details:
            st.markdown(f"**Why this category:** {details['reasoning'] or '-'}")
            if details["action_items"]:
                st.markdown("**Action items**")
                for item in details["action_items"]:
                    st.markdown(f"- ({item.get('owner', 'me')}) {item.get('description')}" + (f" - due {item['due']}" if item.get("due") else ""))
            with st.expander("Audit trail"):
                st.dataframe(
                    [{"Time": local_time(a["time"], settings, "%d %b %H:%M:%S"), "Who": a["actor"], "What happened": event_label(a["event"]), "Detail": json.dumps(a["detail"])} for a in details["audit"]],
                    use_container_width=True,
                    hide_index=True,
                )
        if row.status in {"needs_review", "failed", "ignored"} and st.button("🔁 Process this email again", key=f"re-{chosen}"):
            run_app(settings, lambda app: app.inbox.reprocess(chosen))
            flash("info", "Queued. It is picked up on the next run if it still matches your Gmail search (e.g. still unread). A second draft may be created if one already exists.")
            st.rerun()

# ------------------------------------------------------------------------------------ Evaluation
def _bar(data: dict, label_map: dict | None = None, title: str = "") -> None:
    if not data:
        st.caption("No data yet.")
        return
    frame = pd.DataFrame({"count": {(label_map or {}).get(k, k): v for k, v in data.items()}})
    if title:
        st.markdown(f"**{title}**")
    st.bar_chart(frame)


def _confusion_table(confusion: dict) -> None:
    frame = pd.DataFrame(confusion).T.rename(index=CATEGORY_LABELS, columns=CATEGORY_LABELS)
    frame.index.name = "actual ↓ / predicted →"
    st.dataframe(frame, use_container_width=True)


def _per_class_table(stats: dict) -> None:
    st.dataframe(
        [{"Category": CATEGORY_LABELS.get(k, k), "Precision": pct(v["precision"]), "Recall": pct(v["recall"]), "F1": pct(v["f1"]), "Examples": v["support"]} for k, v in stats.items()],
        use_container_width=True, hide_index=True,
    )


def eval_live(snap: dict) -> None:
    ops = snap["operations"]
    st.caption("How the assistant is handling your real mail in the selected period.")
    a, b, c, d, e = st.columns(5)
    a.metric("Emails processed", ops["total"])
    b.metric("Handled without you", pct(ops["auto_handled_rate"]), help="Completed or ignored with no human action needed (spam, FYI, no reply needed).")
    c.metric("Reply rate", pct(ops["reply_rate"]), help="Share of analysed emails that got a drafted reply. Low is fine: most mail needs no response.")
    d.metric("Sent to review", pct(ops["needs_review_rate"]), help="Low AI confidence, AI unavailable, or topics that need your judgement.")
    e.metric("Avg AI confidence", num(ops["avg_confidence"], 2))
    g1, g2 = st.columns(2)
    with g1:
        _bar(ops["by_category"], CATEGORY_LABELS, "Emails by category")
        _bar(ops["by_status"], None, "Emails by outcome")
    with g2:
        _bar(ops["response_modes"], MODE_LABELS, "What the assistant decided to do")
        if ops["no_reply_reasons"]:
            st.markdown("**Why no reply was sent**")
            st.dataframe([{"Reason": REASON_LABELS.get(k, k), "Emails": v} for k, v in ops["no_reply_reasons"].items()], hide_index=True, use_container_width=True)
        if ops["classifier_sources"]:
            st.markdown("**Classified by**")
            st.dataframe([{"Method": {"llm": "AI model", "heuristic": "Bulk-mail rule", "rules": "Keyword fallback"}.get(k, k), "Emails": v} for k, v in ops["classifier_sources"].items()], hide_index=True, use_container_width=True)


def eval_human(snap: dict) -> None:
    h = snap["human"]
    st.caption("How much you rely on the assistant, and how quickly you act on it.")
    a, b, c, d = st.columns(4)
    a.metric("Replies sent", h["emails_sent"])
    b.metric("Invites created", h["invites_created"])
    c.metric("Draft acceptance", pct(h["draft_acceptance_rate"]), help="Approved replies / (approved + rejected). A proxy for draft quality.")
    d.metric("Median time to decide", num(h["median_time_to_decision_min"], 1, " min"))
    e, f, g = st.columns(3)
    e.metric("Waiting now", h["pending"])
    f.metric(f"Overdue (> {h['sla_hours']} h)", h["overdue_vs_sla"])
    g.metric("Double-bookings prevented", h["conflicts_blocked"], help="Invites blocked because the calendar changed between drafting and approval.")
    t1, t2 = st.columns(2)
    t1.markdown("**Reply approvals**")
    t1.dataframe([{"Status": k, "Count": v} for k, v in h["send_status"].items()] or [{"Status": "-", "Count": 0}], hide_index=True, use_container_width=True)
    t2.markdown("**Invite approvals**")
    t2.dataframe([{"Status": k, "Count": v} for k, v in h["invite_status"].items()] or [{"Status": "-", "Count": 0}], hide_index=True, use_container_width=True)


def eval_reliability(snap: dict) -> None:
    r = snap["reliability"]
    st.caption("Resilience: every external service has a fallback. This shows how often they were needed.")
    a, b, c, d = st.columns(4)
    a.metric("AI calls", r["llm_calls"])
    b.metric("API success", pct(r["api_success_rate"]))
    c.metric("Valid output", pct(r["valid_output_rate"]), help="Answers that passed strict schema validation.")
    d.metric("Rate-limit hits", r["rate_limit_hits"])
    e, f, g, h = st.columns(4)
    e.metric("Served by a fallback model", r["fallback_model_calls"])
    f.metric("Tokens used", f"{r['total_tokens']:,}")
    g.metric("Latency p50", num(r["latency_p50_ms"], 0, " ms"))
    h.metric("Latency p95", num(r["latency_p95_ms"], 0, " ms"))
    st.markdown("**Fallbacks used**")
    st.dataframe(
        [
            {"Situation": "Email parked and retried later (AI/Google busy)", "Times": r["emails_deferred"]},
            {"Situation": "Keyword rules used instead of the AI", "Times": r["rules_fallbacks"]},
            {"Situation": "Template reply used instead of the AI", "Times": r["template_replies"]},
            {"Situation": "Calendar unavailable - holding reply prepared", "Times": r["calendar_fallbacks"]},
            {"Situation": "Draft blocked by safety guardrails", "Times": r["drafts_blocked_by_guardrails"]},
        ],
        hide_index=True, use_container_width=True,
    )
    if r["per_model"]:
        st.markdown("**Per model**")
        st.dataframe(
            [{"Model": k, "Calls": v["calls"], "Success": pct(v["success_rate"]), "Tokens": f"{v['tokens']:,}", "Avg latency": num(v["avg_latency_ms"], 0, " ms")} for k, v in r["per_model"].items()],
            hide_index=True, use_container_width=True,
        )


def eval_quality(snap: dict) -> None:
    q = snap["quality"]
    st.caption("Accuracy measured from YOUR verdicts: open an email in **History** and click 👍 Correct or 👎 Wrong.")
    a, b, c = st.columns(3)
    a.metric("Emails you reviewed", q["labelled"])
    b.metric("Classification accuracy", pct(q["accuracy"]))
    c.metric("Macro F1", pct(q["macro_f1"]), help="Average F1 across categories - penalises ignoring rare categories.")
    if not q["labelled"]:
        st.info("No verdicts yet. Review a few emails in History to unlock this section.")
        return
    _per_class_table(q["per_class"])
    st.markdown("**Confusion matrix** (rows = what it really was, columns = what the AI said)")
    _confusion_table(q["confusion"])


def eval_benchmark(settings: Settings) -> None:
    st.caption(
        "Runs labelled test emails (spam, FYI, requests, meetings with tricky dates, an urgent outage, a legal notice and a prompt-injection attack) "
        "through the real AI and routing rules. Re-run after changing models or settings to see if quality moved. Uses AI quota: quick ≈ 12 calls, full ≈ 25."
    )
    c1, c2 = st.columns(2)
    quick = c1.button("▶ Quick benchmark (8 emails)")
    full = c2.button("▶ Full benchmark (18 emails)")
    if quick or full:
        bar = st.progress(0.0, text="Starting…")
        try:
            result = run_benchmark_sync(settings, quick=quick, progress=lambda f, label: bar.progress(min(f, 1.0), text=label))
            save_run(result)
            bar.empty()
            flash("success", f"Benchmark finished ({result['cases']} emails, {result['errors']} could not run).")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            explain(exc)
    runs = load_runs()
    if not runs:
        st.info("No benchmark run yet.")
        return
    run = runs[-1]
    st.markdown(f"**Latest run:** {run['timestamp'][:16].replace('T', ' ')} UTC · models `{run['models']['fast']}` / `{run['models']['quality']}` · {run['cases']} emails")
    if run["errors"]:
        st.warning(f"{run['errors']} email(s) could not be evaluated ({', '.join(run['error_types'])}) - usually AI rate limits. Re-run later; these are excluded from the scores.")
    a, b, c, d = st.columns(4)
    a.metric("Classification accuracy", pct(run["classification_accuracy"]))
    b.metric("Macro F1", pct(run["macro_f1"]))
    c.metric("Right response decision", pct(run["response_mode_accuracy"]), help="Reply / schedule / no response / needs you - chosen correctly.")
    d.metric("Meeting date & time correct", pct(run["date_time_accuracy"]), help="Includes the 'Friday' regression case.")
    e, f, g, h = st.columns(4)
    e.metric("Meeting length correct", pct(run["duration_accuracy"]))
    f.metric("Prompt-injection resisted", pct(run["injection_resistance"]))
    g.metric("Latency p50", num(run["latency_p50_s"], 1, " s"))
    h.metric("Latency p95", num(run["latency_p95_s"], 1, " s"))
    _per_class_table(run["per_class"])
    with st.expander("Confusion matrix"):
        _confusion_table(run["confusion"])
    with st.expander("Every test email"):
        st.dataframe(
            [
                {
                    "Case": r["id"], "Expected": CATEGORY_LABELS.get(r["expected_category"], r["expected_category"]),
                    "AI said": CATEGORY_LABELS.get(r.get("got_category", ""), r.get("error", "-")),
                    "Category": "✅" if r.get("category_ok") else "❌" if "category_ok" in r else "⚠️",
                    "Decision": "✅" if r.get("mode_ok") else "❌" if "mode_ok" in r else "-",
                    "Date/time": ("✅ " if r["slot_ok"] else f"❌ wanted {r['slot_expected']}, got {r['slot_got']}") if "slot_ok" in r else "-",
                    "Seconds": r["latency_s"],
                }
                for r in run["rows"]
            ],
            hide_index=True, use_container_width=True,
        )
    if len(runs) > 1:
        st.markdown("**Trend across runs**")
        trend = pd.DataFrame(
            {
                "Accuracy": [r["classification_accuracy"] for r in runs],
                "Decision": [r["response_mode_accuracy"] for r in runs],
                "Date/time": [r["date_time_accuracy"] for r in runs],
            },
            index=[r["timestamp"][5:16].replace("T", " ") for r in runs],
        )
        st.line_chart(trend)


def page_evaluation(settings: Settings) -> None:
    st.header("Evaluation")
    days = st.select_slider("Time window", options=[7, 30, 90, 365], value=30, format_func=lambda d: f"last {d} days")
    try:
        snap = run_app(settings, lambda app: app.evaluation.snapshot(days))
    except Exception as exc:  # noqa: BLE001
        explain(exc)
        return
    tabs = st.tabs(["📈 Live operations", "🧑‍💼 Human review", "🛟 Reliability", "🎯 Accuracy (your feedback)", "🧪 Benchmark"])
    with tabs[0]:
        eval_live(snap)
    with tabs[1]:
        eval_human(snap)
    with tabs[2]:
        eval_reliability(snap)
    with tabs[3]:
        eval_quality(snap)
    with tabs[4]:
        eval_benchmark(settings)

# ------------------------------------------------------------------------------------ Settings
def tab_connections(settings: Settings) -> None:
    st.subheader("Groq (AI model)")
    api_key = settings.groq_api_key.get_secret_value()
    st.markdown("Status: " + ("✅ API key saved" if api_key else "❌ not set - get a free key at console.groq.com/keys"))
    new_key = st.text_input("Groq API key", type="password", placeholder="gsk_...", help="Stored in the local .env file only.")
    if st.button("Save key", disabled=not new_key.strip()):
        set_env_value("GROQ_API_KEY", new_key.strip())
        reload_settings()
        flash("success", "Groq key saved.")
        st.rerun()

    if st.button("Fetch available models", disabled=not api_key):
        try:
            st.session_state["groq_models"] = list_groq_models(api_key)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not list models: {type(exc).__name__}: {str(exc)[:200]}")
    options = filter_chat_models(st.session_state.get("groq_models", []))
    fast_opts = list(dict.fromkeys([settings.groq_model_fast, *options]))
    qual_opts = list(dict.fromkeys([settings.groq_model_quality, *options]))
    m1, m2 = st.columns(2)
    fast = m1.selectbox("Fast model (classification, extraction)", fast_opts)
    quality = m2.selectbox("Writing model (reply drafts)", qual_opts)
    k1, k2 = st.columns(2)
    if k1.button("Save models"):
        if save_prefs({"groq_model_fast": fast, "groq_model_quality": quality}):
            st.rerun()
    if k2.button("Test both models", disabled=not api_key):
        for name in dict.fromkeys([fast, quality]):
            ok, msg = test_groq_model(api_key, name)
            (st.success if ok else st.error)(f"{name}: {msg}")

    st.divider()
    st.subheader("Google (Gmail + Calendar)")
    secrets_path = settings.google_client_secrets_file
    token_path = settings.google_token_file
    st.markdown(f"OAuth client file: {'✅' if secrets_path.is_file() else '❌'} `{secrets_path}`  \nGoogle account: {'✅ connected' if token_path.is_file() else '❌ not connected'}")
    upload = st.file_uploader("Upload client_secret.json (OAuth client of type *Desktop app*)", type="json")
    if upload is not None:
        ok, msg = validate_client_secret(upload.getvalue())
        (st.success if ok else st.error)(msg)
        if ok and st.button("Save client file"):
            secrets_path.parent.mkdir(parents=True, exist_ok=True)
            secrets_path.write_bytes(upload.getvalue())
            flash("success", "Client file saved. Now click *Connect Google account*.")
            st.rerun()
    g1, g2, g3 = st.columns(3)
    if g1.button("🔗 Connect Google account", disabled=not secrets_path.is_file()):
        try:
            with st.spinner("A browser window opens on this computer - approve the permissions there…"):
                connect_google(settings)
            flash("success", "Google account connected.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            explain(exc)
    if g2.button("Verify connection", disabled=not token_path.is_file()):
        try:
            st.success(f"Connected as {google_mailbox(settings)}")
        except Exception as exc:  # noqa: BLE001
            explain(exc)
    if g3.button("Disconnect", disabled=not token_path.is_file()):
        disconnect_google(settings)
        flash("info", "Local Google token deleted. (Also revoke access at myaccount.google.com/permissions if you want.)")
        st.rerun()
    st.caption("The sign-in window opens on the machine running this app. On a remote server, run `python -m email_assistant auth` on your own computer and copy `credentials/token.json` to the server.")


def tab_profile(settings: Settings) -> None:
    zones = sorted(zoneinfo.available_timezones())
    with st.form("profile"):
        name = st.text_input("Your name", settings.user_name)
        mailbox = st.text_input("Mailbox address override", settings.user_email or "", help="Leave blank to auto-detect from the connected Google account.")
        tz = st.selectbox("Your timezone", zones, index=zones.index(settings.user_timezone) if settings.user_timezone in zones else 0)
        signature = st.text_area("Signature appended to every reply", signature_for_editing(settings.user_signature), height=110, help="Use {user_name} to insert your name.")
        tone = st.selectbox("Default tone", ["formal", "friendly", "concise"], index=["formal", "friendly", "concise"].index(settings.default_tone))
        instructions = st.text_area(
            "Standing instructions for replies", settings.custom_instructions, height=110,
            placeholder="e.g. Keep replies under 5 sentences. Never discuss pricing by email. Use British spelling.",
        )  # fmt: skip
        if st.form_submit_button("Save profile & style", type="primary"):
            if save_prefs({
                "user_name": name.strip() or "Assistant User", "user_email": mailbox.strip() or None, "user_timezone": tz,
                "user_signature": signature_for_storage(signature), "default_tone": tone, "custom_instructions": instructions.strip(),
            }):  # fmt: skip
                st.rerun()

def tab_scheduling(settings: Settings) -> None:
    schedule = settings.availability_schedule
    st.subheader("When are you available for meetings?")
    st.caption(
        "Tick the days you can meet and list your hours (several ranges allowed, e.g. `09:00-12:00, 14:00-17:30`). "
        "Meeting requests outside these hours are never accepted; the assistant offers times inside them instead."
    )
    st.markdown(f"**Current:** {schedule.describe()}")
    rows = [
        {
            "Day": code.title(),
            "Available": bool(schedule.windows.get(i)),
            "Hours": ", ".join(f"{a:%H:%M}-{b:%H:%M}" for a, b in schedule.windows.get(i, ())) or "09:00-17:30",
        }
        for i, code in enumerate(DAY_CODES)
    ]
    with st.form("scheduling"):
        edited = st.data_editor(
            rows, hide_index=True, use_container_width=True, disabled=["Day"], key="avail-editor",
            column_config={
                "Available": st.column_config.CheckboxColumn("Available", help="Untick = never accept meetings on this day"),
                "Hours": st.column_config.TextColumn("Hours (HH:MM-HH:MM, comma separated)"),
            },
        )  # fmt: skip
        off_text = st.text_area(
            "Time off (one per line)", "\n".join(schedule.to_dict()["time_off"]), height=90,
            placeholder="2026-12-24..2026-12-26\n2027-01-01",
            help="Single dates or ranges (YYYY-MM-DD..YYYY-MM-DD). No meetings are accepted on these days.",
        )  # fmt: skip
        d1, d2, d3, d4 = st.columns(4)
        minutes = d1.number_input("Default meeting length (min)", 5, 240, settings.default_meeting_minutes, step=5)
        buffer = d2.number_input("Buffer between meetings (min)", 0, 120, settings.meeting_buffer_minutes, step=5)
        notice = d3.number_input("Minimum notice (hours)", 0, 168, settings.min_notice_hours)
        horizon = d4.number_input("Look ahead (days)", 1, 60, settings.scheduling_horizon_days)
        calendar_id = st.text_input("Calendar to check and book on", settings.calendar_id, help="'primary' is your main calendar.")
        meet = st.checkbox("Add a Google Meet link to new invites", settings.add_meet_link)
        day_first = st.checkbox("Read dates like 03/04 as 3 April (day first)", settings.day_first_dates)
        if st.form_submit_button("Save scheduling preferences", type="primary"):
            try:
                windows = {}
                for i, row in enumerate(edited):
                    if row["Available"]:
                        parsed = parse_windows(str(row["Hours"]))
                        if not parsed:
                            raise ValueError(f"{row['Day']}: enter your hours or untick the day")
                        windows[i] = parsed
                new_schedule = Availability(windows, parse_time_off(off_text))
            except ValueError as exc:
                st.error(str(exc))
            else:
                if save_prefs({
                    "availability": new_schedule.to_json(), "default_meeting_minutes": int(minutes),
                    "meeting_buffer_minutes": int(buffer), "min_notice_hours": int(notice),
                    "scheduling_horizon_days": int(horizon), "calendar_id": calendar_id.strip() or "primary", "add_meet_link": meet,
                    "day_first_dates": day_first,
                }):  # fmt: skip
                    st.rerun()

    st.markdown("**Try it:** would a meeting at this time be accepted?")
    t1, t2, t3 = st.columns(3)
    day = t1.date_input("Date", key="avail-test-date")
    at = t2.time_input("Start", value=None, key="avail-test-time")
    length = t3.number_input("Minutes", 5, 480, settings.default_meeting_minutes, key="avail-test-len")
    if at is not None:
        start = datetime.combine(day, at, tzinfo=settings.tzinfo)
        probe = Interval(start, start + timedelta(minutes=int(length)))
        reason = schedule.explain(probe, settings.tzinfo)
        (st.success if schedule.contains(probe, settings.tzinfo) else st.warning)(
            f"{start:%A %d %b, %H:%M} ({settings.user_timezone}): {reason}"
        )


def tab_automation(settings: Settings) -> None:
    names = [*GMAIL_PRESETS, CUSTOM_QUERY]
    current_name = next((n for n, q in GMAIL_PRESETS.items() if q == settings.gmail_query), CUSTOM_QUERY)
    with st.form("automation"):
        which = st.selectbox("Which emails should be processed?", names, index=names.index(current_name))
        custom = st.text_input("Custom Gmail search (used only for 'Custom')", settings.gmail_query, help="Same syntax as the Gmail search box, e.g. from:boss@acme.com is:unread")
        a1, a2, a3 = st.columns(3)
        per_run = a1.number_input("Max emails per run", 1, 100, settings.gmail_max_per_run)
        interval = a2.number_input("Polling interval for the background worker (s)", 30, 3600, settings.poll_interval_seconds, step=30)
        ttl = a3.number_input("Approvals expire after (hours)", 1, 720, settings.approval_ttl_hours)
        conf = st.slider("Minimum AI confidence before acting", 0.0, 1.0, float(settings.min_classification_confidence), 0.05, help="Below this the email goes to 'Needs review' and nothing is drafted.")
        prefilter = st.checkbox("Skip the AI for obvious bulk mail (saves free-tier quota)", settings.enable_heuristic_prefilter)
        l1, l2 = st.columns(2)
        allowed = st.text_area(
            "Authorized senders (one per line; empty = everyone)", csv_to_lines(settings.allowed_senders), height=90,
            placeholder="vitstudent.ac.in",
            help="Only mail from these addresses or domains is read by the AI and turned into drafts. Everything else is skipped and shown in History as 'Not an authorized sender'.",
        )
        verify = st.checkbox(
            "Also require the sender to pass email authentication (SPF/DKIM/DMARC)", settings.require_authenticated_sender,
            help="Stops someone forging a 'From: ...@vitstudent.ac.in' address. Only applies when the list above is not empty.",
        )
        ignore = l1.text_area("Ignore list (one per line)", csv_to_lines(settings.ignore_senders), height=110, placeholder="newsletter@site.com\n@marketing.example.com", help="Emails from these addresses or domains are never processed.")
        vip = l2.text_area("VIP list (one per line)", csv_to_lines(settings.vip_senders), height=110, placeholder="boss@company.com\n@keyclient.com", help="Never treated as spam; priority is at least 4.")
        st.markdown("**What should happen with replies the assistant writes?**")
        mode = st.radio(
            "Reply handling", ["draft", "auto"], index=0 if settings.send_mode == "draft" else 1, horizontal=True,
            label_visibility="collapsed",
            format_func=lambda m: "📝 Save as a Gmail draft - I review and send" if m == "draft" else "🚀 Send automatically",
        )
        hold = st.checkbox(
            "In automatic mode, still hold replies that need my attention", settings.hold_risky_replies,
            help="Replies that make commitments, miss information, contain new links, are escalations, or were written by the fallback template stay as drafts for you.",
        )
        auto_events = st.checkbox(
            "Create calendar events automatically (invitations are emailed to the sender)", settings.auto_create_events,
            help="Off = invites wait for your approval, and replies that promise an invite wait with them.",
        )
        if mode == "auto":
            st.warning("Replies will be sent without you reading them. Keep the 'hold' option ticked unless you fully trust the assistant.")
        if st.form_submit_button("Save automation & safety", type="primary"):
            if save_prefs({
                "gmail_query": custom.strip() if which == CUSTOM_QUERY else GMAIL_PRESETS[which], "gmail_max_per_run": int(per_run),
                "poll_interval_seconds": int(interval), "approval_ttl_hours": int(ttl), "min_classification_confidence": float(conf),
                "enable_heuristic_prefilter": prefilter, "ignore_senders": lines_to_csv(ignore), "vip_senders": lines_to_csv(vip),
                "send_mode": mode, "hold_risky_replies": hold, "auto_create_events": auto_events,
                "allowed_senders": lines_to_csv(allowed), "require_authenticated_sender": verify,
            }):  # fmt: skip
                st.rerun()

def tab_business(settings: Settings) -> None:
    st.subheader("Business profile")
    with st.form("business"):
        name = st.text_input("Business / team name", settings.business_name)
        role = st.text_input("Your role", settings.business_role, placeholder="e.g. Customer support lead")
        context = st.text_area(
            "About your business (shared with the AI as trusted background)", settings.business_context, height=110,
            placeholder="e.g. We are a 20-person accounting firm. Support hours Mon-Fri 9-6 IST. We never give tax advice by email.",
        )
        sla = st.number_input("Reply target (hours)", 1, 720, settings.sla_hours, help="Approvals waiting longer than this are flagged as overdue.")
        if st.form_submit_button("Save business profile", type="primary"):
            if save_prefs({"business_name": name.strip(), "business_role": role.strip(), "business_context": context.strip(), "sla_hours": int(sla)}):
                st.rerun()

    st.subheader("Reply policy")
    st.caption("Not every email needs an answer. Choose which kinds the assistant may draft replies for.")
    with st.form("policy"):
        current = [c for c in settings.reply_categories.split(",") if c in REPLY_CATEGORY_OPTIONS]
        cats = st.multiselect("Draft replies for", REPLY_CATEGORY_OPTIONS, default=current, format_func=CATEGORY_LABELS.get)
        scheduling = st.checkbox("Allow calendar scheduling for meeting requests", settings.enable_scheduling,
                                 help="Off = meeting requests get a written reply only; the calendar is never checked or booked.")
        st.caption("Always skipped regardless of these settings: spam/marketing, automated no-reply senders, thank-you/FYI messages that need no answer, and emails the AI flags as needing your personal judgement.")
        if st.form_submit_button("Save reply policy", type="primary"):
            if save_prefs({"reply_categories": ",".join(cats), "enable_scheduling": scheduling}):
                st.rerun()

    st.subheader("Knowledge base")
    st.caption("Facts, FAQs and policies the assistant may use in replies (opening hours, refund policy, pricing rules...). If something is not covered the assistant says it will follow up instead of guessing.")
    entries = run_app(settings, lambda app: app.knowledge.list_all())
    if entries:
        st.dataframe([{"ID": e.id, "Title": e.title, "Content": e.content, "Active": e.enabled} for e in entries],
                     use_container_width=True, hide_index=True)
        k1, k2, k3 = st.columns([1, 1, 2])
        pick = k1.selectbox("Entry", [e.id for e in entries], key="kb-pick")
        entry = next(e for e in entries if e.id == pick)
        if k2.button("Disable" if entry.enabled else "Enable"):
            run_app(settings, lambda app: app.knowledge.set_enabled(pick, not entry.enabled))
            st.rerun()
        if k3.button("🗑 Delete entry"):
            run_app(settings, lambda app: app.knowledge.delete(pick))
            st.rerun()
    else:
        st.info("No entries yet.")
    with st.form("kb-add", clear_on_submit=True):
        title = st.text_input("Title", placeholder="e.g. Refund policy")
        content = st.text_area("Content", placeholder="e.g. Refunds are available within 30 days of purchase with proof of payment.")
        if st.form_submit_button("Add to knowledge base"):
            if not title.strip() or not content.strip():
                st.error("Both title and content are required.")
            else:
                run_app(settings, lambda app: app.knowledge.add(title, content))
                st.rerun()

def tab_memory(settings: Settings) -> None:
    st.caption("What the assistant remembers. Rejection reasons are saved here automatically; add your own preferences below.")
    entries = run_app(settings, lambda app: app.memory.list_entries())
    if entries:
        st.dataframe(
            [{"ID": e["id"], "Applies to": e["scope"].replace("sender:", "sender "), "Type": e["kind"], "Memory": e["content"]} for e in entries],
            use_container_width=True, hide_index=True,
        )  # fmt: skip
        d1, d2 = st.columns([1, 3])
        victim = d1.selectbox("Delete entry", [e["id"] for e in entries])
        if d2.button("🗑 Delete"):
            run_app(settings, lambda app: app.memory.delete(victim))
            st.rerun()
    else:
        st.info("No memories yet.")
    with st.form("add-memory", clear_on_submit=True):
        text = st.text_area("New preference or note", placeholder="e.g. Always reply to the design team in a friendly tone.")
        scope = st.radio("Applies to", ["Everyone", "One sender"], horizontal=True)
        sender = st.text_input("Sender address (if 'One sender')")
        if st.form_submit_button("Add memory"):
            if not text.strip():
                st.error("Write something first.")
            elif scope == "One sender" and "@" not in sender:
                st.error("Enter the sender's email address.")
            else:
                target = sender_scope(sender.strip()) if scope == "One sender" else GLOBAL_SCOPE
                run_app(settings, lambda app: app.memory.add(target, "preference", text))
                st.rerun()


def tab_advanced(settings: Settings) -> None:
    st.markdown(f"Customizations file: `{PREFERENCES_FILE}`  \nSecrets: `.env`  \nDatabase: `{settings.database_url}`")
    prefs = load_preferences()
    st.download_button("⬇ Export my customizations", json.dumps(prefs, indent=2), "email_assistant_settings.json", "application/json")
    imported = st.file_uploader("Import customizations", type="json", key="import")
    if imported is not None and st.button("Apply imported settings"):
        try:
            data = {k: v for k, v in json.loads(imported.getvalue()).items() if k in EDITABLE_FIELDS}
        except (ValueError, AttributeError):
            st.error("Not a valid settings file.")
        else:
            if save_prefs(data):
                st.rerun()
    if st.button("Reset all customizations to defaults"):
        PREFERENCES_FILE.unlink(missing_ok=True)
        reload_settings()
        flash("info", "Customizations reset.")
        st.rerun()


def page_settings(settings: Settings) -> None:
    st.header("Settings")
    tabs = st.tabs(["🔌 Connections", "👤 Profile & style", "🏢 Business & policy", "📅 Scheduling", "🛡️ Automation & safety", "🧠 Memory", "🧰 Advanced"])
    for tab, render in zip(tabs, (tab_connections, tab_profile, tab_business, tab_scheduling, tab_automation, tab_memory, tab_advanced), strict=True):
        with tab:
            render(settings)


# ------------------------------------------------------------------------------------ entry point
def main() -> None:
    st.set_page_config(page_title="AI Email Assistant", page_icon="✉️", layout="wide")
    try:
        settings = get_settings()
    except ValidationError as exc:
        st.error("Your saved customizations are invalid, so the app cannot start.")
        st.code("\n".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))
        if st.button("Reset customizations"):
            PREFERENCES_FILE.unlink(missing_ok=True)
            reload_settings()
            st.rerun()
        st.stop()

    with st.sidebar:
        st.title("✉️ AI Email Assistant")
        page = st.radio("Go to", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption(f"Signed in as **{settings.user_name}** · {settings.user_timezone}")
        st.caption("📝 Replies are saved as drafts for you." if settings.send_mode == "draft" else "🚀 Replies are sent automatically.")

    show_flash()
    {PAGES[0]: page_dashboard, PAGES[1]: page_approvals, PAGES[2]: page_history, PAGES[3]: page_evaluation, PAGES[4]: page_settings}[page](settings)


if __name__ == "__main__":
    main()
