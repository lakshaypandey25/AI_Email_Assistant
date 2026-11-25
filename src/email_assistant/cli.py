"""Command-line interface.

    python -m email_assistant auth            # one-time Google OAuth consent
    python -m email_assistant init-db         # create tables
    python -m email_assistant run [--once]    # process the inbox (drafts only; sends need approval)
    python -m email_assistant pending         # list actions awaiting approval
    python -m email_assistant show ID         # inspect one action
    python -m email_assistant approve ID...   # approve + execute (asks for confirmation)
    python -m email_assistant approve --email N   # approve invite + reply of email N together (safe order)
    python -m email_assistant ui              # open the Streamlit web UI
    python -m email_assistant reject ID [-r why]
    python -m email_assistant remember "text" [--sender a@b.com]
    python -m email_assistant mcp-server      # stdio MCP server (safe tools only)
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys

from pydantic import ValidationError

from email_assistant.config import Settings, get_settings
from email_assistant.exceptions import AssistantError
from email_assistant.logging_config import configure_logging


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="email-assistant", description="AI Email Assistant")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("auth", help="Run Google OAuth consent flow")
    sub.add_parser("logout", help="Delete the local Google token")
    sub.add_parser("init-db", help="Create database tables")
    sub.add_parser("whoami", help="Verify Google auth and show the mailbox address")
    run = sub.add_parser("run", help="Process the inbox")
    run.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    run.add_argument("--max", type=int, default=None, help="Max messages this cycle")
    sub.add_parser("pending", help="List pending approvals")
    show = sub.add_parser("show", help="Show one approval")
    show.add_argument("id", type=int)
    appr = sub.add_parser("approve", help="Approve and execute actions")
    appr.add_argument("ids", type=int, nargs="*")
    appr.add_argument("--email", type=int, default=None, help="Approve all open actions of this email id")
    appr.add_argument("--ignore-conflicts", action="store_true", help="Create the invite even if the slot is now busy")
    appr.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    rej = sub.add_parser("reject", help="Reject an action")
    rej.add_argument("id", type=int)
    rej.add_argument("-r", "--reason", default=None, help="Why (stored as feedback for future drafts)")
    mem = sub.add_parser("remember", help="Add a memory/preference")
    mem.add_argument("text")
    mem.add_argument("--sender", default=None, help="Scope the memory to a sender address")
    mcp = sub.add_parser("mcp-server", help="Start the stdio MCP server (the agent starts it by itself; use this for MCP clients)")
    mcp.add_argument("--no-ai", action="store_true", help="Do not offer the ai_draft_reply tool")
    sub.add_parser("ui", help="Start the Streamlit web UI")
    ev = sub.add_parser("eval", help="Run the labelled evaluation benchmark with the real AI")
    ev.add_argument("--quick", action="store_true", help="Only 8 emails (fewer AI calls)")
    return p


def _print(msg: str = "") -> None:
    print(msg, flush=True)


async def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    from email_assistant.container import build_container
    from email_assistant.services.memory import GLOBAL_SCOPE, sender_scope
    from email_assistant.mcp_tools.names import TOOL_PROFILE

    app = build_container(settings)
    await app.db.create_all()
    actor = f"user:{getpass.getuser()}"
    try:
        if args.command == "init-db":
            _print("Database ready.")
        elif args.command == "whoami":
            _print(f"Authenticated mailbox: {(await app.tools.call(TOOL_PROFILE, {}))['email']}")
        elif args.command == "run":
            if args.once:
                summary = await app.agent.run_once(args.max)
                _print(f"Processed {summary.total} message(s): {dict(summary.counts)}")
            else:
                _print("Polling... press Ctrl+C to stop.")
                await app.agent.run_forever()
        elif args.command == "eval":
            from email_assistant.evaluation.runner import run_benchmark
            from email_assistant.evaluation.store import save_run

            result = await run_benchmark(settings, app.llm, quick=args.quick, progress=lambda f, label: _print(f"  {f:4.0%} {label}"))
            save_run(result)
            for key in ("cases", "errors", "classification_accuracy", "macro_f1", "response_mode_accuracy", "date_time_accuracy", "duration_accuracy", "injection_resistance", "latency_p50_s", "latency_p95_s"):
                _print(f"{key:26} {result[key]}")
        elif args.command == "pending":
            groups = await app.hitl.list_groups()
            if not groups:
                _print("No pending approvals.")
            for g in groups:
                _print(f"--- email#{g.email_id} from {g.sender} | {g.subject} | {g.category} (priority {g.priority})")
                for r in g.items:
                    _print(f"  [{r.id}] {r.action_type} ({r.status}) expires {r.expires_at:%Y-%m-%d %H:%M} UTC")
                    _print("      " + r.preview.replace("\n", "\n      "))
                    if r.error:
                        _print(f"      last error: {r.error}")
                for w in g.warnings:
                    _print(f"  ! {w}")
                if len(g.items) > 1:
                    _print(f"  tip: python -m email_assistant approve --email {g.email_id}")
        elif args.command == "show":
            row = await app.hitl.get(args.id)
            _print(str(row) if row else "Not found.")
        elif args.command == "approve":
            if args.email is not None:
                if not args.yes and input(f"Approve ALL open actions of email#{args.email}? [y/N] ").strip().lower() != "y":
                    _print("skipped")
                    return 0
                group = await app.hitl.approve_email(args.email, actor, ignore_conflicts=args.ignore_conflicts)
                for res in group.results:
                    _print(f"[{res.approval_id}] -> {res.status.value}" + (f": {res.error}" if res.error else ""))
                for skipped in group.skipped:
                    _print(f"[{skipped}] not sent: the calendar invite failed, so the reply was withheld")
                return 0
            if not args.ids:
                _print("Give approval ids, or use --email N.")
                return 1
            for approval_id in args.ids:
                view = await app.hitl.get(approval_id)
                if view is None:
                    _print(f"[{approval_id}] not found")
                    continue
                _print(f"[{view.id}] {view.action_type}\n    " + view.preview.replace("\n", "\n    "))
                if not args.yes and input("Approve and execute? [y/N] ").strip().lower() != "y":
                    _print("  skipped")
                    continue
                res = await app.hitl.approve(approval_id, actor, ignore_conflicts=args.ignore_conflicts)
                _print(f"  -> {res.status.value}" + (f": {res.error}" if res.error else ""))
        elif args.command == "reject":
            await app.hitl.reject(args.id, actor, args.reason)
            if args.reason:
                await app.memory.add(GLOBAL_SCOPE, "feedback", f"Rejected a proposed action: {args.reason}")
            _print("Rejected.")
        elif args.command == "remember":
            scope = sender_scope(args.sender) if args.sender else GLOBAL_SCOPE
            await app.memory.add(scope, "preference", args.text)
            _print("Saved.")
        return 0
    finally:
        await app.aclose()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration error - check your .env file:", file=sys.stderr)
        for err in exc.errors():
            print(f"  - {'.'.join(str(x) for x in err['loc'])}: {err['msg']}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level, settings.log_json)

    try:
        if args.command == "auth":
            from email_assistant.auth.google_auth import GoogleAuthManager

            GoogleAuthManager(settings).run_interactive_flow()
            print("Authentication successful. Token stored.")
            return 0
        if args.command == "logout":
            from email_assistant.auth.google_auth import GoogleAuthManager

            GoogleAuthManager(settings).logout()
            print("Local token deleted. Also revoke access at https://myaccount.google.com/permissions")
            return 0
        if args.command == "ui":
            import subprocess
            from pathlib import Path

            app_path = Path(__file__).parent / "ui" / "app.py"
            return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])
        if args.command == "mcp-server":
            from email_assistant.mcp_tools.server import main as mcp_main

            mcp_main(no_ai=args.no_ai)
            return 0
        return asyncio.run(_dispatch(args, settings))
    except KeyboardInterrupt:
        return 130
    except AssistantError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
