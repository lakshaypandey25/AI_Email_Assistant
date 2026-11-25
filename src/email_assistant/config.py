"""Centralised, validated configuration loaded from environment / ``.env``.

No credential is ever hard-coded: secrets use ``SecretStr`` so they are masked in
``repr`` output and logs.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from datetime import time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from email_assistant.core.senders import parse_list

from email_assistant.scheduling.slots import Availability, default_availability

_DAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


class Settings(BaseSettings):
    """Application settings. Field names map to UPPER_CASE environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # --- General -----------------------------------------------------------
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # --- LLM (Groq) ----------------------------------------------------------
    groq_api_key: SecretStr = SecretStr("")
    groq_model_fast: str = "openai/gpt-oss-20b"
    groq_model_quality: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = 30.0
    groq_max_output_tokens: int = 2048
    groq_requests_per_minute: int = 20
    llm_temperature: float = 0.1
    llm_max_validation_attempts: int = 3
    llm_max_body_chars: int = 4000

    # --- Google ----------------------------------------------------------------
    google_client_secrets_file: Path = Path("credentials/client_secret.json")
    google_token_file: Path = Path("credentials/token.json")
    token_encryption_key: SecretStr | None = None
    google_requests_per_second: float = 5.0
    google_max_concurrency: int = 4
    google_retry_max_attempts: int = 5

    # --- Gmail ingestion -------------------------------------------------------
    gmail_query: str = "in:inbox is:unread newer_than:2d -category:promotions -category:social"
    gmail_max_per_run: int = 15
    max_concurrent_emails: int = 2
    poll_interval_seconds: int = 180
    max_processing_attempts: int = 3
    enable_heuristic_prefilter: bool = True

    # --- Resilience ----------------------------------------------------------------------
    groq_fallback_models: str = ""
    groq_tokens_per_minute: int = 6000
    llm_reasoning_effort: Literal["low", "medium", "high"] = "low"
    defer_retry_minutes: int = 5
    defer_give_up_hours: int = 3

    # --- Reply policy & business profile --------------------------------------------
    reply_categories: str = "action_required,meeting_request,priority_escalation"
    enable_scheduling: bool = True
    business_name: str = ""
    business_role: str = ""
    business_context: str = ""
    sla_hours: int = 24
    kb_max_snippets: int = 3

    # --- Persistence -------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./data/assistant.db"

    # --- User profile --------------------------------------------------------------
    user_name: str = "Assistant User"
    user_email: str | None = None
    user_timezone: str = "UTC"
    user_signature: str = "Best regards,\\n{user_name}"
    default_tone: Literal["formal", "friendly", "concise"] = "formal"
    custom_instructions: str = ""
    ignore_senders: str = ""
    allowed_senders: str = "vitstudent.ac.in"  # empty = anyone; otherwise only these addresses / domains are processed
    require_authenticated_sender: bool = True  # allowed senders must also pass SPF/DKIM/DMARC (anti-spoofing)
    day_first_dates: bool = True  # 03/04 means 3 April (not 4 March)
    vip_senders: str = ""

    # --- Scheduling ------------------------------------------------------------------
    calendar_id: str = "primary"
    working_hours_start: time = time(9, 0)
    working_hours_end: time = time(17, 30)
    work_days: str = "mon,tue,wed,thu,fri"
    availability: str = ""  # JSON edited in the UI; empty = use the working hours / days above
    default_meeting_minutes: int = 30
    meeting_buffer_minutes: int = 10
    min_notice_hours: int = 2
    scheduling_horizon_days: int = 14
    add_meet_link: bool = True

    # --- HITL & guardrails ---------------------------------------------------------------
    approval_ttl_hours: int = 72
    min_classification_confidence: float = 0.6
    max_draft_chars: int = 4000
    send_mode: Literal["draft", "auto"] = "draft"
    auto_create_events: bool = False
    hold_risky_replies: bool = True

    # ------------------------------------------------------------------ validators
    
    @field_validator("user_timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown IANA timezone: {value!r}") from exc
        return value

    @field_validator("availability")
    @classmethod
    def _validate_availability(cls, value: str) -> str:
        if value.strip():
            Availability.from_json(value)  # raises ValueError with a friendly message
        return value

    @field_validator("work_days")
    @classmethod
    def _validate_work_days(cls, value: str) -> str:
        days = [d.strip().lower()[:3] for d in value.split(",") if d.strip()]
        if not days or any(d not in _DAY_INDEX for d in days):
            raise ValueError("WORK_DAYS must be a comma list of mon,tue,wed,thu,fri,sat,sun")
        return ",".join(days)

    @field_validator("reply_categories")
    @classmethod
    def _validate_reply_categories(cls, value: str) -> str:
        allowed = {"action_required", "meeting_request", "information_only", "spam_marketing", "priority_escalation"}
        items = [c.strip().lower() for c in value.split(",") if c.strip()]
        if any(c not in allowed for c in items):
            raise ValueError(f"REPLY_CATEGORIES may only contain: {sorted(allowed)}")
        return ",".join(items)

    @field_validator("min_classification_confidence")
    @classmethod
    def _validate_conf(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("MIN_CLASSIFICATION_CONFIDENCE must be within [0, 1]")
        return value

    # ------------------------------------------------------------------ helpers
    @property
    def tzinfo(self) -> ZoneInfo:
        """The user's timezone as a ``ZoneInfo``."""
        return ZoneInfo(self.user_timezone)

    @property
    def availability_schedule(self) -> Availability:
        """When the user can take meetings (UI schedule, or the old working hours when none is saved)."""
        if self.availability.strip():
            return Availability.from_json(self.availability)
        return default_availability(self.working_hours_start, self.working_hours_end, self.work_days_set)

    @property
    def work_days_set(self) -> frozenset[int]:
        """Working weekdays as Python ``weekday()`` integers (Mon=0)."""
        return frozenset(_DAY_INDEX[d] for d in self.work_days.split(","))

    @property
    def signature(self) -> str:
        """Signature block appended to every draft."""
        return self.user_signature.replace("\\n", "\n").replace("{user_name}", self.user_name)

    @property
    def ignore_list(self) -> list[str]:
        """Senders / domains that are skipped without spending LLM quota."""
        return parse_list(self.ignore_senders)

    @property
    def allowed_list(self) -> list[str]:
        """Only these senders / domains are processed (empty = everyone)."""
        return parse_list(self.allowed_senders)

    @property
    def reply_categories_set(self) -> frozenset[str]:
        """Categories the assistant is allowed to draft replies for."""
        return frozenset(c for c in self.reply_categories.split(",") if c)

    @property
    def fallback_models_list(self) -> list[str]:
        """Extra Groq models tried (in order) when the primary model is rate-limited or unavailable."""
        return [m.strip() for m in self.groq_fallback_models.replace("\n", ",").split(",") if m.strip()]

    @property
    def vip_list(self) -> list[str]:
        """Senders / domains that are never treated as spam and get priority >= 4."""
        return parse_list(self.vip_senders)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


# --------------------------------------------------------------------------- user preferences
#: Where UI customizations are stored (non-secret). Secrets stay in ``.env``.
PREFERENCES_FILE = Path("data/user_settings.json")

#: Only these fields can be overridden from the UI / preferences file.
EDITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "user_name", "user_email", "user_timezone", "user_signature", "default_tone", "custom_instructions",
        "groq_model_fast", "groq_model_quality", "calendar_id", "working_hours_start", "working_hours_end",
        "work_days", "default_meeting_minutes", "meeting_buffer_minutes", "min_notice_hours",
        "scheduling_horizon_days", "add_meet_link", "gmail_query", "gmail_max_per_run", "poll_interval_seconds",
        "min_classification_confidence", "send_mode", "auto_create_events", "hold_risky_replies", "approval_ttl_hours",
        "enable_heuristic_prefilter", "ignore_senders", "vip_senders", "max_draft_chars",
        "groq_fallback_models", "groq_tokens_per_minute", "llm_reasoning_effort", "defer_retry_minutes", "defer_give_up_hours", "groq_requests_per_minute", "availability",
        "reply_categories", "enable_scheduling", "business_name", "business_role", "business_context", "sla_hours", "kb_max_snippets",
        "allowed_senders", "require_authenticated_sender", "day_first_dates",
    }
)  # fmt: skip


def load_preferences(path: Path | None = None) -> dict[str, Any]:
    """Read saved customizations (empty dict if none / unreadable). Unknown keys are dropped."""
    target = path or PREFERENCES_FILE
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if k in EDITABLE_FIELDS} if isinstance(data, dict) else {}


def save_preferences(values: dict[str, Any], path: Path | None = None) -> None:
    """Atomically persist customizations (whitelisted keys only)."""
    target = path or PREFERENCES_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in values.items() if k in EDITABLE_FIELDS}
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, indent=2, sort_keys=True)
    os.replace(tmp, target)


def set_env_value(key: str, value: str, path: Path | None = None) -> None:
    """Create or update ``KEY=value`` in the ``.env`` file (used for secrets entered in the UI)."""
    target = path or Path(".env")
    quoted = f'"{value}"' if any(c in value for c in " #\"'") else value
    lines = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
    replaced = False
    for i, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == key and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={quoted}"
            replaced = True
    if not replaced:
        lines.append(f"{key}={quoted}")
    fd, tmp = tempfile.mkstemp(dir=target.parent if str(target.parent) else ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(tmp, target)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings: environment / ``.env`` first, then UI customizations on top."""
    return Settings(**load_preferences())  # type: ignore[arg-type]


def reload_settings() -> Settings:
    """Drop the cache and re-read ``.env`` and the preferences file."""
    get_settings.cache_clear()
    return get_settings()
