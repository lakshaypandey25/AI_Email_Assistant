"""Schemas the LLM must satisfy via tool / function calling.

The JSON Schema of each model is sent to the provider as a function definition; the
returned arguments are then validated with Pydantic. Before-validators add tolerance
for common small-model quirks (``"85%"`` confidences, ``"Action Required"`` enums).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from email_assistant.schemas.enums import EmailCategory


def _coerce_probability(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100.0
            except ValueError:
                return value
        try:
            value = float(text)
        except ValueError:
            return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 1 < value <= 100:
        return value / 100.0
    return value


class ActionItem(BaseModel):
    """A concrete follow-up extracted from an email."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    description: str = Field(min_length=3, max_length=300, description="Imperative, specific task")
    owner: Literal["me", "sender", "other"] = Field(default="me", description="Who must do it")
    due: str | None = Field(default=None, description="ISO date YYYY-MM-DD if a deadline is stated")


class ClassificationResult(BaseModel):
    """Output of the ``classify_email`` tool."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    category: EmailCategory = Field(
        description=(
            "action_required: the user must do/answer something. "
            "meeting_request: someone wants to schedule, move or cancel a meeting/call. "
            "information_only: FYI, receipts, notifications; no reply needed. "
            "spam_marketing: unsolicited promotion, newsletters, cold sales. "
            "priority_escalation: urgent/critical, executives, outages, legal or security issues, hard deadlines <24h."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated probability the category is right")
    priority_score: int = Field(ge=1, le=5, description="1 = ignorable, 3 = normal, 5 = needs attention today")
    reasoning: str = Field(min_length=3, max_length=1000, description="1-3 sentences; cite concrete evidence")
    summary: str = Field(min_length=3, max_length=600, description="One or two sentence summary")
    requires_reply: bool = Field(description="True if the sender expects a response")
    response_mode: Literal["none", "reply", "schedule", "human_only"] | None = Field(
        default=None,
        description=(
            "What the assistant should do. none: no response needed (thanks/ok/FYI/notifications/receipts). "
            "reply: a written reply is enough (question, request, acknowledgement) - no calendar involved. "
            "schedule: sender wants to schedule/move/cancel a meeting, so check the calendar. "
            "human_only: needs the user's own judgement (legal, HR, money, sensitive, complaints) - do not draft."
        ),
    )
    action_items: list[ActionItem] = Field(default_factory=list, max_length=10)

    @field_validator("category", mode="before")
    @classmethod
    def _normalise_category(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalise_confidence(cls, value: Any) -> Any:
        return _coerce_probability(value)

    @field_validator("priority_score", mode="before")
    @classmethod
    def _normalise_priority(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    @field_validator("response_mode", mode="before")
    @classmethod
    def _normalise_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().lower().replace(" ", "_").replace("-", "_")
            return None if text in {"", "null"} else {"no_response": "none", "human": "human_only", "scheduling": "schedule"}.get(text, text)
        return value

    @property
    def effective_response_mode(self) -> str:
        """The model's explicit choice, or a safe default derived from the category."""
        if self.response_mode:
            return self.response_mode
        if self.category is EmailCategory.SPAM_MARKETING:
            return "none"
        if self.category is EmailCategory.MEETING_REQUEST:
            return "schedule"
        if self.category is EmailCategory.PRIORITY_ESCALATION:
            return "reply"
        return "reply" if self.requires_reply else "none"


class EmailDraft(BaseModel):
    """Output of the ``draft_reply`` tool. Subject/recipients are decided by code, not the model."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    body: str = Field(
        min_length=1,
        max_length=6000,
        description="Reply body only. No subject line, no signature, no sign-off name.",
    )
    tone: Literal["formal", "friendly", "concise", "apologetic"] = "formal"
    confidence: float = Field(ge=0.0, le=1.0, description="How confident you are the reply is correct and complete")
    needs_user_input: bool = Field(
        default=False,
        description="True if information only the user knows is missing (the draft should then ask or hedge).",
    )
    open_questions: list[str] = Field(default_factory=list, description="Questions the user must answer")
    makes_commitments: bool = Field(
        default=False,
        description="True if the reply commits the user to money, deadlines, deliverables, legal terms or attendance.",
    )
    risk_flags: list[str] = Field(default_factory=list, description="Short labels for anything sensitive")

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalise_confidence(cls, value: Any) -> Any:
        return _coerce_probability(value)
