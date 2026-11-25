"""Dependency-injection container: wires settings, tools, services and the agent.

The agent's tools are an MCP client by default (``McpToolClient.stdio``): Gmail and Calendar are reached ONLY through
the MCP server process. Tests may inject fakes (``gmail=`` / ``calendar=``), in which case the server-side registry is
used in-process as a test double, or pass ``tools=`` (for example an in-memory MCP client) to exercise the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from email_assistant.agent.orchestrator import EmailAgent
from email_assistant.config import Settings
from email_assistant.db.session import Database
from email_assistant.evaluation.stats import EvaluationService
from email_assistant.llm.base import StructuredLLM
from email_assistant.llm.groq_client import GroqLLMClient
from email_assistant.mcp_tools.client import McpToolClient
from email_assistant.mcp_tools.names import TOOL_PROFILE
from email_assistant.services.classification import ClassificationService
from email_assistant.services.drafting import DraftingService
from email_assistant.services.hitl import HITLController
from email_assistant.services.inbox import InboxService
from email_assistant.services.knowledge import KnowledgeService
from email_assistant.services.memory import MemoryService
from email_assistant.services.scheduling import SchedulingService
from email_assistant.services.telemetry import MetricsRecorder


@dataclass
class GoogleStack:
    """Google-facing components. Built only inside the MCP server process (or by tests that inject fakes)."""

    auth: Any
    gmail: Any
    calendar: Any
    fetcher: Any
    registry: Any


@dataclass
class Container:
    """Fully-wired application."""

    settings: Settings
    db: Database
    tools: Any  # McpToolClient (production) or a ToolRegistry (test double)
    llm: StructuredLLM
    memory: MemoryService
    hitl: HITLController
    inbox: InboxService
    knowledge: KnowledgeService
    evaluation: EvaluationService
    agent: EmailAgent
    classifier: ClassificationService
    drafter: DraftingService
    google: GoogleStack | None = None

    async def aclose(self) -> None:
        """Stop the MCP server process (if any) and release the database."""
        closer = getattr(self.tools, "aclose", None)
        if closer is not None:
            await closer()
        await self.db.dispose()


def build_google_stack(settings: Settings, *, gmail: Any = None, calendar: Any = None) -> GoogleStack:
    """Build the Google clients + server-side tool registry (used by the MCP server, and by tests with fakes)."""
    from email_assistant.auth.google_auth import GoogleAuthManager
    from email_assistant.integrations.calendar_client import CalendarClient
    from email_assistant.integrations.gmail_client import GmailClient
    from email_assistant.mcp_tools.tools import build_registry
    from email_assistant.services.fetcher import EmailFetcher

    auth = GoogleAuthManager(settings)
    gmail = gmail or GmailClient(auth, settings)
    calendar = calendar or CalendarClient(auth, settings)
    fetcher = EmailFetcher(gmail, settings)
    registry = build_registry(gmail=gmail, calendar=calendar, fetcher=fetcher, settings=settings)
    return GoogleStack(auth, gmail, calendar, fetcher, registry)


class _OwnEmail:
    """The mailbox address, fetched once through the ``gmail_get_profile`` tool."""

    def __init__(self, tools: Any, settings: Settings) -> None:
        self._tools = tools
        self._settings = settings
        self._cached: str | None = None

    async def __call__(self) -> str:
        if self._settings.user_email:
            return self._settings.user_email.lower()
        if self._cached is None:
            self._cached = str((await self._tools.call(TOOL_PROFILE, {}))["email"]).lower()
        return self._cached


def build_container(
    settings: Settings,
    *,
    gmail: Any = None,
    calendar: Any = None,
    llm: StructuredLLM | None = None,
    db: Database | None = None,
    tools: Any = None,
) -> Container:
    """Wire the whole application. By default the agent reaches Google only through the MCP server."""
    google: GoogleStack | None = None
    if tools is None:
        if gmail is not None or calendar is not None:  # tests: fakes behind the server-side registry
            google = build_google_stack(settings, gmail=gmail, calendar=calendar)
            tools = google.registry
        else:
            tools = McpToolClient.stdio()
    database = db or Database(settings.database_url)
    llm_client: StructuredLLM = llm or GroqLLMClient(settings, recorder=MetricsRecorder(database))
    memory = MemoryService(database)
    knowledge = KnowledgeService(database)
    hitl = HITLController(database, tools, settings)
    classifier = ClassificationService(llm_client, settings)
    drafter = DraftingService(llm_client, settings)
    agent = EmailAgent(
        settings=settings,
        db=database,
        registry=tools,
        classifier=classifier,
        drafter=drafter,
        scheduler=SchedulingService(llm_client, tools, settings),
        memory=memory,
        hitl=hitl,
        knowledge=knowledge,
        own_email_provider=_OwnEmail(tools, settings),
    )
    return Container(
        settings, database, tools, llm_client, memory, hitl, InboxService(database), knowledge,
        EvaluationService(database, settings), agent, classifier, drafter, google,
    )