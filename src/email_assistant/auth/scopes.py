"""OAuth scopes - the minimum set required (Principle of Least Privilege).

====================================  ===============================================
Scope                                 Why it is needed
====================================  ===============================================
gmail.readonly                        Read inbox messages (no modify / delete)
gmail.compose                         Create drafts and send *approved* drafts
calendar.freebusy                     Availability lookups (no event contents)
calendar.events                       Create events after human approval
====================================  ===============================================

Deliberately NOT requested: ``gmail.modify``, ``mail.google.com`` (full access, incl. delete),
and the broad ``calendar`` scope.
"""

from __future__ import annotations

GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_COMPOSE = "https://www.googleapis.com/auth/gmail.compose"
CALENDAR_FREEBUSY = "https://www.googleapis.com/auth/calendar.freebusy"
CALENDAR_EVENTS = "https://www.googleapis.com/auth/calendar.events"

REQUIRED_SCOPES: tuple[str, ...] = (GMAIL_READONLY, GMAIL_COMPOSE, CALENDAR_FREEBUSY, CALENDAR_EVENTS)
