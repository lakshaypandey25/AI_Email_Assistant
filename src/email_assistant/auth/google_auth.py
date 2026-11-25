"""Google OAuth2 manager: interactive consent flow, token persistence and automatic refresh."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import structlog
from google.auth.exceptions import RefreshError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from email_assistant.auth.scopes import REQUIRED_SCOPES
from email_assistant.auth.token_store import TokenStore
from email_assistant.config import Settings
from email_assistant.core.retry import RetryPolicy, retry_async
from email_assistant.exceptions import AuthenticationRequiredError, ConfigurationError

logger = structlog.get_logger(__name__)


def _is_transient_refresh_error(exc: BaseException) -> bool:
    """Network blips while refreshing are retryable; ``RefreshError`` (revoked/invalid) is not."""
    return isinstance(exc, (TransportError, ConnectionError, TimeoutError)) and not isinstance(exc, RefreshError)


class GoogleAuthManager:
    """Owns the user's Google credentials.

    * :meth:`run_interactive_flow` - one-time browser consent (``email-assistant auth``).
    * :meth:`get_credentials` - returns valid credentials, refreshing the access token when it
      is expired/near expiry and persisting the refreshed token. Safe for concurrent callers.
    """

    def __init__(
        self,
        settings: Settings,
        token_store: TokenStore | None = None,
        scopes: Sequence[str] = REQUIRED_SCOPES,
    ) -> None:
        self._settings = settings
        self._scopes = tuple(scopes)
        key = (settings.token_encryption_key.get_secret_value() if settings.token_encryption_key else "") or None
        self._store = token_store or TokenStore(settings.google_token_file, key)
        self._lock = asyncio.Lock()
        self._cached: Credentials | None = None
        self._refresh = retry_async(
            RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=15.0),
            is_retryable=_is_transient_refresh_error,
            operation="google_token_refresh",
        )(self._refresh_once)

    # ------------------------------------------------------------------ interactive
    def run_interactive_flow(self) -> Credentials:
        """Open a browser for OAuth consent and persist the resulting credentials (blocking)."""
        secrets = self._settings.google_client_secrets_file
        if not secrets.is_file():
            raise ConfigurationError(
                f"OAuth client file not found at {secrets}. Download it from Google Cloud Console "
                "(APIs & Services > Credentials > OAuth client ID > Desktop app)."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(secrets), scopes=list(self._scopes))
        creds = flow.run_local_server(
            host="localhost",
            port=0,
            access_type="offline",  # request a refresh token
            prompt="consent",  # force refresh-token issuance even on re-consent
            authorization_prompt_message="Opening browser for Google consent...\nURL: {url}",
            success_message="Authentication complete. You can close this tab.",
        )
        self._store.save(creds.to_json())
        self._cached = creds
        logger.info("google_auth_completed", scopes=list(creds.scopes or []))
        return creds

    def logout(self) -> None:
        """Delete the local token (also revoke access at https://myaccount.google.com/permissions)."""
        self._store.delete()
        self._cached = None

    # ------------------------------------------------------------------ runtime
    def _load(self) -> Credentials:
        raw = self._store.load()
        if raw is None:
            raise AuthenticationRequiredError("No Google token found. Run: python -m email_assistant auth")
        try:
            creds = Credentials.from_authorized_user_info(json.loads(raw))
        except (ValueError, KeyError) as exc:
            raise AuthenticationRequiredError("Stored Google token is corrupt. Re-run the auth command.") from exc
        granted = set(creds.scopes or [])
        missing = set(self._scopes) - granted
        if missing:
            raise AuthenticationRequiredError(
                f"Stored token lacks required scopes {sorted(missing)}. Re-run the auth command."
            )
        return creds

    async def _refresh_once(self, creds: Credentials) -> None:
        await asyncio.to_thread(creds.refresh, Request())

    async def get_credentials(self) -> Credentials:
        """Return valid credentials, refreshing (and persisting) them when necessary."""
        async with self._lock:
            creds = self._cached or self._load()
            if not creds.valid:
                if not (creds.expired and creds.refresh_token):
                    raise AuthenticationRequiredError("Google token is invalid and cannot be refreshed. Re-run auth.")
                try:
                    await self._refresh(creds)
                except RefreshError as exc:
                    self._cached = None
                    raise AuthenticationRequiredError(
                        "Google refresh token was rejected (revoked, expired, or OAuth app in 'Testing' mode "
                        "for >7 days). Re-run the auth command."
                    ) from exc
                self._store.save(creds.to_json())
                logger.info("google_token_refreshed")
            self._cached = creds
            return creds
