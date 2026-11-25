import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httplib2
import pytest
from cryptography.fernet import Fernet
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from conftest import make_settings
from email_assistant.auth import google_auth
from email_assistant.auth.scopes import GMAIL_READONLY, REQUIRED_SCOPES
from email_assistant.auth.token_store import TokenStore
from email_assistant.core.retry import RetryPolicy
from email_assistant.exceptions import (
    AuthenticationRequiredError, GoogleAPIError, GoogleRateLimitError,
)
from email_assistant.integrations.gmail_client import GmailClient, build_reply_message
from email_assistant.integrations.google_errors import translate_http_error


def http_error(status, reason="", retry_after=None):
    info = {"status": str(status)}
    if retry_after:
        info["retry-after"] = str(retry_after)
    body = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()
    return HttpError(httplib2.Response(info), body)


# ----------------------------------------------------------------- error translation
def test_429_and_quota_403_are_retryable_rate_limits():
    err = translate_http_error(http_error(429, "rateLimitExceeded", retry_after=7))
    assert isinstance(err, GoogleRateLimitError) and err.retryable and err.retry_after == 7
    assert isinstance(translate_http_error(http_error(403, "userRateLimitExceeded")), GoogleRateLimitError)


def test_permission_403_and_404_are_not_retryable_but_5xx_is():
    assert not translate_http_error(http_error(403, "forbidden")).retryable
    assert not translate_http_error(http_error(404, "notFound")).retryable
    assert translate_http_error(http_error(503, "backendError")).retryable


# ----------------------------------------------------------------- Gmail client
def fake_auth():
    auth = MagicMock()
    auth.get_credentials = AsyncMock(return_value=MagicMock())
    return auth


def service_mock():
    return MagicMock()


async def test_create_draft_builds_threaded_message():
    service = service_mock()
    create = service.users.return_value.drafts.return_value.create
    create.return_value.execute.return_value = {"id": "d1"}
    with patch("email_assistant.integrations.base.build", return_value=service):
        gmail = GmailClient(fake_auth(), make_settings())
        draft_id = await gmail.create_draft(
            thread_id="t9", to="alice@acme.com", subject="Re: Hi", body="Thanks!",
            in_reply_to="<abc@acme.com>", references="<root@acme.com>",
        )
    assert draft_id == "d1"
    body = create.call_args.kwargs["body"]["message"]
    assert body["threadId"] == "t9"
    raw = base64.urlsafe_b64decode(body["raw"]).decode()
    assert "In-Reply-To: <abc@acme.com>" in raw and "References: <root@acme.com> <abc@acme.com>" in raw


def test_header_injection_and_bad_addresses_rejected():
    msg = build_reply_message(to="a@b.com", subject="Hi\r\nBcc: evil@x.com", body="x")
    assert msg["Bcc"] is None
    with pytest.raises(ValueError):
        build_reply_message(to="not-an-address", subject="s", body="x")


async def test_get_raw_message_decodes_base64url():
    service = service_mock()
    raw = base64.urlsafe_b64encode(b"Subject: x\r\n\r\nbody").decode().rstrip("=")
    service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "m1", "threadId": "t1", "raw": raw, "labelIds": ["INBOX"], "internalDate": "1741000000000",
    }
    with patch("email_assistant.integrations.base.build", return_value=service):
        msg = await GmailClient(fake_auth(), make_settings()).get_raw_message("m1")
    assert msg.raw.startswith(b"Subject: x") and msg.internal_date.year == 2025


async def test_transient_errors_are_retried_and_permanent_ones_are_not():
    service = service_mock()
    send = service.users.return_value.drafts.return_value.send
    send.return_value.execute.side_effect = [http_error(503, "backendError"), {"id": "sent", "threadId": "t"}]
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    policy = RetryPolicy(max_attempts=3, base_delay=0.01, jitter=False, sleep=fake_sleep)
    with patch("email_assistant.integrations.base.build", return_value=service):
        gmail = GmailClient(fake_auth(), make_settings(), retry_policy=policy)
        assert (await gmail.send_draft("d1"))["message_id"] == "sent"
        assert len(sleeps) == 1
        send.return_value.execute.side_effect = [http_error(404, "notFound")]
        with pytest.raises(GoogleAPIError) as exc:
            await gmail.send_draft("missing")
    assert exc.value.status == 404 and len(sleeps) == 1


# ----------------------------------------------------------------- OAuth / token handling
class FakeCreds:
    def __init__(self, valid=True, expired=False, scopes=REQUIRED_SCOPES, refresh_error=None):
        self.valid, self.expired, self.refresh_token = valid, expired, "refresh"
        self.scopes, self.refresh_error, self.refreshed = list(scopes), refresh_error, False

    def refresh(self, _request):
        if self.refresh_error:
            raise self.refresh_error
        self.valid, self.expired, self.refreshed = True, False, True

    def to_json(self):
        return json.dumps({"token": "refreshed", "scopes": self.scopes})


def manager(tmp_path, monkeypatch, creds, stored=True):
    store = TokenStore(tmp_path / "token.json")
    if stored:
        store.save(json.dumps({"token": "old"}))
    cls = MagicMock()
    cls.from_authorized_user_info.return_value = creds
    monkeypatch.setattr(google_auth, "Credentials", cls)
    return google_auth.GoogleAuthManager(make_settings(), token_store=store), store


async def test_missing_token_requires_auth(tmp_path, monkeypatch):
    mgr, _ = manager(tmp_path, monkeypatch, FakeCreds(), stored=False)
    with pytest.raises(AuthenticationRequiredError):
        await mgr.get_credentials()


async def test_valid_credentials_returned_without_refresh(tmp_path, monkeypatch):
    creds = FakeCreds()
    mgr, _ = manager(tmp_path, monkeypatch, creds)
    assert await mgr.get_credentials() is creds and not creds.refreshed


async def test_expired_token_is_refreshed_and_persisted(tmp_path, monkeypatch):
    creds = FakeCreds(valid=False, expired=True)
    mgr, store = manager(tmp_path, monkeypatch, creds)
    await mgr.get_credentials()
    assert creds.refreshed and json.loads(store.load())["token"] == "refreshed"


async def test_revoked_refresh_token_requires_reauth(tmp_path, monkeypatch):
    creds = FakeCreds(valid=False, expired=True, refresh_error=RefreshError("invalid_grant"))
    mgr, _ = manager(tmp_path, monkeypatch, creds)
    with pytest.raises(AuthenticationRequiredError):
        await mgr.get_credentials()


async def test_missing_scopes_are_rejected(tmp_path, monkeypatch):
    mgr, _ = manager(tmp_path, monkeypatch, FakeCreds(scopes=[GMAIL_READONLY]))
    with pytest.raises(AuthenticationRequiredError):
        await mgr.get_credentials()


def test_token_store_encrypts_at_rest_and_is_atomic(tmp_path):
    key = Fernet.generate_key().decode()
    store = TokenStore(tmp_path / "t.json", key)
    store.save('{"secret": "abc"}')
    assert b"secret" not in (tmp_path / "t.json").read_bytes()
    assert json.loads(store.load()) == {"secret": "abc"}
    assert not list(tmp_path.glob(".token-*"))
