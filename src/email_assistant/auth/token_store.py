"""Token persistence: atomic writes, ``0600`` permissions and optional Fernet encryption at rest."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from email_assistant.exceptions import ConfigurationError


class TokenStore:
    """Stores the serialised OAuth credentials JSON.

    Args:
        path: Token file location.
        encryption_key: Optional Fernet key. When provided the file content is encrypted;
            a plaintext file left over from before encryption was enabled is still readable
            and is transparently re-written encrypted on the next save.
    """

    def __init__(self, path: Path, encryption_key: str | None = None) -> None:
        self._path = Path(path)
        try:
            self._fernet = Fernet(encryption_key.encode()) if encryption_key else None
        except (ValueError, TypeError) as exc:
            raise ConfigurationError("TOKEN_ENCRYPTION_KEY is not a valid Fernet key") from exc

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def load(self) -> str | None:
        """Return the stored credentials JSON, or ``None`` if nothing is stored."""
        if not self.exists():
            return None
        data = self._path.read_bytes()
        if self._fernet is not None:
            try:
                return self._fernet.decrypt(data).decode("utf-8")
            except InvalidToken:
                # Fall back to a legacy plaintext token (re-encrypted on next save).
                text = data.decode("utf-8", errors="strict")
                if text.lstrip().startswith("{"):
                    return text
                raise ConfigurationError("Token file cannot be decrypted with TOKEN_ENCRYPTION_KEY") from None
        return data.decode("utf-8")

    def save(self, credentials_json: str) -> None:
        """Atomically write the token file with owner-only permissions."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = credentials_json.encode("utf-8")
        if self._fernet is not None:
            payload = self._fernet.encrypt(payload)
        fd, tmp_name = tempfile.mkstemp(dir=self._path.parent, prefix=".token-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            with contextlib.suppress(OSError):  # not supported on some filesystems (e.g. Windows)
                os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def delete(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()
