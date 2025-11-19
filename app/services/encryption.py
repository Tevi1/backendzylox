"""AES-256-GCM helpers for encrypting uploaded document blobs."""
from __future__ import annotations

import base64
import os
from functools import lru_cache
from typing import Dict

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptionConfigError(RuntimeError):
    """Raised when the FILE_ENC_KEY_B64 env var is missing or invalid."""


@lru_cache(maxsize=1)
def _load_key() -> bytes:
    key_b64 = os.getenv("FILE_ENC_KEY_B64")
    if not key_b64:
        raise EncryptionConfigError("FILE_ENC_KEY_B64 env var is required for AES-256-GCM.")

    key = base64.b64decode(key_b64)
    if len(key) != 32:
        raise EncryptionConfigError("FILE_ENC_KEY_B64 must decode to exactly 32 bytes.")
    return key


def encrypt_bytes(data: bytes) -> Dict[str, str]:
    """Encrypt bytes and return base64-encoded nonce + ciphertext payload."""

    aes = AESGCM(_load_key())
    nonce = os.urandom(12)
    encrypted = aes.encrypt(nonce, data, None)
    return {
        "nonce": base64.b64encode(nonce).decode("utf-8"),
        "blob": base64.b64encode(encrypted).decode("utf-8"),
    }


def decrypt_bytes(payload: Dict[str, str]) -> bytes:
    """Decrypt bytes previously produced by :func:`encrypt_bytes`."""

    aes = AESGCM(_load_key())
    nonce = base64.b64decode(payload["nonce"])
    ciphertext = base64.b64decode(payload["blob"])
    return aes.decrypt(nonce, ciphertext, None)

