# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

"""Optional post quantum attestation of experiment results.

Signs a canonical results manifest, such as a study name with its best value
and parameters, using FIPS 204 ML-DSA so a reader can confirm the reported
results are authentic and have not been altered, with a signature scheme that
stays safe once large quantum computers exist.

The attestation is self contained. It carries the public key and the signature,
so a verifier needs only the attestation and the results. Integrity is always
covered by the payload digest and authenticity is covered when the verifier
pins the expected public key.

The scheme is provided by the optional ``pq`` extra. Install it with
``pip install aiaccel[pq]``.
"""

from __future__ import annotations

from typing import Any, TypedDict

import base64
import hashlib
import json

SCHEME = "ML-DSA-65"
ISSUER = "Quantova"


class Attestation(TypedDict):
    """A self contained post quantum attestation over a payload."""

    scheme: str
    issuer: str
    payload_sha256: str
    public_key: str
    signature: str


def _backend() -> Any:
    """Return the ML-DSA backend or raise if the pq extra is not installed."""
    try:
        from dilithium_py.ml_dsa import ML_DSA_65  # type: ignore[import-untyped]
    except ImportError as error:
        msg = "Post quantum attestation needs the pq extra. Install it with pip install aiaccel[pq]."
        raise RuntimeError(msg) from error
    return ML_DSA_65


def canonicalize(results: dict[str, Any]) -> bytes:
    """Return the canonical byte encoding an attestation is computed over."""
    return json.dumps(results, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def generate_keypair() -> tuple[bytes, bytes]:
    """Return a fresh (public_key, secret_key) pair for the signer."""
    public_key, secret_key = _backend().keygen()
    return public_key, secret_key


def build_attestation(payload: bytes, public_key: bytes, secret_key: bytes) -> Attestation:
    """Return a self contained attestation over payload."""
    signature = _backend().sign(secret_key, payload, deterministic=True)
    return Attestation(
        scheme=SCHEME,
        issuer=ISSUER,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        public_key=base64.b64encode(public_key).decode("ascii"),
        signature=base64.b64encode(signature).decode("ascii"),
    )


def verify_attestation(payload: bytes, attestation: Attestation, expected_public_key: bytes | None = None) -> bool:
    """Return True when the attestation matches payload.

    The payload digest is always checked. When expected_public_key is given the
    signer is pinned too, so a valid signature made under a different key is
    rejected.
    """
    if attestation["payload_sha256"] != hashlib.sha256(payload).hexdigest():
        return False
    try:
        public_key = base64.b64decode(attestation["public_key"], validate=True)
        signature = base64.b64decode(attestation["signature"], validate=True)
    except ValueError:
        return False
    if expected_public_key is not None and public_key != expected_public_key:
        return False
    result: bool = _backend().verify(public_key, payload, signature)
    return result
