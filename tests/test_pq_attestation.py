# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

import base64

import pytest

pytest.importorskip("dilithium_py")

from aiaccel import pq_attestation as pqa

RESULTS = {"study": "cnn-sweep", "best_value": 0.9123, "params": {"lr": 0.01, "batch": 64}}


def test_build_and_verify_roundtrip() -> None:
    payload = pqa.canonicalize(RESULTS)
    public_key, secret_key = pqa.generate_keypair()
    att = pqa.build_attestation(payload, public_key, secret_key)
    assert att["scheme"] == "ML-DSA-65"
    assert att["issuer"] == "Quantova"
    assert pqa.verify_attestation(payload, att) is True
    assert pqa.verify_attestation(payload, att, public_key) is True


def test_tampered_results_rejected() -> None:
    payload = pqa.canonicalize(RESULTS)
    public_key, secret_key = pqa.generate_keypair()
    att = pqa.build_attestation(payload, public_key, secret_key)
    altered = pqa.canonicalize({**RESULTS, "best_value": 0.0})
    assert pqa.verify_attestation(altered, att) is False


def test_wrong_signer_rejected_when_pinned() -> None:
    payload = pqa.canonicalize(RESULTS)
    public_key, secret_key = pqa.generate_keypair()
    other_public_key, _ = pqa.generate_keypair()
    att = pqa.build_attestation(payload, public_key, secret_key)
    assert pqa.verify_attestation(payload, att, other_public_key) is False


def test_forged_signature_rejected() -> None:
    payload = pqa.canonicalize(RESULTS)
    public_key, secret_key = pqa.generate_keypair()
    att = pqa.build_attestation(payload, public_key, secret_key)
    att["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    assert pqa.verify_attestation(payload, att) is False


def test_malformed_field_rejected() -> None:
    payload = pqa.canonicalize(RESULTS)
    public_key, secret_key = pqa.generate_keypair()
    att = pqa.build_attestation(payload, public_key, secret_key)
    att["public_key"] = "!!!not base64!!!"
    assert pqa.verify_attestation(payload, att) is False


def test_canonicalize_is_order_independent() -> None:
    assert pqa.canonicalize({"a": 1, "b": 2}) == pqa.canonicalize({"b": 2, "a": 1})
