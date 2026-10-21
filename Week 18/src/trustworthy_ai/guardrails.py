"""Small, auditable guardrails for a grounded policy FAQ assistant.

This demo intentionally uses deterministic rules. It is not a substitute for
context-aware classifiers, a production PII detector, or human review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)"),
    "id_number": re.compile(r"\b(?:SSN\s*[:#]?\s*)?\d{3}-\d{2}-\d{4}\b", re.I),
}

INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b", re.I),
    re.compile(r"\b(?:reveal|print|show|expose|leak)\b.{0,50}\b(?:system prompt|secret|confidential|hidden instructions?)\b", re.I),
    re.compile(r"\b(?:bypass|override|disable)\b.{0,40}\b(?:safety|guardrails?|restrictions?|policy)\b", re.I),
    re.compile(r"\bdeveloper message\b.{0,50}\b(?:ignore|override|replace)\b", re.I),
]

SENSITIVE_TERMS = {
    "medical": ("diagnose", "dosage", "prescription", "medical advice"),
    "self_harm": ("suicide", "kill myself", "self-harm", "hurt myself"),
    "violence": ("build a bomb", "make a weapon", "kill someone", "attack someone"),
    "financial": ("invest all my savings", "guaranteed return", "buy this stock"),
}

FACTS = {
    "returns": "Items can be returned within 30 days of delivery if unused and in their original packaging.",
    "shipping": "Standard shipping usually takes 3 to 5 business days. Tracking is sent after dispatch.",
    "refunds": "Approved refunds are returned to the original payment method within 5 to 10 business days.",
}

SAFE_FALLBACK = "I could not safely verify that answer from the available policy information. Please contact support."


@dataclass
class Response:
    answer: str
    category: str
    status: str
    sources: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_pii(text: str) -> list[tuple[str, str]]:
    """Return (kind, matched value) pairs for supported PII patterns."""
    hits = []
    for kind, pattern in PII_PATTERNS.items():
        hits.extend((kind, match.group(0)) for match in pattern.finditer(text))
    return hits


def redact_pii(text: str) -> str:
    for kind, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{kind.upper()}]", text)
    return text


def detect_injection(text: str) -> list[str]:
    return [pattern.pattern for pattern in INJECTION_PATTERNS if pattern.search(text)]


def classify_sensitive_topic(text: str) -> str | None:
    lowered = text.casefold()
    for category, terms in SENSITIVE_TERMS.items():
        if any(term in lowered for term in terms):
            return category
    return None


def validate_response(value: Any, reference: str) -> tuple[Response, list[str]]:
    """Validate structure, groundedness, and leakage; fail closed to fallback."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return Response(SAFE_FALLBACK, "general", "safe_fallback", []), ["schema:type"]
    required = {"answer", "category", "status", "sources"}
    if set(value) != required:
        errors.append("schema:keys")
    if not isinstance(value.get("answer"), str) or not value.get("answer", "").strip():
        errors.append("schema:answer")
    if value.get("category") not in {"returns", "shipping", "refunds", "general", "medical", "self_harm", "violence", "financial"}:
        errors.append("schema:category")
    if value.get("status") not in {"answered", "refused", "safe_fallback"}:
        errors.append("schema:status")
    if not isinstance(value.get("sources"), list) or any(not isinstance(s, str) for s in value.get("sources", [])):
        errors.append("schema:sources")

    answer = value.get("answer", "") if isinstance(value.get("answer"), str) else ""
    if detect_pii(answer):
        errors.append("output:pii")
    if re.search(r"\b(?:api[_ -]?key|password|secret token)\s*[:=]\s*\S+", answer, re.I):
        errors.append("output:sensitive")
    # A transparent lexical entailment proxy: meaningful answer terms must
    # appear in the supplied reference, allowing a small amount of filler.
    if reference and answer not in {SAFE_FALLBACK}:
        content = {t.casefold() for t in re.findall(r"[a-zA-Z]{4,}", answer)}
        supported = {t.casefold() for t in re.findall(r"[a-zA-Z]{4,}", reference)}
        unsupported = content - supported - {"please", "contact", "support", "available", "information", "policy", "could", "safely", "verify", "that", "from", "the"}
        if len(unsupported) > max(2, int(len(content) * 0.35)):
            errors.append("groundedness:low_overlap")

    if errors:
        return Response(SAFE_FALLBACK, "general", "safe_fallback", []), errors
    return Response(answer, value["category"], value["status"], value["sources"]), []


class GuardrailEngine:
    """Policy FAQ pipeline: screen input, answer from a tiny local corpus, validate output."""

    def respond(self, user_input: str) -> tuple[Response, dict[str, Any]]:
        audit: dict[str, Any] = {"pii_types": [], "injection": False, "sensitive_topic": None, "output_errors": []}
        pii = detect_pii(user_input)
        audit["pii_types"] = sorted({kind for kind, _ in pii})
        clean = redact_pii(user_input)
        if detect_injection(clean):
            audit["injection"] = True
            return Response("I can help with policy questions, but I cannot follow requests to bypass safeguards or reveal hidden instructions.", "general", "refused", []), audit
        sensitive = classify_sensitive_topic(clean)
        if sensitive:
            audit["sensitive_topic"] = sensitive
            message = ("I cannot provide crisis support here. If you may be in immediate danger, contact local emergency services or a trusted person now."
                       if sensitive == "self_harm" else
                       "I cannot provide individualized high stakes advice. Please consult a qualified professional."
                       if sensitive in {"medical", "financial"} else
                       "I cannot assist with harming people or making weapons.")
            return Response(message, sensitive, "refused", []), audit

        terms = {"returns": ("return", "returns"), "shipping": ("shipping", "ship", "delivery"), "refunds": ("refund", "refunds")}
        key = next((k for k, variants in terms.items() if any(term in clean.casefold() for term in variants)), None)
        if key:
            candidate = {"answer": FACTS[key], "category": key, "status": "answered", "sources": [key]}
            response, errors = validate_response(candidate, FACTS[key])
        else:
            candidate = {"answer": SAFE_FALLBACK, "category": "general", "status": "safe_fallback", "sources": []}
            response, errors = validate_response(candidate, "")
        audit["output_errors"] = errors
        return response, audit

