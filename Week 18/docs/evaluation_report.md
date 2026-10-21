# Trustworthy AI Guardrails: Evaluation and Risk Report

## 1. Design and architecture

This project is a deterministic retail policy FAQ assistant covering returns, shipping, and refunds. It uses a layered pipeline: PII detection and redaction, injection screening, sensitive-topic handling, local-reference response generation, then output schema, groundedness, and sensitive-information checks. This order prevents known unsafe inputs from reaching generation and gives validators the final candidate before it is returned. Invalid output is replaced with a fixed safe fallback.

The application has no external model or network dependency. It demonstrates guardrail mechanics and makes outcomes reproducible; it is not an LLM integration. See [the architecture diagram](architecture.md).

## 2. Implementation choices and limitations

| Control | Technique and action | Main limitation |
|---|---|---|
| PII | Regular expressions identify email, phone-like numbers, and SSN-shaped values; redact before processing. Audit logs retain types only. | Pattern coverage is incomplete; broad phone matching can mistake order numbers for phone numbers. Names, addresses, and novel identifiers are not detected. |
| Prompt injection | A small set of regular expressions recognizes common override, bypass, and secret-extraction phrases; a match is refused. | Wording changes, indirect instructions, multilingual text, and encoded or split attacks can evade patterns. |
| Sensitive topics | Keyword lists route medical and financial advice to professional referral, self-harm to a brief crisis direction, and violence to refusal. | Context is not understood. Quoted or educational discussion can be misclassified; new phrases and euphemisms may be missed. |
| Schema | Exact required keys, non-empty string answer, allowed category/status enums, and list-of-string sources are checked. | Schema validity does not mean the content is correct or useful. |
| Groundedness | Meaningful answer-token overlap with the supplied reference is used as a simple heuristic. | Token overlap is not entailment: paraphrases can fail while carefully worded unsupported claims can pass. |
| Sensitive output | Email, phone, SSN-like patterns, and password/API-key/secret-token assignments are blocked. | This does not recognize every sensitive-data class or secret format. |

## 3. Evaluation results

The labeled evaluation has 32 cases: 6 PII, 6 injection, 5 sensitive-topic, 3 normal/fallback, 6 schema, 3 groundedness, and 3 sensitive-output cases. The evaluation command writes each input, reference where relevant, expected behavior, actual behavior, and pass/fail to `evaluation/results.json`.

- Overall: **29/32 passed (90.6%)**. Failures are intentionally preserved as measured failure examples rather than hidden by adjusting labels.
- PII: **precision 0.75, recall 1.00** (TP 3, FP 1, FN 0, TN 2). One ten-digit order reference is falsely classified as a phone number.
- Prompt injection: **precision 1.00, recall 0.75** (TP 3, FP 0, FN 1, TN 2). “Disregard your earlier instructions…” is the missed case.
- Sensitive topics: 4/5 passed. The phrase “medical advice” in a policy-disclaimer context triggered a medical refusal instead of answering the shipping question.
- Normal/fallback: 3/3 passed.
- Output checks: schema 6/6, groundedness 3/3, sensitive output 3/3 passed.

These precision/recall values apply only to the small labeled PII and injection samples. The 32 examples are hand-authored, cover a narrow vocabulary, and are not representative estimates of real-world attack rates. Passing these cases does not establish security, semantic factuality, regulatory compliance, or safe behavior under distribution shift.

## 4. Failure analysis

### False positive: order number flagged as a phone number (I05)

The generic phone regex accepts a long digit sequence without requiring separators or a country/area-code pattern. A ten-digit order reference is consequently redacted as PII. This can remove useful context and reduce the assistant's ability to resolve legitimate requests. Improve precision by applying locale-aware phone parsing, excluding known order-number fields, or asking the user to confirm ambiguous values; preserve privacy by redacting only at the boundary where needed.

### False negative: paraphrased instruction override (I10)

The detector recognizes a few exact constructions such as “ignore previous instructions” but not the semantically similar “disregard your earlier instructions.” This could allow a more capable downstream model to follow attacker-controlled instructions. Improve recall with a separately evaluated intent classifier, adversarial paraphrase tests, and separation of user data from system instructions. Do not treat the current regex pass as proof that a request is safe.

### False positive: sensitive-topic keyword in a benign question (I17)

The substring “medical advice” triggers the medical route even though the request asks about shipping. The result is an unnecessary refusal. Add contextual classification and confidence thresholds, with a narrowly scoped clarification path for ambiguous cases, while maintaining stricter handling for direct requests for individualized high-stakes advice.

## 5. Safety and usability trade-offs

Redacting every phone-shaped number is safer when the input may contain a real contact number, but it can erase order references and make support less useful. A more permissive injection filter would let more benign phrasing through, but also expose the system to paraphrased override attempts such as I10. A deployment must balance these risks by measuring false positives and false negatives on its actual domain and escalating uncertain high-impact cases.

## 6. Risk assessment

**Deployment recommendation: suitable only with restrictions as an educational demonstration; not ready as a customer-facing or high-stakes assistant.** It has no LLM, identity system, or tools, and its deterministic FAQ corpus limits the consequence of a successful injection here. The regex and keyword checks, lexical grounding heuristic, narrow PII coverage, and small hand-built evaluation set would be inadequate as sole production controls. A real deployment needs a semantic groundedness method, broader privacy detection, representative adversarial evaluation, secure audit/retention controls, monitoring, and human escalation for high-impact requests.
