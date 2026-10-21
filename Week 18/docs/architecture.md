# Architecture

```mermaid
flowchart LR
    U[User prompt] --> P[PII detector and redactor]
    P --> J{Prompt injection?}
    J -- yes --> R[Refusal]
    J -- no --> S{Sensitive topic classifier}
    S -- flagged --> H[Safe high-stakes response]
    S -- ordinary --> G[Policy FAQ response generator]
    K[(Local policy facts)] --> G
    G --> V[Schema validation]
    V --> Q[Groundedness check]
    Q --> O[Sensitive output scan]
    O --> D{All checks pass?}
    D -- yes --> A[Validated answer]
    D -- no --> F[Safe fallback]
    P -. category only .-> L[Audit metadata]
    J -. decision .-> L
    S -. category .-> L
    V -. errors .-> L
    Q -. errors .-> L
    O -. findings .-> L
```

## Stage decisions

1. **PII scan/redaction runs first** so subsequent stages and the deterministic generator operate on masked text. The demo reports PII kinds, never matched values.
2. **Injection and topic checks precede generation** so flagged requests cannot reach the answer path. Injection attempts are refused. Medical and financial requests receive a professional-referral message; self-harm gets a brief crisis direction; violent requests are refused.
3. **The response generator reads only local policy facts** about returns, shipping, and refunds. Unsupported questions receive a fallback.
4. **Output validators run last** because schema, groundedness, and leakage checks must inspect the final candidate. Any validation error fails closed to the same safe fallback.

The audit metadata is returned to the caller for demonstration and is designed not to contain prompt text or PII values. A real service would also need access controls, retention limits, tamper-evident storage, and monitoring.
