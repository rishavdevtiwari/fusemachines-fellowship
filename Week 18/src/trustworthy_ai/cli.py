"""Interactive command line demonstration."""

import json
from .guardrails import GuardrailEngine


def main() -> None:
    engine = GuardrailEngine()
    print("Policy FAQ demo. Ask about returns, shipping, or refunds; type 'quit' to exit.")
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().casefold() in {"quit", "exit"}:
            break
        response, audit = engine.respond(prompt)
        print(json.dumps({"response": response.to_dict(), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
