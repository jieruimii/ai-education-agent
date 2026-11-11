"""Run deterministic, offline tutor regression cases without an API key."""

from __future__ import annotations

import json
from pathlib import Path

from learning_core import MockEducationService, StudentProfile


def main() -> int:
    cases = json.loads((Path(__file__).parent / "evals" / "tutor_cases.json").read_text())
    service = MockEducationService()
    failures = []
    for case in cases:
        profile = StudentProfile()
        answer = service.tutor_reply(profile, case["question"]).text
        lowered = answer.lower()
        missing = [term for term in case["must_include"] if term.lower() not in lowered]
        forbidden = [term for term in case["must_not_include"] if term.lower() in lowered]
        if missing or forbidden or len(answer.split()) < case["minimum_words"]:
            failures.append({
                "id": case["id"],
                "missing": missing,
                "forbidden": forbidden,
                "words": len(answer.split()),
            })
    if failures:
        print(json.dumps({"passed": False, "failures": failures}, indent=2))
        return 1
    print(json.dumps({"passed": True, "cases": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

