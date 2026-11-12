# AI Education Agent repository map

This repository is the system of record for the Tkinter education-agent prototype.

## Non-negotiable constraints

- Preserve the Python/Tkinter desktop UI.
- Keep Mock Mode usable without credentials or network access.
- Never commit `.env`, API keys, student records, or private course material.
- Python, not the model, owns workflow transitions and release gates.
- Problem-Solving mode remains hint-first and must not reveal final assessed answers.
- Tutor claims linked to a concept card must stay within that approved evidence.
- Do not present a model self-critique as proof of factual correctness.

## Architecture map

- `main.py`: Tkinter UI, background model worker, and safe display boundary.
- `learning_core.py`: deterministic Mock Mode and problem-solving state machine.
- `meta_education_service.py`: provider adapter and education-service boundary.
- `education_harness.py`: single-agent Tutor workflow and quality release gate.
- `knowledge_base.py`: small instructor-reviewable concept cards and source URLs.
- `resource_catalog.py`: verified public learning-resource recommendations.
- `evals/`: versioned regression cases based on observed failures.
- `tests/`: deterministic unit tests; no live API calls.
- `docs/AGENT_HARNESS_DESIGN.md`: rationale, workflow, and evaluation plan.
- `backend/math_verification/`: optional FastAPI service that confines engineering
  calculations to an AST-validated SymPy/Pint worker and grounded tool loop;
  the Student UI exposes it as Verified Math Solver.

## Verification commands

```bash
python3 -m unittest discover -s tests -v
python3 run_evals.py
python3 -m py_compile *.py tests/*.py
python3 -m unittest discover -s backend/math_verification/tests -v
```

Live Meta evaluation is a separate, explicitly authorized activity because it may consume
paid tokens and transmit prompts to the provider.

