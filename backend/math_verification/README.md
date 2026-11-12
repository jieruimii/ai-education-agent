# Deterministic Math Verification Backend

This FastAPI service keeps engineering mathematics outside the language model.
The model may parse a problem and write a constrained SymPy/Pint program, but a
separate worker validates and executes that program before any answer is released.

The same engine is integrated into the packaged Tkinter application as
**Verified Math Solver**. The standalone FastAPI boundary remains available for
future web or service integrations.

## Run locally

From the repository root:

```bash
python3 -m venv .math-backend-venv
source .math-backend-venv/bin/activate
pip install -r backend/math_verification/requirements.txt
export MATH_PROVIDER=openai
export MATH_MODEL=gpt-5.2
export OPENAI_API_KEY='your-provider-key'
export BACKEND_API_KEY='choose-a-long-random-local-secret'
uvicorn backend.math_verification.main:app --host 127.0.0.1 --port 8000
```

For an OpenAI-compatible service, set `MATH_PROVIDER=openai-compatible`,
`MATH_BASE_URL`, `MATH_MODEL`, and `MATH_API_KEY`. For Anthropic, set
`MATH_PROVIDER=anthropic`, `MATH_MODEL`, and `ANTHROPIC_API_KEY`.

Retrieve and submit the included RLC example:

```bash
curl -H "X-API-Key: $BACKEND_API_KEY" http://127.0.0.1:8000/api/examples/rlc
curl -X POST http://127.0.0.1:8000/api/solve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $BACKEND_API_KEY" \
  --data-binary @rlc-request.json
```

## Security boundary

The local worker uses an AST allowlist, isolated Python mode, a temporary working
directory, a minimal environment, output limits, wall-clock timeout, and POSIX
resource limits. Generated code cannot import files, networking, subprocess, or
dynamic-execution modules. This is defense in depth, not a kernel isolation
boundary. Production deployment must additionally use an unprivileged container
or microVM with no network, a read-only root filesystem, a writable temporary
mount only, dropped Linux capabilities, process/memory/CPU limits, and request
rate limits at the ingress.

## Verification

```bash
python -m unittest discover -s backend/math_verification/tests -v
```

No live provider call is required by the test suite.
