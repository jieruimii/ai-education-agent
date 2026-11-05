# AI Education Agent

A desktop prototype for instructor-governed AI learning support, built with Python and Tkinter.

## Features

- Student, TA, and Research Team role views
- Tutor Agent for explanations and comprehension checks
- Hint-first Problem-Solving Agent with a four-stage workflow
- Direct-answer refusal and TA flag review
- Cross-course telemetry and prompt-governance controls
- Local mock responses or streaming through Meta's official Llama API client
- Light and dark modes

## Run in local mock mode

```bash
python3 main.py
```

Mock mode requires no API key, network connection, or third-party package.

## Enable Meta Model API streaming

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add a valid key to `.env`:

```text
MODEL_API_KEY=your_key_here
```

Then run:

```bash
python main.py
```

The status bar will show whether the app is using Meta Model API or local simulation.

## Security

- Never commit `.env` or an API key.
- The repository includes only `.env.example`.
- Provider calls use low reasoning effort and a 1,024-token output cap by default.
- Only the six most recent conversation turns are sent to the provider.
- Direct-answer and prompt-bypass requests are refused locally before a model call.

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for architecture, token optimization, authorization, evaluation, and pilot guidance.
