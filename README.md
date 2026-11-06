# AI Education Agent

A desktop prototype for instructor-governed AI learning support, built with Python and Tkinter.

## Features

- Student, TA, and Research Team role views
- Tutor Agent for explanations and comprehension checks
- Hint-first Problem-Solving Agent with a four-stage workflow
- Local attachment picker with provider upload disabled pending privacy approval
- Complete worked solutions in Tutor mode; hint-only coaching in Problem-Solving mode
- Direct-answer refusal and TA flag review
- Cross-course telemetry and prompt-governance controls
- Local mock responses or Muse Wonder streaming through Meta Model API
- Rendered Markdown, headings, lists, code, and LaTeX math with responsive word wrapping
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
AI_AGENT_MODE=meta
```

Then run:

```bash
python main.py
```

The status bar will show whether the app is using Meta Model API or local simulation.
Live mode uses `https://api.meta.ai/v1`, the Responses API, and
`muse-wonder-1.0` by default. Files remain local and are not included in Meta
requests because a general provider retention period has not been verified.

Model responses are rendered locally inside a read-only Tkinter text widget.
Markdown formatting is styled rather than shown as raw punctuation; inline and
display LaTeX are converted to local math images. If a formula is malformed, the
expression falls back to readable text instead of breaking the response stream.

## Security

- Never commit `.env` or an API key.
- The repository includes only `.env.example`.
- Provider calls use low reasoning effort and a 2,048-token output cap by default.
- Only the six most recent conversation turns are sent to the provider.
- External file uploads are disabled; selected files remain on the device.
- Direct-answer requests are answered in Tutor mode and refused only in Problem-Solving mode.

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for architecture, token optimization, authorization, evaluation, and pilot guidance.
