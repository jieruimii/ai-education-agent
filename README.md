# AI Education Agent

A desktop prototype for instructor-governed AI learning support, built with Python and Tkinter.

## Features

- Student, TA, and Research Team role views
- Tutor Agent for explanations and comprehension checks
- Release-gated Tutor harness: analyze, ground, draft, evaluate, revise once, then display
- Instructor-reviewable concept evidence for physics, thermodynamics, mathematics, chemistry, computer science, literature, and history
- Question-first topic routing; no Active Course selector can bias an unrelated question
- Bounded Wikipedia research in Meta Mode, followed by evidence-bounded drafting and release checks
- Visible action trace for classification, retrieval, quality checks, revision, and release
- Hint-first Problem-Solving Agent with a four-stage workflow
- Local attachment picker with provider upload disabled pending privacy approval
- Complete worked solutions in Tutor mode; hint-only coaching in Problem-Solving mode
- Direct-answer refusal and TA flag review
- Cross-course telemetry and prompt-governance controls
- Local mock responses or Muse Wonder streaming through Meta Model API
- Rendered Markdown, headings, lists, code, and LaTeX math with responsive word wrapping
- Light and dark modes

## Answer-quality harness

Live Tutor answers are no longer displayed directly from a single model call. Python builds a
question contract, retrieves exact approved concept evidence, requests a structured explanation,
checks it with a deterministic rubric and a schema-checked critic, and revises once when needed.
Only the checked final text reaches the Tkinter chat.

Reviewed local evidence now suppresses unnecessary web searches. If the Muse critic returns no
visible text, the harness retries it once with a larger allowance and then safely falls back to
the deterministic release rubric instead of aborting the student answer. The UI reports separate
automated checks for correctness/evidence agreement, grounding, directness, tool precision, and
reliability. These are harness diagnostics, not official benchmark or ground-truth scores.

Run the offline quality regression suite:

```bash
python3 run_evals.py
```

See [docs/AGENT_HARNESS_DESIGN.md](docs/AGENT_HARNESS_DESIGN.md) for the workflow, limitations,
and instructor-evaluation plan. See [docs/EVALUATION_STRATEGY.md](docs/EVALUATION_STRATEGY.md)
for the HLE, MMLU, GAIA, BFCL, SWE-bench, GDPval, and AA-Briefcase evaluation map.

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
ENABLE_WEB_RESEARCH=true
```

Then run:

```bash
python main.py
```

The status bar will show whether the app is using Meta Model API or local simulation.
Live mode uses `https://api.meta.ai/v1`, the Responses API, and
`muse-wonder-1.0` by default. Files remain local and are not included in Meta
requests because a general provider retention period has not been verified.
The displayed Agent activity panel reports observable actions and quality checks;
it does not expose private model chain-of-thought.

Model responses are rendered locally inside a read-only Tkinter text widget.
Markdown formatting is styled rather than shown as raw punctuation; inline and
display LaTeX are converted to local math images. If a formula is malformed, the
expression falls back to readable text instead of breaking the response stream.

## Security

- Never commit `.env` or an API key.
- The repository includes only `.env.example`.
- Provider calls use low reasoning effort and a 2,048-token output cap by default.
- The legacy provider seam retains six turns; the harnessed Tutor path sends at most four prior turns.
- Harnessed Tutor mode normally uses two model calls, with one bounded critic retry or one revision when needed.
- External file uploads are disabled; selected files remain on the device.
- Direct-answer requests are answered in Tutor mode and refused only in Problem-Solving mode.

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for architecture, token optimization, authorization, evaluation, and pilot guidance.
