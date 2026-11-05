# AI Education Agent — Build and Delivery Guide

This guide takes the included stakeholder demo into a safe, token-efficient pilot. It runs in mock mode without a key and can stream real responses through Meta's official Llama API client when configured.

## 1. Run the local demo

### Prerequisites

- Python 3.10 or newer
- Tk support (included with the standard Python installers for Windows and macOS)
- No API key for mock mode; a Meta Llama API key for live mode

### Start it

1. Put `main.py` in a folder you control.
2. Open Terminal, PowerShell, or Cursor's terminal in that folder.
3. Run:

   ```bash
   python main.py
   ```

   If your computer uses `python3`, run `python3 main.py` instead.

4. Start the demo in **Student** view. Send one message to the Tutor Agent.
5. Open **Problem Solver**, ask for a final answer, and show the refusal plus workflow stages.
6. Open **TA** and show the new interaction/flag in the review log.
7. Open **Research Team** and show aggregate telemetry and prompt governance.
8. Toggle dark mode.

The app keeps conversations and demo records in memory. Closing it resets them.

### Enable Meta Llama streaming

The credential previously pasted into chat should be treated as exposed. Revoke it in the Meta developer console and create a replacement before continuing.

1. Create a virtual environment in the same folder as `main.py`:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate it with `.venv\Scripts\activate`.

2. Install the official client and environment-file loader:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`.
4. Put the **newly rotated** key after `LLAMA_API_KEY=` in `.env`.
5. Keep `AI_AGENT_MODE=auto` and start the app:

   ```bash
   python main.py
   ```

6. Confirm the bottom status bar says `Meta Llama connected` before sending a real request.
7. To return to the free local demo, set `AI_AGENT_MODE=mock` and restart.

The `.gitignore` excludes `.env`. Never put the key directly in `main.py`, a screenshot, a commit, or a stakeholder build.

## 2. What the prototype proves

- Role-based product surfaces for Student, TA, and Research Team
- Two distinct pedagogical agents, not just two chat labels
- A visible problem-solving policy: Diagnose → Design Hint → Gather Report → Iterate
- Local streaming behavior without an external model
- Direct-answer and prompt-bypass flagging
- TA review workflow
- Cross-course aggregate telemetry
- Version-ready prompt settings with basic validation
- Light and dark presentation modes

It does **not** yet prove model quality, real authentication, true authorization, retrieval quality, persistent storage, or FERPA compliance. Those are production workstreams, not UI features.

## 3. Recommended production architecture

Keep the desktop client thin. Do not call the model provider directly from Tkinter because that exposes the API key and makes access control difficult.

```text
Tkinter client
    |
    | authenticated HTTPS
    v
Application API
    |-- role and course authorization
    |-- conversation/session service
    |-- policy engine and answer-leakage checks
    |-- retrieval service (course resources only)
    |-- provider adapter (Meta/Llama or another model)
    |-- usage/cost meter
    |-- audit events
    |
    +--> PostgreSQL: users, courses, sessions, flags, prompt versions
    +--> Object storage: approved course documents
    +--> Vector index: chunk embeddings and metadata
```

The backend owns the provider key. The desktop app receives only the response stream and UI-safe metadata.

## 4. Build in controlled phases

### Phase A — define the pilot before adding an API

1. Pick one course and one module.
2. Name the content owner (usually the instructor).
3. Decide which resources are approved: syllabus, lecture notes, textbook excerpts with permission, worked examples, and rubric.
4. Define the Tutor Agent's allowed actions.
5. Define the Problem Solver's prohibited behavior, especially graded-answer disclosure.
6. Write five normal, five ambiguous, and five adversarial test conversations.
7. Choose success measures: learning gain, useful-hint rate, citation correctness, answer leakage, latency, cost per resolved question, and TA workload.

### Phase B — extract the UI from the service layer

1. Keep the current Tkinter views.
2. Create an `AgentService` interface with a method such as `stream_reply(request)`.
3. Move the current predefined responses into `MockAgentService`.
4. Make the UI depend only on `AgentService`, not on provider-specific code.
5. Run all stakeholder demos with `MockAgentService` until the product behavior is stable.

This separation lets the team change model vendors without rewriting the application.

### Phase C — add the backend

1. Create a small FastAPI service in a separate backend folder.
2. Add `/sessions`, `/messages/stream`, `/flags`, `/telemetry`, and `/prompt-versions` endpoints.
3. Add server-side role checks to every endpoint. Hiding a tab is not authorization.
4. Use server-sent events or newline-delimited JSON to stream responses.
5. Store secrets in environment variables or a managed secret store.
6. Never commit `.env`, credentials, raw student exports, or provider keys.
7. Add request IDs so a UI event, retrieval trace, model call, and audit record can be correlated.

### Phase D — connect the Meta/Llama provider

Use the exact endpoint and request schema shown in the provider documentation for your account; do not assume every Meta or Llama host uses the same format.

1. Create a `MetaLlamaAdapter` on the backend only.
2. Read `LLAMA_API_KEY`, `LLAMA_MODEL`, and other provider configuration from the backend environment.
3. Set strict connect/read timeouts.
4. Request streaming if the provider supports it.
5. Normalize provider events into your own internal event format:

   ```json
   {"type": "token", "text": "..."}
   {"type": "citation", "source_id": "lecture-04", "page": 7}
   {"type": "usage", "input_tokens": 820, "output_tokens": 210}
   {"type": "done", "request_id": "..."}
   ```

6. Map timeouts, rate limits, and safety rejections to friendly client messages.
7. Log counts, timing, route, policy version, and retrieval IDs—not API keys or unnecessary student text.
8. Keep the mock adapter selectable through a configuration flag for demos and tests.

### Phase E — add instructor-governed retrieval

1. Ingest only instructor-approved files.
2. Extract text and preserve course, module, document, section, page, and access metadata.
3. Chunk by meaningful section rather than a fixed character count alone.
4. Create embeddings once during ingestion, not on every question.
5. Filter retrieval by the student's authorized course before similarity search.
6. Retrieve a small number of candidates, rerank them, and pass only the best evidence to the model.
7. Require inline citations that map to actual retrieved chunks.
8. If evidence is weak, respond with uncertainty or escalate instead of filling the gap from model memory.

## 5. Reduce token consumption deliberately

Token usage is usually driven by repeatedly sending long instructions, complete chat history, and too many retrieved chunks. Treat the context window as a budget.

### Use a request budget

For every request, allocate a maximum across:

- System and policy instructions
- Student's current question
- Short conversation state
- Retrieved evidence
- Output allowance

Reject or compress the request before it exceeds that budget. Record estimated and actual token counts.

### Do not resend the whole conversation

Maintain structured state instead:

```json
{
  "goal": "find reaction forces",
  "knowns": ["beam length 4 m", "point load 10 kN"],
  "student_attempt": "sum of vertical forces written",
  "misconception": "moment arm measured from wrong support",
  "workflow_stage": "gather_report",
  "open_question": "which point should define moments?"
}
```

Send the last two to four turns plus this compact state. Summarize only when the state materially changes.

### Retrieve less, then retrieve again

- Start with three or four small, high-confidence chunks.
- Filter by course/module/document before vector search.
- Deduplicate overlapping chunks.
- Ask a clarifying question before retrieving broadly.
- Cache retrieval for repeated or semantically equivalent questions.
- Never insert an entire PDF into each prompt.

### Route work to the cheapest adequate operation

- Use local rules for direct-answer detection, empty messages, workflow transitions, and UI copy.
- Use embeddings only for retrieval, not for every classification.
- Use a smaller model for intent classification and conversation summarization.
- Use the stronger model only for pedagogical generation that needs it.
- Cache stable explanations, but do not cache personalized or sensitive content across students.

### Limit output without damaging learning

- Ask for one explanation, one hint, or one comprehension question per turn.
- Set a conservative output-token ceiling.
- Stop generation after the requested pedagogical unit.
- Let the student request expansion.
- Prefer structured state fields over asking the model to repeat its reasoning history.

### Measure the right unit

Track:

- Input, output, cached, and total tokens per request
- Tokens per resolved student question
- Cost per active student and per course week
- Retrieval chunks used versus cited
- Latency to first token and completion
- Learning outcome and hint usefulness alongside cost

The objective is not the fewest tokens; it is the lowest cost for a safe, useful learning outcome.

## 6. Enforce role and data boundaries

Use a server-side authorization matrix:

| Capability | Student | TA | Research Team |
|---|---:|---:|---:|
| Use assigned-course agents | Yes | Optional | No |
| Read own session | Yes | Yes, assigned sections | Pseudonymized only |
| Read flagged reports | No | Yes, assigned sections | Aggregate counts |
| Read cross-course telemetry | No | No | Yes, aggregate |
| Edit active prompt | No | No | Yes, approval required |
| Export identifiable logs | No | Restricted | No by default |

Add course membership checks and least-privilege queries. A role name alone is insufficient; a TA should not automatically see every course.

## 7. Guard against direct-answer leakage

Use multiple layers:

1. Detect assessment context and direct-answer language before generation.
2. Give the Problem Solver a short, explicit hint-first policy.
3. Use workflow state so it cannot skip directly to a solution.
4. Scan generated output for final-answer patterns before returning it.
5. Replace unsafe output with a refusal and the next smallest hint.
6. Flag uncertain cases for TA review.
7. Maintain an adversarial test set: jailbreaks, fake instructor claims, role-play, answer extraction, and repeated hint escalation.

Do not store or display hidden chain-of-thought. Store concise pedagogical state and observable actions instead.

## 8. Test before a real class pilot

### Automated tests

- Role checks deny every unauthorized endpoint.
- Course filters prevent cross-course retrieval.
- Direct-answer prompts produce a refusal and a useful next step.
- Provider timeout and rate-limit paths recover cleanly.
- Streaming can be cancelled without corrupting session state.
- Every citation points to a retrieved, authorized source.
- Prompt changes create immutable versions and audit events.

### Evaluation set

Create a versioned set of representative questions with instructor-reviewed expectations. Score separately:

- Retrieval relevance
- Citation correctness and faithfulness
- Conceptual correctness
- Pedagogical usefulness
- Answer leakage
- Bias/accessibility concerns
- Latency and cost

Do not collapse these into a single “accuracy” score.

### Human pilot

1. Run internally with synthetic students.
2. Run with instructors and TAs.
3. Conduct a small, consented pilot in one module.
4. Compare learning outcomes with a defined baseline.
5. Review flags and failure cases weekly.
6. Expand only after predefined safety and quality thresholds are met.

## 9. Package the desktop app for stakeholders

After testing the plain Python version, package it so stakeholders do not need Python:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
python -m pip install pyinstaller
pyinstaller --onefile --windowed --name AI-Education-Agent main.py
```

The result appears in `dist/`. Build separately on each target operating system. Code-sign and notarize production distributions where required. Do not package an API key into the executable.

## 10. Stakeholder demo script

1. **Student / Tutor:** “Explain why moments can be taken about any point.” Show streaming and the comprehension question.
2. **Student / Problem Solver:** “Give me the final answer.” Show the explicit refusal and hint workflow.
3. **TA:** Show the interaction as a flagged event and mark it reviewed.
4. **Research Team:** Show aggregate course usage and the prompt policy—not student-identifiable text.
5. **Token story:** Point to “0 external tokens” in demo mode, then explain the production budget, compact state, selective retrieval, caching, and model routing.

## Definition of done for the first production pilot

- One approved course module is indexed and permission-filtered.
- Student, TA, and research permissions are enforced server-side.
- Model keys exist only on the backend.
- Every response records model, prompt version, retrieval IDs, latency, and token usage.
- Direct-answer leakage tests meet the instructor-approved threshold.
- Citations are traceable to approved sources.
- The UI handles timeout, cancellation, retry, and provider failure.
- A retention policy and consent/notice language are approved.
- Learning, safety, cost, and staff-workload metrics have owners and review cadence.

