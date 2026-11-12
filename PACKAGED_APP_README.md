# AI Education Agent for macOS

This package contains the standalone `AI_Education_Agent.app` desktop prototype
for Professor Penkova. It was built for Apple Silicon Macs and does not require
Python to be installed.

The interface is intentionally student-only; the former TA review and Research
Team dashboards are not included.

## Start in Mock Mode

1. Unzip the package.
2. Move `AI_Education_Agent.app` to `Applications` if desired.
3. Double-click the application.

Mock Mode is the safe default when no Meta configuration is present. It makes
no external model request. The Tutor Agent includes a detailed offline
throttling explanation for the meeting regression case. Problem-Solving mode,
student profiles, verified recommendations, Markdown, and LaTeX remain available.
The **Verified Math Solver** includes a reviewed local RLC demonstration in Mock
Mode. In Meta Mode it requires a Muse function call before executing
AST-validated SymPy/Pint code and releasing a numerical result.

## Tutor answer-quality harness

In Meta Mode, Tutor responses pass through a single-agent workflow:

`analyze → retrieve approved evidence → draft → evaluate → revise once → release`

Unchecked provider text is not displayed. A passing answer normally uses two
model calls; a rejected draft may use a third revision call. This improves
answer structure and reduces unsupported claims, but it cannot guarantee factual
correctness. The instructor-reviewable local cards currently cover Newton's
first law, throttling, integration by substitution, SN1 versus SN2, supervised
versus unsupervised learning, dramatic irony in *Oedipus Rex*, and causes of the
French Revolution. Other concepts still need instructor-approved evidence.

When a local card matches, the app skips unnecessary web research. If the Muse
critic returns no visible text, it retries once and then uses the deterministic
rubric and reviewed-card fallback rather than aborting. Responses include
separate automated checks for correctness/evidence agreement, grounding,
directness, tool precision, and reliability; these are not official benchmark
or ground-truth scores.

The Student Profile intentionally has no Active Course selector. The current
question controls topic detection, evidence retrieval, and resource ranking so a
previous course choice cannot contaminate an unrelated answer.

## macOS Gatekeeper

This prototype is not signed with an Apple Developer ID and is not notarized.
If macOS blocks the first launch, Control-click the app, choose **Open**, and
confirm. You may instead need **System Settings → Privacy & Security → Open Anyway**.

## Enable Meta Mode with your own key

The app never stores a key inside its application bundle. It reads:

`~/Library/Application Support/AI Education Agent/config.env`

Create it in Terminal:

```bash
mkdir -p "$HOME/Library/Application Support/AI Education Agent"
umask 077
touch "$HOME/Library/Application Support/AI Education Agent/config.env"
chmod 600 "$HOME/Library/Application Support/AI Education Agent/config.env"
open -e "$HOME/Library/Application Support/AI Education Agent/config.env"
```

Add these lines, replacing only the placeholder key:

```text
MODEL_API_KEY=replace_with_your_own_key
META_BASE_URL=https://api.meta.ai/v1
META_MODEL=muse-wonder-1.0
META_MAX_COMPLETION_TOKENS=2048
AI_AGENT_MODE=meta
ENABLE_WEB_RESEARCH=true
```

Quit and reopen the app. Never email, commit, or put this configuration inside
the `.app` or ZIP. Change `AI_AGENT_MODE=mock` to return to Mock Mode.

In Meta Mode, the Tutor shows a compact **Agent activity** trace: detected
subject and intent, evidence tools used, quality checks, revisions, and the
release score. This is an auditable action trace, not the model's private
chain-of-thought. Conditional online retrieval is restricted to Wikipedia's
public API; reviewed local concept cards remain the preferred evidence.

## Privacy and compatibility

- Selected attachments remain local; external file analysis is disabled.
- Do not enter real student records, credentials, or private teaching materials.
- Meta Mode sends the question, learner profile, approved evidence, and limited
  conversation context to the configured provider.
- Target: Apple Silicon (`arm64`). Intel Macs require a separate tested build.
- The app is unsigned and not notarized.

