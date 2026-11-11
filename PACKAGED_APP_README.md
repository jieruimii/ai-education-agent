# AI Education Agent for macOS

This package contains the standalone `AI_Education_Agent.app` desktop prototype
for Professor Penkova. It was built for Apple Silicon Macs and does not require
Python to be installed.

## Start in Mock Mode

1. Unzip the package.
2. Move `AI_Education_Agent.app` to `Applications` if desired.
3. Double-click the application.

Mock Mode is the safe default when no Meta configuration is present. It makes
no external model request. The Tutor Agent includes a detailed offline
throttling explanation for the meeting regression case. Problem-Solving mode,
student profiles, verified recommendations, Markdown, and LaTeX remain available.

## Tutor answer-quality harness

In Meta Mode, Tutor responses pass through a single-agent workflow:

`analyze → retrieve approved evidence → draft → evaluate → revise once → release`

Unchecked provider text is not displayed. A passing answer normally uses two
model calls; a rejected draft may use a third revision call. This improves
answer structure and reduces unsupported claims, but it cannot guarantee factual
correctness. Only the throttling-process concept currently has a reviewed local
knowledge card; other courses need instructor-approved evidence and evaluations.

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
```

Quit and reopen the app. Never email, commit, or put this configuration inside
the `.app` or ZIP. Change `AI_AGENT_MODE=mock` to return to Mock Mode.

## Privacy and compatibility

- Selected attachments remain local; external file analysis is disabled.
- Do not enter real student records, credentials, or private teaching materials.
- Meta Mode sends the question, learner profile, approved evidence, and limited
  conversation context to the configured provider.
- Target: Apple Silicon (`arm64`). Intel Macs require a separate tested build.
- The app is unsigned and not notarized.

