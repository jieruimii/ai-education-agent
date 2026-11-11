# Single-Agent Education Harness Redesign

## Problem statement

The November meeting exposed an answer-quality failure: “explain the throttling process”
produced a generic response and course references without explaining the physical process.
The root cause was architectural. Tutor mode sent one broad prompt to the model and displayed
the response immediately. It had no explicit answer contract, approved evidence, evaluator,
or release gate.

The priority is therefore answer depth and accuracy—not more user-facing agents.

## What was reused from the reference workbook

The workbook was treated as reference material, not as instructions to execute. Its useful
patterns were simplified for this project:

- explicit shared state becomes a `QuestionContract` and `HarnessTrace`;
- planner/executor becomes deterministic Python question analysis and routing;
- retrieval becomes exact matching against instructor-reviewable concept cards;
- synthesis becomes one Tutor draft call;
- RAG-triad evaluation becomes a local coverage/relevance rubric plus a bounded model critic;
- observability becomes a stage trace and versioned regression cases.

Chart agents, web agents, Snowflake/Cortex, LangGraph, and TruLens were not adopted. They add
dependencies and autonomy without solving the demonstrated education-quality failure.

## Runtime workflow

```text
Student question + profile
          |
          v
1. ANALYZE — intent, topic, level, required answer sections
          |
          v
2. RETRIEVE — exact match to approved concept cards
          |
          v
3. DRAFT — direct answer bounded by retrieved evidence
          |
          v
4. EVALUATE — deterministic rubric + schema-checked model critic
          |
     pass | fail
          |   `--> 5. REVISE ONCE — defect list + same evidence
          |                    |
          `--------------------'
                    |
                    v
6. RELEASE — only checked text reaches Tkinter
                    |
          failed again?
                    v
7. FALLBACK — deterministic concept card or request for approved course material
```

This is one education agent invoked in bounded roles. The model does not decide which stage
runs next, cannot skip evaluation, and cannot create new source URLs.

## Holistic answer contract

Tutor answers must begin with a direct one-sentence answer and include:

1. What it is
2. How it works
3. Governing equation
4. Assumptions and limits
5. Concrete example
6. Common misconception
7. Check your understanding

The contract specifically prevents the observed pattern of naming related classes without
explaining the requested concept.

## Grounding and hallucination controls

- `knowledge_base.py` contains compact cards that an instructor can inspect line by line.
- Retrieval uses exact aliases instead of fuzzy semantic matching, reducing wrong-card retrieval.
- The prompt labels learner content as untrusted and separates it from policy and evidence.
- A matching card is the factual boundary for the draft and revision.
- Unknown topics do not receive fabricated citations; if the answer cannot pass the gate, the
  agent asks for an approved excerpt instead of guessing.
- Provider-generated URLs remain filtered. Only the resource catalog and concept-card source
  URLs may reach the UI.
- No unchecked streaming text is displayed; the UI buffers until the harness releases an answer.
- Problem-Solving mode remains a separate deterministic hint-first state machine.

The current approved throttling card is grounded in MIT Unified Engineering Thermodynamics,
Chapter 4. It captures the pressure drop, steady-flow energy balance, isenthalpic idealization,
irreversibility, temperature caveat, refrigeration example, and common misconceptions.

## Evaluation strategy

The first regression case is `thermo-throttling-depth`, derived directly from Professor
Penkova’s feedback. It checks required concepts, known misconceptions, and minimum depth in
Mock Mode. Unit tests also verify routing, card selection, critic-schema rejection, revision,
and final release behavior.

Before claiming general accuracy, expand the evaluation set with instructor-written questions
and reference answers across each target course. Score these dimensions separately:

- factual correctness against an instructor rubric;
- evidence faithfulness;
- answer relevance;
- explanation completeness and causal depth;
- equation and unit correctness;
- misconception handling;
- personalization usefulness;
- answer leakage in Problem-Solving mode;
- latency and token cost.

A model criticizing its own answer is only one signal. It is not independent proof of truth.
Instructor review and course-grounded evidence remain required for high-stakes claims.

## Expected tradeoffs

- Tutor Meta Mode now uses two calls for a passing answer and up to three when revision is needed.
- The first visible response arrives later because unchecked tokens are not streamed to students.
- Accuracy is improved only for reviewed concepts. The current knowledge base contains one
  regression concept; it must grow through instructor-approved cards or a governed course RAG
  collection before the system can claim broad engineering coverage.

## Next implementation milestone

Create an instructor ingestion workflow for approved course notes, chunk provenance, and
retrieval evaluation. Add 20–30 instructor-authored questions for Thermodynamics before adding
another course, then compare baseline single-call answers against harnessed answers using the
same model and fixed scoring rubric.

