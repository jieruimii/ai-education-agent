# Evaluation strategy

The prototype uses layered evaluation. A single benchmark score cannot show
that a tutor is correct, pedagogically useful, grounded, and able to call tools
properly.

## Release gate used now

Every Tutor response follows `analyze → retrieve → draft → evaluate → revise →
release`. The local regression suite checks subject routing, reviewed evidence,
required explanation sections, misconceptions, resource relevance, and leakage
guards. These tests are offline and deterministic; they are the deployment gate.

The Student UI reports five separate automated harness diagnostics:

- **Correctness:** agreement with the reviewed evidence and critic, not independent proof of truth.
- **Grounding:** whether supplied evidence is identified in the released answer.
- **Directness:** whether the first line answers the question and the response stays within its word budget.
- **Tool precision:** whether research was necessary and whether retrieved titles passed relevance filtering.
- **Reliability:** whether the normal critic path completed or required retry/fallback.

These diagnostics must never be reported as HLE, MMLU, GAIA, BFCL, or factual-accuracy scores.

## External benchmark map

| Layer | Benchmark | What it tests here | Status |
| --- | --- | --- | --- |
| Academic knowledge | HLE and STEM MMLU subsets | Closed-ended math, physics, chemistry, computer-science, engineering, and humanities correctness | Planned adapter; licensed/public items must be acquired separately |
| Tool use | GAIA and BFCL | Correct decision to research, tool selection, arguments, multi-turn behavior, and refusal to call irrelevant tools | Local trajectory tests implemented; official suites not bundled |
| Coding | SWE-bench | Repository-level software-engineering tasks | Use only for computer-science or autonomous-lab modes, not general tutoring |
| Work products | GDPval | Real professional deliverables, including engineering-related work | Optional capability study; not a tutor-accuracy substitute |
| File-based office work | AA-Briefcase | Agent work over documents, spreadsheets, slides, and PDFs | Optional external evaluation; its private task set is not bundled |

GDPval and AA-Briefcase are useful for future autonomous lab or professional
work-product modes. They do not directly measure whether a first-year student
received a clear, sourced explanation of Newton's first law. The product's
primary scorecard therefore remains instructor-reviewed course questions.

## Metrics

- Factual correctness and equation/unit correctness
- Evidence grounding and citation validity
- Subject-routing and retrieval relevance
- Required explanation coverage
- Tool-call selection, argument accuracy, and unnecessary-call rate
- Hallucination/unsupported-claim rate
- Hint leakage in Problem-Solving mode
- Latency and model/tool-call count
- Instructor rating and student comprehension-check performance

## Reporting rule

Never report an HLE, MMLU, BFCL, GAIA, SWE-bench, GDPval, or AA-Briefcase score
unless the official or licensed task set and grading method were actually run.
Local smoke tests must be labeled as local tests, not benchmark scores.

## References

- HLE: https://arxiv.org/abs/2501.14249
- MMLU: https://arxiv.org/abs/2009.03300
- GAIA: https://arxiv.org/abs/2311.12983
- BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html
- SWE-bench: https://www.swebench.com/
- GDPval: https://openai.com/index/gdpval/
- AA-Briefcase: https://www.mypriors.com/benchmarks/aa-briefcase
