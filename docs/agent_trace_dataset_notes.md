# Agent Trace Dataset Notes

These datasets are curated from the latest 100 root traces in the Braintrust project `pydantic-supervisor`.

Span choices:

- `SupervisorAgent`: use the root `invocation [supervisor_with_critic]` span. It already matches the supervisor entrypoint used in [eval_supervisor.py](/Users/curtisjgalione/git/pydantic-supervisor/evals/eval_supervisor.py): original user query in, critic-validated final answer out.
- `ResearchAgent`: use `handoff [ResearchAgent]` spans. They isolate the delegate agent boundary, preserve the agent's own answer, and keep the associated serialized messages that contain the web-search evidence.
- `MathAgent`: use `handoff [MathAgent]` spans. They are better than the nested delegate invocation spans because the math invocation prompt is templated by the supervisor, while the handoff span preserves the canonical math task and result together. For the checked-in dataset rows, the `input.query` is normalized back to the original root query so the examples are usable for direct-agent evals.

Dataset shape:

- [supervisor_trace_dataset.jsonl](/Users/curtisjgalione/git/pydantic-supervisor/datasets/supervisor_trace_dataset.jsonl): `input.query` plus expected final answer and routing label.
- [research_trace_dataset.jsonl](/Users/curtisjgalione/git/pydantic-supervisor/datasets/research_trace_dataset.jsonl): `input.query` plus expected answer and citation expectations.
- [math_trace_dataset.jsonl](/Users/curtisjgalione/git/pydantic-supervisor/datasets/math_trace_dataset.jsonl): `input.query` plus expected answer text and a normalized reference answer.

Supervisor curation:

- Favor delegation-heavy examples that clearly require either `ResearchAgent` or `MathAgent`.
- Keep only a small number of direct-response controls, and make them realistic supervisor use cases rather than creative-writing or joke prompts.
- Exclude ambiguous prompts where the observed route is weak training signal for the intended agent boundary.

Refresh flow:

1. Export the recent traces to `/tmp/pydantic_supervisor_roots.json` and `/tmp/pydantic_supervisor_candidate_spans.json`.
2. Run `python3 scripts/build_trace_datasets.py`.
