# Braintrust Gateway Demo Walkthrough

**Audience**: Prospect currently using LangFuse, not yet on Braintrust.
**Goal**: Show the gateway's value through a real multi-agent system.
**Project**: pydantic-supervisor (PydanticAI multi-agent with supervisor, research, math, and critic agents)

---

## Pre-Demo Checklist (Dry Run)

Before the demo, verify each of these:

| # | Check | How to verify | Expected result |
|---|-------|---------------|-----------------|
| 1 | Multi-model trace visible | Run eval, open trace in Braintrust UI | Spans show models from 2-3 different providers |
| 2 | Span tree is nested | Click into a trace | Agent → tool → LLM hierarchy, not flat |
| 3 | Cache headers surfaced | Run eval twice with same inputs, check span metadata | `gateway_cache_status` populated on 2nd run |
| 4 | Cache latency difference | Compare span durations across the two runs | Cached spans <100ms vs 1-3s uncached |
| 5 | Scorer spans present | Check trace | Routing Accuracy, Response Quality, etc. visible |
| 6 | Gateway URLs correct | Check .env | All base URLs point to `gateway.braintrust.dev`, not `api.braintrust.dev` |

**If check #3 fails**: The gateway header patch may not be firing for the SDK path being used. Debug by checking `braintrust_gateway_header_patch.py` and whether the `log_headers` function is being called. Also verify temperature=0 is being sent in requests (required for `auto` cache mode, or use `x-bt-use-cache: always`).

---

## Demo Flow

### Beat 1: "One client, one key, three providers" (~3 min)

**What to show**: The code — this is the proof.

1. Open `src/modeling.py` — show the `AsyncOpenAI` client construction:
   - `base_url` → `gateway.braintrust.dev`
   - `api_key` → `BRAINTRUST_API_KEY` (not an OpenAI key)
   - `x-bt-use-cache: always` header
   - `x-bt-project-id` header for automatic trace routing

2. Open `.env.example` — show:
   - `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL` both pointing at the gateway
   - No provider-specific API keys needed in the application
   - *"The SDKs pick these up automatically — zero code changes to route through the gateway."*

3. Briefly flash the Braintrust project settings → AI Providers tab:
   - *"Provider credentials are configured here, centrally managed. If I rotate an OpenAI key, I do it once — not in every service."*

**Key line**: *"My application has one API key and one gateway URL. The gateway resolves the right provider credentials based on the model I request."*

### Beat 2: "Here's what that produces" (~3 min)

**What to show**: A trace in the Braintrust UI.

1. Open the `pydantic-supervisor` project logs
2. Click into a trace showing the full supervisor flow
3. Walk the span tree:
   - SupervisorAgent makes routing decisions
   - Delegates to MathAgent (tool calls: add, multiply, divide)
   - Delegates to ResearchAgent (web search)
   - CriticAgent validates the response
4. Point out different models on different spans (`metadata.provider` showing `openai` vs `anthropic` vs `google`)
5. *"All of this came through the same gateway endpoint. Three providers, one unified trace."*

**Contrast with LangFuse**: *"In LangFuse, you'd need separate provider SDK integrations and manual instrumentation to get this unified view. Here, routing through the gateway gives you tracing as a side effect."*

### Beat 3: "Caching saves time and money during iteration" (~4 min)

**What to show**: Two eval runs, side by side.

1. Run the eval: `braintrust eval evals/eval_supervisor.py`
2. Wait for completion, note the wall-clock time
3. Run the exact same eval again (same inputs, same prompts)
4. Open both experiments in the Braintrust UI
5. Click into a trace from the second run — show:
   - `gateway_cache_status` in span metadata (should show cached)
   - Compare LLM span durations: first run 1-3s, second run <100ms on cached calls
6. *"When you're iterating on prompts or scorers, you're not re-paying for identical LLM calls. The gateway caches at the edge and returns encrypted results in under 100ms."*

**Key differentiator from provider caching**:
- Gateway cache is cross-provider (same mechanism for OpenAI, Anthropic, Google)
- Cached responses never hit the provider — no API billing
- Encrypted per-user with AES-GCM using your API key
- Works for all supported endpoints (chat completions, embeddings, etc.)

### Beat 4 (if time): "Zero-code tracing" (~1 min, talking point)

**What to say** (no live demo needed):
- *"If you don't want to use our SDK at all — say you just want basic observability — you set one header (`x-bt-parent: project_id:<ID>`) on your gateway requests and every LLM call is auto-logged to Braintrust."*
- *"You get a flat list of LLM calls — not the rich tree we just saw — but it's zero instrumentation. Then when you're ready for the full experience, you add the SDK and get the hierarchical traces, tool calls, and scores."*

---

## What NOT to emphasize

- **BTQL / per-model usage queries**: This works because of Braintrust logging, not the gateway specifically. Don't frame it as a gateway feature. Use it casually if it comes up.
- **Playground model comparison**: The prospect will discover this naturally. Not a gateway story.
- **Fallback / model swapping**: Good narrative but hard to demo live. Mention only if asked.

---

## Talking Points for Q&A

**"How is this different from just using OpenAI/Anthropic directly?"**
→ Unified credential management, cross-provider caching, automatic tracing, and you can swap models by changing a string — no SDK changes, no redeployment.

**"What about latency overhead?"**
→ The gateway runs on Cloudflare Workers globally. Overhead is typically <50ms, and cached responses return in <100ms — often faster than hitting the provider directly.

**"Can we self-host this?"**
→ Yes. In hybrid/BYOC mode, the gateway is bundled into your data plane containers. LLM traffic goes directly from your infrastructure to the provider — never through Braintrust's cloud.

**"What if we already have provider API keys everywhere?"**
→ You can migrate incrementally. The gateway accepts provider keys directly too — you don't need a Braintrust API key to use caching and the unified interface. The Braintrust key adds credential centralization and auto-tracing.
