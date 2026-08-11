# Architecture Deep-Dive

## LangGraph StateGraph

The agent is a compiled `StateGraph` with five nodes. The shared `AgentState` TypedDict
flows through every node — no global variables, crash-safe, deterministically replayable.

```
plan_story
    │
    ▼
lock_bible
    │
    ▼
generate_shot ◄──────────────┐
    │                        │ repair (if QC failed
    ▼                        │ and repair_count < 2)
qc_shot ──── passed ────────►│ (next shot)
    │                        │
    └──── failed ────────────┘
    │ (max repairs exhausted)
    ▼
assemble ──► END
```

### Conditional edges

| From | Condition | To |
|------|-----------|----|
| `plan_story` | Status is FAILED | `assemble` (partial delivery) |
| `plan_story` | Status is OK | `lock_bible` |
| `qc_shot` | Shot passed | `generate_shot` (next shot) |
| `qc_shot` | Shot failed, repair budget ok | `generate_shot` (repair) |
| `qc_shot` | Shot failed, max repairs | `assemble` |
| `qc_shot` | Budget exhausted | `assemble` |
| `qc_shot` | All shots done | `assemble` |

## Frame Chaining

Shot N's final frame is passed as `conditioning_image_url` to shot N+1's
generation request. This preserves protagonist identity, wardrobe, and
lighting across cuts — the core mechanism that makes the 40-second story coherent.

```
Shot 0 ──[final frame]──► Shot 1 ──[final frame]──► Shot 2 ──[final frame]──► Shot 3
```

The deliberate trade-off: parallel generation would be ~4× faster, but
breaks frame chaining. Latency was traded for the core value proposition.

## ContinuityBible

The bible is produced by a single LLM call after planning and is **immutable**
for the life of the job. It contains:

- `protagonist` — detailed physical description
- `wardrobe` — exact clothing, colours, accessories  
- `location` — precise setting with architecture and textures
- `lighting` — time of day, quality, direction
- `colour_palette` — hex codes or film reference
- `lens_language` — focal length, depth-of-field, aspect ratio
- `style_tags` — cinematic style descriptors

Every shot prompt = bible + beat action + camera move.

## QC Loop

After each shot is generated, a vision model (alias: `vision-default`) evaluates
the clip against the bible and returns a `continuity_score` [0.0–1.0].

- `≥ 0.75` → shot passes, advance to next
- `< 0.75` → shot fails, trigger repair (max 2 attempts)
- Same failure signature twice → `FAILED_NO_PROGRESS` (stop immediately)

The QC model is calibrated on a labelled set; its own failures fall back to
passing the shot rather than burning repair budget on QC overhead.

## Budget Enforcement

Hard caps (all configurable via env vars):

| Cap | Default |
|-----|---------|
| `MAX_JOB_BUDGET_USD` | $2.00 |
| `MAX_JOB_ITERATIONS` | 20 |
| `MAX_JOB_WALL_CLOCK_SECONDS` | 480s (8 min) |

When any cap is hit, the agent routes to `assemble` with whatever shots it has
(PARTIAL outcome). The p90 target is ≤ 8 minutes end-to-end.

## Provider Abstraction

```python
class AbstractVideoProvider(ABC):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...
    async def health_check(self) -> bool: ...
    name: str
```

Code never names a provider. `_get_provider()` in the generator node reads
`settings.video_provider` and returns the appropriate implementation.
Swapping providers = one config line change.

Idempotency keys (SHA-256 of `job_id + shot_index`) prevent double-billing on
network retries.

## Failure Taxonomy

Per the Common Platform Specification:

```
retryable error → exponential backoff + jitter (max 3 attempts)
fallback        → alternate model within alias group
circuit break   → per dependency, 5 failures in 30s
degrade         → cached/partial result, always flagged
fail honestly   → what happened, what was preserved, what to do next
```

Every error response carries:
- `stable_code` — machine-readable
- `trace_id` — opens the exact Langfuse trace instantly

## Observability

```
Trace = one job
  └── Span: plan_story
        └── Generation: LLM call (model, tokens, cost, prompt version)
  └── Span: lock_bible
        └── Generation: LLM call
  └── Span: generate_shot (×4)
        └── External: Higgsfield MCP call
  └── Span: qc_shot (×4)
        └── Generation: vision model call
  └── Span: assemble
        └── Score: continuity_score
```

Every log line carries `trace_id` as a structured field, so any log line
joins to its Langfuse trace in one click.
