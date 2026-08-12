# 🎬 Video Agent

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C1C1C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**One prompt becomes a continuous 40-second story — four 10-second shots with enforced narrative and visual continuity.**

[🎥 Watch Demo](#demo) · [📖 Architecture](#architecture) · [🚀 Quick Start](#quick-start) · [📡 API Reference](#api-reference)

</div>

---

## The Problem

Text-to-video models generate clips of 5–10 seconds **in isolation**. Generate four clips from four prompts and you get four unrelated clips: the protagonist changes face, the room changes colour, the story never moves.

**Generation is solved. Continuity is not.**

## The Solution

Video Agent takes **one prompt** and produces a **continuous 40-second MP4** composed of four 10-second shots that tell a coherent story — same character, same wardrobe, same location, same visual style, connected by frame chaining.

```
"A lone astronaut discovers a hidden garden on Mars at sunset"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Beat 1 [setup]        │  Beat 2 [development]              │
│  She lands. Dust. Red  │  She walks. Footprints in          │
│  horizon. Alone.       │  the ochre soil. Silence.          │
│  ─────────────────     │  ─────────────────────────         │
│  Beat 3 [turn]         │  Beat 4 [resolution]               │
│  Green shoots between  │  She kneels. Removes helmet.       │
│  rocks. Impossible.    │  Breathes Earth air. Smiles.       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   🎬 40-second stitched MP4  +  per-shot clips  +  JSON artifacts
```

---

## Architecture

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                              │
│                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────┐   ┌───────┐   │
│  │  Planner │ → │  Bible   │ → │Generator │ → │  QC  │ → │Assem- │   │
│  │  Node    │   │  Node    │   │  Node    │   │ Node │   │bler   │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────┘   └───────┘   │
│  4-beat arc     Continuity     Shot prompt     Vision      ffmpeg    │
│  (LLM)          Bible (LLM)    + generation    model QC    stitch    │
│                 immutable      (sequential)    ≥ 0.75      + deliver │
│                                frame chain     or repair             │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Common Platform (Guidelines.pdf)                               │
│                                                                 │
│  FastAPI (async)  ·  LiteLLM proxy  ·  Redis checkpoints        │
│  Langfuse traces  ·  PostgreSQL+pgvector  ·  MinIO/S3           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Sequential shot generation** | Deliberate trade-off. Parallel is ~4× faster but breaks frame chaining. Frame chaining is what makes the product work. |
| **Immutable ContinuityBible** | Locked after planning, embedded in every shot prompt. Zero character drift. |
| **Provider abstraction** | Code never names a provider. Aliases resolve at the LiteLLM gateway. Swapping models = config change, zero code diff. |
| **Never returns nothing** | If one shot succeeded, a stitched partial is delivered with a resume handle. |
| **Checkpoint every node** | Crashes resume. Completed shots are never regenerated or re-billed. |

---

## Quick Start

### Prerequisites
- Python 3.12+
- ffmpeg (`brew install ffmpeg`)
- Docker + Docker Compose (for full stack)

### Option A — Local dev (no Docker, no API keys)

```bash
git clone https://github.com/YOUR_USERNAME/video-agent
cd video-agent

# Install dependencies
pip install -e ".[dev]"

# Copy env template (no keys needed for mock provider)
cp .env.example .env

# Run the demo (uses mock video provider)
python demo/run_demo.py --prompt "A detective uncovers a century-old conspiracy in London"
```

### Option B — Full API server (local, mock provider)

```bash
# Start API server
uvicorn video_agent.main:app --reload

# In another terminal — create a job
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A lone astronaut discovers a garden on Mars", "provider": "mock"}'

# Poll for status
curl http://localhost:8000/api/v1/jobs/{job_id}

# Open interactive docs
open http://localhost:8000/docs
```

### Option C — Full Docker stack

```bash
cp .env.example .env
# Add your LLM and Higgsfield API keys to .env

docker compose up -d

# Langfuse observability at http://localhost:3000
# MinIO console at http://localhost:9001
# API docs at http://localhost:8000/docs
```

---

## Demo

> 🎥 **[Demo Recording Link — Coming Soon]**
>
> The demo shows: prompt submission → StoryPlan generation → ContinuityBible locking → 4 shots generated sequentially with frame chaining → QC scoring → ffmpeg stitch → final MP4 delivery.

**Sample output** for prompt `"A lone astronaut discovers a hidden garden on Mars at sunset"`:

```json
{
  "job_id": "a3f2e1b0",
  "status": "success",
  "story_plan": {
    "title": "The Garden at the Edge of Space",
    "genre": "sci-fi drama",
    "beats": [
      {"index": 0, "label": "setup", "action": "Astronaut lands, surveys barren red landscape"},
      {"index": 1, "label": "development", "action": "Suit sensors spike — oxygen detected nearby"},
      {"index": 2, "label": "turn", "action": "She rounds a boulder — green shoots, impossible"},
      {"index": 3, "label": "resolution", "action": "She kneels, removes glove, touches living leaf"}
    ]
  },
  "artifacts": {
    "stitched_mp4_url": "https://storage.../a3f2e1b0_stitched.mp4",
    "story_plan_url": "https://storage.../a3f2e1b0_story_plan.json",
    "continuity_bible_url": "https://storage.../a3f2e1b0_continuity_bible.json"
  },
  "budget": {
    "cost_usd": 0.42,
    "elapsed_seconds": 187
  }
}
```

---

## API Reference

### `POST /api/v1/jobs`
Submit a story prompt. Returns immediately with `job_id`. Agent runs asynchronously.

**Request:**
```json
{
  "prompt": "Your story idea (10–500 chars)",
  "provider": "mock"  // "mock" | "higgsfield" — auto-selected if omitted
}
```

**Response `202 Accepted`:**
```json
{ "job_id": "uuid", "status": "planning", "message": "Poll GET /jobs/{job_id}" }
```

### `GET /api/v1/jobs/{job_id}`
Poll job status and partial results.

**Status values:** `planning` → `generating` → `qc` → `assembling` → `success` | `partial` | `failed`

### `GET /api/v1/health`
```json
{ "status": "ok", "version": "1.0.0", "provider": "mock", "langfuse": false }
```

Full interactive docs at `/docs` (Swagger UI) and `/redoc`.

---

## Deliverables per Job

| Artifact | Description |
|----------|-------------|
| `stitched_mp4_url` | The 40-second continuous story |
| `individual_clips[]` | Each 10-second shot separately |
| `thumbnail_url` | First frame of the story |
| `continuity_frames[]` | Final frames used for chaining |
| `story_plan_url` | 4-beat narrative arc as JSON |
| `continuity_bible_url` | Full visual contract as JSON |

Every job is **fully reproducible**: per-shot cost, model, seed, and prompt are stored.

---

## Failure Behaviour

Per the **Common Platform Specification** (Guidelines.pdf):

| Condition | Outcome |
|-----------|---------|
| Evaluator satisfied | `SUCCESS` |
| Budget exhausted | `PARTIAL` — best-so-far, flagged |
| Same failure signature twice | `FAILED_NO_PROGRESS` — stop immediately |
| Non-retryable error | `FAILED` / `ESCALATED` |

- **Retry:** exponential backoff + jitter, max 3 attempts, retryable errors only
- **Circuit break:** per dependency, 5 failures in 30s
- **Checkpoint:** after every LangGraph node — crashes resume, never restart

---

## Observability

All traces go to **Langfuse** (self-hosted). Every log line carries `trace_id` so any log joins to its Langfuse trace instantly.

Never logged: credentials, raw PII, full media payloads, row-level query results.

```
Langfuse UI → http://localhost:3000
```

---

## Running Tests

```bash
# Unit tests (no API keys needed)
pytest tests/unit/ -v

# Integration tests (mock provider)
pytest tests/integration/ -v

# Full coverage report
pytest --cov=video_agent --cov-report=html
```

---

## Project Structure

```
video-agent/
├── src/video_agent/
│   ├── agent/
│   │   ├── graph.py          # LangGraph StateGraph
│   │   ├── state.py          # AgentState TypedDict
│   │   └── nodes/
│   │       ├── planner.py    # Beat arc generation
│   │       ├── bible.py      # ContinuityBible locking
│   │       ├── generator.py  # Sequential shot generation
│   │       ├── qc.py         # Vision model QC + repair
│   │       └── assembler.py  # ffmpeg stitch + deliver
│   ├── providers/
│   │   ├── base.py           # AbstractVideoProvider
│   │   ├── higgsfield.py     # Higgsfield MCP adapter
│   │   └── mock.py           # Mock (coloured clips, no API key)
│   ├── gateway/
│   │   └── llm.py            # LiteLLM alias gateway
│   ├── api/
│   │   ├── routes.py         # FastAPI endpoints
│   │   └── schemas.py        # Pydantic models
│   └── observability/
│       └── langfuse_client.py
├── tests/
│   ├── unit/                 # Planner, Bible, QC node tests
│   └── integration/          # Full job lifecycle test
├── demo/run_demo.py          # CLI demo with rich UI
├── docker-compose.yml        # Full local stack
└── litellm_config.yaml       # Model alias configuration
```

---

## Built on

| Component | Role |
|-----------|------|
| **LangGraph** | Stateful orchestration, crash-safe checkpointing |
| **LiteLLM** | Single egress for all model calls (alias-based) |
| **Higgsfield MCP** | Video generation provider |
| **Langfuse** | Traces, generations, scores, prompt registry |
| **ffmpeg** | Clip concatenation and normalisation |
| **PostgreSQL + pgvector** | System of record, RLS per tenant |
| **Redis 7** | Checkpoints, locks, rate limits, idempotency |
| **FastAPI** | Async REST API |
| **MinIO** | Local S3-compatible storage |

---

*Entermind · Video Agent · v1.0.0 · August 2026*
