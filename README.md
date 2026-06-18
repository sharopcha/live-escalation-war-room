# renggo-band-bridge

**Live Escalation War Room** — Band of Agents Hackathon (June 12–19, 2026)

> Live voice AI that escalates hard calls to a Band room of specialist agents + human approval — and resolves them before the caller hangs up.

[![MIT License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org)
[![Band SDK](https://img.shields.io/badge/band-sdk-purple)](https://docs.band.ai)

---

## Architecture

```
Caller ──► Renggo Gemini Live Agent (closed-source, hosted)
                │
                │ escalate_to_band tool
                ▼
         ┌──────────────┐
         │   Bridge     │  FastAPI + Band agent (this repo)
         │  :8080       │  • Creates Band room
         └──────┬───────┘  • Recruits 4 agents
                │          • Stores resolution
         ┌──────▼──────────────────────────────────┐
         │          Band Chat Room                  │
         │  @Triage → @Knowledge + @Compliance      │
         │                      ↓                   │
         │         requires_human_approval?          │
         │         NO → auto-resolve (~10-15s)       │
         │         YES → @Supervisor (human, Band UI)│
         └──────────────────────────────────────────┘
                │
         check_resolution tool (Renggo polls)
                │
         ◄──── resolution_text spoken mid-call
               (or: "I'll call you back shortly" + outbound call)
```

### Agents

| Agent | Framework | Model | Role |
|-------|-----------|-------|------|
| **Bridge** | FastAPI + httpx | — | Orchestrates room, routes messages, stores resolution |
| **Triage** | LangGraph | GPT-4o-mini | Classifies case, emits `requires_human_approval` flag |
| **Knowledge** | Standalone RAG | GPT-4o-mini (optional) | Searches KB, returns answer + confidence |
| **Compliance** | OpenAI API | gpt-4o-mini | Checks resolution against policy/regulations |
| **Supervisor** | Human | — | Reviews in Band web UI, posts `@Bridge {resolution JSON}` |

### Routing

```
Triage: requires_human_approval = false
  → In-call path: Knowledge + Compliance deliberate in parallel
  → Bridge synthesises resolution → Renggo speaks it (~10-15s)

Triage: requires_human_approval = true
  → Callback path: Voice agent offers callback, ends call gracefully
  → Band room continues async, human approves in Band UI
  → Bridge fires POST /calls/outbound with resolution injected
```

---

## Quick Start

### 1. Register Band agents

Go to [app.band.ai](https://app.band.ai), log in, and register 4 agents:

```
Settings → Agents → New Agent

Name: renggo-bridge    → copy BRIDGE_AGENT_KEY + BRIDGE_AGENT_ID
Name: renggo-triage    → copy TRIAGE_AGENT_KEY + TRIAGE_AGENT_ID
Name: renggo-knowledge → copy KNOWLEDGE_AGENT_KEY + KNOWLEDGE_AGENT_ID
Name: renggo-compliance→ copy COMPLIANCE_AGENT_KEY + COMPLIANCE_AGENT_ID
```

> **Note:** Agent API keys are shown only once. Save them immediately.

### 2. Configure environment

```bash
cp .env.example .env
# Fill in all REPLACE_ME values
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

Or run services individually (useful for debugging):

```bash
# Terminal 1 — bridge
uvicorn bridge.main:app --port 8080

# Terminal 2 — triage agent
python agents/triage/main.py

# Terminal 3 — knowledge agent
python agents/knowledge/main.py

# Terminal 4 — compliance agent
python agents/compliance/main.py
```

### 4. Test end-to-end (text input)

```bash
# Trigger an escalation
curl -X POST http://localhost:8080/escalations \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "call_test_001",
    "caller_id": "+40722000001",
    "issue_description": "Customer was charged twice for the same invoice in March",
    "transcript": "Agent: How can I help you today?\nCaller: I was charged twice for the same bill last month!",
    "context": {"account_id": "ACC-12345", "tier": "standard"}
  }'

# Response: {"escalation_id": "esc_abc123", "status": "queued"}

# Poll for resolution
curl http://localhost:8080/escalations/esc_abc123/resolution
```

---

## Renggo Integration

Add these two tools to your Renggo orchestrator agent config (see `flows/tool_spec.json` for full schema):

```python
# In your orchestrator tool definitions, set BRIDGE_URL to where the bridge is running
BRIDGE_URL = "http://localhost:8080"  # or your hosted URL

tools = [
    {
        "name": "escalate_to_band",
        "http": {"url": f"{BRIDGE_URL}/escalations", "method": "POST"}
    },
    {
        "name": "check_resolution",
        "http": {"url": f"{BRIDGE_URL}/escalations/{{escalation_id}}/resolution", "method": "GET"}
    }
]
```

The voice agent polls `check_resolution` every 3–5 seconds (up to `IN_CALL_TIMEOUT_S`). On `status=resolved`, it speaks `resolution.resolution_text`. On `status=callback_scheduled`, it says "I'll call you back shortly" and ends the call gracefully.

---

## Demo Scenarios

### Fast path — telecom billing dispute (~15s in-call)

```
Caller: "I was charged twice for my internet plan in March."
→ Triage: billing_dispute, severity=medium, requires_human_approval=false
→ Knowledge: finds DEF-2024-087 or double-charge policy, confidence=0.9
→ Compliance: compliant=true
→ Resolution: "I can see a duplicate charge on your March invoice.
   I've applied a full credit of €29.99 which will appear on your next bill."
```

### Slow path — hardship refund with human approval

```
Caller: "I lost my job and can't pay this month's bill."
→ Triage: hardship_refund, severity=high, requires_human_approval=true
→ Voice agent: "I need to get authorisation for this. I'll call you back
   within 2 business hours with a solution."
→ Band room continues async → Supervisor approves in Band UI
→ Bridge fires outbound call with resolution
```

---

## Open-source split

This repo contains **only the new bridge + agents code** (MIT licensed).

The Renggo voice platform (`call-center-ai`, `platform-api`, `orchestrator`) is closed-source hosted infrastructure and is not included here. The bridge connects to it as an HTTP client — no platform files are copied into this repo.

---

## Project structure

```
renggo-band-bridge/
├── bridge/          # FastAPI bridge + Band polling loop
│   ├── main.py      # App + escalation endpoints + room orchestration
│   ├── config.py    # Settings from env
│   └── store.py     # In-memory escalation store
├── agents/
│   ├── triage/      # LangGraph triage agent
│   ├── knowledge/   # TF-IDF RAG knowledge agent
│   └── compliance/  # OpenAI compliance agent
├── shared/
│   ├── band_client.py  # Band REST API client
│   ├── band_agent.py   # Base polling loop
│   └── models.py       # Shared Pydantic models
├── kb/sample/       # Demo knowledge base (public data)
├── flows/           # Tool spec for Renggo orchestrator
├── docker-compose.yml
└── .env.example
```

---

## License

MIT — see [LICENSE](LICENSE).

Built for the [Band of Agents Hackathon](https://lablab.ai/ai-hackathons/band-of-agents-hackathon) (June 12–19, 2026).
