# RevOps Autonomous Engine (Forge & Fire)

AI-native RevOps runtime built around:
- **Google ADK patterning** for orchestration
- **MCP** for decoupled tool interfaces
- **Gemini-family agent roles** for planning, enrichment, scoring, and CRM sync

This repo provides a runnable MVP of a hub-and-spoke, agent-to-agent architecture where:
- intelligence is implemented by specialized agents
- capabilities are provided by swappable MCP tool servers
- governance callbacks enforce data hygiene before CRM writes

## Why this architecture

Traditional lead-gen automation breaks when APIs or CRMs change.  
This design decouples agent logic from integrations so you can swap a tool server or config instead of rewriting orchestration logic.

## Project structure

```text
revops_engine/
  agents/
    orchestrator.py
    research_agent.py
    scoring_agent.py
    verifier.py
    crm_agent.py
  core/
    adk_compat.py
    governance.py
    memory.py
    schemas.py
  mcp_servers/
    crm_server.py
    research_server.py
    enrichment_server.py
  main.py
```

## Agent roles

- **Orchestrator**: strategic router (research -> scoring -> verifier -> CRM sync)
- **Research Agent**: enrichment + deep-intel loop
- **Scoring Agent**: assigns points and tier (A/B/C) with reason codes
- **Verifier**: pass/fail quality gate before write
- **CRM Agent**: sequential pattern (`formatter` -> `pusher`) with callback guardrails

## MCP servers

- **CRM Server (declarative style)**
  - `upsert_lead`
  - `get_account_history`
  - `update_deal_stage`
  - read-only win/loss resource for lookalike analysis
- **Research Server (imperative style)**
  - `search_linkedin`
  - `scrape_company_news`
  - `find_decision_makers`
- **Enrichment Server (imperative style)**
  - `verify_email`
  - `get_technographics`
  - `fetch_funding_data`

## Governance

`core/governance.py` includes interceptors/callbacks such as:
- `validate_crm_write` (blocks malformed emails)
- `hitl_required` (example human-in-the-loop gate for risky writes)

## Run locally

```bash
python3 revops_engine/main.py
```

Run with custom payload:

```bash
python3 revops_engine/main.py --objective-json '{"email":"ops@signalworks.ai","firmographics":{"employees":400,"revenue":"$100M","industry":"SaaS"}}'
```

Run with a direct email:

```bash
python3 revops_engine/main.py --email "alex@forgesaas.com"
```

## Example outcomes

- **synced**: lead passes verification and is written to CRM
- **blocked**: verifier rejects output before CRM write

## Notes

- `core/adk_compat.py` provides local fallback primitives when the real ADK package is unavailable.
- Memory is currently in-memory (`InMemoryStore`) with a clean abstraction point for BigQuery/Pinecone-backed history.
