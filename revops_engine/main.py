"""Entrypoint for the RevOps Autonomous Engine MVP."""

from __future__ import annotations

import argparse
import json

from agents.crm_agent import CRMAgent
from agents.orchestrator import RevOpsOrchestrator
from agents.research_agent import ResearchAgent
from agents.scoring_agent import ScoringAgent
from agents.verifier import Verifier
from core.memory import InMemoryStore, MemoryLayer
from mcp_servers.crm_server import CRMServer
from mcp_servers.enrichment_server import EnrichmentServer
from mcp_servers.research_server import ResearchServer


def build_orchestrator() -> RevOpsOrchestrator:
    crm_server = CRMServer()
    research_server = ResearchServer()
    enrichment_server = EnrichmentServer()
    memory = MemoryLayer(store=InMemoryStore())

    research_agent = ResearchAgent(research_server=research_server, enrichment_server=enrichment_server)
    scoring_agent = ScoringAgent()
    verifier = Verifier()
    crm_agent = CRMAgent(crm_server=crm_server)
    return RevOpsOrchestrator(
        research_agent=research_agent,
        scoring_agent=scoring_agent,
        verifier=verifier,
        crm_agent=crm_agent,
        memory=memory,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Forge & Fire RevOps engine.")
    parser.add_argument(
        "--objective-json",
        type=str,
        default="",
        help="Raw JSON payload for a lead objective.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="alex@forgesaas.com",
        help="Email to use when objective JSON is not supplied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orchestrator = build_orchestrator()

    if args.objective_json:
        objective = json.loads(args.objective_json)
    else:
        objective = {
            "email": args.email,
            "firmographics": {"employees": 250, "revenue": "$25M-$50M", "industry": "SaaS"},
        }

    result = orchestrator.run_objective(objective)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
