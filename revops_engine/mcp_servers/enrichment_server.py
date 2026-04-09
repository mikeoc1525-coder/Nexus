"""Imperative enrichment MCP server tools."""

from __future__ import annotations


class EnrichmentServer:
    def verify_email(self, email: str) -> dict[str, bool | str]:
        is_valid = "@" in email and "." in email.split("@")[-1]
        return {"email": email, "is_valid": is_valid}

    def get_technographics(self, domain: str) -> list[str]:
        mapping = {
            "forgesaas.com": ["Salesforce", "HubSpot", "Segment"],
            "signalworks.ai": ["HubSpot", "Snowflake", "Apollo"],
        }
        return mapping.get(domain.lower(), ["Salesforce", "Slack"])

    def fetch_funding_data(self, company: str) -> dict[str, str]:
        return {"company": company, "stage": "Series B", "last_round": "$45M"}
