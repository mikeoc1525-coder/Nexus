"""Memory layer for lead history and agent events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from core.schemas import Lead


class MemoryStore(Protocol):
    def upsert_lead(self, lead: Lead) -> None:
        ...

    def get_lead_by_email(self, email: str) -> Lead | None:
        ...

    def list_leads(self) -> list[Lead]:
        ...

    def append_event(self, event: dict[str, Any]) -> None:
        ...

    def events(self) -> list[dict[str, Any]]:
        ...


class InMemoryStore:
    def __init__(self) -> None:
        self._leads_by_email: dict[str, Lead] = {}
        self._events: list[dict[str, Any]] = []

    def upsert_lead(self, lead: Lead) -> None:
        self._leads_by_email[lead.email.lower()] = lead

    def get_lead_by_email(self, email: str) -> Lead | None:
        return self._leads_by_email.get(email.lower())

    def list_leads(self) -> list[Lead]:
        return list(self._leads_by_email.values())

    def append_event(self, event: dict[str, Any]) -> None:
        self._events.append(event)

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)


@dataclass
class ExternalMemoryConfig:
    provider: str
    connection_uri: str


class MemoryLayer:
    """Abstraction point for long-term memory backends (BigQuery, Pinecone, etc.)."""

    def __init__(self, store: MemoryStore, external: ExternalMemoryConfig | None = None):
        self.store = store
        self.external = external

    def upsert_lead(self, lead: Lead) -> None:
        self.store.upsert_lead(lead)
        self.store.append_event({"type": "lead_upserted", "lead_email": lead.email})

    def get_lead_by_email(self, email: str) -> Lead | None:
        return self.store.get_lead_by_email(email)

    def list_leads(self) -> list[Lead]:
        return self.store.list_leads()

    def append_event(self, event: dict[str, Any]) -> None:
        self.store.append_event(event)
