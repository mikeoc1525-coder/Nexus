"""
mcp_servers/crm_server.py — Nexus CRM MCP Server

SQLite-backed CRM tool provider. Exposes a generic tool surface that agents
call via the MCP protocol. No Zoho, HubSpot, or Salesforce code lives here —
swap the backend by pointing _db_path at a different adapter.

Tools:
    upsert_lead          — create or update a lead (keyed on tenant_id + email)
    get_lead             — fetch a single lead by id or email
    update_lead_status   — change a lead's status (validated)
    list_leads           — query leads for a tenant with optional filters

Invariants:
    - Every operation requires tenant_id — no cross-tenant data leakage
    - Email is normalised to lowercase on write
    - Status values are validated against the schema allowlist
    - updated_at is refreshed on every write

Run standalone (stdio transport, for ADK agent wiring):
    python mcp_servers/crm_server.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import aiosqlite
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Ensure project root is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schemas import Lead, Firmographics  # noqa: E402 — path insert must come first

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", "./data/nexus.db"))

VALID_STATUSES = {
    "new", "researched", "verified", "scored",
    "qualified", "junk", "pending_review",
}

# ─── MCP Server ───────────────────────────────────────────────────────────────

mcp = FastMCP(
    "nexus-crm",
    instructions=(
        "Generic CRM tool server for Nexus. "
        "All tools require tenant_id. "
        "Email is the unique lead identifier within a tenant."
    ),
)

# ─── Database Helpers ─────────────────────────────────────────────────────────

async def _ensure_db() -> None:
    """Create tables if they don't exist. Safe to call on every request."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id            TEXT PRIMARY KEY,
                tenant_id     TEXT NOT NULL,
                email         TEXT NOT NULL,
                first_name    TEXT,
                last_name     TEXT,
                company       TEXT,
                title         TEXT,
                phone         TEXT,
                status        TEXT NOT NULL DEFAULT 'new',
                firmographics TEXT NOT NULL DEFAULT '{}',
                technographics TEXT NOT NULL DEFAULT '[]',
                source        TEXT,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                UNIQUE(tenant_id, email)
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_leads_tenant ON leads(tenant_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(tenant_id, status)"
        )
        await db.commit()


def _row_to_dict(row: aiosqlite.Row, cursor: aiosqlite.Cursor) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict using column names."""
    columns = [desc[0] for desc in cursor.description]
    d = dict(zip(columns, row))
    # Deserialise JSON columns
    d["firmographics"] = json.loads(d.get("firmographics") or "{}")
    d["technographics"] = json.loads(d.get("technographics") or "[]")
    return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def upsert_lead(
    tenant_id: str,
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    company: Optional[str] = None,
    title: Optional[str] = None,
    phone: Optional[str] = None,
    status: str = "new",
    employees: Optional[int] = None,
    revenue: Optional[str] = None,
    industry: Optional[str] = None,
    location: Optional[str] = None,
    technographics: Optional[list[str]] = None,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create or update a lead record.

    Keyed on (tenant_id, email) — if a lead with this email already exists
    for this tenant, it is updated in place. Otherwise a new record is created.

    Returns the full lead record after the write.

    Args:
        tenant_id:      Required. Identifies the tenant this lead belongs to.
        email:          Required. Unique identifier within the tenant.
        first_name:     Lead's first name.
        last_name:      Lead's last name.
        company:        Company name.
        title:          Job title.
        phone:          Phone number.
        status:         One of: new, researched, verified, scored, qualified,
                        junk, pending_review. Defaults to 'new'.
        employees:      Firmographic — employee count.
        revenue:        Firmographic — annual revenue string (e.g. "$5M-$10M").
        industry:       Firmographic — industry vertical.
        location:       Firmographic — city/country.
        technographics: List of technology names used by this company.
        source:         Where this lead originated (e.g. "apollo", "form").

    Returns:
        dict: The full lead record.

    Raises:
        ValueError: If email is malformed or status is not in the allowlist.
    """
    await _ensure_db()

    email = email.lower().strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise ValueError(f"Malformed email: {email!r}")

    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r}. Must be one of: {VALID_STATUSES}"
        )

    firmographics = json.dumps({
        "employees": employees,
        "revenue": revenue,
        "industry": industry,
        "location": location,
    })
    technographics_json = json.dumps(technographics or [])
    now = _now_iso()

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Check for existing record
        async with db.execute(
            "SELECT id FROM leads WHERE tenant_id = ? AND email = ?",
            (tenant_id, email),
        ) as cur:
            existing = await cur.fetchone()

        if existing:
            lead_id = existing["id"]
            await db.execute(
                """
                UPDATE leads SET
                    first_name     = COALESCE(?, first_name),
                    last_name      = COALESCE(?, last_name),
                    company        = COALESCE(?, company),
                    title          = COALESCE(?, title),
                    phone          = COALESCE(?, phone),
                    status         = ?,
                    firmographics  = ?,
                    technographics = ?,
                    source         = COALESCE(?, source),
                    updated_at     = ?
                WHERE tenant_id = ? AND id = ?
                """,
                (
                    first_name, last_name, company, title, phone,
                    status, firmographics, technographics_json,
                    source, now,
                    tenant_id, lead_id,
                ),
            )
        else:
            lead_id = str(uuid4())
            await db.execute(
                """
                INSERT INTO leads (
                    id, tenant_id, email, first_name, last_name,
                    company, title, phone, status,
                    firmographics, technographics, source,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead_id, tenant_id, email, first_name, last_name,
                    company, title, phone, status,
                    firmographics, technographics_json, source,
                    now, now,
                ),
            )

        await db.commit()

        async with db.execute(
            "SELECT * FROM leads WHERE tenant_id = ? AND id = ?",
            (tenant_id, lead_id),
        ) as cur:
            row = await cur.fetchone()
            return _row_to_dict(row, cur)


@mcp.tool()
async def get_lead(
    tenant_id: str,
    lead_id: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Fetch a single lead by id or email.

    Exactly one of lead_id or email must be provided.

    Args:
        tenant_id:  Required. Scopes the lookup to this tenant.
        lead_id:    UUID of the lead record.
        email:      Email address of the lead.

    Returns:
        dict: The lead record, or None if not found.

    Raises:
        ValueError: If neither or both of lead_id/email are provided.
    """
    if not lead_id and not email:
        raise ValueError("Provide either lead_id or email.")
    if lead_id and email:
        raise ValueError("Provide lead_id OR email, not both.")

    await _ensure_db()

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if lead_id:
            query = "SELECT * FROM leads WHERE tenant_id = ? AND id = ?"
            params = (tenant_id, lead_id)
        else:
            query = "SELECT * FROM leads WHERE tenant_id = ? AND email = ?"
            params = (tenant_id, email.lower().strip())

        async with db.execute(query, params) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            return _row_to_dict(row, cur)


@mcp.tool()
async def update_lead_status(
    tenant_id: str,
    lead_id: str,
    status: str,
) -> dict[str, Any]:
    """
    Update the status of a lead record.

    This is the primary state-transition tool. Agents should call this
    at each pipeline stage rather than calling upsert_lead for status-only changes.

    Valid transitions (not enforced here — enforced by agents):
        new → researched → verified → scored → qualified | junk | pending_review

    Args:
        tenant_id:  Required. Must match the lead's tenant.
        lead_id:    UUID of the lead to update.
        status:     New status. Must be one of the valid status values.

    Returns:
        dict: The updated lead record.

    Raises:
        ValueError: If status is invalid or the lead is not found.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r}. Must be one of: {VALID_STATUSES}"
        )

    await _ensure_db()

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT id FROM leads WHERE tenant_id = ? AND id = ?",
            (tenant_id, lead_id),
        ) as cur:
            if not await cur.fetchone():
                raise ValueError(
                    f"Lead {lead_id!r} not found for tenant {tenant_id!r}"
                )

        await db.execute(
            "UPDATE leads SET status = ?, updated_at = ? WHERE tenant_id = ? AND id = ?",
            (status, _now_iso(), tenant_id, lead_id),
        )
        await db.commit()

        async with db.execute(
            "SELECT * FROM leads WHERE tenant_id = ? AND id = ?",
            (tenant_id, lead_id),
        ) as cur:
            row = await cur.fetchone()
            return _row_to_dict(row, cur)


@mcp.tool()
async def list_leads(
    tenant_id: str,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List leads for a tenant, with optional status filtering and pagination.

    Args:
        tenant_id:  Required. Returns only leads belonging to this tenant.
        status:     Optional. Filter by status value.
        limit:      Max records to return (default 50, max 500).
        offset:     Pagination offset (default 0).

    Returns:
        dict: {
            "leads": [...],       # list of lead dicts
            "total": int,         # total matching records (for pagination)
            "limit": int,
            "offset": int,
        }

    Raises:
        ValueError: If status filter is not a valid status value.
    """
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status filter {status!r}. Must be one of: {VALID_STATUSES}"
        )

    limit = min(max(1, limit), 500)  # clamp to [1, 500]

    await _ensure_db()

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if status:
            count_query = (
                "SELECT COUNT(*) FROM leads WHERE tenant_id = ? AND status = ?",
                (tenant_id, status),
            )
            rows_query = (
                "SELECT * FROM leads WHERE tenant_id = ? AND status = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (tenant_id, status, limit, offset),
            )
        else:
            count_query = (
                "SELECT COUNT(*) FROM leads WHERE tenant_id = ?",
                (tenant_id,),
            )
            rows_query = (
                "SELECT * FROM leads WHERE tenant_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (tenant_id, limit, offset),
            )

        async with db.execute(*count_query) as cur:
            total = (await cur.fetchone())[0]

        async with db.execute(*rows_query) as cur:
            rows = await cur.fetchall()
            leads = [_row_to_dict(row, cur) for row in rows]

        return {
            "leads": leads,
            "total": total,
            "limit": limit,
            "offset": offset,
        }


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run()  # stdio transport — ADK agents connect via subprocess
