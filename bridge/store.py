"""
In-memory escalation store with asyncio locks.

Sufficient for a hackathon demo. Replace with Redis/SQLite for production.
"""
from __future__ import annotations

import asyncio
from typing import Dict

from shared.models import EscalationTicket, EscalationStatus


class EscalationStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tickets: Dict[str, EscalationTicket] = {}

    async def create(self, ticket: EscalationTicket) -> None:
        async with self._lock:
            self._tickets[ticket.escalation_id] = ticket

    async def get(self, escalation_id: str) -> EscalationTicket | None:
        async with self._lock:
            return self._tickets.get(escalation_id)

    async def update(self, ticket: EscalationTicket) -> None:
        async with self._lock:
            self._tickets[ticket.escalation_id] = ticket

    async def set_status(
        self, escalation_id: str, status: EscalationStatus
    ) -> EscalationTicket | None:
        async with self._lock:
            t = self._tickets.get(escalation_id)
            if t:
                t.status = status
                self._tickets[escalation_id] = t
            return t

    async def all(self) -> list[EscalationTicket]:
        async with self._lock:
            return list(self._tickets.values())


# Module-level singleton — shared across FastAPI lifespan
store = EscalationStore()
