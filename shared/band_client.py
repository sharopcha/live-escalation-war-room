"""
Band REST API client.

Wraps the Thenvoi/Band HTTP API used by the bridge and agents.
Each method maps to one documented endpoint.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default poll interval for the work-queue poller
DEFAULT_POLL_INTERVAL = 1.5   # seconds
DEFAULT_TIMEOUT = 20.0


class BandClient:
    """
    Async Band REST client.

    Args:
        api_key:  Agent API key (band_a_...) for agent-scoped endpoints,
                  or User API key (band_u_...) for human-scoped endpoints.
        base_url: Band base URL, e.g. https://app.band.ai
    """

    def __init__(self, api_key: str, base_url: str = "https://app.band.ai") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BandClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=DEFAULT_TIMEOUT,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("BandClient must be used as an async context manager")
        return self._client

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        resp = await self.http.request(method, path, **kwargs)
        if resp.status_code >= 400:
            logger.error("Band API %s %s → %s: %s", method, path, resp.status_code, resp.text)
            resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    async def get_agent_me(self) -> dict[str, Any]:
        """Get this agent's own profile (validates connection)."""
        return await self._request("GET", "/api/v1/agent/me")

    async def get_user_me(self) -> dict[str, Any]:
        """Get the user's own profile."""
        return await self._request("GET", "/api/v1/user/me")

    async def list_agent_peers(self) -> list[dict[str, Any]]:
        """List collaborators this agent can interact with."""
        data = await self._request("GET", "/api/v1/agent/peers")
        return data.get("data", [])

    # ------------------------------------------------------------------
    # Chat / Room management
    # ------------------------------------------------------------------

    async def create_chat(
        self,
        name: str,
        description: str = "",
        participant_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new Band chat room."""
        payload: dict[str, Any] = {"name": name, "description": description}
        if participant_ids:
            payload["participant_ids"] = participant_ids
        return await self._request("POST", "/api/v1/agent/chats", json=payload)

    async def get_chat(self, chat_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/v1/agent/chats/{chat_id}")

    async def list_chats(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/api/v1/agent/chats")
        return data.get("data", [])

    # ------------------------------------------------------------------
    # Participants
    # ------------------------------------------------------------------

    async def add_participant(self, chat_id: str, participant_id: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v1/agent/chats/{chat_id}/participants",
            json={"participant_id": participant_id},
        )

    async def remove_participant(self, chat_id: str, participant_id: str) -> None:
        await self._request(
            "DELETE",
            f"/api/v1/agent/chats/{chat_id}/participants/{participant_id}",
        )

    async def list_participants(self, chat_id: str) -> list[dict[str, Any]]:
        data = await self._request("GET", f"/api/v1/agent/chats/{chat_id}/participants")
        return data.get("data", [])

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        chat_id: str,
        text: str,
        mention_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Send a message to a chat room.
        `mention_ids` are agent/user IDs to @mention (required for them to see it).
        """
        payload: dict[str, Any] = {"text": text}
        if mention_ids:
            payload["mention_ids"] = mention_ids
        return await self._request(
            "POST", f"/api/v1/agent/chats/{chat_id}/messages", json=payload
        )

    async def get_chat_context(
        self, chat_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Fetch recent message history for context rehydration."""
        data = await self._request(
            "GET",
            f"/api/v1/agent/chats/{chat_id}/messages",
            params={"limit": limit},
        )
        return data.get("data", [])

    # ------------------------------------------------------------------
    # Work-queue (agent message polling)
    # ------------------------------------------------------------------

    async def get_next_message(self) -> dict[str, Any] | None:
        """
        Poll for the next unprocessed message directed at this agent.
        Returns None if the queue is empty.
        """
        data = await self._request("GET", "/api/v1/agent/messages/next")
        return data.get("data") or None

    async def mark_processing(self, message_id: str) -> None:
        await self._request(
            "PATCH",
            f"/api/v1/agent/messages/{message_id}/status",
            json={"status": "processing"},
        )

    async def mark_processed(self, message_id: str) -> None:
        await self._request(
            "PATCH",
            f"/api/v1/agent/messages/{message_id}/status",
            json={"status": "processed"},
        )

    async def mark_failed(self, message_id: str, reason: str = "") -> None:
        await self._request(
            "PATCH",
            f"/api/v1/agent/messages/{message_id}/status",
            json={"status": "failed", "reason": reason},
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def post_event(
        self,
        chat_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Post a structured event to a chat room.
        event_type: "tool_call" | "tool_result" | "thought" | "error" | "task"
        """
        return await self._request(
            "POST",
            f"/api/v1/agent/chats/{chat_id}/events",
            json={"type": event_type, "payload": payload},
        )
