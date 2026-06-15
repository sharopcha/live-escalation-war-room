"""
Base Band agent with async work-queue polling loop.

Subclass this, implement `handle_message()`, and call `run()`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from shared.band_client import BandClient
from shared.models import BandMessage

logger = logging.getLogger(__name__)


class BaseBandAgent(ABC):
    """
    Async Band agent base class.

    Polls the Band work-queue, marks messages processing/processed/failed,
    and dispatches to `handle_message()`.

    Args:
        agent_key:     Agent API key (band_a_...) for this specific agent.
        base_url:      Band API base URL.
        poll_interval: Seconds between poll attempts when queue is empty.
        agent_name:    Human-readable name for logging.
    """

    def __init__(
        self,
        agent_key: str,
        base_url: str = "https://app.band.ai",
        poll_interval: float = 1.5,
        agent_name: str = "agent",
    ) -> None:
        self._agent_key = agent_key
        self._base_url = base_url
        self._poll_interval = poll_interval
        self._agent_name = agent_name
        self._running = False

    # ------------------------------------------------------------------
    # Subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    async def handle_message(
        self,
        message: BandMessage,
        client: BandClient,
    ) -> None:
        """
        Process one message from the work queue.

        `client` is already open — use it to send replies, post events, etc.
        Raise an exception to mark the message as failed.
        """

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the polling loop. Runs until `stop()` is called."""
        self._running = True
        logger.info("[%s] Starting Band polling loop", self._agent_name)

        async with BandClient(self._agent_key, self._base_url) as client:
            # Validate connection on startup
            try:
                me = await client.get_agent_me()
                logger.info(
                    "[%s] Connected as agent: %s",
                    self._agent_name,
                    me.get("data", {}).get("name", "?"),
                )
            except Exception as exc:
                logger.error("[%s] Failed to connect: %s", self._agent_name, exc)
                raise

            while self._running:
                try:
                    raw = await client.get_next_message()
                    if raw is None:
                        await asyncio.sleep(self._poll_interval)
                        continue

                    message = self._parse_message(raw)
                    logger.info(
                        "[%s] Received message %s from %s",
                        self._agent_name,
                        message.message_id,
                        message.sender_id,
                    )

                    await client.mark_processing(message.message_id)

                    try:
                        await self.handle_message(message, client)
                        await client.mark_processed(message.message_id)
                    except Exception as exc:
                        logger.error(
                            "[%s] Error handling message %s: %s",
                            self._agent_name,
                            message.message_id,
                            exc,
                            exc_info=True,
                        )
                        await client.mark_failed(message.message_id, str(exc))

                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(
                        "[%s] Poll error: %s", self._agent_name, exc, exc_info=True
                    )
                    await asyncio.sleep(self._poll_interval * 2)

        logger.info("[%s] Polling loop stopped", self._agent_name)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_message(raw: dict[str, Any]) -> BandMessage:
        return BandMessage(
            message_id=raw.get("id", ""),
            chat_id=raw.get("chat_id", ""),
            sender_id=raw.get("sender_id", ""),
            sender_type=raw.get("sender_type", "agent"),
            text=raw.get("text", ""),
            mentions=raw.get("mention_ids", []),
            created_at=raw.get("created_at", ""),
            raw=raw,
        )

    @staticmethod
    def extract_json_block(text: str) -> dict[str, Any] | None:
        """
        Pull the first JSON object out of a message like:
          @Bridge {"escalation_id": "esc_abc", "resolution_text": "..."}
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None
