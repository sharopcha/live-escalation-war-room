"""
Renggo Band Bridge — FastAPI service + Band agent logic.

Endpoints consumed by Renggo's orchestrator:
  POST /escalations                      → create escalation, spin up Band room
  GET  /escalations/{id}/resolution      → poll for resolution (check_resolution)

The bridge also runs as a registered Band agent, polling its own work-queue
for @Bridge messages from the supervisor (human approval path).
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from bridge.config import get_settings
from bridge.store import store
from shared.band_client import BandClient
from shared.models import (
    BandMessage,
    ComplianceResult,
    EscalationCreated,
    EscalationPath,
    EscalationRequest,
    EscalationResolutionResponse,
    EscalationStatus,
    EscalationTicket,
    KnowledgeResult,
    Resolution,
    TriageResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("bridge")


# ---------------------------------------------------------------------------
# Band agent logic — bridge as an agent
# ---------------------------------------------------------------------------

class BridgeAgent:
    """
    The bridge registers as a Band agent.
    It creates rooms, recruits specialist agents, and receives the final
    @Bridge {resolution JSON} from the supervisor.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._running = False
        self._poll_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Bridge Band polling started")

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()

    # ------------------------------------------------------------------
    # Room orchestration — called when a new escalation arrives
    # ------------------------------------------------------------------

    async def open_room(self, ticket: EscalationTicket) -> str:
        """
        Create a Band room, add all agents, and post the call context.
        Returns the room/chat ID.
        """
        cfg = self._settings
        async with BandClient(cfg.bridge_agent_key, cfg.band_base_url) as client:
            # 1. Create room
            room = await client.create_chat(
                name=f"Escalation {ticket.escalation_id}",
                description=ticket.issue_description[:200],
            )
            room_data = room.get("data", room)
            chat_id: str = room_data["id"]
            logger.info("Created Band room %s for %s", chat_id, ticket.escalation_id)

            # 2. Add specialist agents
            for agent_id in [
                cfg.triage_agent_id,
                cfg.knowledge_agent_id,
                cfg.compliance_agent_id,
            ]:
                try:
                    await client.add_participant(chat_id, agent_id)
                except Exception as exc:
                    logger.warning("Could not add agent %s: %s", agent_id, exc)

            # 3. Post call context, mentioning the triage agent first
            context_msg = self._format_context(ticket)
            await client.send_message(
                chat_id,
                context_msg,
                mention_ids=[cfg.triage_agent_id],
            )
            logger.info("Posted context to room %s, mentioned triage", chat_id)

        return chat_id

    @staticmethod
    def _format_context(ticket: EscalationTicket) -> str:
        ctx = json.dumps(ticket.context, ensure_ascii=False) if ticket.context else "{}"
        return (
            f"🚨 ESCALATION {ticket.escalation_id}\n\n"
            f"**Caller:** {ticket.caller_id}\n"
            f"**Issue:** {ticket.issue_description}\n"
            f"**Context:** {ctx}\n\n"
            f"**Recent transcript:**\n{ticket.transcript}\n\n"
            f"@Triage please classify this case and emit your TriageResult JSON."
        )

    async def recruit_supervisor(self, chat_id: str) -> None:
        """Mention the human supervisor in the room (callback path)."""
        cfg = self._settings
        async with BandClient(cfg.bridge_agent_key, cfg.band_base_url) as client:
            await client.send_message(
                chat_id,
                "⚠️ Human approval required. @Supervisor please review and post "
                "@Bridge {resolution JSON} when ready.",
                mention_ids=[],  # supervisor is a human user — mention by name in text
            )

    async def post_parallel_task(
        self, chat_id: str, ticket: EscalationTicket, triage: TriageResult
    ) -> None:
        """After triage → in-call path: kick off knowledge + compliance in parallel."""
        cfg = self._settings
        async with BandClient(cfg.bridge_agent_key, cfg.band_base_url) as client:
            await client.send_message(
                chat_id,
                f"✅ Triage complete (auto-approved path) for {ticket.escalation_id}.\n\n"
                f"**Issue:** {ticket.issue_description}\n"
                f'suggested_resolution: "{triage.suggested_resolution or ""}"\n\n'
                f"@Knowledge @Compliance please deliberate in parallel and each post "
                f"your result JSON.",
                mention_ids=[cfg.knowledge_agent_id, cfg.compliance_agent_id],
            )

    async def synthesise_and_resolve(
        self,
        ticket: EscalationTicket,
        knowledge: KnowledgeResult,
        compliance: ComplianceResult,
    ) -> None:
        """Auto-synthesise resolution from knowledge + compliance results."""
        resolution = Resolution(
            escalation_id=ticket.escalation_id,
            resolution_text=knowledge.answer,
            approved_by="auto",
            requires_callback=False,
            confidence=min(knowledge.confidence, 1.0),
            notes=(
                f"Compliance: {'OK' if compliance.compliant else 'Issues: ' + ', '.join(compliance.issues)}"
            ),
        )
        ticket.resolution = resolution
        ticket.status = EscalationStatus.RESOLVED
        ticket.path = EscalationPath.IN_CALL
        ticket.resolved_at = datetime.utcnow()
        await store.update(ticket)
        logger.info("Auto-resolved escalation %s", ticket.escalation_id)

    # ------------------------------------------------------------------
    # Band polling loop — receives @Bridge messages from supervisor
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """
        Band's agent work-queue is scoped per chat room — there is no global
        "next message" endpoint. So instead of polling one queue, we poll each
        room belonging to a still-active escalation.
        """
        cfg = self._settings
        async with BandClient(cfg.bridge_agent_key, cfg.band_base_url) as client:
            logger.info("Bridge polling Band work-queues…")
            while self._running:
                try:
                    tickets = await store.all()
                    room_ids = {
                        t.band_room_id
                        for t in tickets
                        if t.band_room_id
                        and t.status not in (EscalationStatus.RESOLVED, EscalationStatus.FAILED)
                    }

                    got_message = False
                    for chat_id in room_ids:
                        raw = await client.get_next_message(chat_id)
                        if raw is None:
                            continue
                        got_message = True

                        msg = _parse_message(raw, chat_id)
                        await client.mark_processing(chat_id, msg.message_id)
                        try:
                            await self._handle_band_message(msg, client)
                            await client.mark_processed(chat_id, msg.message_id)
                        except Exception as exc:
                            logger.error("Error handling Band message: %s", exc, exc_info=True)
                            await client.mark_failed(chat_id, msg.message_id, str(exc))

                    if not got_message:
                        await asyncio.sleep(1.5)

                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error("Poll error: %s", exc, exc_info=True)
                    await asyncio.sleep(3)

    async def _handle_band_message(
        self, msg: BandMessage, client: BandClient
    ) -> None:
        """
        Route incoming Band messages to the right handler.

        Messages may be:
        - TriageResult JSON from the triage agent
        - KnowledgeResult JSON from the knowledge agent
        - ComplianceResult JSON from the compliance agent
        - Resolution JSON from the supervisor (@Bridge {...})
        """
        raw_json = _extract_json(msg.text)
        if not raw_json:
            logger.debug("Non-JSON message from %s, skipping", msg.sender_id)
            return

        esc_id = raw_json.get("escalation_id")
        if not esc_id:
            logger.debug("No escalation_id in message, skipping")
            return

        ticket = await store.get(esc_id)
        if not ticket:
            logger.warning("Unknown escalation_id: %s", esc_id)
            return

        # Detect message type by its keys
        if "requires_human_approval" in raw_json:
            await self._on_triage(ticket, raw_json, client)
        elif "confidence" in raw_json and "answer" in raw_json:
            await self._on_knowledge(ticket, raw_json)
        elif "compliant" in raw_json:
            await self._on_compliance(ticket, raw_json)
        elif "resolution_text" in raw_json:
            await self._on_supervisor_resolution(ticket, raw_json)
        else:
            logger.debug("Unrecognised message shape: %s", list(raw_json.keys()))

    async def _on_triage(
        self,
        ticket: EscalationTicket,
        data: dict,
        client: BandClient,
    ) -> None:
        result = TriageResult(**data)
        logger.info(
            "Triage for %s: approval=%s severity=%s category=%s",
            result.escalation_id,
            result.requires_human_approval,
            result.severity,
            result.category,
        )

        if result.requires_human_approval:
            # → Callback path
            ticket.path = EscalationPath.CALLBACK
            ticket.status = EscalationStatus.AWAITING_HUMAN
            await store.update(ticket)
            await self.recruit_supervisor(ticket.band_room_id or "")
            logger.info("Escalation %s → callback path (human approval)", ticket.escalation_id)
        else:
            # → In-call path
            ticket.status = EscalationStatus.DELIBERATING
            await store.update(ticket)
            if ticket.band_room_id:
                await self.post_parallel_task(ticket.band_room_id, ticket, result)
            logger.info("Escalation %s → in-call path (auto)", ticket.escalation_id)

    async def _on_knowledge(self, ticket: EscalationTicket, data: dict) -> None:
        result = KnowledgeResult(**data)
        ticket.context["_knowledge"] = result.model_dump()
        await store.update(ticket)
        logger.info("Knowledge result received for %s (confidence=%.2f)", ticket.escalation_id, result.confidence)
        await self._try_auto_resolve(ticket)

    async def _on_compliance(self, ticket: EscalationTicket, data: dict) -> None:
        result = ComplianceResult(**data)
        ticket.context["_compliance"] = result.model_dump()
        await store.update(ticket)
        logger.info("Compliance result received for %s (compliant=%s)", ticket.escalation_id, result.compliant)

        # Edge case: compliance flips needs_escalation after triage said auto
        if result.needs_escalation and ticket.path == EscalationPath.IN_CALL:
            logger.warning(
                "Compliance escalated %s mid-deliberation → switching to callback",
                ticket.escalation_id,
            )
            ticket.path = EscalationPath.CALLBACK
            ticket.status = EscalationStatus.AWAITING_HUMAN
            await store.update(ticket)
            if ticket.band_room_id:
                await self.recruit_supervisor(ticket.band_room_id)
            return

        await self._try_auto_resolve(ticket)

    async def _try_auto_resolve(self, ticket: EscalationTicket) -> None:
        """Resolve once BOTH knowledge and compliance results are in."""
        k = ticket.context.get("_knowledge")
        c = ticket.context.get("_compliance")
        if not (k and c):
            return
        if ticket.status == EscalationStatus.RESOLVED:
            return
        knowledge = KnowledgeResult(**k)
        compliance = ComplianceResult(**c)
        await self.synthesise_and_resolve(ticket, knowledge, compliance)

    async def _on_supervisor_resolution(self, ticket: EscalationTicket, data: dict) -> None:
        """Human supervisor posted @Bridge {resolution JSON}."""
        resolution = Resolution(**data)
        ticket.resolution = resolution
        ticket.status = EscalationStatus.RESOLVED
        ticket.resolved_at = datetime.utcnow()
        await store.update(ticket)
        logger.info("Human-approved resolution for %s", ticket.escalation_id)

        if ticket.path == EscalationPath.CALLBACK:
            await self._fire_outbound_call(ticket)

    async def _fire_outbound_call(self, ticket: EscalationTicket) -> None:
        """Trigger Renggo's outbound call API with the resolution."""
        cfg = get_settings()
        if not cfg.renggo_outbound_url:
            logger.warning("No RENGGO_OUTBOUND_URL configured — skipping callback")
            return
        payload = {
            "caller_id": ticket.caller_id,
            "context": {
                "escalation_id": ticket.escalation_id,
                "resolution": ticket.resolution.model_dump() if ticket.resolution else {},
            },
        }
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.post(
                    cfg.renggo_outbound_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {cfg.renggo_api_key}"},
                )
                resp.raise_for_status()
                logger.info("Outbound call triggered for %s", ticket.escalation_id)
        except Exception as exc:
            logger.error("Failed to fire outbound call: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

bridge_agent = BridgeAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await bridge_agent.start()
    yield
    await bridge_agent.stop()


app = FastAPI(
    title="Renggo Band Bridge",
    description="Escalation bridge connecting Renggo voice AI to a Band multi-agent war room.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "renggo-band-bridge"}


@app.post("/escalations", response_model=EscalationCreated, status_code=201)
async def create_escalation(req: EscalationRequest) -> EscalationCreated:
    """
    Called by Renggo's `escalate_to_band` tool.
    Creates a Band room, recruits agents, posts context.
    Returns an escalation_id for polling.
    """
    ticket = EscalationTicket(
        call_id=req.call_id,
        caller_id=req.caller_id,
        issue_description=req.issue_description,
        transcript=req.transcript,
        context=req.context,
        language=req.language,
    )
    await store.create(ticket)
    logger.info("Escalation created: %s", ticket.escalation_id)

    # Open Band room async — don't block the response
    async def _open():
        try:
            chat_id = await bridge_agent.open_room(ticket)
            ticket.band_room_id = chat_id
            ticket.status = EscalationStatus.TRIAGING
            await store.update(ticket)
        except Exception as exc:
            logger.error("Failed to open Band room: %s", exc, exc_info=True)
            ticket.status = EscalationStatus.FAILED
            await store.update(ticket)

    asyncio.create_task(_open())

    return EscalationCreated(
        escalation_id=ticket.escalation_id,
        status=EscalationStatus.QUEUED,
    )


@app.get("/escalations/{escalation_id}/resolution", response_model=EscalationResolutionResponse)
async def get_resolution(escalation_id: str) -> EscalationResolutionResponse:
    """
    Called by Renggo's `check_resolution` tool.
    Returns current status and resolution (if available).
    """
    import asyncio
    
    ticket = await store.get(escalation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Escalation not found")

    # Long-polling: wait up to 25 seconds for deliberation to finish
    for _ in range(25):
        if ticket.status in (
            EscalationStatus.RESOLVED,
            EscalationStatus.AWAITING_HUMAN,
            EscalationStatus.CALLBACK_SCHEDULED,
            EscalationStatus.FAILED,
        ):
            break
        await asyncio.sleep(1.0)
        ticket = await store.get(escalation_id)

    res = ticket.resolution
    if not res:
        # Provide a default empty resolution to prevent JSONPath errors in the call center flow
        # when it tries to extract $.resolution.resolution_text during polling.
        from shared.models import Resolution
        res = Resolution(escalation_id=escalation_id, resolution_text="")

    return EscalationResolutionResponse(
        escalation_id=escalation_id,
        status=ticket.status,
        resolution=res,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_message(raw: dict[str, Any], default_chat_id: str = "") -> BandMessage:
    # Band's message body comes back under "content" (matches the key
    # send_message() POSTs as) — not "text". Keep "text" as a fallback
    # in case an endpoint ever differs.
    mentions_raw = raw.get("mentions", raw.get("mention_ids", []))
    return BandMessage(
        message_id=raw.get("id", ""),
        chat_id=raw.get("chat_id") or default_chat_id,
        sender_id=raw.get("sender_id", ""),
        sender_type=raw.get("sender_type", "agent"),
        text=raw.get("content") or raw.get("text", ""),
        mentions=[m.get("id") if isinstance(m, dict) else m for m in mentions_raw],
        created_at=raw.get("created_at", ""),
        raw=raw,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(), strict=False)
        except json.JSONDecodeError:
            return None
    return None


if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run("bridge.main:app", host=cfg.host, port=cfg.port, reload=False)
