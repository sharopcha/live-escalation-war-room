"""Shared Pydantic models used across bridge and all agents."""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


# ---------------------------------------------------------------------------
# Escalation lifecycle
# ---------------------------------------------------------------------------

class EscalationStatus(str, Enum):
    QUEUED = "queued"
    TRIAGING = "triaging"
    DELIBERATING = "deliberating"
    AWAITING_HUMAN = "awaiting_human"
    RESOLVED = "resolved"
    CALLBACK_SCHEDULED = "callback_scheduled"
    FAILED = "failed"


class EscalationPath(str, Enum):
    IN_CALL = "in_call"       # resolve within same call (~10-15s)
    CALLBACK = "callback"     # async, call back when resolved


class EscalationRequest(BaseModel):
    """Payload Renggo sends to POST /escalations."""
    call_id: str
    caller_id: str
    issue_description: str
    transcript: str                    # last N turns of the call
    context: dict[str, Any] = Field(default_factory=dict)
    language: str = "en"


class EscalationTicket(BaseModel):
    """Internal escalation record stored in the bridge."""
    escalation_id: str = Field(default_factory=lambda: f"esc_{uuid.uuid4().hex[:12]}")
    call_id: str
    caller_id: str
    issue_description: str
    transcript: str
    context: dict[str, Any] = Field(default_factory=dict)
    language: str = "en"
    status: EscalationStatus = EscalationStatus.QUEUED
    path: EscalationPath | None = None
    band_room_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    resolution: "Resolution | None" = None


class EscalationCreated(BaseModel):
    """Response to POST /escalations."""
    escalation_id: str
    status: EscalationStatus


class EscalationResolutionResponse(BaseModel):
    """Response to GET /escalations/{id}/resolution."""
    escalation_id: str
    status: EscalationStatus
    resolution: "Resolution | None" = None


# ---------------------------------------------------------------------------
# Agent outputs — strict contracts validated by the bridge
# ---------------------------------------------------------------------------

class TriageResult(BaseModel):
    """Output contract for the triage agent."""
    escalation_id: str
    requires_human_approval: bool
    severity: str = Field(pattern="^(low|medium|high|critical)$")
    category: str              # e.g. "billing_dispute", "hardship_refund"
    reasoning: str
    suggested_resolution: str | None = None


class KnowledgeResult(BaseModel):
    """Output contract for the knowledge agent."""
    escalation_id: str
    answer: str
    source: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_passages: list[str] = Field(default_factory=list)


class ComplianceResult(BaseModel):
    """Output contract for the compliance agent."""
    escalation_id: str
    compliant: bool
    issues: list[str] = Field(default_factory=list)
    needs_escalation: bool = False    # flip triggers callback mode
    recommended_resolution: str | None = None


class Resolution(BaseModel):
    """
    Final resolution — posted by supervisor as @Bridge {json} or
    synthesised by bridge from knowledge + compliance on the auto path.
    """
    escalation_id: str
    resolution_text: str
    approved_by: str = "auto"          # "auto" | "human:<username>"
    requires_callback: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str | None = None


# ---------------------------------------------------------------------------
# Band message envelope (what the bridge parses from the work queue)
# ---------------------------------------------------------------------------

class BandMessage(BaseModel):
    """Normalised Band chat message as returned by the REST API."""
    message_id: str
    chat_id: str
    sender_id: str
    sender_type: str           # "user" | "agent"
    text: str
    mentions: list[str] = Field(default_factory=list)
    created_at: str
    raw: dict[str, Any] = Field(default_factory=dict)
