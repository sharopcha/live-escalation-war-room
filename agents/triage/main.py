"""
Triage Agent — LangGraph framework adapter.

Classifies each escalation and emits a TriageResult JSON to the Band room,
including the critical `requires_human_approval` flag that drives routing.

Band SDK: LangGraph adapter (thenvoi[langgraph])
"""
from __future__ import annotations

import json
import logging
import os
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from shared.band_agent import BaseBandAgent
from shared.band_client import BandClient
from shared.models import BandMessage, TriageResult

logger = logging.getLogger("agent.triage")

# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class TriageState(TypedDict):
    escalation_id: str
    issue_description: str
    transcript: str
    context: str
    # Outputs
    requires_human_approval: bool
    severity: str
    category: str
    reasoning: str
    suggested_resolution: str | None


SYSTEM_PROMPT = """You are an expert call-centre triage agent for Renggo AI.
Your job is to classify a customer escalation and determine whether it requires human approval.

Severity levels:
- low: routine query, no policy risk
- medium: non-standard request within normal policy bounds
- high: exception required, compliance risk, or significant financial impact
- critical: legal, safety, or regulatory risk

Categories (use the closest match):
billing_dispute, hardship_refund, service_interruption, compliance_concern,
fraud_suspicion, accessibility_request, regulatory_inquiry, general_complaint

Requires human approval = true when ANY of:
- Financial exception > standard refund policy
- Legal or regulatory language used
- Fraud indicators
- severity is critical
- No KB answer likely to satisfy the caller

Output ONLY valid JSON, nothing else:
{
  "escalation_id": "...",
  "requires_human_approval": false,
  "severity": "medium",
  "category": "billing_dispute",
  "reasoning": "Short explanation",
  "suggested_resolution": "Optional one-liner for auto path"
}"""


def classify_node(state: TriageState) -> TriageState:
    llm = ChatOpenAI(
        model=os.getenv("TRIAGE_MODEL", "gpt-4o-mini"),
        temperature=0,
        max_tokens=512,
    )
    user_msg = (
        f"Escalation ID: {state['escalation_id']}\n"
        f"Issue: {state['issue_description']}\n"
        f"Context: {state['context']}\n\n"
        f"Transcript (last turns):\n{state['transcript']}"
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw)
    return {
        **state,
        "requires_human_approval": parsed.get("requires_human_approval", True),
        "severity": parsed.get("severity", "medium"),
        "category": parsed.get("category", "general_complaint"),
        "reasoning": parsed.get("reasoning", ""),
        "suggested_resolution": parsed.get("suggested_resolution"),
    }


def build_triage_graph():
    graph = StateGraph(TriageState)
    graph.add_node("classify", classify_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", END)
    return graph.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_triage_graph()
    return _graph


# ---------------------------------------------------------------------------
# Band agent
# ---------------------------------------------------------------------------

class TriageAgent(BaseBandAgent):
    """LangGraph-powered triage agent."""

    async def handle_message(self, message: BandMessage, client: BandClient) -> None:
        raw_json = self.extract_json_block(message.text)

        # Context is embedded in the room's initial message (no JSON block)
        # Parse it from the structured text the bridge posted
        escalation_id, issue, transcript, context = _parse_context_message(message.text)
        if not escalation_id:
            logger.debug("No escalation context in message, skipping")
            return

        logger.info("Triaging escalation %s", escalation_id)

        state: TriageState = {
            "escalation_id": escalation_id,
            "issue_description": issue,
            "transcript": transcript,
            "context": context,
            "requires_human_approval": True,
            "severity": "medium",
            "category": "general_complaint",
            "reasoning": "",
            "suggested_resolution": None,
        }

        result_state = get_graph().invoke(state)

        result = TriageResult(
            escalation_id=escalation_id,
            requires_human_approval=result_state["requires_human_approval"],
            severity=result_state["severity"],
            category=result_state["category"],
            reasoning=result_state["reasoning"],
            suggested_resolution=result_state["suggested_resolution"],
        )

        # Get bridge agent ID from env to @mention it
        bridge_id = os.getenv("BRIDGE_AGENT_ID", "")
        reply = f"@Bridge {result.model_dump_json()}"
        await client.send_message(
            message.chat_id,
            reply,
            mention_ids=[bridge_id] if bridge_id else [],
        )
        logger.info(
            "Triage done for %s: approval=%s severity=%s",
            escalation_id,
            result.requires_human_approval,
            result.severity,
        )


def _parse_context_message(text: str) -> tuple[str, str, str, str]:
    """Extract escalation fields from the structured context message the bridge posts."""
    import re
    esc_id = ""
    issue = ""
    transcript = ""
    context = "{}"

    m = re.search(r"ESCALATION\s+(esc_\w+)", text)
    if m:
        esc_id = m.group(1)

    m = re.search(r"\*\*Issue:\*\*\s*(.+)", text)
    if m:
        issue = m.group(1).strip()

    m = re.search(r"\*\*Context:\*\*\s*(.+)", text)
    if m:
        context = m.group(1).strip()

    m = re.search(r"transcript:\*\*\n(.*?)(?:\n\n|$)", text, re.DOTALL)
    if m:
        transcript = m.group(1).strip()

    return esc_id, issue, transcript, context


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    import asyncio
    agent = TriageAgent(
        agent_key=os.environ["TRIAGE_AGENT_KEY"],
        base_url=os.getenv("BAND_BASE_URL", "https://app.band.ai"),
        agent_name="triage",
    )
    await agent.run()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
