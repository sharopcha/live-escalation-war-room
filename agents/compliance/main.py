"""
Compliance Agent — powered by OpenAI.

Uses the OpenAI API endpoint to run an LLM for compliance checking.

Band SDK: Pydantic AI-style (direct HTTP, clean structured output)
"""
from __future__ import annotations

import json
import logging
import os
import re

import httpx

from shared.band_agent import BaseBandAgent
from shared.band_client import BandClient
from shared.models import BandMessage, ComplianceResult

logger = logging.getLogger("agent.compliance")

# OpenAI API endpoint
OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a regulatory compliance agent for a call-centre AI platform.

Your job is to evaluate whether a proposed resolution complies with:
- Consumer protection regulations (EU/RO context)
- GDPR data handling requirements
- Financial services refund policies
- Telecom service level agreements

Output ONLY valid JSON, no extra text:
{
  "escalation_id": "...",
  "compliant": true,
  "issues": [],
  "needs_escalation": false,
  "recommended_resolution": "Optional improved wording if issues found"
}

Set needs_escalation=true ONLY if the proposed resolution would violate a hard regulatory rule.
issues should list specific policy/regulation concerns (empty list if none).
"""


async def check_compliance(
    escalation_id: str,
    issue_description: str,
    proposed_resolution: str | None,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> ComplianceResult:
    """Call OpenAI API and parse the compliance result."""
    user_content = (
        f"Escalation ID: {escalation_id}\n"
        f"Issue: {issue_description}\n"
        f"Proposed resolution: {proposed_resolution or 'Not yet determined'}\n\n"
        "Please evaluate compliance and output JSON."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": 400,
    }

    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(
            f"{OPENAI_BASE}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw.strip())
    return ComplianceResult(
        escalation_id=escalation_id,
        compliant=parsed.get("compliant", True),
        issues=parsed.get("issues", []),
        needs_escalation=parsed.get("needs_escalation", False),
        recommended_resolution=parsed.get("recommended_resolution"),
    )


# ---------------------------------------------------------------------------
# Band agent
# ---------------------------------------------------------------------------

class ComplianceAgent(BaseBandAgent):
    """OpenAI-powered compliance agent."""

    def __init__(self, openai_key: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._openai_key = openai_key
        self._model = os.getenv("COMPLIANCE_MODEL", DEFAULT_MODEL)

    async def handle_message(self, message: BandMessage, client: BandClient) -> None:
        esc_id = _extract_esc_id(message.text)
        issue = _extract_issue(message.text)
        if not esc_id or not issue:
            logger.debug("Cannot parse compliance context, skipping")
            return

        # Extract suggested resolution from the message if present
        proposed = _extract_proposed(message.text)

        logger.info("Compliance check for %s", esc_id)

        try:
            result = await check_compliance(
                escalation_id=esc_id,
                issue_description=issue,
                proposed_resolution=proposed,
                api_key=self._openai_key,
                model=self._model,
            )
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc, exc_info=True)
            # Fail safe: return compliant=True so we don't block the call
            result = ComplianceResult(
                escalation_id=esc_id,
                compliant=True,
                issues=[f"Compliance check unavailable: {exc}"],
                needs_escalation=False,
            )

        bridge_id = os.getenv("BRIDGE_AGENT_ID", "")
        await client.send_message(
            message.chat_id,
            f"@Bridge {result.model_dump_json()}",
            mention_ids=[bridge_id] if bridge_id else [],
        )
        logger.info(
            "Compliance result for %s: compliant=%s needs_escalation=%s",
            esc_id,
            result.compliant,
            result.needs_escalation,
        )


def _extract_esc_id(text: str) -> str | None:
    m = re.search(r"(esc_\w+)", text)
    return m.group(1) if m else None


def _extract_issue(text: str) -> str | None:
    m = re.search(r"\*\*Issue:\*\*\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return text[:300]


def _extract_proposed(text: str) -> str | None:
    m = re.search(r"suggested_resolution[\":\s]+([^\n\"{}]+)", text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    agent = ComplianceAgent(
        agent_key=os.environ["COMPLIANCE_AGENT_KEY"],
        openai_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("BAND_BASE_URL", "https://app.band.ai"),
        agent_name="compliance",
    )
    await agent.run()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
