"""
Run the Profile Intelligence Engine against the real Cerebras API.

From the backend folder (digital-marketing-ai/backend):

  PowerShell:
    $env:PYTHONPATH = (Get-Location).Path
    python scripts/run_profile_engine_cerebras.py

  cmd:
    set PYTHONPATH=%CD%
    python scripts/run_profile_engine_cerebras.py

Loads backend/.env if present (CEREBRAS_API_KEY, optional CEREBRAS_MODEL, etc.).
Uses preset answers for follow-up questions so the run is non-interactive.

Requires: httpx, pydantic (same as the app).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path


def _load_dotenv() -> None:
    backend = Path(__file__).resolve().parents[1]
    env_path = backend / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# Short answers so the engine can finish without stdin
_PRESET_ANSWERS: dict[str, str] = {
    "monthly_revenue": "About $8k MRR.",
    "team_size": "Four people including founders.",
    "primary_channel": "Mostly LinkedIn and some content marketing.",
    "biggest_bottleneck": "Converting traffic into qualified demos.",
    "time_in_market": "Roughly one year.",
    "target_market_geo": "United States and Canada.",
    "budget_range": "Around $2k per month on marketing.",
}


async def _run_demo() -> None:
    backend = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend))

    from app.schemas.profile_engine import SessionState, SessionStatus
    from app.services.llm.client import get_llm_client
    from app.services.profile_engine import question_selector
    from app.services.profile_engine.orchestrator import ProfileEngineOrchestrator

    sid = uuid.uuid4()
    raw = (
        "We are a B2B SaaS startup selling marketing automation to SMBs. "
        "MVP launched a few months ago; we need more qualified leads."
    )

    orch = ProfileEngineOrchestrator(llm=get_llm_client())
    state = SessionState(
        session_id=sid,
        status=SessionStatus.COLLECTING_INPUT,
        raw_input=raw,
        answers={},
    )

    print("Running extract -> classify -> readiness -> need routing (Cerebras)...\n")
    state = await orch.run_initial_pipeline(state)

    snapshot = {
        "extracted": state.extracted.model_dump() if state.extracted else None,
        "classified": state.classified.model_dump() if state.classified else None,
        "readiness": state.readiness.model_dump() if state.readiness else None,
        "need_routing": state.need_routing.model_dump() if state.need_routing else None,
        "confidence": state.confidence_score,
    }
    print(json.dumps(snapshot, indent=2, default=str))
    print()

    while True:
        q = question_selector.select_question(state)
        if q is None:
            break
        ans = _PRESET_ANSWERS.get(q.key, "Not sure yet.")
        print(f"Q [{q.key}]: {q.text}\nA: {ans}\n")
        step = await orch.process_answer(state, q.key, ans)
        if step.error:
            print("Error:", step.error, step.message, file=sys.stderr)
            sys.exit(1)

    print("Building final profile (Cerebras)...\n")
    final_step = await orch.build_final_profile(state)
    if final_step.error or not final_step.profile:
        print("Build failed:", final_step.error, final_step.message, file=sys.stderr)
        sys.exit(1)

    print(json.dumps(final_step.profile.model_dump(), indent=2, default=str))


def main() -> None:
    _load_dotenv()
    if not os.getenv("CEREBRAS_API_KEY"):
        print(
            "Set CEREBRAS_API_KEY (e.g. in backend/.env) or export it in the shell.",
            file=sys.stderr,
        )
        sys.exit(1)
    asyncio.run(_run_demo())


if __name__ == "__main__":
    main()
