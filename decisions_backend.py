"""Pass-1 window scoring on a decisions model (OpenRouter /api/alpha/decisions).

``llm_backend`` and ``vision_backend`` both speak ``/chat/completions``. This
one does not, and that is the whole point: a decisions model answers a typed
question about a state instead of generating prose, so it returns in ~0.3s and
costs ~$0.000016 per window where a reasoning chat model spends tens of seconds
and thousands of thinking tokens deciding the same thing.

It only fits pass 1. Scoring a window is a closed question; pass 2 has to write
titles and hooks and choose cut points, which is generation, and the frame
stages need eyes. Both stay on the chat backends. openshorts already splits
score-then-detail, so this drops into the seam that was already there.

Measured on six hand-picked windows: every filler window scored 0.00 at
confidence 1.00, every strong one scored 0.54-0.99, so the separation is clean
at the bottom end, which is what a shortlist needs.

Config:
  SCORE_BACKEND=decisions        turn it on (or set DECISIONS_MODEL)
  DECISIONS_MODEL                default "typesafe/jev-1.13"
  DECISIONS_URL                  default OpenRouter's alpha endpoint
  DECISIONS_API_KEY              falls back to LLM_API_KEY
  DECISIONS_CONCURRENCY          default 8; calls are short, so fan out wider
                                 than the chat path does

This rides OpenRouter's *alpha* path. The request schema is validated strictly
server-side (``questions`` entries need ``type`` of noul|choice|score plus
``instructions`` and ``criteria``), so treat a 400 here as "the shape moved"
and let the caller fall back to the chat backend rather than failing the job.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import httpx

DEFAULT_MODEL = "typesafe/jev-1.13"
DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_TIMEOUT = 120.0

# Two criteria rather than a 0-100 instruction: the model answers with a
# distribution over the criteria, so naming the failure mode explicitly is what
# makes the bottom of the range trustworthy.
CRITERIA = [
    {"name": "weak",
     "description": "filler, mid-sentence, rambling, or no payoff on its own"},
    {"name": "strong",
     "description": "a gripping hook with a clear payoff that stands alone"},
]
INSTRUCTIONS = ("How strong is this transcript window as a standalone "
                "short-form vertical clip for TikTok, Reels or Shorts?")


def model_name() -> str:
    return (os.environ.get("DECISIONS_MODEL") or "").strip() or DEFAULT_MODEL


def url() -> str:
    return (os.environ.get("DECISIONS_URL") or "").strip() or DEFAULT_URL


def _api_key() -> str:
    return (os.environ.get("DECISIONS_API_KEY")
            or os.environ.get("LLM_API_KEY") or "").strip()


def active() -> bool:
    """True when pass-1 scoring should go to the decisions endpoint."""
    explicit = (os.environ.get("SCORE_BACKEND") or "").strip().lower()
    if explicit in ("decisions", "jev"):
        return bool(_api_key())
    if explicit:  # any other explicit value means "not this one"
        return False
    return bool(os.environ.get("DECISIONS_MODEL")) and bool(_api_key())


def describe() -> Optional[dict]:
    if not active():
        return None
    return {"provider": "decisions", "model": model_name(), "url": url()}


def _concurrency() -> int:
    try:
        return max(1, int(os.environ.get("DECISIONS_CONCURRENCY", "")))
    except ValueError:
        return 8


def _timeout() -> float:
    try:
        return float(os.environ.get("DECISIONS_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _score_one(client: httpx.Client, window: dict) -> tuple:
    """``(score_0_100, cost)`` for one window. Raises on transport failures."""
    body = {
        "model": model_name(),
        "state": {"window": str(window.get("text") or "")},
        "questions": {"viral": {"type": "score",
                                "instructions": INSTRUCTIONS,
                                "criteria": CRITERIA}},
    }
    resp = client.post(url(), json=body,
                       headers={"Authorization": f"Bearer {_api_key()}",
                                "Content-Type": "application/json"})
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Decisions server {resp.status_code} from {url()}: {resp.text[:300]}")
    data = resp.json()
    answer = (data.get("answers") or {}).get("viral") or {}
    # The endpoint returns 0..1 over the criteria; the rest of the pipeline
    # sorts on the same 0-100 scale the chat prompt asks for.
    score = float(answer.get("score") or 0.0) * 100.0
    return score, (data.get("usage") or {}).get("cost") or 0.0


def score_windows(windows: List[dict], costs: Optional[list] = None) -> List[dict]:
    """``[{"id": ..., "score": 0-100}]`` for every window, order preserved.

    Raises ``RuntimeError`` when more than half the windows fail, so the caller
    can fall back to the chat backend instead of shortlisting from noise. A
    smaller number of failures keeps its windows at a neutral score: dropping
    them silently would quietly delete candidates the model never judged.
    """
    if not windows:
        return []

    results: List[Optional[tuple]] = [None] * len(windows)

    def _run(i_w):
        i, w = i_w
        try:
            return i, _score_one(client, w)
        except Exception as e:  # noqa: BLE001 - one bad window must not kill the pass
            return i, e

    with _client() as client:
        workers = min(_concurrency(), len(windows))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, outcome in pool.map(_run, enumerate(windows)):
                results[i] = outcome

    failures = [r for r in results if isinstance(r, Exception)]
    if len(failures) * 2 > len(windows):
        raise RuntimeError(
            f"Decisions scoring failed for {len(failures)}/{len(windows)} windows; "
            f"first error: {failures[0]}")

    scored = []
    total_cost = 0.0
    for w, outcome in zip(windows, results):
        if isinstance(outcome, Exception):
            scored.append({"id": w.get("id"), "score": 50.0})
            continue
        score, cost = outcome
        total_cost += float(cost or 0.0)
        scored.append({"id": w.get("id"), "score": score})

    if costs is not None and total_cost:
        costs.append({
            "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
            "input_cost": 0.0, "output_cost": 0.0,
            "total_cost": total_cost, "model": model_name(),
            "price_estimated": False, "local": False,
        })
    if failures:
        print(f"   ⚠️ {len(failures)} window(s) kept at a neutral score "
              f"(decisions call failed: {failures[0]})")
    return scored
