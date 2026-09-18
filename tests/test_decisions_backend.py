"""Pass-1 scoring on a decisions model (decisions_backend.py).

Unlike the other two backends this one does not speak /chat/completions, so
these tests pin the request shape the alpha endpoint validates strictly
(state + questions, type "score", instructions, criteria), the 0..1 -> 0-100
rescale the rest of the pipeline sorts on, and the failure policy: a few bad
windows keep a neutral score, a majority raises so the caller can fall back to
the chat backend instead of shortlisting from noise.
"""
import json

import httpx
import pytest

import decisions_backend as db


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("SCORE_BACKEND", "decisions")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.delenv("DECISIONS_MODEL", raising=False)
    monkeypatch.delenv("DECISIONS_CONCURRENCY", raising=False)


def _serve(handler, monkeypatch):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(db, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _answer(score, cost=0.0000157):
    return httpx.Response(200, json={
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"viral": {"type": "score", "score": score, "confidence": 1.0}},
        "usage": {"input_tokens": 372, "output_tokens": 18, "cost": cost},
    })


WINDOWS = [{"id": "w0", "text": "filler filler"},
           {"id": "w1", "text": "a gripping hook"}]


# --- activation -----------------------------------------------------------

def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SCORE_BACKEND", raising=False)
    monkeypatch.delenv("DECISIONS_MODEL", raising=False)
    assert db.active() is False
    assert db.describe() is None


def test_on_when_asked_for(enabled):
    assert db.active() is True
    assert db.describe()["model"] == "typesafe/jev-1.13"


def test_another_score_backend_wins(monkeypatch):
    monkeypatch.setenv("SCORE_BACKEND", "chat")
    monkeypatch.setenv("DECISIONS_MODEL", "typesafe/jev-1.13")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    assert db.active() is False


def test_needs_a_key(monkeypatch):
    monkeypatch.setenv("SCORE_BACKEND", "decisions")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DECISIONS_API_KEY", raising=False)
    assert db.active() is False


# --- request shape --------------------------------------------------------

def test_sends_the_schema_the_alpha_endpoint_validates(enabled, monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _answer(0.9)

    _serve(handler, monkeypatch)
    db.score_windows([WINDOWS[1]])

    assert seen["url"].endswith("/api/alpha/decisions")
    assert seen["auth"] == "Bearer sk-test"
    q = seen["body"]["questions"]["viral"]
    assert q["type"] == "score"                      # noul|choice|score
    assert q["instructions"]                          # required
    assert isinstance(q["criteria"], list) and q["criteria"]   # required
    assert seen["body"]["state"]["window"] == "a gripping hook"


# --- response contract ----------------------------------------------------

def test_rescales_to_the_0_100_the_pipeline_sorts_on(enabled, monkeypatch):
    _serve(lambda r: _answer(0.34), monkeypatch)
    out = db.score_windows([WINDOWS[0]])
    assert out == [{"id": "w0", "score": 34.0}]


def test_order_is_preserved(enabled, monkeypatch):
    # Concurrent fan-out must not reorder; the shortlist maps back by id.
    def handler(request):
        text = json.loads(request.content)["state"]["window"]
        return _answer(0.1 if text == "filler filler" else 0.99)

    _serve(handler, monkeypatch)
    out = db.score_windows(WINDOWS)
    assert [w["id"] for w in out] == ["w0", "w1"]
    assert out[0]["score"] < out[1]["score"]


def test_cost_is_accumulated(enabled, monkeypatch):
    _serve(lambda r: _answer(0.5, cost=0.000002), monkeypatch)
    costs = []
    db.score_windows(WINDOWS, costs)
    assert len(costs) == 1
    assert costs[0]["total_cost"] == pytest.approx(0.000004)
    assert costs[0]["local"] is False


def test_empty_input_is_a_no_op(enabled):
    assert db.score_windows([]) == []


# --- failure policy -------------------------------------------------------

def test_a_minority_of_failures_keeps_neutral_scores(enabled, monkeypatch):
    """Dropping them would silently delete candidates nothing ever judged."""
    windows = [{"id": f"w{i}", "text": f"t{i}"} for i in range(4)]

    def handler(request):
        if json.loads(request.content)["state"]["window"] == "t2":
            return httpx.Response(500, text="boom")
        return _answer(0.8)

    _serve(handler, monkeypatch)
    out = db.score_windows(windows)
    assert len(out) == 4
    assert next(w for w in out if w["id"] == "w2")["score"] == 50.0


def test_a_majority_of_failures_raises_so_the_caller_falls_back(enabled, monkeypatch):
    windows = [{"id": f"w{i}", "text": f"t{i}"} for i in range(4)]
    _serve(lambda r: httpx.Response(400, text="schema moved"), monkeypatch)
    with pytest.raises(RuntimeError, match="Decisions scoring failed"):
        db.score_windows(windows)
