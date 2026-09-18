"""The OpenAI-compatible vision backend (vision_backend.py).

The frame-reading half of the pipeline used to be Gemini-only, which meant a
self-hoster on OpenRouter lost the layout picker, the screencast detector and
the silent-video path. These tests pin the contract those three rely on: frames
ride as base64 ``image_url`` parts, the answer comes back in the same
``(parsed, cost)`` shape as the Gemini stage, and the vision model is
selectable apart from the text one.
"""
import base64
import json

import httpx
import pytest

import gemini_worker
import llm_backend
import vision_backend


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)


def _serve(handler, monkeypatch):
    """Route the shared httpx client through an in-process handler."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 900, "completion_tokens": 30},
    })


LAYOUT_ANSWER = {"layout": "none", "why": "talking head", "confidence": 0.9}


# --- activation -----------------------------------------------------------

def test_inactive_without_a_base_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert vision_backend.active() is False
    assert vision_backend.describe() is None


def test_active_follows_the_text_backend(local):
    assert vision_backend.active() is True
    assert vision_backend.describe()["baseUrl"] == "http://llm.test/v1"


# --- model selection ------------------------------------------------------

def test_model_falls_back_to_the_text_model(local):
    # One id usually does both on OpenRouter, so an unset VISION_MODEL is the
    # common case and must not be an error.
    assert vision_backend.model_name() == "qwen2.5:14b"


def test_vision_model_overrides_the_text_model(local, monkeypatch):
    # The text model often has no eyes (llama3.1:8b); this is the escape hatch.
    monkeypatch.setenv("VISION_MODEL", "openai/gpt-5-mini")
    assert vision_backend.model_name() == "openai/gpt-5-mini"
    assert llm_backend.model_name() == "qwen2.5:14b"


# --- request shape --------------------------------------------------------

def test_frames_ride_as_base64_image_parts(local, monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion(LAYOUT_ANSWER)

    _serve(handler, monkeypatch)
    vision_backend.generate_json("pick one", [b"\xff\xd8jpeg-a", b"\xff\xd8jpeg-b"],
                                 gemini_worker.LayoutChoice)

    content = seen["body"]["messages"][-1]["content"]
    images = [p for p in content if p["type"] == "image_url"]
    texts = [p for p in content if p["type"] == "text"]
    assert len(images) == 2
    assert texts[-1]["text"] == "pick one"
    # Frames before the prompt: the prompts refer to "these frames" already shown.
    assert content[0]["type"] == "image_url"
    head, _, payload = images[0]["image_url"]["url"].partition(",")
    assert head == "data:image/jpeg;base64"
    assert base64.b64decode(payload) == b"\xff\xd8jpeg-a"


def test_uses_the_vision_model_in_the_request(local, monkeypatch):
    monkeypatch.setenv("VISION_MODEL", "anthropic/claude-sonnet-5")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion(LAYOUT_ANSWER)

    _serve(handler, monkeypatch)
    vision_backend.generate_json("pick one", [b"f"], gemini_worker.LayoutChoice)
    assert seen["body"]["model"] == "anthropic/claude-sonnet-5"


# --- response contract ----------------------------------------------------

def test_returns_the_gemini_stage_shape(local, monkeypatch):
    _serve(lambda r: _completion(LAYOUT_ANSWER), monkeypatch)
    parsed, cost = vision_backend.generate_json("pick one", [b"f"],
                                                gemini_worker.LayoutChoice)
    assert parsed["layout"] == "none"
    assert cost["input_tokens"] == 900
    assert cost["local"] is True
    assert cost["total_cost"] == 0.0


def test_schema_violations_raise(local, monkeypatch):
    # A model that drops a field must fail here, not deep in the clip pipeline.
    _serve(lambda r: _completion({"why": "no layout key"}), monkeypatch)
    with pytest.raises(Exception):
        vision_backend.generate_json("pick one", [b"f"], gemini_worker.LayoutChoice)


def test_http_errors_raise_with_the_url(local, monkeypatch):
    _serve(lambda r: httpx.Response(500, text="boom"), monkeypatch)
    with pytest.raises(RuntimeError, match="Vision server 500"):
        vision_backend.generate_json("pick one", [b"f"], gemini_worker.LayoutChoice)


def test_response_format_ladder_falls_back(local, monkeypatch):
    # Same ladder as llm_backend: json_schema, then json_object, then nothing.
    attempts = []

    def handler(request):
        body = json.loads(request.content)
        attempts.append((body.get("response_format") or {}).get("type"))
        if len(attempts) == 1:
            return httpx.Response(400, text="unsupported response_format")
        return _completion(LAYOUT_ANSWER)

    _serve(handler, monkeypatch)
    parsed, _ = vision_backend.generate_json("pick one", [b"f"],
                                             gemini_worker.LayoutChoice)
    assert attempts == ["json_schema", "json_object"]
    assert parsed["layout"] == "none"


# --- routing: the three callers actually reach this backend ---------------

def test_layout_picker_routes_to_the_vision_server(local, monkeypatch):
    """The whole point: no GEMINI_API_KEY, and the layout picker still answers."""
    import layout_picker

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(layout_picker, "ENABLED", True)
    monkeypatch.setattr(layout_picker, "sample_frames", lambda *a, **k: [b"f1", b"f2"])
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return _completion({"layout": "split", "why": "two speakers", "confidence": 0.8})

    _serve(handler, monkeypatch)
    assert layout_picker.pick("/video.mp4", 60) == "split"
    assert len(calls) == 1
    assert sum(1 for p in calls[0]["messages"][-1]["content"]
               if p["type"] == "image_url") == 2


def test_screencast_routes_to_the_vision_server(local, monkeypatch):
    import layout_picker
    import screencast_layout

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(screencast_layout, "ENABLED", True)
    monkeypatch.setattr(layout_picker, "sample_frames_timed",
                        lambda *a, **k: [(0.0, b"f1"), (10.0, b"f2")])
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"ranges": [
            {"start": 0.0, "end": 10.0, "what": "slides", "width_fraction": 0.9}]})

    _serve(handler, monkeypatch)
    ranges = screencast_layout.detect_content_ranges("/video.mp4", 30)
    assert ranges == [(0.0, 10.0, "slides", 0.9)]
    # The timestamps have to reach the model or it cannot answer in ranges.
    assert "0s, 10s" in seen["body"]["messages"][-1]["content"][-1]["text"]


def test_screencast_never_uploads_the_source_on_this_path(local, monkeypatch):
    """Strictly better privacy than the Gemini path: stills only, no file."""
    import layout_picker
    import screencast_layout

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(screencast_layout, "ENABLED", True)
    monkeypatch.setattr(layout_picker, "sample_frames_timed",
                        lambda *a, **k: [(0.0, b"f1")])
    _serve(lambda r: _completion({"ranges": []}), monkeypatch)

    def explode(*a, **k):
        raise AssertionError("the Gemini Files API must not be touched here")

    monkeypatch.setattr("google.genai.Client", explode)
    assert screencast_layout.detect_content_ranges("/video.mp4", 30) == []
