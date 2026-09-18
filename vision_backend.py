"""Frame-based vision backend: the same OpenAI-compatible server, with images.

``llm_backend`` covers the text-only half of the pipeline (transcript scoring).
This module covers the half that has to *look* at the footage: the layout
picker (``layout_picker.py``), the on-screen content detector
(``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). All three sample JPEG frames, so they map onto the
``image_url`` content parts every OpenAI-compatible vision server accepts
(OpenRouter, vLLM, LM Studio, llama.cpp's server, Ollama's vision models) and
none of them need Gemini's Files API.

Configuration rides on ``llm_backend``'s: same ``LLM_BASE_URL``,
``LLM_API_KEY``, ``LLM_TIMEOUT``. The one addition is ``VISION_MODEL``, because
the model that scores a transcript is often not the one that can read a frame
(``llama3.1:8b`` has no eyes). When ``VISION_MODEL`` is unset we fall back to
``LLM_MODEL``, which is right for OpenRouter, where one model id usually does
both.

Without either a vision endpoint or a Gemini key the callers degrade exactly as
they did before: the layout picker and screencast detector return "none" and
the silent-video path fails with a clear message.
"""

from __future__ import annotations

import base64
import os
from typing import Optional, Sequence, Tuple, Type

from pydantic import BaseModel

import llm_backend


def model_name() -> str:
    """The vision model id, falling back to the text one."""
    return (os.environ.get("VISION_MODEL") or "").strip() or llm_backend.model_name()


def active() -> bool:
    """True when frame analysis should call the OpenAI-compatible server."""
    return llm_backend.active()


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": llm_backend.base_url()}


def _image_part(frame: bytes) -> dict:
    # Data URLs rather than hosted links: the frames are transient and a
    # self-hosted install has nowhere public to put them.
    encoded = base64.b64encode(frame).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def generate_json(prompt: str, frames: Sequence[bytes], schema: Type[BaseModel],
                  model: Optional[str] = None) -> Tuple[dict, Optional[dict]]:
    """One vision chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the shape
    ``main._run_gemini_stage`` returns, so callers do not branch on the
    provider. Frames go before the prompt, matching the order the Gemini calls
    used, because the prompts refer to "these frames" in the past tense.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{llm_backend.base_url()}/chat/completions"
    model = model or model_name()
    content = [_image_part(f) for f in frames]
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": content},
    ]
    last_rejection: Optional[str] = None
    with llm_backend._client() as client:
        for fmt in llm_backend._response_formats(schema):
            body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=llm_backend._headers())
            if fmt is not None and llm_backend._is_format_rejection(resp):
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Vision server {resp.status_code} from {url}: {resp.text[:300]}")
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"Vision server rejected every response_format variant: {last_rejection}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost
