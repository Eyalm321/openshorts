"""Ask Gemini which layout a video needs, once per video.

The four previous attempts at this problem asked the model (or a pixel
heuristic) to MEASURE something — edge density, MSER text density, what fraction
of the duration had content on screen, what fraction of the width it spanned —
and let thresholds turn that number into a routing decision. All four failed the
same way: they could not separate a spreadsheet from a corner scoreboard.

This asks for the decision itself. Measured over the 48-clip corpus against
hand-checked labels (`labels-layout.json` in the reframe-testing skill), three
runs:

    run 1   45/48  94%   18/20 content found   1 false positive of 28
    run 2   44/48  92%   17/20                 1 of 28
    run 3   46/48  96%   18/20                 0 of 28

Two clips out of 48 changed answer between runs. That matters because the
earlier note in this repo said Gemini was too non-deterministic to build on —
the same video scored 1% and 97% coverage on consecutive runs. The variance was
in asking for a continuous measurement, not in the model: a categorical choice
between closed options is stable.

It samples FRAMES instead of uploading the video. Gemini bills video at ~300
tokens per second, so an hour of source is ~1.08M tokens — past a 1M context
window before it starts — and means pushing a 1-2GB upload to get back one word.
This pipeline ingests hour-long podcasts, so that scales badly in exactly the
normal case. Twelve stills cost ~3k tokens whatever the source runs to.

Measured on the same 48 clips (whole video vs sampled frames):

    whole video          94% / 92% / 96%    18,17,18 of 20 found
    12 frames @ 640px    90% / 90%          15,16 of 20
    12 frames @ 1024px   92%                17 of 20
    24 frames @ 1024px   90%                16 of 20

Resolution was the gap, not frame count: at 640px a spreadsheet is unreadable,
and doubling the frames made it slightly worse rather than better. At 1024px the
difference from sending the whole video sits inside the run-to-run variance the
whole-video mode already has, at 2.2s per clip instead of ~15s.

Off by default (``AUTO_LAYOUT=1``). A caller that already switched layouts on
by hand wins: this only ever ADDS, so an explicit choice is never overridden.
"""
import json
import os

# AUTO_LAYOUT=1 decides and applies. AUTO_LAYOUT=shadow decides, logs, and
# applies NOTHING: the render is byte-for-byte what it would have been.
#
# Shadow exists because everything measured about this picker was measured on 48
# YouTube clips chosen by hand, and the material users actually upload is a
# different distribution nobody has looked at. A week of shadow answers "what
# does it say about OUR videos" for 0.002 USD and ~2s per video, with no way to
# damage a clip somebody paid for.
_MODE = os.environ.get("AUTO_LAYOUT", "0").strip().lower()
SHADOW = _MODE == "shadow"
ENABLED = _MODE == "1" or SHADOW

# 12 frames at 1024px wide. Both numbers are measured, not guessed: see above.
SAMPLE_FRAMES = int(os.environ.get("LAYOUT_SAMPLE_FRAMES", "12"))
SAMPLE_WIDTH = int(os.environ.get("LAYOUT_SAMPLE_WIDTH", "1024"))

# What each decision turns on. Keys match the layout names in the prompt.
DECISION_FLAGS = {
    "none": [],
    "screencast": ["screencast_layout"],
    "split": ["split_layout", "active_speaker"],
}

VALID = set(DECISION_FLAGS)


def _module_flags(decision):
    """Modules to enable for a decision, ignoring anything unrecognised."""
    return DECISION_FLAGS.get(str(decision or "none").strip().lower(), [])


def apply(decision):
    """Switch on the modules a decision needs. Returns the modules touched.

    Deliberately additive: an operator who set SPLIT_LAYOUT=1 for a job wants
    stacking regardless of what the model thinks, and a model that says "none"
    must not quietly undo that.
    """
    import active_speaker
    import screencast_layout
    import split_layout

    modules = {"split_layout": split_layout,
               "screencast_layout": screencast_layout,
               "active_speaker": active_speaker}

    touched = []
    for name in _module_flags(decision):
        module = modules.get(name)
        if module is not None and not getattr(module, "ENABLED", False):
            module.ENABLED = True
            touched.append(name)
    return touched


def sample_frames(video_path, n=None, width=None):
    """JPEG bytes for ``n`` frames spread evenly across the video."""
    import cv2

    n = n or SAMPLE_FRAMES
    width = width or SAMPLE_WIDTH
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    try:
        if total <= 0:
            return out
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (width, max(2, int(h * width / w))),
                                interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", scaled,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out.append(buf.tobytes())
    finally:
        cap.release()
    return out


def sample_frames_timed(video_path, duration, n=None, width=None):
    """``(timestamp_seconds, jpeg_bytes)`` for ``n`` frames spread evenly.

    The stages that used to hand Gemini the whole file need to answer in time
    ranges, so they need to know *when* each frame is. Same sampling as
    ``sample_frames``; only the timestamp comes back alongside.
    """
    import cv2

    n = n or SAMPLE_FRAMES
    width = width or SAMPLE_WIDTH
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    out = []
    try:
        if total <= 0:
            return out
        for i in range(n):
            index = int(i * total / n)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            # Prefer the container's own clock; fall back to frame index over
            # fps, then to an even split of the known duration.
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= 0:
                ts = index / fps if fps > 0 else (i * float(duration or 0) / n)
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (width, max(2, int(h * width / w))),
                                interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", scaled,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out.append((round(ts, 2), buf.tobytes()))
    finally:
        cap.release()
    return out


def pick(video_path, video_duration):
    """The layout Gemini picks for this video, or "none" on any failure.

    Never raises: a missing answer has to degrade to today's routing rather
    than break the job.
    """
    if not ENABLED:
        return "none"
    import vision_backend
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not vision_backend.active():
        return "none"

    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
    print("🎛️  Choosing a layout for this video…")
    try:
        # Inside the try on purpose: the contract above is that this never
        # raises, and an unimportable SDK is just one more reason to fall back.
        import gemini_worker

        frames = sample_frames(video_path)
        if not frames:
            print("   ⚠️ No readable frames — keeping the default layout.")
            return "none"

        if vision_backend.active():
            # The OpenAI-compatible path takes the same frames; only the
            # envelope differs, so the answer parses identically below.
            answer, _ = vision_backend.generate_json(
                gemini_worker.LAYOUT_CHOICE_PROMPT, frames, gemini_worker.LayoutChoice)
        else:
            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=api_key)
            parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg")
                     for b in frames]
            response = client.models.generate_content(
                model=model_name,
                contents=parts + [gemini_worker.LAYOUT_CHOICE_PROMPT],
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=gemini_worker.LayoutChoice,
                ))
            gemini_worker.raise_if_blocked(response)
            answer = json.loads(response.text) or {}
    except Exception as e:
        print(f"   ⚠️ Layout choice failed ({e}) — keeping the default layout.")
        return "none"

    decision = str(answer.get("layout", "none")).strip().lower()
    if decision not in VALID:
        print(f"   ⚠️ Unknown layout '{decision}' — keeping the default layout.")
        return "none"

    why = str(answer.get("why", ""))[:80]
    confidence = answer.get("confidence")
    print(f"   🎬 Layout: {decision} (confianza {confidence}) — {why}")
    return decision


def pick_and_apply(video_path, video_duration):
    """Decide, switch on (unless shadowing), report what changed."""
    decision = pick(video_path, video_duration)

    if SHADOW:
        # One greppable line per job. Deliberately not routed through the
        # analytics module: that one is opt-in and host-scoped, and a shadow
        # run has to work on any deployment, including self-hosted.
        would = _module_flags(decision)
        print(f"[layout-shadow] decision={decision} "
              f"would_enable={','.join(would) if would else 'none'} "
              f"duration={video_duration:.0f}s")
        return decision

    touched = apply(decision)
    if touched:
        print(f"   ✅ Enabled: {', '.join(touched)}")
    return decision
