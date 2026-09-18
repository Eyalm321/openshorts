"""The CUDA preload that makes GPU transcription work on a plain pip install.

ctranslate2 dlopens libcublas by bare soname; the nvidia wheels install it
somewhere no loader searches. Without the preload a GPU box falls back to CPU
(~10x slower on the sample this was measured against) with a confusing
"libcublas.so.12 not found". The preload is best-effort by design, so these
tests pin that it is safe, idempotent, and never raises.
"""
import transcribe_backends as tb


def test_preload_is_idempotent_and_never_raises(monkeypatch):
    monkeypatch.setattr(tb, "_CUDA_PRELOADED", False)
    tb._preload_cuda_libs()
    assert tb._CUDA_PRELOADED is True
    tb._preload_cuda_libs()  # second call is a no-op, not a reload
    assert tb._CUDA_PRELOADED is True


def test_missing_libraries_are_not_fatal(monkeypatch):
    """A CPU-only box has none of these; that must stay a silent no-op."""
    import ctypes

    monkeypatch.setattr(tb, "_CUDA_PRELOADED", False)

    def boom(*a, **k):
        raise OSError("no such file")

    monkeypatch.setattr(ctypes, "CDLL", boom)
    tb._preload_cuda_libs()  # must not propagate


def test_cublaslt_is_loaded_before_cublas(monkeypatch):
    """libcublas needs libcublasLt already resolved; order is load-bearing."""
    import ctypes
    import glob

    monkeypatch.setattr(tb, "_CUDA_PRELOADED", False)
    monkeypatch.setattr(glob, "glob",
                        lambda pat: [pat.replace("nvidia/*", "nvidia/x")])
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda p, mode=0: loaded.append(p))
    tb._preload_cuda_libs()

    names = [p.rsplit("/", 1)[-1] for p in loaded]
    assert any("cublasLt" in n for n in names)
    lt = next(i for i, n in enumerate(names) if "cublasLt" in n)
    plain = next(i for i, n in enumerate(names) if n.startswith("libcublas.so"))
    assert lt < plain
