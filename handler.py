"""RunPod serverless handler for the Qwen-Image-Edit Rapid AIO LoRAs pipeline.

Reuses the Space's own app.py (same pipeline, same LoRA logic, same canvas maths)
so results match the HF ZeroGPU path bit for bit — only the GPU that runs it
changes. Point the Space at this endpoint to spend RunPod balance instead of the
HF quota.

Input:
  { "image_1": b64, "image_2": b64|null, "extras": [b64,...], "prompt": str,
    "lora": str, "seed": int, "randomize_seed": bool, "cfg": float, "steps": int,
    "megapixels": float, "extras_condition_only": bool, "pad_to_canvas": bool,
    "vae_tiling": bool, "res_multiple": int, "ref_megapixels": float,
    "decoder": "vae"|"2x", "keep_decoder_2x": bool,
    "highlight": bool, "highlight_strength": float }
Output: { "image": b64 png, "seed": int, "backend": "runpod" }
"""

import base64
import io
import os
import time
import traceback

import runpod
from PIL import Image

t0 = time.time()
import app as A  # noqa: E402  (loads the pipeline once per worker)

print(f"[handler] pipeline ready in {time.time() - t0:.0f}s | AIO_VERSION={A.AIO_VERSION}")


def _pil(b64):
    if not b64:
        return None
    if isinstance(b64, str) and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def handler(job):
    inp = job.get("input") or {}
    try:
        img1 = _pil(inp.get("image_1"))
        if img1 is None:
            return {"error": "image_1 is required"}
        extras = [_pil(x) for x in (inp.get("extras") or []) if x]
        t = time.time()
        result, seed, _last = A.infer(
            True,
            img1,
            _pil(inp.get("image_2")),
            extras,
            inp.get("prompt") or "",
            inp.get("lora") or A.NONE_LORA,
            int(inp.get("seed") or 0),
            bool(inp.get("randomize_seed", True)),
            float(inp.get("cfg", 1.0)),
            int(inp.get("steps", 4)),
            float(inp.get("megapixels", 1.0)),
            bool(inp.get("extras_condition_only", True)),
            bool(inp.get("pad_to_canvas", True)),
            bool(inp.get("vae_tiling", False)),
            int(inp.get("res_multiple", 16)),
            float(inp.get("ref_megapixels", 0.0)),
            inp.get("decoder", "vae"),
            bool(inp.get("keep_decoder_2x", False)),
            bool(inp.get("highlight", True)),
            float(inp.get("highlight_strength", 1.0)),
        )
        return {
            "image": _b64(result),
            "seed": seed,
            "backend": "runpod",
            "seconds": round(time.time() - t, 2),
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
