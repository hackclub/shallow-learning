from typing import Tuple
import os
from PIL import Image

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def optimize_sprite(src_name: str, dst_basename: str, target_px_h: int = 32) -> str:
    """Create an 8-bitty optimized copy of a sprite in the static dir.

    - Preserves aspect ratio and transparency
    - Scales height to target_px_h with nearest-neighbor
    - Quantizes to 32 colors
    - Saves as '<dst_basename>_optimized.png'
    """
    _ensure_dir(STATIC_DIR)
    src = os.path.join(STATIC_DIR, src_name)
    out = os.path.join(STATIC_DIR, f"{os.path.splitext(dst_basename)[0]}_optimized.png")
    if not os.path.exists(src):
        return out
    img = Image.open(src).convert('RGBA')
    w, h = img.size
    h = max(1, h)
    scale = float(target_px_h) / float(h)
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    img_small = img.resize((tw, th), Image.NEAREST)
    # Use FASTOCTREE which supports RGBA; disable dithering for crisp pixels
    img_quant = img_small.quantize(colors=32, method=Image.FASTOCTREE, dither=Image.Dither.NONE)
    img_out = img_quant.convert('RGBA')
    img_out.save(out)
    return out

def optimize_all(target_px_h: int = 32) -> None:
    for name in ('dino.png', 'coin.png', 'boots.png', 'raccoon.png'):
        optimize_sprite(name, name, target_px_h)


