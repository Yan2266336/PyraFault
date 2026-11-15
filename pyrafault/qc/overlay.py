"""Provenance overlays (guideline §13.3).

GREEN is where the query pixels actually came from; RED is where the pair claims they
came from. On a clean pair the two rectangles must coincide exactly; on a fault pair they
must separate by the amount the record says.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from pyrafault.geom import BBoxL0

TRUE_COLOR = (0, 200, 0)
CLAIMED_COLOR = (220, 0, 0)
CAPTION_H = 28


def bbox_to_context_px(
    bbox: BBoxL0, context: BBoxL0, output_size: int
) -> tuple[int, int, int, int]:
    """Level-0 bbox to pixel rectangle inside the rendered context image.

    This is a *visualisation* coordinate, derived from the canonical level-0 frame and
    never stored — guideline §4 forbids mixing the two.
    """
    sx = output_size / context.w
    sy = output_size / context.h
    return (
        int(round((bbox.x - context.x) * sx)),
        int(round((bbox.y - context.y) * sy)),
        int(round((bbox.x1 - context.x) * sx)),
        int(round((bbox.y1 - context.y) * sy)),
    )


def render_pair(
    context_img: Image.Image,
    query_img: Image.Image,
    true_px: tuple[int, int, int, int],
    claimed_px: tuple[int, int, int, int],
    caption: str,
    panel: int = 1024,
) -> Image.Image:
    """Context with both boxes on the left, the query pixels on the right."""
    ctx = context_img.convert("RGB").resize((panel, panel), Image.LANCZOS)
    scale = panel / context_img.width
    draw = ImageDraw.Draw(ctx)

    # Claimed first so that on a clean pair the green true box lands on top and is visible.
    for box, color in ((claimed_px, CLAIMED_COLOR), (true_px, TRUE_COLOR)):
        x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
        draw.rectangle([x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)], outline=color, width=3)

    qry = query_img.convert("RGB").resize((panel, panel), Image.LANCZOS)

    out = Image.new("RGB", (panel * 2 + 12, panel + CAPTION_H), (255, 255, 255))
    out.paste(ctx, (0, CAPTION_H))
    out.paste(qry, (panel + 12, CAPTION_H))
    ImageDraw.Draw(out).text((4, 8), caption, fill=(0, 0, 0))
    return out
