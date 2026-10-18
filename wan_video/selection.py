"""Region selections for video editing: masks built from polygons, boxes and mask images.

A selection is drawn on (or uploaded for) the video's frame size and applies to every frame.
White = the region to regenerate.
"""

from __future__ import annotations

import json

from PIL import Image, ImageChops, ImageDraw, ImageFilter


def mask_from_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """White or opaque pixels count as selected."""
    if "A" in img.getbands() and img.getchannel("A").getextrema()[0] < 255:
        m = img.getchannel("A")
    else:
        m = img.convert("L")
    m = m.point(lambda v: 255 if v >= 128 else 0)
    return m.resize(size, Image.Resampling.NEAREST) if m.size != size else m


def build_mask(size, mask_img=None, polygons=(), boxes=()) -> Image.Image | None:
    """The union of a mask image, lasso polygons and boxes, or None when nothing is selected."""
    m = mask_from_image(mask_img, size) if mask_img is not None else None
    if polygons or boxes:
        drawn = Image.new("L", size, 0)
        draw = ImageDraw.Draw(drawn)
        for poly in polygons:
            if len(poly) >= 3:
                draw.polygon([(float(x), float(y)) for x, y in poly], fill=255)
        for x0, y0, x1, y1 in boxes:
            draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
        m = drawn if m is None else ImageChops.lighter(m, drawn)
    return None if is_empty(m) else m


def is_empty(mask: Image.Image | None) -> bool:
    return mask is None or mask.getbbox() is None


def parse_selection_json(text: str) -> dict:
    """``{"polygons": [[[x, y], ...]], "boxes": [[x0, y0, x1, y1]]}`` or a bare list of polygons."""
    data = json.loads(text)
    if isinstance(data, list):
        data = {"polygons": data}
    if not isinstance(data, dict):
        raise ValueError("selection must be an object or a list of polygons")
    polygons, boxes = data.get("polygons") or [], data.get("boxes") or []
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 3 or any(len(p) != 2 for p in poly):
            raise ValueError("each polygon needs at least 3 [x, y] points")
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("each box must be [x0, y0, x1, y1]")
    return {"polygons": polygons, "boxes": boxes}


def grow(mask: Image.Image, px: int) -> Image.Image:
    """Dilate a mask a little so edits blend past the selection's edge."""
    if px <= 0:
        return mask
    return mask.filter(ImageFilter.MaxFilter(px * 2 + 1))
