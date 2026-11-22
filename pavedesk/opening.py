import numpy as np
from PIL import Image, ImageDraw

from .analysis import capture_provenance, specimen_provenance
from .art import ACCENT, SHADOW, base_rgba, curve_points, stored_preview
from .geometry import polygon_mask, region_domain, specimen_frame
from .metrology import polygon_geometry
from .model import (
    InputError,
    MAX_REGIONS,
    dictionary,
    identity,
    model_version,
    scalar,
    string,
    unique_ids,
)
from .store import make_id


PASSAGE_TINT = (126, 187, 218)
LEFTOVER_TINT = (237, 122, 110)
COVERED_OUTLINE = (199, 212, 151)
INSIDE_RATIO = 99.5
OUTSIDE_RATIO = 0.5
MAX_LOOSE_BLOCKS = 12


def opening_fields(payload):
    data = dictionary(payload, "Opening plan")
    block_ids = unique_ids(data.get("block_ids"), "Loose blocks", minimum=1, maximum=MAX_LOOSE_BLOCKS)
    deepest_id = identity(data.get("deepest_id"), "Deepest block")
    if deepest_id not in block_ids:
        raise InputError("The deepest block must be one of the loose blocks.")
    passage_id = identity(data.get("passage_id"), "Passage region")
    if passage_id in block_ids:
        raise InputError("The keep-clear passage cannot also be a loose block.")
    return {
        "block_ids": block_ids,
        "deepest_id": deepest_id,
        "passage_id": passage_id,
        "template_w": scalar(data.get("template_w", 900), "Template width", lower=150, upper=6000),
        "template_h": scalar(data.get("template_h", 600), "Template height", lower=150, upper=6000),
    }


def integral_table(mask):
    table = mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    return np.pad(table, ((1, 0), (1, 0)))


def window_sums(table, width, height):
    return table[height:, width:] - table[:-height, width:] - table[height:, :-width] + table[:-height, :-width]


def compute_layout(tile, frame, spec):
    if tile.get("plane") is None:
        raise InputError("Set a base photo and mark the survey plane first.")
    if frame.scale is None:
        raise InputError("Enter the specimen dimensions in millimetres before planning an opening.")
    regions = {region["id"]: region for region in tile["regions"]}
    wanted = [*spec["block_ids"], spec["passage_id"]]
    if any(key not in regions for key in wanted):
        raise InputError("A marked region was removed. Draw it again before planning the opening.")
    blocks = [regions[key] for key in spec["block_ids"]]
    passage = regions[spec["passage_id"]]
    deepest = regions[spec["deepest_id"]]
    width_px, height_px = frame.extent
    scale_x, scale_y = frame.scale
    passage_mask = polygon_mask(frame.extent, passage["vertices"])
    if not passage_mask.any():
        raise InputError("The passage region has no area in the survey plane.")
    prow, pcol = np.nonzero(passage_mask)
    passage_wide = (pcol.max() + 1 - pcol.min()) * scale_x
    passage_tall = (prow.max() + 1 - prow.min()) * scale_y
    long_side = max(spec["template_w"], spec["template_h"])
    short_side = min(spec["template_w"], spec["template_h"])
    if passage_tall >= passage_wide:
        win_w_mm, win_h_mm = short_side, long_side
    else:
        win_w_mm, win_h_mm = long_side, short_side
    win_w_px = int(round(win_w_mm / scale_x))
    win_h_px = int(round(win_h_mm / scale_y))
    if win_w_px > width_px or win_h_px > height_px:
        raise InputError("The template is larger than the surveyed plane.")
    if min(win_w_px, win_h_px) < 2:
        raise InputError("The template is smaller than the survey resolution.")
    max_x = width_px - win_w_px
    max_y = height_px - win_h_px
    clear = window_sums(integral_table(passage_mask), win_w_px, win_h_px) == 0
    if not clear.any():
        raise InputError("No template position fits inside the plane without crossing the passage.")
    block_masks = [(block, polygon_mask(frame.extent, block["vertices"])) for block in blocks]
    loose_union = np.zeros((height_px, width_px), dtype=bool)
    for _, mask in block_masks:
        loose_union |= mask
    coverage = window_sums(integral_table(loose_union), win_w_px, win_h_px)
    masks_by_id = {block["id"]: mask for block, mask in block_masks}
    drow, dcol = np.nonzero(masks_by_id[deepest["id"]])
    x0, x1 = dcol.min(), dcol.max() + 1
    y0, y1 = drow.min(), drow.max() + 1
    contain = np.zeros((max_y + 1, max_x + 1), dtype=bool)
    lo_x, hi_x = max(0, x1 - win_w_px), min(x0, max_x)
    lo_y, hi_y = max(0, y1 - win_h_px), min(y0, max_y)
    if lo_x <= hi_x and lo_y <= hi_y:
        contain[lo_y:hi_y + 1, lo_x:hi_x + 1] = True
    enclosable = bool((clear & contain).any())
    search = clear & contain if enclosable else clear
    best = int(coverage[search].max())
    tolerance = max(4, best // 1000)
    candidates = search & (coverage >= best - tolerance)
    centre_y = float(drow.mean())
    centre_x = float(dcol.mean())
    grid_y, grid_x = np.mgrid[0:max_y + 1, 0:max_x + 1]
    distance = ((grid_x + win_w_px / 2 - centre_x) * scale_x) ** 2 + (
        (grid_y + win_h_px / 2 - centre_y) * scale_y
    ) ** 2
    chosen = int(np.argmin(np.where(candidates, distance, np.inf)))
    win_y, win_x = divmod(chosen, max_x + 1)
    vertices = [
        [round(win_x / width_px, 6), round(win_y / height_px, 6)],
        [round((win_x + win_w_px) / width_px, 6), round(win_y / height_px, 6)],
        [round((win_x + win_w_px) / width_px, 6), round((win_y + win_h_px) / height_px, 6)],
        [round(win_x / width_px, 6), round((win_y + win_h_px) / height_px, 6)],
    ]
    px_area = scale_x * scale_y
    window_mask = np.zeros((height_px, width_px), dtype=bool)
    window_mask[win_y:win_y + win_h_px, win_x:win_x + win_w_px] = True
    leftover_mask = np.zeros((height_px, width_px), dtype=bool)
    reports = []
    left_out = []
    for block, mask in block_masks:
        total = int(np.count_nonzero(mask))
        inside = int(np.count_nonzero(mask & window_mask))
        percent = round(100 * inside / total, 2) if total else 0.0
        status = "inside" if percent >= INSIDE_RATIO else "outside" if percent <= OUTSIDE_RATIO else "partial"
        if status != "inside":
            left_out.append(block["name"])
            leftover_mask |= mask & ~window_mask
        reports.append({
            "region_id": block["id"],
            "name": block["name"],
            "area_mm2": round(total * px_area, 1),
            "covered_mm2": round(inside * px_area, 1),
            "covered_percent": percent,
            "status": status,
        })
    enclosed = bool(contain[win_y, win_x])
    total_loose = int(np.count_nonzero(loose_union))
    notes = []
    if enclosed:
        notes.append("The deepest block sits fully inside the window.")
    elif not contain.any():
        notes.append("The deepest block is larger than the template, so no position can enclose it completely.")
    else:
        notes.append("Every position enclosing the deepest block would cross the passage, so the window takes the largest loose-area share instead.")
    if left_out:
        notes.append("Left outside the window: " + ", ".join(left_out) + ".")
    else:
        notes.append("All marked loose blocks are fully inside the window.")
    plan = {
        "template_mm": [win_w_mm, win_h_mm],
        "plane_mm": [round(width_px * scale_x, 1), round(height_px * scale_y, 1)],
        "window": {
            "vertices": vertices,
            "width_mm": win_w_mm,
            "height_mm": win_h_mm,
            "center_mm": [
                round((win_x + win_w_px / 2) * scale_x, 1),
                round((win_y + win_h_px / 2) * scale_y, 1),
            ],
        },
        "passage": {"region_id": passage["id"], "name": passage["name"]},
        "deepest": {
            "region_id": deepest["id"],
            "name": deepest["name"],
            "enclosed": enclosed,
            "enclosable": enclosable,
        },
        "loose": {
            "total_mm2": round(total_loose * px_area, 1),
            "covered_mm2": round(best * px_area, 1),
            "leftover_mm2": round((total_loose - best) * px_area, 1),
            "covered_percent": round(100 * best / total_loose, 2) if total_loose else 0.0,
            "blocks": reports,
        },
        "left_out": left_out,
        "notes": notes,
    }
    return {
        "plan": plan,
        "passage": passage,
        "passage_mask": passage_mask,
        "block_masks": block_masks,
        "reports": reports,
        "deepest_id": deepest["id"],
        "leftover_mask": leftover_mask,
    }


def opening_overlay(frame, layout):
    image = base_rgba(frame)
    raw = np.asarray(image, dtype=np.float32).copy()
    passage_tint = np.array(PASSAGE_TINT, dtype=np.float32)
    leftover_tint = np.array(LEFTOVER_TINT, dtype=np.float32)
    raw[layout["passage_mask"], :3] = raw[layout["passage_mask"], :3] * 0.74 + passage_tint * 0.26
    raw[layout["leftover_mask"], :3] = raw[layout["leftover_mask"], :3] * 0.52 + leftover_tint * 0.48
    image = Image.fromarray(raw.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(min(image.size) / 300))
    status_by_id = {report["region_id"]: report["status"] for report in layout["reports"]}
    for block, _ in layout["block_masks"]:
        color = COVERED_OUTLINE if status_by_id[block["id"]] == "inside" else LEFTOVER_TINT
        points = curve_points(block["vertices"], image.size)
        draw.line([*points, points[0]], fill=SHADOW, width=stroke + 2, joint="curve")
        draw.line([*points, points[0]], fill=color, width=stroke, joint="curve")
        label = block["name"][:36] + (" · deepest" if block["id"] == layout["deepest_id"] else "")
        draw.text((points[0][0] + 5, points[0][1] + 4), label, fill=color, stroke_width=2, stroke_fill=SHADOW)
    passage_points = curve_points(layout["passage"]["vertices"], image.size)
    draw.line([*passage_points, passage_points[0]], fill=SHADOW, width=stroke + 2, joint="curve")
    draw.line([*passage_points, passage_points[0]], fill=PASSAGE_TINT, width=stroke, joint="curve")
    draw.text(
        (passage_points[0][0] + 5, passage_points[0][1] + 4),
        "Passage · keep clear",
        fill=PASSAGE_TINT,
        stroke_width=2,
        stroke_fill=SHADOW,
    )
    window_points = curve_points(layout["plan"]["window"]["vertices"], image.size)
    draw.line([*window_points, window_points[0]], fill=SHADOW, width=stroke + 4, joint="curve")
    draw.line([*window_points, window_points[0]], fill=ACCENT, width=stroke + 2, joint="curve")
    label = f"{layout['plan']['template_mm'][0]:g} x {layout['plan']['template_mm'][1]:g} mm"
    draw.text((window_points[0][0] + 6, window_points[0][1] + 6), label, fill=ACCENT, stroke_width=2, stroke_fill=SHADOW)
    return image


def plan_opening(store, specimen_id, payload):
    spec = opening_fields(payload)
    tile = store.tile(specimen_id)
    frame = specimen_frame(store.blobs, tile)
    layout = compute_layout(tile, frame, spec)
    return layout["plan"], opening_overlay(frame, layout)


def adopt_opening(store, specimen_id, payload):
    data = dictionary(payload, "Opening adoption")
    title = string(data.get("title"), "Opening title", maximum=90)
    expected = model_version(data)
    spec = opening_fields(data)
    tile = store.tile(specimen_id)
    frame = specimen_frame(store.blobs, tile)
    layout = compute_layout(tile, frame, spec)
    region = {
        "id": make_id(),
        "name": title[:80],
        "purpose": "inspection",
        "vertices": layout["plan"]["window"]["vertices"],
    }

    def change(current):
        if not current["plane"]:
            raise InputError("Set a base photo before saving an opening.")
        if len(current["regions"]) >= MAX_REGIONS:
            raise InputError(f"A specimen supports at most {MAX_REGIONS} regions.")
        current["regions"].append(region)

    updated = store.edit_tile(specimen_id, expected, change, "region", f"Saved repair opening region: {region['name']}")
    available = region_domain(frame.extent, updated) & frame.valid
    shape, _ = polygon_geometry(updated, frame, region, available)
    record = store.save_metrology({
        "title": title,
        "mode": "area",
        "tile_id": updated["id"],
        "specimen": specimen_provenance(updated),
        "source": capture_provenance(frame),
        "region": region,
        "exclusions": [candidate for candidate in updated["regions"] if candidate["purpose"] == "ignore"],
        "shape": shape,
        "opening": layout["plan"],
        "image": stored_preview(opening_overlay(frame, layout), store.blobs),
    })
    return updated, record
