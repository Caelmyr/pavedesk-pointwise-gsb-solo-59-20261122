"""Two-visit polarized moisture recheck.

Combines a morning and a later parallel/cross polarization pair registered to
the same plane. The per-visit signal is the same parallel-minus-cross residual
used by the polarized glare measurement; the second visit turns one-off
specular responses (old paint, dried patches) into reviewable evidence classes.
"""

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .art import base_rgba, font, uint8_image
from .color import linear_lab
from .geometry import region_domain, specimen_frame
from .lighting import captured_pair, glare_scale, registration_pair
from .model import InputError
from .photos import luminance


VISITS = ("morning", "later")
ZONE_CONFIRMED = "confirmed"
ZONE_MORNING_MISSING = "morning-only-missing-later"
ZONE_TRANSIENT = "morning-only-later-dry"
ZONE_PAINT = "old-paint-reflection"
ZONE_UNMEASURED = "unmeasured"
DISPATCH_ZONES = (ZONE_CONFIRMED, ZONE_MORNING_MISSING)
ZONE_META = {
    ZONE_CONFIRMED: {
        "label": "两次可核 · 早图与晚图均见潮痕",
        "map_label": "Confirmed wet - seen in both visits",
        "short": "两次可核",
        "color": (72, 149, 214),
        "dispatch": True,
    },
    ZONE_MORNING_MISSING: {
        "label": "仅早图可见 · 晚图右角漏拍（仍需复查）",
        "map_label": "Morning-only - later corner missed",
        "short": "仅早图·证据不全",
        "color": (238, 155, 63),
        "dispatch": True,
    },
    ZONE_TRANSIENT: {
        "label": "仅早图可见 · 晚图同位置已干（不派复查）",
        "map_label": "Morning-only - dry later, not dispatched",
        "short": "仅早图·晚图已干",
        "color": (224, 205, 92),
        "dispatch": False,
    },
    ZONE_PAINT: {
        "label": "老漆反光 · 按颜色剔除，不派复查",
        "map_label": "Old-paint glare - excluded",
        "short": "老漆反光",
        "color": (191, 105, 178),
        "dispatch": False,
    },
    ZONE_UNMEASURED: {
        "label": "证据不全/无有效像素",
        "map_label": "No usable evidence",
        "short": "无数据",
        "color": (96, 104, 110),
        "dispatch": False,
    },
}
MIN_STRUCTURE = np.ones((3, 3), dtype=bool)


def visit_frames(blobs, tile, photo_ids):
    """Resolve and validate the four captures as two polarization pairs."""
    frames = {}
    for visit in VISITS:
        pair = photo_ids[visit]
        parallel = specimen_frame(blobs, tile, pair["parallel"])
        cross = specimen_frame(blobs, tile, pair["cross"])
        if parallel.photo["role"] != "parallel" or cross.photo["role"] != "cross":
            raise InputError(f"The {visit} pair needs one parallel-polarized and one cross-polarized photo.")
        registration_pair(parallel, cross)
        frames[visit] = {"parallel": parallel, "cross": cross}
    extent = frames[VISITS[0]]["parallel"].extent
    if any(
        frames[visit][role].extent != extent
        for visit in VISITS for role in ("parallel", "cross")
    ):
        raise InputError("All four captures must be registered to the same specimen plane.")
    return frames


def visit_residual(parallel, cross, area, balance, fixed_gain):
    """Polarized residual fraction for one visit, mirroring measure_glare."""
    common, measure, overlap = captured_pair(parallel, cross, area)
    direct = luminance(parallel.linear)
    subdued = luminance(cross.linear)
    ties = area & parallel.valid & cross.valid & ~parallel.clipped & ~cross.clipped
    gain, method = glare_scale(direct, subdued, ties, balance, fixed_gain)
    fitted = subdued * gain
    remaining = np.maximum(direct - fitted, 0)
    fraction = remaining / np.maximum(direct, 0.025)
    return {
        "fraction": fraction.astype(np.float32),
        "measure": measure,
        "common": common,
        "gain": float(gain),
        "balance": method,
        "overlap": overlap,
    }


def smooth_mask(mask, radius):
    if radius <= 0.5:
        return mask.copy()
    size = max(1, int(round(radius)))
    structure = np.ones((2 * size + 1, 2 * size + 1), dtype=bool)
    opened = ndimage.binary_opening(mask, structure=structure, iterations=1)
    return ndimage.binary_closing(opened, structure=structure, iterations=1)


def pavement_reference(frames, common_all):
    """Median Lab of dry, low-glare pavement across both parallel captures."""
    samples = []
    for visit in VISITS:
        parallel = frames[visit]["parallel"]
        lab = linear_lab(parallel.linear)
        samples.append(lab[common_all])
    pooled = np.concatenate(samples, axis=0)
    stride = max(1, len(pooled) // 60000)
    return np.median(pooled[::stride], axis=0).astype(np.float32)


def paint_mask(frames, support, pavement_lab, threshold):
    """Chromatic/light specular old-paint mask from both parallel captures."""
    combined = np.zeros(support.shape, dtype=bool)
    for visit in VISITS:
        lab = linear_lab(frames[visit]["parallel"].linear)
        distance = np.linalg.norm(lab - pavement_lab[None, None, :], axis=-1)
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        painted = support & (distance >= threshold) & (lab[..., 0] >= pavement_lab[0] - 6) & (chroma >= 9)
        combined |= smooth_mask(painted, 1.2)
    return combined


def classify_zones(morning, later, area, later_support, common_support, paint, exclude_paint, minimum_island):
    """Turn per-visit hot masks into evidence classes on the shared plane."""
    hot_am = morning
    hot_pm = later
    hot_any = (hot_am | hot_pm) & area
    # Paint pixels are always reported as their own class. When exclusion is on
    # they are also removed from every dispatchable wet class.
    paint_zone = hot_any & paint
    removable_paint = paint if exclude_paint else np.zeros_like(paint)
    confirmed = hot_am & hot_pm & common_support & ~removable_paint
    morning_missing = hot_am & ~later_support & area & ~removable_paint
    transient = hot_am & ~hot_pm & common_support & ~removable_paint
    # A later-only hot pixel never enters the dispatch range; keep it diagnostic.
    later_only = hot_pm & ~hot_am & common_support & ~removable_paint
    zones = {
        ZONE_CONFIRMED: confirmed,
        ZONE_MORNING_MISSING: morning_missing,
        ZONE_TRANSIENT: transient,
        ZONE_PAINT: paint_zone,
        ZONE_UNMEASURED: area & ~common_support & ~morning_missing,
    }
    # Enforce the minimum island size on the two dispatchable classes.
    for key in DISPATCH_ZONES:
        if minimum_island > 1:
            zones[key] = keep_islands(zones[key], minimum_island)
    diagnostic_later_only = keep_islands(later_only, minimum_island)
    return zones, diagnostic_later_only


def keep_islands(mask, minimum):
    labels, count = ndimage.label(mask, structure=MIN_STRUCTURE)
    if not count:
        return mask
    kept = np.zeros_like(mask)
    for index in range(1, count + 1):
        component = labels == index
        if int(np.count_nonzero(component)) >= minimum:
            kept |= component
    return kept


def component_summaries(mask, frame, limit=24):
    labels, count = ndimage.label(mask, structure=MIN_STRUCTURE)
    width, height = frame.extent
    scale_x, scale_y = frame.scale or (1.0, 1.0)
    entries = []
    for index, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        rows, cols = region
        pixels = int(np.count_nonzero(labels[region] == index))
        if pixels == 0:
            continue
        entries.append({
            "zone_pixels": pixels,
            "area_mm2": round(pixels * scale_x * scale_y, 3) if frame.scale else None,
            "bounds": [
                round(cols.start / width, 5), round(rows.start / height, 5),
                round(cols.stop / width, 5), round(rows.stop / height, 5),
            ],
        })
    entries.sort(key=lambda item: -item["zone_pixels"])
    return entries[:limit]


def area_block(mask, frame, denominator=None):
    pixels = int(np.count_nonzero(mask))
    scale_x, scale_y = frame.scale or (1.0, 1.0)
    return {
        "pixels": pixels,
        "area_mm2": round(pixels * scale_x * scale_y, 3) if frame.scale else None,
        "percent_of_plane": round(100 * pixels / mask.size, 3),
        "percent_of_review_area": round(100 * pixels / denominator, 3) if denominator else None,
    }


def moisture_overlay(frames, zones, later_support):
    """Categorical recheck map drawn over the morning parallel capture."""
    base_frame = frames[VISITS[0]]["parallel"]
    image = base_rgba(base_frame)
    width, height = image.size
    shown = np.asarray(image, dtype=np.float32).copy()
    # Dim the underlying photo so the zone colors read clearly.
    shown[:, :, :3] *= 0.55

    def fill(mask, color, alpha=0.62):
        shown[mask, :3] = shown[mask, :3] * (1 - alpha) + np.asarray(color, dtype=np.float32) * alpha

    # Paint first so dispatch classes win overlaps; unmeasured last as hatching.
    fill(zones[ZONE_PAINT], ZONE_META[ZONE_PAINT]["color"], 0.5)
    fill(zones[ZONE_TRANSIENT], ZONE_META[ZONE_TRANSIENT]["color"], 0.55)
    fill(zones[ZONE_MORNING_MISSING], ZONE_META[ZONE_MORNING_MISSING]["color"], 0.68)
    fill(zones[ZONE_CONFIRMED], ZONE_META[ZONE_CONFIRMED]["color"], 0.72)

    yy, xx = np.ogrid[:height, :width]
    unmeasured = zones[ZONE_UNMEASURED]
    stripe = ((xx + yy) % 16 <= 4) & unmeasured
    shown[unmeasured, :3] = shown[unmeasured, :3] * 0.45 + np.array((46, 50, 54), dtype=np.float32) * 0.55
    shown[stripe, :3] = shown[stripe, :3] * 0.35 + np.array(ZONE_META[ZONE_UNMEASURED]["color"], dtype=np.float32) * 0.65
    # Mark the exact later-visit coverage limit as a single vertical boundary.
    covered_cols = np.nonzero(later_support.any(axis=0))[0]
    if covered_cols.size:
        limit = int(covered_cols.max())
        edge_rows = np.nonzero(later_support[:, max(0, limit - 1)])[0]
        if edge_rows.size:
            shown[edge_rows, limit, :3] = (245, 240, 225)

    result = uint8_image(np.clip(shown[:, :, :3], 0, 255) / 255.0, base_frame.valid.astype(np.float32)).convert("RGBA")
    draw = ImageDraw.Draw(result)
    typeface = font(max(12, min(20, round(min(width, height) / 34))))
    legend_keys = (ZONE_CONFIRMED, ZONE_MORNING_MISSING, ZONE_TRANSIENT, ZONE_PAINT, ZONE_UNMEASURED)
    box_height = 26 * len(legend_keys) + 16
    draw.rounded_rectangle((12, 12, 12 + 360, 12 + box_height), radius=8, fill=(28, 34, 38, 214))
    for index, key in enumerate(legend_keys):
        top = 20 + index * 26
        draw.rectangle((26, top, 46, top + 16), fill=ZONE_META[key]["color"] + (255,), outline=(245, 240, 225, 255))
        draw.text((54, top - 1), ZONE_META[key]["map_label"], font=typeface, fill=(245, 240, 225, 255))
    draw.text((26, 12 + box_height - 2), "White line: later-visit coverage boundary", font=font(11), fill=(220, 216, 205, 255))
    return result


def zone_legend():
    return {
        "mode": "categorical",
        "labels": [
            {"name": ZONE_META[key]["label"], "color": "#%02x%02x%02x" % ZONE_META[key]["color"]}
            for key in (ZONE_CONFIRMED, ZONE_MORNING_MISSING, ZONE_TRANSIENT, ZONE_PAINT, ZONE_UNMEASURED)
        ],
    }


def provenance_for(frame):
    source = frame.photo
    return {
        "id": source["id"],
        "display_name": source["name"],
        "role": source["role"],
        "visit": source.get("visit"),
        "original_digest": source["source_digest"],
        "working_digest": source["working_digest"],
        "raw_size": source["original_size"],
        "upright_size": [source["width"], source["height"]],
        "working_color_space": source["working_color_space"],
        "plane_to_capture": list(frame.transform),
        "synthetic": source["synthetic"],
    }


def measure_moisture(store, tile, photo_ids, region, options):
    frames = visit_frames(store.blobs, tile, photo_ids)
    base_frame = frames[VISITS[0]]["parallel"]
    area = region_domain(base_frame.extent, tile, region)
    residual = {
        visit: visit_residual(
            frames[visit]["parallel"], frames[visit]["cross"], area,
            options["balance"], options["fixed_gain"],
        )
        for visit in VISITS
    }
    am, pm = residual["morning"], residual["later"]
    later_support = pm["measure"]
    morning_support = am["measure"]
    common_support = morning_support & later_support
    if int(np.count_nonzero(common_support)) < 200:
        raise InputError("The two visits share too little supported, unclipped pavement to cross-check moisture.")

    pavement_lab = pavement_reference(frames, common_support)
    # The old-paint map is always computed and reported; the option only
    # controls whether paint pixels are removed from the dispatchable classes.
    paint = paint_mask(frames, morning_support | later_support, pavement_lab, options["paint_color_delta"])

    hot_am = smooth_mask(morning_support & (am["fraction"] >= options["revisit_fraction"]), 1.0)
    hot_pm = smooth_mask(later_support & (pm["fraction"] >= options["revisit_fraction"]), 1.0)
    zones, later_only = classify_zones(
        hot_am, hot_pm, area, later_support, common_support, paint,
        options["exclude_paint"], options["minimum_island"],
    )

    review_total = int(np.count_nonzero(area & morning_support))
    zone_areas = {
        key: area_block(zones[key], base_frame, review_total)
        for key in (ZONE_CONFIRMED, ZONE_MORNING_MISSING, ZONE_TRANSIENT, ZONE_PAINT, ZONE_UNMEASURED)
    }
    dispatch_mask = zones[ZONE_CONFIRMED] | zones[ZONE_MORNING_MISSING]
    dispatch_area = area_block(dispatch_mask, base_frame, review_total)
    components = {
        key: component_summaries(zones[key], base_frame)
        for key in DISPATCH_ZONES
    }
    support = {
        "morning_support_percent": round(100 * np.count_nonzero(morning_support) / area.size, 2),
        "later_support_percent": round(100 * np.count_nonzero(later_support) / area.size, 2),
        "shared_support_percent": round(100 * np.count_nonzero(common_support) / area.size, 2),
        "later_missing_pixels": int(np.count_nonzero(area & ~later_support)),
        "morning_hot_pixels": int(np.count_nonzero(hot_am)),
        "later_hot_pixels": int(np.count_nonzero(hot_pm)),
        "paint_excluded_pixels": int(np.count_nonzero(paint & area)),
        "later_only_hot_pixels": int(np.count_nonzero(later_only)),
    }
    metrics = {
        "unit": "mm²" if base_frame.scale else "px²",
        "revisit_fraction": options["revisit_fraction"],
        "paint_color_delta": options["paint_color_delta"],
        "exclude_paint": options["exclude_paint"],
        "minimum_island": options["minimum_island"],
        "pavement_reference_lab": [round(float(v), 3) for v in pavement_lab],
        "visits": {
            visit: {
                "gain": round(residual[visit]["gain"], 5),
                "balance": residual[visit]["balance"],
                "overlap": residual[visit]["overlap"],
            }
            for visit in VISITS
        },
        "support": support,
        "zone_areas": zone_areas,
        "dispatch_area": dispatch_area,
        "dispatch_classes": [ZONE_META[key]["label"] for key in DISPATCH_ZONES],
        "components": components,
        "later_only_diagnostic": area_block(later_only, base_frame, review_total),
        "note": (
            "Dispatch range = both-visit confirmed plus morning-only wet where the later visit missed the "
            "corner. Morning-only pixels that are dry in the later shared view and old-paint reflections are "
            "reported but excluded from dispatch."
        ),
    }
    overlay = moisture_overlay(frames, zones, later_support)
    visits_provenance = {
        visit: {
            "parallel": provenance_for(frames[visit]["parallel"]),
            "cross": provenance_for(frames[visit]["cross"]),
        }
        for visit in VISITS
    }
    return metrics, overlay, frames, visits_provenance, zone_legend()
