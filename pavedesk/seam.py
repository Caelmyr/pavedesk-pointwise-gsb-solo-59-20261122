"""Repair seam color matching with a shared neutral-card calibration.

Two differently lit captures of one repair patch each carry a grey card from
the same physical batch.  A single neutral target drives one channel-gain
record per capture: the gains remove the *lighting* cast but the two ends are
never pulled toward each other, so a genuine color change at the seam stays
visible.

An end is only declared color-comparable when, after the shared calibration,
the red and blue channels stay within 5% of the green channel on that end's
grey card.  An end that fails the card check is reported as non-comparable and
is never folded into a normal delta-E figure.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .art import INK, SHADOW, font, stored_preview
from .color import (
    delta_e,
    float_list,
    reference_fingerprint,
    robust_neutral,
)
from .geometry import polygon_mask, region_domain, specimen_frame
from .model import InputError, dictionary, identity, scalar, string
from .photos import linear_to_srgb, uint8_image


CARD_CHANNEL_TOLERANCE = 0.05
MIN_EDGE_PIXELS = 80

EDGES = (
    {"key": "old", "label": "Existing paving edge", "short": "Old material"},
    {"key": "new", "label": "Repair fill edge", "short": "New fill"},
)
EDGE_TINT = {"old": (235, 180, 130), "new": (133, 199, 179)}
NOT_COMPARABLE = "Card color not comparable"
COMPARABLE = "Card color comparable"


@dataclass(frozen=True)
class EdgeEnd:
    key: str
    label: str
    photo_id: str
    card_region_id: str
    edge_region_id: str


def end_fields(payload):
    data = dictionary(payload, "Seam calibration request")
    target = scalar(data.get("target", 0.18), "Neutral reflectance target", lower=0.08, upper=0.8)
    ends = []
    for edge in EDGES:
        key = edge["key"]
        ends.append(
            EdgeEnd(
                key=key,
                label=edge["label"],
                photo_id=identity(data.get(f"{key}_photo_id"), f"{edge['short']} photo ID"),
                card_region_id=identity(data.get(f"{key}_card_region_id"), f"{edge['short']} grey card region"),
                edge_region_id=identity(data.get(f"{key}_edge_region_id"), f"{edge['short']} seam edge region"),
            )
        )
    if ends[0].photo_id == ends[1].photo_id:
        raise InputError("The two seam ends must use two different captures.")
    return target, ends


def find_region(tile, region_id, purpose):
    region = next((item for item in tile["regions"] if item["id"] == region_id), None)
    if region is None:
        raise InputError(f"The {purpose} region is no longer on this specimen.")
    return region


def card_neutrality(card_linear, gains):
    """Red/blue deviation relative to green for this end's calibration.

    The grey card is physically neutral, so any imbalance in the recorded
    channels is illumination cast.  The recorded gains undo that cast: the
    ratio of the red (or blue) gain to the green gain states how far the
    correction itself had to move relative to green.  A cast needing more
    than a 5% relative gain shift fails the field comparability rule.  This
    is computed from the gains rather than the corrected median, which the
    gains force to exactly the shared target by construction.
    """
    gains = np.asarray(gains, dtype=np.float32)
    green_gain = max(float(gains[1]), 1e-9)
    red_to_green = float(gains[0] / green_gain)
    blue_to_green = float(gains[2] / green_gain)
    red_dev = abs(red_to_green - 1.0)
    blue_dev = abs(blue_to_green - 1.0)
    comparable = red_dev <= CARD_CHANNEL_TOLERANCE and blue_dev <= CARD_CHANNEL_TOLERANCE
    return {
        "red_to_green": round(red_to_green, 5),
        "blue_to_green": round(blue_to_green, 5),
        "red_deviation_percent": round(red_dev * 100, 3),
        "blue_deviation_percent": round(blue_dev * 100, 3),
        "tolerance_percent": round(CARD_CHANNEL_TOLERANCE * 100, 1),
        "comparable": comparable,
        "status": COMPARABLE if comparable else NOT_COMPARABLE,
        "failed_channels": [
            name
            for name, dev in (("red", red_dev), ("blue", blue_dev))
            if dev > CARD_CHANNEL_TOLERANCE
        ],
    }


def calibrated_card_pixels(frame, card_region):
    candidate = region_domain(frame.extent, {"regions": [card_region]})
    measured = polygon_mask(frame.extent, card_region["vertices"]) & candidate
    support = measured & frame.valid & ~frame.clipped
    count = int(np.count_nonzero(support))
    if count < 40:
        raise InputError("A grey card region needs at least 40 supported, unclipped pixels.")
    return frame.linear[support], count


def calibrate_end(store, tile, end, target):
    photo = next((item for item in tile["photos"] if item["id"] == end.photo_id), None)
    if photo is None:
        raise InputError(f"The capture for the {end.label} is no longer attached to this specimen.")
    card_region = find_region(tile, end.card_region_id, "grey card")
    if card_region["purpose"] != "neutral":
        raise InputError(f"The {end.label} card region must be marked as a neutral color card.")
    edge_region = find_region(tile, end.edge_region_id, "seam edge")
    frame = specimen_frame(store.blobs, tile, end.photo_id)
    full_area = region_domain(frame.extent, tile)
    record = robust_neutral(frame, card_region, full_area, target)
    card_linear, card_pixels = calibrated_card_pixels(frame, card_region)
    gains = np.asarray(record["gains"], dtype=np.float32)
    check = card_neutrality(card_linear, gains)
    record.update(
        {
            "edge_key": end.key,
            "edge_label": end.label,
            "photo_id": end.photo_id,
            "card_region_id": end.card_region_id,
            "edge_region_id": end.edge_region_id,
            "card_neutrality": check,
            "plane_size": list(frame.extent),
        }
    )
    return frame, edge_region, record, card_pixels


def edge_fingerprint(store, tile, frame, edge_region, correction):
    area = region_domain(frame.extent, tile, edge_region)
    support = area & frame.valid & ~frame.clipped
    count = int(np.count_nonzero(support))
    if count < MIN_EDGE_PIXELS:
        raise InputError(
            f"The seam edge region '{edge_region['name']}' needs at least {MIN_EDGE_PIXELS} supported pixels."
        )
    corrected = frame.linear * np.asarray(correction["gains"], dtype=np.float32)
    fingerprint = reference_fingerprint(corrected, support)
    clipped = int(np.count_nonzero(area & frame.clipped))
    return {
        "region_id": edge_region["id"],
        "region_name": edge_region["name"],
        "supported_pixels": count,
        "clipped_pixels": clipped,
        "available_percent": round(100 * count / max(int(np.count_nonzero(area)), 1), 2),
        "fingerprint": fingerprint,
        "support": support,
        "corrected": corrected,
    }


def build_seam_calibration(store, tile, payload):
    target, ends = end_fields(payload)
    assembled = []
    for end in ends:
        frame, edge_region, record, card_pixels = calibrate_end(store, tile, end, target)
        edge = edge_fingerprint(store, tile, frame, edge_region, record)
        record["card_supported_pixels"] = card_pixels
        assembled.append(
            {
                "tile": tile,
                "end": end,
                "frame": frame,
                "card": record,
                "edge": edge,
            }
        )
    targets = {item["card"]["target"] for item in assembled}
    if len(targets) != 1:
        raise InputError("Both seam ends must share one neutral target.")
    return target, assembled


def public_card(record):
    return {
        key: record[key]
        for key in (
            "edge_key", "edge_label", "photo_id", "card_region_id", "edge_region_id",
            "target", "gains", "card_samples", "card_supported_pixels",
            "card_median_linear_rgb", "before_swatch", "after_swatch",
            "card_lightness_range", "plane_size", "card_neutrality",
        )
    }


def public_edge(edge):
    return {key: value for key, value in edge.items() if key not in ("support", "corrected")}


def seam_reference_document(tile, title, family, notes, target, assembled):
    ends = []
    for item in assembled:
        record = item["card"]
        ends.append(
            {
                "edge_key": item["end"].key,
                "edge_label": item["end"].label,
                "card": public_card(record),
                "edge": public_edge(item["edge"]),
            }
        )
    return {
        "kind": "seam",
        "tile_id": tile["id"],
        "title": title,
        "family": family,
        "notes": notes,
        "shared_target": target,
        "ends": ends,
        "calibration_note": (
            "One neutral target drives both grey-card gain records. The ends are not pulled toward "
            "each other; a real color change at the seam remains in the edge fingerprints."
        ),
    }


def _edge_outline(draw, frame, region, color, comparable):
    width, height = frame.extent
    points = [(float(x * width - 0.5), float(y * height - 0.5)) for x, y in region["vertices"]]
    line = max(2, round(min(width, height) / 220))
    draw.line([*points, points[0]], fill=SHADOW, width=line + 3, joint="curve")
    draw.line([*points, points[0]], fill=color, width=line, joint="curve")
    if not comparable:
        x0 = min(point[0] for point in points)
        y0 = min(point[1] for point in points)
        x1 = max(point[0] for point in points)
        y1 = max(point[1] for point in points)
        draw.line((x0, y0, x1, y1), fill=(236, 121, 125), width=line + 2)
        draw.line((x0, y1, x1, y0), fill=(236, 121, 125), width=line + 2)


def seam_panel(assembled, title, subtitle):
    """Side-by-side calibrated views with each edge outlined and marked."""
    panels = []
    for item in assembled:
        frame = item["frame"]
        record = item["card"]
        edge = item["edge"]
        end_meta = item["end"]
        corrected = frame.linear * np.asarray(record["gains"], dtype=np.float32)
        image = uint8_image(linear_to_srgb(corrected), frame.valid.astype(np.float32)).convert("RGB")
        draw = ImageDraw.Draw(image)
        color = EDGE_TINT[end_meta.key]
        edge_region = next(
            region for region in item["tile"]["regions"] if region["id"] == end_meta.edge_region_id
        )
        _edge_outline(draw, frame, edge_region, color, record["card_neutrality"]["comparable"])
        width, height = image.size
        head_font = font(max(13, min(20, round(min(width, height) / 26))))
        body_font = font(max(11, min(16, round(min(width, height) / 34))))
        check = record["card_neutrality"]
        lines = [
            f"{end_meta.label} · {frame.photo['name']}",
            f"R/G {check['red_deviation_percent']:.2f}%  B/G {check['blue_deviation_percent']:.2f}%  (limit 5%)",
            check["status"],
            f"Edge median {edge['fingerprint']['swatch']}  ·  {edge['supported_pixels']} px",
        ]
        block_h = 12 + len(lines) * (head_font.size + 6)
        strip = Image.new("RGB", (width, block_h), (25, 36, 34))
        strip_draw = ImageDraw.Draw(strip)
        strip_draw.text((10, 6), lines[0], font=head_font, fill=INK)
        for index, text_line in enumerate(lines[1:], start=1):
            tone = (246, 169, 142) if index == 2 and not check["comparable"] else (188, 203, 193)
            strip_draw.text((10, 6 + index * (head_font.size + 5)), text_line, font=body_font, fill=tone)
        combined = Image.new("RGB", (width, height + block_h), (25, 36, 34))
        combined.paste(image, (0, 0))
        combined.paste(strip, (0, height))
        panels.append(combined)
    gap = 18
    total_width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    total_height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (total_width, total_height + 64), (24, 31, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, font=font(22), fill=INK)
    draw.text((10, 38), subtitle, font=font(14), fill=(167, 181, 174))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 64))
        x += panel.width + gap
    return sheet


def approve_seam_reference(store, tile, payload):
    data = dictionary(payload, "Seam reference approval")
    title = string(data.get("title"), "Reference name", maximum=100)
    family = string(data.get("family"), "Reference series", maximum=100, required=False)
    notes = string(data.get("notes"), "Reference notes", maximum=1200, required=False)
    target, assembled = build_seam_calibration(store, tile, payload)
    document = seam_reference_document(tile, title, family, notes, target, assembled)
    subtitle = "Shared neutral target " + f"{target:.3f}" + " · calibrated seam edges"
    panel = seam_panel(assembled, title, subtitle)
    document["annotated_digest"] = stored_preview(panel, store.blobs)
    return store.add_reference(document)


def compare_seam_ends(observed, approved):
    """Compare one end at a time; non-comparable ends never yield a delta-E."""
    result = []
    for observed_end in observed:
        key = observed_end["end"].key
        approved_end = next((item for item in approved["ends"] if item["edge_key"] == key), None)
        if approved_end is None:
            raise InputError(f"The approved reference has no recorded {key} seam edge.")
        check = observed_end["card"]["card_neutrality"]
        reference_check = approved_end["card"]["card_neutrality"]
        entry = {
            "edge_key": key,
            "edge_label": observed_end["end"].label,
            "observed_card": public_card(observed_end["card"]),
            "reference_card": approved_end["card"],
            "card_check": check,
            "reference_card_check": reference_check,
        }
        if not check["comparable"] or not reference_check["comparable"]:
            entry.update(
                {
                    "comparable": False,
                    "status": NOT_COMPARABLE,
                    "delta_e76": None,
                    "delta_lab": None,
                    "note": (
                        "This end failed the 5% red/blue-to-green grey-card check in the current sample "
                        "or in the approved reference; no color difference is reported."
                    ),
                }
            )
        else:
            observed_fp = observed_end["edge"]["fingerprint"]
            approved_fp = approved_end["edge"]["fingerprint"]
            delta = np.asarray(observed_fp["median_lab"]) - np.asarray(approved_fp["median_lab"])
            entry.update(
                {
                    "comparable": True,
                    "status": COMPARABLE,
                    "delta_e76": round(
                        float(delta_e(np.asarray(observed_fp["median_lab"]), np.asarray(approved_fp["median_lab"]))), 3
                    ),
                    "delta_lab": float_list(delta),
                    "observed_swatch": observed_fp["swatch"],
                    "reference_swatch": approved_fp["swatch"],
                    "observed_dispersion": observed_fp["dispersion"],
                    "reference_dispersion": approved_fp["dispersion"],
                    "note": "Both cards pass the shared 5% channel check; the edge color distance is reported.",
                }
            )
        result.append(entry)
    return result


def seam_match(store, tile, reference, payload):
    if reference.get("kind") != "seam":
        raise InputError("The selected reference is not a repair seam sample.")
    target, assembled = build_seam_calibration(store, tile, payload)
    if abs(target - reference["shared_target"]) > 1e-9:
        raise InputError(
            f"This reference was approved with neutral target {reference['shared_target']}. "
            f"Re-run the match with that shared target."
        )
    ends = compare_seam_ends(assembled, reference)
    panel = seam_panel(
        assembled,
        reference["title"],
        "Reference match · shared target " + f"{target:.3f}",
    )
    comparable_count = sum(1 for end in ends if end["comparable"])
    return {
        "reference_id": reference["id"],
        "reference_title": reference["title"],
        "reference_image": reference["annotated_digest"],
        "shared_target": target,
        "comparable_ends": comparable_count,
        "total_ends": len(ends),
        "ends": ends,
        "calibration_note": reference["calibration_note"],
    }, panel, assembled
