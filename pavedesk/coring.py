"""Plan drill-core positions inside marked material zones on the survey plane."""

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .analysis import evaluate
from .art import INK, SHADOW, base_rgba, curve_points, font, stored_preview
from .color import corrected_linear
from .geometry import exclusion_mask, polygon_mask, specimen_frame
from .model import (
    MAX_REGIONS,
    ConflictError,
    InputError,
    collection,
    dictionary,
    enumeration,
    identity,
    integer,
    scalar,
)
from .store import make_id, timestamp
from .study import STUDY_CHOICES, create_study


DEFAULT_DIAMETER_MM = 100.0
SEVERE_COUNT = 2
CORE_SIDES = 28
MAX_CORES = 4
ZONE_TONES = ((238, 184, 120), (120, 231, 185), (166, 196, 240), (228, 168, 200))
CORE_TONE = (255, 245, 225)
HIT_TONE = (162, 225, 171)
MISS_TONE = (237, 122, 110)
CALM_TONE = (159, 162, 154)
# A 100 mm core covers few registered pixels, so core measurements run at a fitting scale.
CORE_OPTIONS = {
    "color": {"grid": 3},
    "texture": {"background_radius": 2, "edge_smooth": 0.5, "maximum_lag": 4},
    "coverage": {},
    "glare": {"minimum_hotspot": 3},
}


def plan_fields(payload):
    data = dictionary(payload, "Core plan")
    return {
        "measurement_id": identity(data.get("measurement_id"), "Defects measurement"),
        "diameter_mm": scalar(data.get("diameter_mm", DEFAULT_DIAMETER_MM), "Core diameter", lower=20, upper=600),
        "cores": integer(data.get("cores", 2), "Core count", lower=1, upper=MAX_CORES),
    }


def confirmed_distresses(result):
    markers = []
    for observation in result["metrics"]["observations"]:
        if result["decisions"].get(observation["key"], {}).get("decision") != "confirm":
            continue
        markers.append({
            "key": observation["key"],
            "type": observation["type"],
            "strength": observation["strength"],
            "center": observation["normalized_center"],
        })
    markers.sort(key=lambda item: (-item["strength"], item["key"]))
    return markers


def material_zones(tile, extent):
    zones = []
    for region in tile["regions"]:
        if region["purpose"] != "material":
            continue
        mask = polygon_mask(extent, region["vertices"])
        if np.count_nonzero(mask) >= 36:
            zones.append({
                "label": region["name"],
                "region_id": region["id"],
                "vertices": region["vertices"],
                "mask": mask,
            })
    return zones


def pixel_of(point, extent):
    width, height = extent
    return (
        min(height - 1, max(0, int(point[1] * height))),
        min(width - 1, max(0, int(point[0] * width))),
    )


def to_mm(point, dims):
    return np.array([point[0] * dims[0], point[1] * dims[1]])


def mm_distance(first, second, dims):
    return float(np.hypot(*(to_mm(first, dims) - to_mm(second, dims))))


def zone_at(zones, point, extent):
    row, col = pixel_of(point, extent)
    for index, zone in enumerate(zones):
        if zone["mask"][row, col]:
            return index
    return None


def clearance_map(zone, support, scale):
    clear = zone["mask"] & support
    if not clear.any():
        return None
    return ndimage.distance_transform_edt(clear, sampling=(scale[1], scale[0]))


def center_record(pick, distance, radius_mm, extent, zone_index, origin):
    row, col = pick
    width, height = extent
    return {
        "center": [(col + 0.5) / width, (row + 0.5) / height],
        "clearance": round(float(distance[pick]) - radius_mm, 3),
        "zone": zone_index,
        "origin": origin,
    }


def distress_center(mark, feasible, scale, radius_mm, dims):
    height, width = feasible.shape
    target = to_mm(mark["center"], dims)
    rows = (np.arange(height) + 0.5) * scale[1]
    cols = (np.arange(width) + 0.5) * scale[0]
    span = (rows[:, None] - target[1]) ** 2 + (cols[None, :] - target[0]) ** 2
    reach = feasible & (span <= radius_mm**2)
    if not reach.any():
        return None
    return np.unravel_index(np.argmin(np.where(reach, span, np.inf)), span.shape)


def circle_vertices(center, radius_mm, dims):
    return [
        [
            round(center[0] + radius_mm * math.cos(theta) / dims[0], 6),
            round(center[1] + radius_mm * math.sin(theta) / dims[1], 6),
        ]
        for theta in np.linspace(0, 2 * math.pi, CORE_SIDES, endpoint=False)
    ]


def set_score(chosen):
    materials = {item["zone"] for item in chosen}
    severe = set().union(*(item["covers_severe"] for item in chosen))
    other = set().union(*(item["covers_other"] for item in chosen))
    clearance = sum(min(max(item["clearance"], 0.0), 500.0) for item in chosen)
    return (len(severe), len(materials), len(other), round(clearance, 3))


def choose_centers(candidates, cores, dims, diameter_mm):
    if cores == 2:
        best, best_score = None, None
        for first in range(len(candidates)):
            for second in range(first + 1, len(candidates)):
                pair = [candidates[first], candidates[second]]
                if mm_distance(pair[0]["center"], pair[1]["center"], dims) < diameter_mm:
                    continue
                score = set_score(pair)
                if best_score is None or score > best_score:
                    best, best_score = pair, score
        if best is not None:
            return best
    chosen = []
    while len(chosen) < cores:
        pool = [
            item for item in candidates
            if item not in chosen
            and all(mm_distance(item["center"], kept["center"], dims) >= diameter_mm for kept in chosen)
        ]
        if not pool:
            break
        chosen.append(max(pool, key=lambda item: set_score([*chosen, item])))
    return chosen


def plan_notes(circles, unmet):
    lines = [
        f"Core {circle['key']} · {circle['material']}"
        + (f" — covers {', '.join(circle['covers'])}" if circle["covers"] else "")
        for circle in circles
    ]
    missed = [f"{item['key']} ({item['reason']})" for item in unmet["distresses"]]
    if missed:
        lines.append("Not cored: " + "; ".join(missed))
    if unmet["materials"]:
        lines.append("Material not sampled: " + ", ".join(unmet["materials"]))
    if not missed and not unmet["materials"]:
        lines.append("All severe distresses and every marked material are sampled.")
    return lines


def plan_overlay(frame, tile, zones, circles, distress_view, radius_mm, notes):
    linear, _ = corrected_linear(tile, frame, True)
    image = base_rgba(frame, linear)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    stroke = max(1, round(min(width, height) / 300))
    for index, zone in enumerate(zones):
        tone = ZONE_TONES[index % len(ZONE_TONES)]
        points = curve_points(zone["vertices"], image.size)
        draw.line([*points, points[0]], fill=SHADOW, width=stroke + 2, joint="curve")
        draw.line([*points, points[0]], fill=tone, width=stroke, joint="curve")
        draw.text((points[0][0] + 4, points[0][1] + 3), zone["label"][:36], fill=tone, stroke_width=2, stroke_fill=SHADOW)
    for mark in distress_view:
        x, y = curve_points([mark["center"]], image.size)[0]
        radius = max(4, round(min(width, height) / 110))
        tone = HIT_TONE if mark["covered_by"] else MISS_TONE if mark["severe"] else CALM_TONE
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=SHADOW, width=stroke + 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=tone, width=stroke)
        if mark["severe"]:
            outer = radius + 2 * stroke + 2
            draw.ellipse((x - outer, y - outer, x + outer, y + outer), outline=tone, width=stroke)
        if not mark["covered_by"] and mark["severe"]:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=tone, width=stroke)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=tone, width=stroke)
        draw.text((x + radius + 4, y - 6), mark["key"], fill=tone, stroke_width=2, stroke_fill=SHADOW)
    for index, circle in enumerate(circles):
        cx, cy = curve_points([circle["center"]], image.size)[0]
        rx = radius_mm / frame.scale[0]
        ry = radius_mm / frame.scale[1]
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=SHADOW, width=stroke + 3)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=CORE_TONE, width=stroke + 1)
        above = index % 2 == 0
        for row, text in enumerate((f"Core {circle['key']}", circle["material"][:28])):
            tx = cx - draw.textlength(text) / 2
            ty = cy - ry - 26 + row * 13 if above else cy + ry + 6 + row * 13
            draw.text((tx, ty), text, fill=CORE_TONE, stroke_width=2, stroke_fill=SHADOW)
    typeface = font(max(12, round(min(width, height) / 34)))
    pitch = typeface.size + 8
    sheet = Image.new("RGB", (width, height + 10 + pitch * len(notes)), (24, 31, 34))
    sheet.paste(image.convert("RGB"), (0, 0))
    pen = ImageDraw.Draw(sheet)
    for row, line in enumerate(notes):
        pen.text((10, height + 6 + row * pitch), line[:150], font=typeface, fill=INK)
    return sheet


def plan_cores(store, specimen_id, payload):
    fields = plan_fields(payload)
    tile = store.tile(specimen_id)
    if tile.get("plane") is None:
        raise InputError("Set a base photo and survey plane before planning cores.")
    if tile.get("width_mm") is None or tile.get("height_mm") is None:
        raise InputError("Enter the specimen dimensions in millimetres before planning cores.")
    result = store.result(fields["measurement_id"])
    if result["tile_id"] != tile["id"] or result["kind"] != "defects":
        raise InputError("Choose a visible-defects measurement from this specimen.")
    distresses = confirmed_distresses(result)
    if not distresses:
        raise InputError("Confirm at least one defect observation in that measurement first.")
    frame = specimen_frame(store.blobs, tile)
    extent = frame.extent
    support = frame.valid & ~frame.clipped & ~exclusion_mask(extent, tile["regions"])
    zones = material_zones(tile, extent)
    if not zones:
        raise InputError("Draw at least one material zone on the survey plane before planning cores.")
    radius_mm = fields["diameter_mm"] / 2
    dims = (float(tile["width_mm"]), float(tile["height_mm"]))
    distances = [clearance_map(zone, support, frame.scale) for zone in zones]
    feasibles = [None if distance is None else distance >= radius_mm for distance in distances]
    severe = distresses[:SEVERE_COUNT]
    severe_keys = {mark["key"] for mark in severe}
    reasons = {}
    candidates = []
    for mark in severe:
        index = zone_at(zones, mark["center"], extent)
        if index is None:
            reasons[mark["key"]] = "outside every marked material zone"
            continue
        if not support[pixel_of(mark["center"], extent)]:
            reasons[mark["key"]] = "sits on unclear or excluded photo area"
            continue
        if feasibles[index] is None:
            reasons[mark["key"]] = f"no {fields['diameter_mm']:.0f} mm circle fits inside the {zones[index]['label']} zone"
            continue
        pick = distress_center(mark, feasibles[index], frame.scale, radius_mm, dims)
        if pick is None:
            reasons[mark["key"]] = "no drillable centre within one core radius (seam, edge, or unclear photo)"
            continue
        candidates.append(center_record(pick, distances[index], radius_mm, extent, index, mark["key"]))
    for index, distance in enumerate(distances):
        if distance is None or not feasibles[index].any():
            continue
        pick = np.unravel_index(np.argmax(distance), distance.shape)
        candidates.append(center_record(pick, distance, radius_mm, extent, index, None))
    unique = []
    for item in candidates:
        if all(mm_distance(item["center"], kept["center"], dims) >= 1.0 for kept in unique):
            unique.append(item)
    if not unique:
        raise InputError("No drillable core position: every material zone is narrower than one core or lacks clear photo support.")
    for item in unique:
        item["covers_severe"] = {mark["key"] for mark in severe if mm_distance(item["center"], mark["center"], dims) <= radius_mm}
        item["covers_other"] = {
            mark["key"] for mark in distresses[SEVERE_COUNT:]
            if mm_distance(item["center"], mark["center"], dims) <= radius_mm
        }
    chosen = choose_centers(unique, fields["cores"], dims, fields["diameter_mm"])
    chosen.sort(key=lambda item: (item["center"][1], item["center"][0]))
    circles = []
    for letter, item in zip("ABCD", chosen):
        covers = [mark["key"] for mark in distresses if mm_distance(item["center"], mark["center"], dims) <= radius_mm]
        vertices = circle_vertices(item["center"], radius_mm, dims)
        mask = polygon_mask(extent, vertices)
        circles.append({
            "key": letter,
            "center": [round(value, 6) for value in item["center"]],
            "material": zones[item["zone"]]["label"],
            "zone_region_id": zones[item["zone"]]["region_id"],
            "clearance_mm": item["clearance"],
            "covers": covers,
            "support_percent": round(100 * float(np.count_nonzero(mask & support)) / max(int(np.count_nonzero(mask)), 1), 2),
            "vertices": vertices,
        })
    if len(circles) < fields["cores"]:
        reasons["__partial__"] = f"only {len(circles)} of {fields['cores']} requested cores fit the constraints"
    distress_view = []
    missed = []
    for mark in distresses:
        hit = next((circle["key"] for circle in circles if mark["key"] in circle["covers"]), None)
        reason = None
        if hit is None and mark["key"] in severe_keys:
            reason = reasons.get(mark["key"], "cores went to higher-priority targets")
            missed.append({"key": mark["key"], "reason": reason})
        distress_view.append({
            "key": mark["key"],
            "type": mark["type"],
            "strength": mark["strength"],
            "center": mark["center"],
            "severe": mark["key"] in severe_keys,
            "covered_by": hit,
            "reason": reason,
        })
    sampled = {circle["material"] for circle in circles}
    materials = [
        {"label": zone["label"], "region_id": zone["region_id"], "sampled": zone["label"] in sampled}
        for zone in zones
    ]
    unmet = {
        "distresses": missed,
        "materials": [entry["label"] for entry in materials if not entry["sampled"]],
    }
    notes = plan_notes(circles, unmet)
    if "__partial__" in reasons:
        notes.append("Only a partial set of cores fits the marked zones and clear photo support.")
    overlay = stored_preview(plan_overlay(frame, tile, zones, circles, distress_view, radius_mm, notes), store.blobs)
    return store.save_core_plan({
        "tile_id": tile["id"],
        "tile_version": tile["version"],
        "status": "proposed",
        "measurement_id": result["id"],
        "diameter_mm": fields["diameter_mm"],
        "requested_cores": fields["cores"],
        "circles": circles,
        "distresses": distress_view,
        "materials": materials,
        "unmet": unmet,
        "notes": notes,
        "overlay": overlay,
        "confirmed_at": None,
        "outcome": None,
    })


def confirm_plan(store, plan_id, payload):
    data = dictionary(payload, "Core plan confirmation")
    expected = integer(data.get("version"), "Version", lower=1, upper=2_000_000_000)
    kinds = collection(data.get("kinds", ["color", "texture"]), "Measurement kinds", minimum=1, maximum=len(STUDY_CHOICES))
    kinds = list(dict.fromkeys(enumeration(kind, STUDY_CHOICES, "Measurement kind") for kind in kinds))
    plan = store.core_plan(plan_id)
    if plan["version"] != expected:
        raise ConflictError(f"The plan changed in another operation. Refresh and retry. Current version is {plan['version']}.")
    if plan["status"] != "proposed":
        raise InputError("This core plan is already confirmed.")
    tile = store.tile(plan["tile_id"])
    region_ids = {}

    def change(record):
        if len(record["regions"]) + len(plan["circles"]) > MAX_REGIONS:
            raise InputError(f"A specimen supports at most {MAX_REGIONS} regions.")
        for circle in plan["circles"]:
            region_id = make_id()
            region_ids[circle["key"]] = region_id
            record["regions"].append({
                "id": region_id,
                "name": f"Core {circle['key']} · {circle['material']}",
                "purpose": "inspection",
                "vertices": circle["vertices"],
            })

    store.edit_tile(tile["id"], tile["version"], change, "coring", "Marked drilling circles as sampling regions")
    results_by_kind = {}
    study_ids = []
    for kind in kinds:
        result_ids = []
        labels = {}
        for circle in plan["circles"]:
            result = evaluate(store, tile["id"], {
                "kind": kind,
                "region_id": region_ids[circle["key"]],
                "options": CORE_OPTIONS[kind],
            })
            result_ids.append(result["id"])
            labels[result["id"]] = circle["material"]
        study = create_study(store, {
            "title": f"{tile['name']} cores {plan['id'][:6]} · {kind}",
            "group_by": "body",
            "metric": next(iter(STUDY_CHOICES[kind])),
            "result_ids": result_ids,
            "group_labels": labels,
        })
        results_by_kind[kind] = result_ids
        study_ids.append(study["id"])

    def apply(record):
        record["status"] = "confirmed"
        record["confirmed_at"] = timestamp()
        record["outcome"] = {"region_ids": region_ids, "result_ids": results_by_kind, "study_ids": study_ids}

    return store.update_core_plan(plan_id, expected, apply, "Confirmed core plan and built survey comparisons")
