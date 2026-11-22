#!/usr/bin/env python3
"""Two-visit loose-edge carryover scenario.

`build` creates one road section measured twice with "Visible defects":
last week's visit has two separate elongated loose-edge marks, this
week's aligned visit has the same two marks joined into one block.
Last week's field-tagged mark is saved with an explicit review decision.

`report` ranks last week's observations by outline overlap with this
week's merged block so a carryover number can be chosen.

Run from the repository root, e.g.
    PYTHONPATH=. python3 scripts/defect_carryover.py build --data data
    PYTHONPATH=. python3 scripts/defect_carryover.py report --data data
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from pavedesk.analysis import evaluate
from pavedesk.carryover import overlap_scores
from pavedesk.demo import (
    DEMO_ROLES,
    PHOTO_SIZE,
    PLANE_SIZE,
    POSES,
    RoadField,
    light_field,
    neutral_card,
    random_field,
    surface_palette,
    training_regions,
    warp_to_photo,
)
from pavedesk.geometry import canvas_dimensions
from pavedesk.photos import png_bytes, store_capture
from pavedesk.review import save_reviews
from pavedesk.store import Store, make_id

SEED = 20261115
LAST_WEEK = "2026-11-15 visit"
THIS_WEEK = "2026-11-22 visit"
MANIFEST = "scenario.json"

# Loose-edge marks in normalized plane coordinates (polyline, width).
MARK_A = ([(0.270, 0.240), (0.326, 0.258), (0.382, 0.276)], 0.012)  # field-tagged last week
MARK_B = ([(0.462, 0.320), (0.546, 0.342), (0.632, 0.364)], 0.012)
BRIDGE = ([(0.382, 0.276), (0.422, 0.298), (0.462, 0.320)], 0.012)  # appeared this week

DEFECT_OPTIONS = {
    "sensitivity": 0.63,
    "spot_radius": 5,
    "polarity": "dark",
    "minimum_area": 6,
    "maximum_area": 1500,
}

FIELD_TAG_NOTE = "现场已贴编号牌，等复查人员回来；松边需复核"


def draw_marks(shape, polylines):
    height, width = shape
    sheet = Image.new("L", (width, height), 0)
    pen = ImageDraw.Draw(sheet)
    for points, width_norm in polylines:
        pixels = [(round(px * width), round(py * height)) for px, py in points]
        line_width = max(2, round(width_norm * height))
        pen.line(pixels, fill=255, width=line_width, joint="curve")
        radius = line_width / 2
        for cx, cy in (pixels[0], pixels[-1]):
            pen.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    mask = np.asarray(sheet, dtype=np.float32) / 255.0
    return np.clip(ndimage.gaussian_filter(mask, 0.6), 0, 1)


def bake_visit(week):
    """Same section both weeks; only the loose-edge marks change."""
    width, height = PLANE_SIZE
    shape = (height, width)
    generator = np.random.default_rng(SEED)
    y, x = np.indices(shape, dtype=np.float32)
    y /= height
    x /= width
    broad = random_field(generator, shape, 64, 82)
    medium = random_field(generator, shape, 20, 29)
    pebble = random_field(generator, shape, 2.8, 3.2)
    small = generator.normal(size=shape).astype(np.float32)
    tones = 0.16 * broad + 0.20 * medium + 0.29 * pebble + 0.19 * small
    diffuse = surface_palette(((52, 50, 46), (86, 82, 75), (128, 122, 110)), tones)
    pits = np.zeros_like(x)
    for _ in range(18):
        px, py = generator.uniform(0.12, 0.90), generator.uniform(0.12, 0.78)
        rad_x, rad_y = generator.uniform(0.004, 0.024), generator.uniform(0.005, 0.018)
        score = ((x - px) / rad_x) ** 2 + ((y - py) / rad_y) ** 2
        pits = np.maximum(pits, np.clip(1 - score, 0, 1) * generator.uniform(0.30, 0.72))
    polylines = [MARK_A, MARK_B] + ([BRIDGE] if week == 2 else [])
    marks = draw_marks(shape, polylines)
    dark = np.clip(0.61 * pits + 0.88 * marks, 0, 0.90)
    diffuse *= (1 - dark)[..., None]
    wet = np.exp(-(((x - 0.70) / 0.25) ** 2 + ((y - 0.62) / 0.22) ** 2))
    roughness = np.clip(0.68 + pebble * 0.13 - wet * 0.19, 0.15, 0.99)
    neutral_card(diffuse, roughness, x, y)
    heights = (
        0.15 * broad + 0.08 * medium + 0.05 * pebble + 0.008 * small
        - pits * 0.30 - marks * 0.18
    )
    return RoadField(np.clip(diffuse, 0, 1), heights, wet, dark, roughness, x, y)


def render(role, field):
    linear = light_field(field, role)
    image = warp_to_photo(linear, POSES[role], PHOTO_SIZE, SEED)
    return Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))


def build_visit(store, week, visit_label):
    field = bake_visit(week)
    photos = []
    for role in DEMO_ROLES:
        image = render(role, field)
        photos.append(store_capture(
            store.blobs, png_bytes(image), role,
            f"section-7-{visit_label.split()[0]}-{role}.png",
            "Aligned repeat survey photograph, rendered locally.",
            make_id(), synthetic=True,
        ))
    base = photos[0]
    size = canvas_dimensions(base, POSES["diffuse"], 3600, 2400)
    return store.create_tile({
        "name": "Riverside walk · Section 7",
        "batch": "North loop",
        "body": "Asphalt",
        "glaze": "Yellow lane paint",
        "firing": visit_label,
        "notes": "Repeat visible-defects survey of one section; visits share the registered plane.",
        "width_mm": 3600,
        "height_mm": 2400,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": POSES["diffuse"], "size": list(size)},
        "regions": training_regions(None),
        "registrations": {
            photo["id"]: {"base_photo_id": base["id"], "corners": POSES[photo["role"]], "support_percent": 100}
            for photo in photos[1:]
        },
    })


def measure_defects_run(store, tile):
    region = next(item for item in tile["regions"] if item["purpose"] == "inspection")
    return evaluate(store, tile["id"], {
        "kind": "defects",
        "region_id": region["id"],
        "options": DEFECT_OPTIONS,
    })


def closest_observation(record, point):
    best, best_distance = None, float("inf")
    for observation in record["metrics"]["observations"]:
        cx, cy = observation["normalized_center"]
        distance = (cx - point[0]) ** 2 + (cy - point[1]) ** 2
        if distance < best_distance:
            best, best_distance = observation, distance
    return best, float(best_distance) ** 0.5


def centroid(polylines):
    points = [point for polyline, _ in polylines for point in polyline]
    return [round(sum(p[0] for p in points) / len(points), 4), round(sum(p[1] for p in points) / len(points), 4)]


def observation_row(record, observation):
    mark = record["decisions"].get(observation["key"], {})
    return {
        "key": observation["key"],
        "type": observation["type"],
        "center": observation["normalized_center"],
        "area_pixels": observation["area_pixels"],
        "strength": observation["strength"],
        "decision": mark.get("decision", "pending"),
        "notes": mark.get("notes", ""),
    }


def build(data_dir):
    store = Store(data_dir)
    tile_last = build_visit(store, 1, LAST_WEEK)
    tile_this = build_visit(store, 2, THIS_WEEK)
    last = measure_defects_run(store, tile_last)
    this = measure_defects_run(store, tile_this)

    mark_a, dist_a = closest_observation(last, centroid([MARK_A]))
    mark_b, dist_b = closest_observation(last, centroid([MARK_B]))
    merged, dist_m = closest_observation(this, centroid([MARK_A, BRIDGE, MARK_B]))
    for name, found, distance in (("A", mark_a, dist_a), ("B", mark_b, dist_b), ("merged", merged, dist_m)):
        if distance > 0.04:
            raise SystemExit(f"Mark {name} was not detected where expected (nearest {distance:.3f} away).")

    # Last week's field triage: the tagged loose edge is flagged for recheck,
    # a few obvious pits are confirmed and one is dismissed.
    decisions = {mark_a["key"]: {"decision": "review", "notes": FIELD_TAG_NOTE}}
    pits = [o for o in last["metrics"]["observations"] if o["key"] not in {mark_a["key"], mark_b["key"]}]
    pits.sort(key=lambda o: -o["strength"])
    for pit in pits[:2]:
        decisions[pit["key"]] = {"decision": "confirm", "notes": "旧坑洞，形态稳定"}
    if len(pits) > 2:
        decisions[pits[2]["key"]] = {"decision": "dismiss", "notes": "阴影，非缺陷"}
    last = save_reviews(store, last["id"], last["version"], decisions)

    manifest = {
        "data_dir": str(data_dir),
        "last_week": {"visit": LAST_WEEK, "tile_id": tile_last["id"], "measurement_id": last["id"]},
        "this_week": {"visit": THIS_WEEK, "tile_id": tile_this["id"], "measurement_id": this["id"]},
        "field_tagged_key": mark_a["key"],
        "other_old_key": mark_b["key"],
        "merged_key": merged["key"],
    }
    out = Path(data_dir) / MANIFEST
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Last week  {LAST_WEEK}: tile {tile_last['id']}  measurement {last['id']}")
    for observation in last["metrics"]["observations"]:
        print("  ", json.dumps(observation_row(last, observation), ensure_ascii=False))
    print(f"This week  {THIS_WEEK}: tile {tile_this['id']}  measurement {this['id']}")
    for observation in this["metrics"]["observations"]:
        print("  ", json.dumps(observation_row(this, observation), ensure_ascii=False))
    print(f"Manifest written to {out}")


def report(data_dir):
    store = Store(data_dir)
    manifest = json.loads((Path(data_dir) / MANIFEST).read_text(encoding="utf-8"))
    last = store.result(manifest["last_week"]["measurement_id"])
    this = store.result(manifest["this_week"]["measurement_id"])
    merged_key = manifest["merged_key"]
    rows = overlap_scores(last, this, merged_key)
    extent = this["support"]["registered_image_extent"]
    print(f"This week's merged block: {merged_key} on measurement {this['id']} (plane {extent[0]}x{extent[1]})")
    print(f"Last week's measurement: {last['id']} ({manifest['last_week']['visit']})")
    print(f"{'old key':<8}{'type':<16}{'decision':<10}{'shared px':>10}{'IoU':>8}{'of new':>8}{'of old':>8}  notes")
    for row in rows:
        print(f"{row['key']:<8}{row['type']:<16}{row['decision']:<10}{row['shared_pixels']:>10}"
              f"{row['iou']:>8.3f}{row['share_of_new']:>8.3f}{row['share_of_old']:>8.3f}  {row['notes']}")
    tagged = [key for key, mark in last["decisions"].items() if mark["decision"] == "review"]
    print("Field-tagged for recheck last week:", ", ".join(tagged) or "(none)")
    if rows:
        print("Max outline overlap:", rows[0]["key"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "report"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.command == "build":
        build(args.data)
    else:
        report(args.data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
