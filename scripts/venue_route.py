#!/usr/bin/env python3
"""Build the cleaning-crew venue map and plan the scrubber-cart route between the gates.

Creates the "Venue cleaning map" specimen in a PaveDesk store (idempotently), plans a
route for the 600 mm scrubber cart between the two marked gates, saves it as a Geometry
trace record, and exports the duty-desk map PNG.

Run from the repository root:

    PYTHONPATH=. python3 scripts/venue_route.py
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

from pavedesk.art import font
from pavedesk.geometry import region_domain, specimen_frame
from pavedesk.metrology import create_metrology
from pavedesk.photos import png_bytes
from pavedesk.store import Store
from pavedesk.workspace import import_photo, save_region

VENUE_NAME = "Venue cleaning map"
ROUTE_TITLE = "Scrubber cart route - West gate to East gate"
PLANE_W, PLANE_H = 1500, 1000
WIDTH_MM, HEIGHT_MM = 30000.0, 20000.0
MM_PER_PX = WIDTH_MM / PLANE_W
PX_PER_M = 1000.0 / MM_PER_PX
CART_WIDTH_MM = 600.0

# Planning rule: keep at least HARD_FLOOR_MM of room on each side of the cart
# centreline, prefer DESIRED_MM where it costs little, and verify the final
# simplified line at VERIFY_MM.
HARD_FLOOR_MM = 700.0
VERIFY_MM = 780.0
DESIRED_MM = 850.0
PENALTY_SPAN = 400.0
COMPARE_FLOORS_MM = (900.0, 1250.0)


def rect(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# Venue layout in metres, plane origin at the north-west corner, y pointing south.
PALLETS = [
    (6.0, 3.5, 11.0, 9.2),
    (6.0, 10.8, 11.0, 16.5),
    (13.5, 6.5, 18.5, 13.5),
    (21.0, 3.5, 26.0, 8.5),
    (21.0, 11.5, 26.0, 16.5),
    (11.0, 0.0, 14.0, 6.5),
]
VENUE_REGIONS = [
    ("West gate", "inspection", rect(0.8, 9.0, 2.4, 11.0)),
    ("East gate", "inspection", rect(27.6, 9.0, 29.2, 11.0)),
    *[("Pallet %d" % (index + 1), "ignore", rect(*box)) for index, box in enumerate(PALLETS)],
    ("NW corner not photographed", "ignore", [[0.0, 0.0], [3.8, 0.0], [0.0, 2.8]]),
    ("SE corner not photographed", "ignore", [[30.0, 20.0], [26.2, 20.0], [30.0, 17.2]]),
]
GATE_WEST = (1.6, 10.0)
GATE_EAST = (28.4, 10.0)


def render_venue_photo():
    """Top-down venue map photo (RGBA); the two unphotographed corners are transparent."""
    generator = np.random.default_rng(20261122)
    height, width = PLANE_H, PLANE_W
    yy, xx = np.indices((height, width), dtype=np.float32)
    xm = xx / PX_PER_M
    ym = yy / PX_PER_M

    def field(space_y, space_x):
        rows = int(np.ceil(height / space_y)) + 4
        cols = int(np.ceil(width / space_x)) + 4
        coarse = generator.normal(size=(rows, cols)).astype(np.float32)
        return ndimage.map_coordinates(coarse, (yy / space_y + 1, xx / space_x + 1), order=3)

    tones = 0.45 * field(95, 130) + 0.35 * field(26, 34) + 0.20 * generator.normal(size=(height, width)).astype(np.float32)
    stops = (-1.1, 0.05, 1.2)
    dark, mid, light = (104, 101, 94), (130, 127, 118), (152, 149, 140)
    rgb = np.stack(
        [np.interp(tones, stops, (dark[c], mid[c], light[c])) for c in range(3)],
        axis=2,
    ).astype(np.float32)
    joint = ((xm % 5.0) < 0.05) | ((ym % 5.0) < 0.05)
    rgb[joint] *= 0.9
    for _ in range(9):
        cx, cy = generator.uniform(2, 28), generator.uniform(2, 18)
        radius = generator.uniform(0.5, 1.8)
        stain = np.exp(-(((xm - cx) ** 2 + (ym - cy) ** 2) / (2 * radius * radius)))
        rgb *= (1 - 0.10 * stain)[..., None]
    dash = (np.abs(ym - 10.0) < 0.08) & ((xm % 2.4) < 1.4)
    wear = np.clip(0.55 + 0.3 * field(40, 55), 0, 1)
    paint = np.array([198, 164, 74], dtype=np.float32)
    blend = (dash * wear * 0.6)[..., None]
    rgb = rgb * (1 - blend) + paint * blend

    def paint_rect(x1m, y1m, x2m, y2m, color, alpha):
        c1, r1 = int(round(x1m * PX_PER_M)), int(round(y1m * PX_PER_M))
        c2, r2 = int(round(x2m * PX_PER_M)), int(round(y2m * PX_PER_M))
        rgb[r1:r2, c1:c2] = rgb[r1:r2, c1:c2] * (1 - alpha) + np.asarray(color, dtype=np.float32) * alpha

    gate_paint = (96, 142, 102)
    for x1, y1, x2, y2 in ((0.8, 9.0, 2.4, 11.0), (27.6, 9.0, 29.2, 11.0)):
        edge = 0.12
        for side in ((x1, y1, x1 + edge, y2), (x2 - edge, y1, x2, y2), (x1, y1, x2, y1 + edge), (x1, y2 - edge, x2, y2)):
            paint_rect(*side, gate_paint, 0.8)
    wood = np.array([150, 112, 70], dtype=np.float32)
    wood_dark = np.array([104, 76, 48], dtype=np.float32)
    for x1, y1, x2, y2 in PALLETS:
        paint_rect(x1 + 0.12, y1 + 0.18, x2 + 0.12, y2 + 0.18, (38, 36, 33), 0.35)
    for x1, y1, x2, y2 in PALLETS:
        tone = wood + generator.uniform(-8, 8)
        paint_rect(x1, y1, x2, y2, tone, 1.0)
        slat = y1 + 0.55
        while slat < y2 - 0.2:
            paint_rect(x1, slat, x2, slat + 0.07, wood_dark, 0.8)
            slat += 0.55
        for side in ((x1, y1, x2, y1 + 0.09), (x1, y2 - 0.09, x2, y2), (x1, y1, x1 + 0.09, y2), (x2 - 0.09, y1, x2, y2)):
            paint_rect(*side, wood_dark, 0.9)
    wall = (52, 54, 56)
    paint_rect(0, 0, 30.0, 0.12, wall, 1.0)
    paint_rect(0, 19.88, 30.0, 20.0, wall, 1.0)
    paint_rect(0, 0, 0.12, 20.0, wall, 1.0)
    paint_rect(29.88, 0, 30.0, 20.0, wall, 1.0)
    paint_rect(0.0, 9.0, 0.35, 11.0, gate_paint, 0.85)
    paint_rect(29.65, 9.0, 30.0, 11.0, gate_paint, 0.85)
    dist = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
    rgb *= (1.0 - 0.10 * np.clip(dist - 0.4, 0, 1))[..., None]

    alpha = np.full((height, width), 255, dtype=np.uint8)
    # The transparent wedges sit slightly inside their marked regions so every
    # unsupported pixel is also inside the named "not photographed" region.
    nw = (xm / 3.55 + ym / 2.6) < 1.0
    se = ((30.0 - xm) / 3.55 + (20.0 - ym) / 2.6) < 1.0
    alpha[nw | se] = 0
    image = Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha]))
    draw = ImageDraw.Draw(image)
    draw.text((40, 150), "WEST GATE", font=font(30), fill=(232, 230, 222))
    draw.text((width - 215, 150), "EAST GATE", font=font(30), fill=(232, 230, 222))
    for index, (x1, y1, x2, y2) in enumerate(PALLETS, 1):
        draw.text((x1 * PX_PER_M + 10, y1 * PX_PER_M + 8), f"P{index}", font=font(26), fill=(240, 236, 228))
    return png_bytes(image)


def build_specimen(store):
    tile = next((item for item in store.list_tiles() if item["name"] == VENUE_NAME), None)
    created = tile is None
    if created:
        tile = store.create_tile({
            "name": VENUE_NAME,
            "batch": "Cleaning crew",
            "body": "Concrete hall floor",
            "glaze": "Aisle and gate markings",
            "firing": "November survey",
            "notes": (
                "Top-down venue map for the cleaning crew. Both gates, six pallet stacks and the two "
                "corners the camera did not cover are marked. Pallet stacks and unphotographed corners "
                "are exclusion regions. The scrubber cart is 600 mm wide."
            ),
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM,
            "synthetic": True,
        })
    if not tile["photos"]:
        tile = import_photo(
            store, tile["id"], tile["version"], render_venue_photo(),
            "venue-map.png", "diffuse",
            "Top-down venue map; the NW and SE corners were not photographed.",
        )
    present = {region["name"] for region in tile["regions"]}
    for name, purpose, vertices_m in VENUE_REGIONS:
        if name in present:
            continue
        vertices = [[round(x / 30.0, 6), round(y / 20.0, 6)] for x, y in vertices_m]
        tile = save_region(store, tile["id"], {
            "version": tile["version"], "name": name, "purpose": purpose, "vertices": vertices,
        })
    return tile, created


def clearance_grid(available, scale):
    padded = np.pad(available, 1, constant_values=False)
    return ndimage.distance_transform_edt(padded, sampling=(scale[1], scale[0]))[1:-1, 1:-1]


def dijkstra_path(coarse, hard, start, target, step_mm):
    height, width = hard.shape
    flat = np.arange(height * width, dtype=np.int64).reshape(height, width)
    rows, cols, data = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)):
        r0, r1 = max(0, -dr), height - max(0, dr)
        c0, c1 = max(0, -dc), width - max(0, dc)
        both = hard[r0:r1, c0:c1] & hard[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        if not both.any():
            continue
        destination = coarse[r0 + dr:r1 + dr, c0 + dc:c1 + dc][both]
        weight = step_mm * math.hypot(dr, dc) * (1.0 + np.maximum(0.0, DESIRED_MM - destination) / PENALTY_SPAN)
        rows.append(flat[r0:r1, c0:c1][both])
        cols.append(flat[r0 + dr:r1 + dr, c0 + dc:c1 + dc][both])
        data.append(weight)
    graph = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(height * width, height * width),
    )
    source = flat[start]
    dist, pred = dijkstra(graph, directed=True, indices=source, return_predecessors=True)
    goal = flat[target]
    if not np.isfinite(dist[goal]):
        return None
    path = []
    node = goal
    while node != source:
        path.append(node)
        node = pred[node]
    path.append(source)
    path.reverse()
    return [(int(node % width), int(node // width)) for node in path]


def simplify(points, segment_ok):
    kept = list(points)
    index = 1
    while index < len(kept) - 1:
        if segment_ok(kept[index - 1], kept[index + 1]):
            del kept[index]
            if index > 1:
                index -= 1
        else:
            index += 1
    return kept


def plan_route(clearance, scale, floor_mm, verify_mm, want_points=False):
    """Shortest penalized centreline between the gates with per-side room >= floor_mm."""
    height, width = clearance.shape
    factor = 2
    ch, cw = height // factor, width // factor
    coarse = clearance[:ch * factor, :cw * factor].reshape(ch, factor, cw, factor).min(axis=(1, 3))
    hard = coarse >= floor_mm
    start_px = (int(GATE_WEST[0] * 1000 / scale[0]), int(GATE_WEST[1] * 1000 / scale[1]))
    target_px = (int(GATE_EAST[0] * 1000 / scale[0]), int(GATE_EAST[1] * 1000 / scale[1]))
    start = (start_px[1] // factor, start_px[0] // factor)
    target = (target_px[1] // factor, target_px[0] // factor)
    if not (hard[start] and hard[target]):
        return None
    labels, _ = ndimage.label(hard)
    if labels[start] != labels[target]:
        return None
    path = dijkstra_path(coarse, hard, start, target, MM_PER_PX * factor)
    if path is None:
        return None

    def to_normalized(col, row):
        return [round((col * factor + 1) / width, 6), round((row * factor + 1) / height, 6)]

    stride = max(1, len(path) // 220)
    points = [to_normalized(col, row) for col, row in path[::stride]]
    if to_normalized(*path[-1]) != points[-1]:
        points.append(to_normalized(*path[-1]))
    points[0] = [round(GATE_WEST[0] / 30.0, 6), round(GATE_WEST[1] / 20.0, 6)]
    points[-1] = [round(GATE_EAST[0] / 30.0, 6), round(GATE_EAST[1] / 20.0, 6)]

    def segment_ok(a, b):
        ax, ay = a[0] * width - 0.5, a[1] * height - 0.5
        bx, by = b[0] * width - 0.5, b[1] * height - 0.5
        length_mm = math.hypot((bx - ax) * scale[0], (by - ay) * scale[1])
        for t in np.linspace(0.0, 1.0, max(2, int(length_mm / 10.0))):
            col = min(max(int(round(ax + (bx - ax) * t)), 0), width - 1)
            row = min(max(int(round(ay + (by - ay) * t)), 0), height - 1)
            if clearance[row, col] < verify_mm:
                return False
        return True

    points = simplify(points, segment_ok)
    while len(points) > 44:
        points = simplify(points[::2] + [points[-1]], segment_ok)
    length_mm = sum(
        math.hypot((b[0] - a[0]) * WIDTH_MM, (b[1] - a[1]) * HEIGHT_MM)
        for a, b in zip(points, points[1:])
    )
    result = {"length_mm": length_mm, "floor_mm": floor_mm}
    if want_points:
        result["points"] = [[round(x, 5), round(y, 5)] for x, y in points]
    return result


def feasible_route(clearance, scale, requested_mm, verify_mm, want_points=False):
    floor = requested_mm
    while floor >= 300.0:
        result = plan_route(clearance, scale, floor, verify_mm, want_points)
        if result is not None:
            return result
        floor -= 50.0
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/workspace/data", help="PaveDesk data directory")
    parser.add_argument("--out", default="/workspace/out", help="Directory for the exported map")
    args = parser.parse_args()
    store = Store(Path(args.data))
    tile, created = build_specimen(store)
    print(f"Specimen '{VENUE_NAME}': {tile['id']} ({'created' if created else 'existing'})")
    frame = specimen_frame(store.blobs, tile)
    available = region_domain(frame.extent, tile) & frame.valid
    clearance = clearance_grid(available, frame.scale)

    straight_mm = math.hypot(GATE_EAST[0] - GATE_WEST[0], GATE_EAST[1] - GATE_WEST[1]) * 1000
    print(f"Straight-line gate distance: {straight_mm / 1000:.2f} m")
    chosen = feasible_route(clearance, frame.scale, HARD_FLOOR_MM, VERIFY_MM, want_points=True)
    if chosen is None:
        raise SystemExit("No route with at least 300 mm per side exists - the cart cannot pass.")
    print(f"Chosen route (per-side room kept >= {chosen['floor_mm']:.0f} mm): {chosen['length_mm'] / 1000:.2f} m")
    for floor in COMPARE_FLOORS_MM:
        alt = feasible_route(clearance, frame.scale, floor, floor - 10)
        note = f"{alt['length_mm'] / 1000:.2f} m (floor {alt['floor_mm']:.0f} mm)" if alt else "no route"
        print(f"Alternative with per-side room >= {floor:.0f} mm: {note}")

    existing = {record["title"]: record for record in store.list_metrology(tile["id"])}
    if ROUTE_TITLE in existing:
        record = existing[ROUTE_TITLE]
        print(f"Geometry record kept: {record['id']}")
    else:
        record = create_metrology(store, tile["id"], {
            "title": ROUTE_TITLE, "mode": "trace", "points": chosen["points"],
        })
        print(f"Geometry record saved: {record['id']}")
    shape = record["shape"]
    profile = shape.get("clearance", {})
    print(f"  trace length: {shape['trace_length'] / 1000:.2f} m, support along line: {shape['overall_support']}%")
    if profile:
        tight = profile["tightest"]
        print(
            f"  tightest at {tight['along'] / 1000:.2f} m: left {tight['left']:.0f} mm ({tight['left_limit']}), "
            f"right {tight['right']:.0f} mm ({tight['right_limit']})"
        )
        print(f"  min left room: {profile['min_left']['value']:.0f} mm ({profile['min_left']['limit']})")
        print(f"  min right room: {profile['min_right']['value']:.0f} mm ({profile['min_right']['limit']})")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / "venue_route_map.png"
    map_path.write_bytes(store.blobs.get(record["image"]))
    print(f"Duty-desk map: {map_path}")


if __name__ == "__main__":
    main()
