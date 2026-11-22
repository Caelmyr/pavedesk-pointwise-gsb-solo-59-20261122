"""Adopt one main fissure sealing line between two surveyor-marked endpoints.

The automatic Surface fissures trace follows every dark line it can see,
including side branches, and loses the main fissure where a wet mark or other
obscuration suppresses contrast. This module keeps a single line: the cheapest
route between the two endpoints on a weighted grid, where traced (photographed)
line pixels are cheap and every other supported pixel carries a penalty. The
returned line therefore stays on real photographed linework whenever a
continuous trace exists and only leaves it to bridge obscured gaps, with
photographed and inferred lengths accounted separately.
"""
import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

from .art import INK, SHADOW, base_rgba, font, stored_preview
from .analysis import capture_provenance, specimen_provenance
from .craze import (
    PIXEL_LIMIT,
    contrast_suppress,
    gradient_field,
    threshold_centerline,
)
from .geometry import region_domain, specimen_frame
from .metrology import MIN_SEGMENT
from .model import InputError, collection, dictionary, identity, integer, numeric_pair, scalar, string
from .photos import luminance

GRID_NEIGHBORS = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)
# Off-trace travel costs this multiple of traced travel, so the route detours
# onto photographed linework whenever one is reachable, while still being able
# to cross a wet mark where no trace survives.
TRACE_PENALTY = 3.5
RDP_EPSILON_PX = 1.5
ENDPOINT_TRACE_TOLERANCE_PX = 80
MIN_PHOTOGRAPHED_PX = 4
LOW_SUPPORT_WARNING = 25.0
DEFAULT_REVIEW_TOLERANCE_PX = 12


def fracture_parameters(payload):
    data = dictionary(payload, "Main fissure line")
    raw_points = collection(data.get("points"), "Line endpoints", minimum=2, maximum=2)
    points = [numeric_pair(item, "Line endpoint", lower=0, upper=1) for item in raw_points]
    if points[0] == points[1]:
        raise InputError("Mark two different endpoints for the main fissure.")
    settings = {
        "radius": scalar(data.get("radius", 1.4), "Line width radius px", lower=0.6, upper=5),
        "sensitivity": scalar(data.get("sensitivity", 0.72), "Line sensitivity", lower=0, upper=1),
        "join_px": integer(data.get("join_px", 1), "Bridge width px", lower=0, upper=4),
        "component_minimum": integer(data.get("component_minimum", 12), "Minimum trace px", lower=2, upper=150),
        "minimum_cell": 24,
    }
    review_id = data.get("review_result_id")
    review = identity(review_id, "Visible defects review") if review_id else None
    tolerance = integer(
        data.get("review_tolerance_px", DEFAULT_REVIEW_TOLERANCE_PX),
        "Review match tolerance px", lower=4, upper=40,
    )
    title = string(data.get("title"), "Measurement title", maximum=90)
    return {
        "title": title,
        "anchors": points,
        "options": settings,
        "review_result_id": review,
        "review_tolerance_px": tolerance,
    }


def trace_skeleton(frame, available, options):
    source = luminance(frame.linear)
    candidate = available & frame.valid & ~frame.clipped
    if source.size > PIXEL_LIMIT:
        raise InputError("Main fissure selection supports registered planes up to 3.1 megapixels.")
    filtered, valid = gradient_field(source, candidate, options["radius"])
    fine, support = contrast_suppress(source, candidate, options["radius"])
    coarse, margin = contrast_suppress(source, candidate, 2 * options["radius"])
    support &= valid & margin
    if np.count_nonzero(support) < 100:
        raise InputError("Choose a larger area with contiguous, unclipped image support.")
    thin, _, thresholds = threshold_centerline(filtered, fine, coarse, support, options)
    return thin, fine, thresholds


def weighted_grid(thin, passable, scale):
    height, width = thin.shape
    scale_x, scale_y = scale or (1.0, 1.0)
    rows_src, cols_src = np.nonzero(passable)
    edge_rows, edge_cols, edge_weights = [], [], []
    for dy, dx in GRID_NEIGHBORS:
        row2 = rows_src + dy
        col2 = cols_src + dx
        inside = (row2 >= 0) & (row2 < height) & (col2 >= 0) & (col2 < width)
        r1, c1, rr, cc = rows_src[inside], cols_src[inside], row2[inside], col2[inside]
        keep = passable[rr, cc]
        r1, c1, rr, cc = r1[keep], c1[keep], rr[keep], cc[keep]
        step = float(math.hypot(dy * scale_y, dx * scale_x))
        weight_a = np.where(thin[r1, c1], 1.0, TRACE_PENALTY)
        weight_b = np.where(thin[rr, cc], 1.0, TRACE_PENALTY)
        edge_rows.append(r1 * width + c1)
        edge_cols.append(rr * width + cc)
        edge_weights.append(step * (weight_a + weight_b) / 2)
    total = height * width
    return sparse.csr_matrix(
        (np.concatenate(edge_weights), (np.concatenate(edge_rows), np.concatenate(edge_cols))),
        shape=(total, total),
    )


def anchor_pixel(anchor, extent, passable):
    width, height = extent
    x = min(max(int(round(anchor[0] * width)), 0), width - 1)
    y = min(max(int(round(anchor[1] * height)), 0), height - 1)
    if passable[y, x]:
        return x, y
    distance, (nearest_y, nearest_x) = ndimage.distance_transform_edt(~passable, return_indices=True)
    return int(nearest_x[y, x]), int(nearest_y[y, x])


def shortest_route(graph, start_px, end_px, extent):
    width, _ = extent
    source = start_px[1] * width + start_px[0]
    target = end_px[1] * width + end_px[0]
    distances, predecessors = sparse_dijkstra(
        graph, directed=True, indices=source, return_predecessors=True,
    )
    if not np.isfinite(distances[target]):
        raise InputError("No supported surface connects the two endpoints. Choose endpoints inside the survey plane.")
    route_ids = []
    current = target
    while current != source:
        route_ids.append(current)
        previous = predecessors[current]
        if previous < 0:
            raise InputError("No supported surface connects the two endpoints. Choose endpoints inside the survey plane.")
        current = previous
    route_ids.append(source)
    route_ids = route_ids[::-1]
    rows = np.array([item // width for item in route_ids], dtype=np.int64)
    cols = np.array([item % width for item in route_ids], dtype=np.int64)
    return rows, cols


def ramer_douglas_peucker(points, epsilon):
    if len(points) < 3:
        return list(points)
    coordinates = np.asarray(points, dtype=np.float64)
    keep = np.zeros(len(coordinates), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(coordinates) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        start = coordinates[first]
        end = coordinates[last]
        vector = end - start
        denominator = float(np.hypot(vector[0], vector[1]))
        segment = coordinates[first:last + 1]
        if denominator == 0:
            deviation = np.hypot(segment[:, 0] - start[0], segment[:, 1] - start[1])
        else:
            deviation = np.abs(
                (vector[0] * (segment[:, 1] - start[1]))
                - (vector[1] * (segment[:, 0] - start[0]))
            ) / denominator
        # Exclude the interval endpoints so the chosen split strictly shrinks both sides.
        deviation[0] = deviation[-1] = -1.0
        offset = int(np.argmax(deviation))
        if deviation[offset] > epsilon:
            index = first + offset
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return [tuple(point) for point, kept in zip(coordinates.tolist(), keep.tolist()) if kept]


def classify_steps(rows, cols, thin, scale):
    scale_x, scale_y = scale or (1.0, 1.0)
    observed = np.array([
        bool(thin[rows[k], cols[k]]) and bool(thin[rows[k + 1], cols[k + 1]])
        for k in range(len(rows) - 1)
    ])
    lengths = np.hypot(np.diff(rows) * scale_y, np.diff(cols) * scale_x)
    return observed, lengths


def inferred_runs(rows, cols, observed, lengths, contrast, extent, contrast_floor):
    width, height = extent
    runs = []
    index = 0
    while index < len(observed):
        if observed[index]:
            index += 1
            continue
        stop = index
        while stop + 1 < len(observed) and not observed[stop + 1]:
            stop += 1
        run_rows = rows[index:stop + 2]
        run_cols = cols[index:stop + 2]
        run_length = float(lengths[index:stop + 1].sum())
        mean_contrast = float(contrast[run_rows, run_cols].mean())
        runs.append({
            "from": [round(float(run_cols[0] / width), 6), round(float(run_rows[0] / height), 6)],
            "to": [round(float(run_cols[-1] / width), 6), round(float(run_rows[-1] / height), 6)],
            "length": round(run_length, 5),
            "mean_contrast": round(mean_contrast, 6),
            "obscured": bool(mean_contrast < contrast_floor),
            "pixel_steps": int(stop + 1 - index),
        })
        index = stop + 1
    return runs


def simplified_segments(rows, cols, observed, lengths, extent, scale):
    width, height = extent
    scale_x, scale_y = scale or (1.0, 1.0)
    units = "mm" if scale else "px"
    points_px = ramer_douglas_peucker(list(zip(cols.tolist(), rows.tolist())), RDP_EPSILON_PX)
    # Map each simplified vertex onto the nearest route pixel in monotonic order.
    route = np.column_stack([cols, rows]).astype(np.float64)
    search_from = 0
    vertex_indices = []
    for x, y in points_px:
        distances = np.hypot(route[search_from:, 0] - x, route[search_from:, 1] - y)
        offset = int(np.argmin(distances))
        search_from = min(search_from + offset, len(route) - 1)
        vertex_indices.append(search_from)
    raw = []
    for first, second in zip(vertex_indices, vertex_indices[1:]):
        step_observed = observed[first:second]
        step_lengths = lengths[first:second]
        photographed = float(step_lengths[step_observed].sum()) if len(step_observed) else 0.0
        inferred = float(step_lengths[~step_observed].sum()) if len(step_observed) else 0.0
        raw.append({
            "first": first, "second": second,
            "photographed": photographed, "inferred": inferred,
        })
    # Merge any undersized piece into its predecessor so the table reconciles
    # exactly with the full route instead of silently dropping geometry.
    merged = []
    for piece in raw:
        if merged and piece["photographed"] + piece["inferred"] < MIN_SEGMENT:
            merged[-1]["second"] = piece["second"]
            merged[-1]["photographed"] += piece["photographed"]
            merged[-1]["inferred"] += piece["inferred"]
        else:
            merged.append(piece)
    segments = []
    for piece in merged:
        first, second = piece["first"], piece["second"]
        photographed, inferred = piece["photographed"], piece["inferred"]
        total = photographed + inferred
        basis = "photographed" if inferred <= 0.05 * total else "inferred" if photographed <= 0.05 * total else "mixed"
        x0, y0 = route[first]
        x1, y1 = route[second]
        dx, dy = (x1 - x0) * scale_x, (y1 - y0) * scale_y
        segments.append({
            "index": len(segments) + 1,
            "length": round(total, 5),
            "photographed_length": round(photographed, 5),
            "inferred_length": round(inferred, 5),
            "basis": basis,
            "units": units,
            "dx": round(float(dx), 5),
            "dy": round(float(dy), 5),
            "clockwise_heading": round(math.degrees(math.atan2(dy, dx)) % 360, 3),
            "from": [round(float(x0 / width), 6), round(float(y0 / height), 6)],
            "to": [round(float(x1 / width), 6), round(float(y1 / height), 6)],
        })
    normalized = []
    for segment in segments:
        if not normalized:
            normalized.append(segment["from"])
        normalized.append(segment["to"])
    return normalized, segments


def select_main_line(frame, available, anchors, options):
    extent = frame.extent
    thin, contrast, thresholds = trace_skeleton(frame, available, options)
    passable = available & frame.valid
    graph = weighted_grid(thin, passable, frame.scale)
    start_px = anchor_pixel(anchors[0], extent, passable)
    end_px = anchor_pixel(anchors[1], extent, passable)
    snap = [
        round(float(np.hypot(start_px[0] - anchors[0][0] * extent[0], start_px[1] - anchors[0][1] * extent[1])), 3),
        round(float(np.hypot(end_px[0] - anchors[1][0] * extent[0], end_px[1] - anchors[1][1] * extent[1])), 3),
    ]
    trace_distance = ndimage.distance_transform_edt(~thin, return_indices=False)
    endpoint_distances = (
        float(trace_distance[start_px[1], start_px[0]]),
        float(trace_distance[end_px[1], end_px[0]]),
    )
    if max(endpoint_distances) > ENDPOINT_TRACE_TOLERANCE_PX:
        raise InputError(
            f"An endpoint is more than {ENDPOINT_TRACE_TOLERANCE_PX} px from any photographed fissure trace. "
            "Mark the ends on or next to the visible crack.",
        )
    rows, cols = shortest_route(graph, start_px, end_px, extent)
    observed, lengths = classify_steps(rows, cols, thin, frame.scale)
    photographed_length = float(lengths[observed].sum())
    inferred_length = float(lengths[~observed].sum())
    total_length = photographed_length + inferred_length
    if np.count_nonzero(observed) < MIN_PHOTOGRAPHED_PX:
        raise InputError("The chosen route has almost no photographed line support. Choose endpoints that follow a visible crack.")
    points, segments = simplified_segments(rows, cols, observed, lengths, extent, frame.scale)
    runs = inferred_runs(
        rows, cols, observed, lengths, contrast, extent, thresholds["contrast_floor"],
    )
    photographed_percent = round(100.0 * photographed_length / total_length, 3)
    warnings = []
    if photographed_percent < LOW_SUPPORT_WARNING:
        warnings.append(
            f"Only {photographed_percent:.1f}% of the adopted line follows photographed linework; "
            "the remainder is inferred across obscured or unsupported surface.",
        )
    line = {
        "points": points,
        "anchors": anchors,
        "segments": segments,
        "trace_length": round(total_length, 5),
        "photographed_length": round(photographed_length, 5),
        "inferred_length": round(inferred_length, 5),
        "photographed_percent": photographed_percent,
        "length_unit": "mm" if frame.scale else "px",
        "inferred_runs": runs,
        "anchor_snap_px": snap,
        "warnings": warnings,
        "detector": {
            "options": options,
            "thresholds": thresholds,
            "traced_pixels": int(np.count_nonzero(thin)),
            "trace_penalty": TRACE_PENALTY,
        },
    }
    return line, (rows, cols, observed)


def review_basis(record, route_rows, route_cols, extent, tolerance_px):
    if record["kind"] != "defects":
        raise InputError("The basis review must be a Visible defects measurement.")
    width, height = extent
    route = np.column_stack([route_cols, route_rows]).astype(np.float64)
    decisions = record.get("decisions", {})
    review_dates = [
        entry.get("updated_at") for entry in decisions.values() if entry.get("updated_at")
    ]
    reviewed_at = max(review_dates) if review_dates else record["created_at"]
    matched = []
    for observation in record["metrics"]["observations"]:
        key = observation["key"]
        decision = decisions.get(key)
        if not decision or decision["decision"] != "confirm":
            continue
        center_x = observation["normalized_center"][0] * width
        center_y = observation["normalized_center"][1] * height
        distances = np.hypot(route[:, 0] - center_x, route[:, 1] - center_y)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        if distance > tolerance_px:
            continue
        matched.append({
            "key": key,
            "type": observation["type"],
            "decision": "confirm",
            "notes": decision.get("notes", ""),
            "decided_at": decision.get("updated_at", record["created_at"]),
            "normalized_center": observation["normalized_center"],
            "distance_px": round(distance, 3),
        })
    return {
        "result_id": record["id"],
        "label": record["label"],
        "result_version": record["version"],
        "created_at": record["created_at"],
        "reviewed_at": reviewed_at,
        "overlay_digest": record["pictures"]["overlay"],
        "source_digest": record["pictures"]["source"],
        "match_tolerance_px": tolerance_px,
        "matched_decisions": matched,
    }


PHOTOGRAPHED_COLOR = (110, 214, 150)
INFERRED_COLOR = (240, 181, 82)
BASIS_COLOR = (162, 225, 171)


def fracture_overlay(frame, route, anchors, basis):
    rows, cols, observed = route
    image = base_rgba(frame)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    stroke = max(1, round(min(width, height) / 320))
    typeface = font(max(12, min(18, round(min(width, height) / 38))))
    # Dark outline first, then colored classified centerline.
    pixels = list(zip(cols.tolist(), rows.tolist()))
    draw.line(pixels, fill=SHADOW, width=stroke + 3, joint="curve")
    for index in range(len(pixels) - 1):
        color = PHOTOGRAPHED_COLOR if observed[index] else INFERRED_COLOR
        draw.line([pixels[index], pixels[index + 1]], fill=color, width=stroke + 1)
    for label, anchor in zip(("A", "B"), anchors):
        x, y = round(anchor[0] * width), round(anchor[1] * height)
        box = (x - 6, y - 6, x + 6, y + 6)
        draw.rectangle(box, fill=INK, outline=SHADOW, width=2)
        draw.text((x + 8, y - 8), label, fill=INK, stroke_width=2, stroke_fill=SHADOW)
    if basis:
        occupied = []
        for item in basis["matched_decisions"]:
            x = round(item["normalized_center"][0] * width)
            y = round(item["normalized_center"][1] * height)
            radius = max(5, round(min(width, height) / 110))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline=SHADOW, width=stroke + 2)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline=BASIS_COLOR, width=stroke)
            label_w = int(draw.textlength(item["key"], font=typeface))
            label_x = x + radius + 6
            label_y = y - 7

            def box_at(ly):
                return (label_x - 2, ly - 2, label_x + label_w + 3, ly + 14)

            # Cascade below any label whose box would overlap this one.
            label_box = box_at(label_y)
            while any(not (label_box[2] <= other[0] or other[2] <= label_box[0]
                           or label_box[3] <= other[1] or other[3] <= label_box[1])
                      for other in occupied):
                label_y = max(other[3] + 2 for other in occupied
                              if not (box_at(label_y)[2] <= other[0] or other[2] <= box_at(label_y)[0]
                                      or box_at(label_y)[3] <= other[1] or other[3] <= box_at(label_y)[1]))
                label_box = box_at(label_y)
            occupied.append(label_box)
            draw.text((label_x, label_y), item["key"], fill=BASIS_COLOR,
                      stroke_width=2, stroke_fill=SHADOW)
    legend = ("A/B surveyor endpoints   green: photographed line   "
              "amber: inferred across obscured surface")
    draw.rounded_rectangle((10, 10, 12 + draw.textlength(legend, font=typeface) + 12, 38),
                           radius=4, fill=SHADOW)
    draw.text((16, 14), legend, font=typeface, fill=INK)
    return image


def create_fracture_line(store, specimen_id, payload):
    params = fracture_parameters(payload)
    tile = store.tile(specimen_id)
    review_record = None
    if params["review_result_id"]:
        review_record = store.result(params["review_result_id"])
        if review_record["tile_id"] != tile["id"]:
            raise InputError("The basis review belongs to another specimen.")
        if review_record["kind"] != "defects":
            raise InputError("The basis review must be a Visible defects measurement.")
    frame = specimen_frame(store.blobs, tile)
    available = region_domain(frame.extent, tile) & frame.valid
    line, route = select_main_line(frame, available, params["anchors"], params["options"])
    basis = None
    if review_record is not None:
        basis = review_basis(
            review_record, route[0], route[1],
            frame.extent, params["review_tolerance_px"],
        )
    overlay = fracture_overlay(frame, route, params["anchors"], basis)
    return store.save_metrology({
        "title": params["title"],
        "mode": "fracture-line",
        "tile_id": tile["id"],
        "specimen": specimen_provenance(tile),
        "source": capture_provenance(frame),
        "exclusions": [candidate for candidate in tile["regions"] if candidate["purpose"] == "ignore"],
        "shape": line,
        "review_basis": basis,
        "image": stored_preview(overlay, store.blobs),
    })
