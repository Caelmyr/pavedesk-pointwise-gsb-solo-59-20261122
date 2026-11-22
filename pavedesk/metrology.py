import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .analysis import capture_provenance, specimen_provenance
from .art import ACCENT, INK, SHADOW, base_rgba, curve_points, stored_preview
from .geometry import active_region, polygon_mask, region_domain, signed_area, specimen_frame, validate_polygon
from .model import InputError, collection, dictionary, enumeration, identity, numeric_pair, string
from .route import route_frame, source_palette


MIN_SEGMENT = 2.0
MIN_MARK_PIXELS = 20
REGION_OUTLINE = (199, 212, 151)
EXCLUDED_OUTLINE = (237, 122, 110)
MASK_THROUGH = (102, 180, 147, 94)
UNPHOTOGRAPHED = (236, 112, 104)
MAX_TRACE_SAMPLES = 12_000


def path_points(payload):
    raw = collection(payload, "Trace points", minimum=2, maximum=45)
    return [numeric_pair(item, "Trace point", lower=0, upper=1) for item in raw]


def calibrated_dimensions(frame):
    width, height = frame.extent
    if frame.scale is not None:
        return np.array([width * frame.scale[0], height * frame.scale[1]]), "mm"
    return np.array([width, height]), "px"


def path_segments(frame, points, closed=False):
    factors, units = calibrated_dimensions(frame)
    point_pixels = np.asarray(points, dtype=np.float64) * frame.extent
    vertices = np.asarray(points, dtype=np.float64) * factors
    if closed:
        vertices = np.vstack((vertices, vertices[0]))
        point_pixels = np.vstack((point_pixels, point_pixels[0]))
    deltas = np.diff(vertices, axis=0)
    deltas_px = np.diff(point_pixels, axis=0)
    lengths_px = np.linalg.norm(deltas_px, axis=1)
    if np.any(lengths_px < MIN_SEGMENT):
        raise InputError("Each trace segment needs at least two registered image pixels.")
    length = np.linalg.norm(deltas, axis=1)
    segments = []
    for index, (delta, value, pixels) in enumerate(zip(deltas, length, lengths_px)):
        vector = delta.tolist()
        segments.append({
            "index": index + 1,
            "length": round(float(value), 5),
            "length_px": round(float(pixels), 5),
            "units": units,
            "dx": round(vector[0], 5),
            "dy": round(vector[1], 5),
            "clockwise_heading": round(math.degrees(math.atan2(vector[1], vector[0])) % 360, 3),
        })
    return segments, units


def path_cumulative(segments):
    distance = 0
    totals = [0]
    for segment in segments:
        distance += segment["length"]
        totals.append(round(distance, 5))
    return totals


def path_turns(segments, closed=False):
    angles = [segment["clockwise_heading"] for segment in segments]
    linked = angles[1:] + [angles[0]] if closed else angles[1:]
    return [round((after - before + 180) % 360 - 180, 3) for before, after in zip(angles, linked)]


def sample_support_along(available, frame, points, closed=False):
    positions = points + [points[0]] if closed else points
    xy = np.asarray(positions, dtype=np.float32) * frame.extent - 0.5
    mask = available.astype(np.float32)
    pieces = []
    for entry, (start, stop) in enumerate(zip(xy, xy[1:]), 1):
        length = float(np.linalg.norm(stop - start))
        count = min(1800, int(np.ceil(length)) + 1)
        positions = np.linspace(start, stop, count)
        row = np.clip(positions[:, 1], 0, mask.shape[0] - 1)
        col = np.clip(positions[:, 0], 0, mask.shape[1] - 1)
        sampled = ndimage.map_coordinates(mask, (row, col), order=1, mode="nearest")
        actual = sampled >= 0.995
        transitions = np.diff(np.pad(actual.astype(np.int8), 1))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        pieces.append({
            "index": entry,
            "samples": count,
            "supported_samples": int(np.count_nonzero(actual)),
            "support_percent": round(100 * float(np.mean(actual)), 3),
            "supported_intervals": [
                [round(int(lo) / count, 4), round(int(hi) / count, 4)] for lo, hi in zip(starts, stops)
            ],
        })
    return pieces


def region_outline_bounds(vertices, totals):
    values = np.asarray(vertices, dtype=np.float64)
    xy = values * totals
    minimum = np.min(xy, axis=0)
    maximum = np.max(xy, axis=0)
    extent = maximum - minimum
    return {
        "bounds": np.round(np.r_[minimum, maximum], 5).tolist(),
        "axis_width": round(float(extent[0]), 5),
        "axis_height": round(float(extent[1]), 5),
    }


def shape_center(vertices, total_area):
    points = np.asarray(vertices, dtype=np.float64)
    next_points = np.roll(points, -1, axis=0)
    area_terms = points[:, 0] * next_points[:, 1] - points[:, 1] * next_points[:, 0]
    weighted = np.sum((points + next_points) * area_terms[:, None], axis=0)
    return np.round(weighted / (6 * total_area), 6).tolist()


def proper_axis(support, scale):
    rows, cols = np.nonzero(support)
    if len(rows) < 7:
        return None
    points = np.stack((cols + 0.5, rows + 0.5), axis=1)
    if scale:
        points *= np.asarray(scale)
    centroid = np.mean(points, axis=0)
    cov = np.cov((points - centroid).T)
    eigen, directions = np.linalg.eigh(cov)
    major = directions[:, -1]
    minor_width = max(float(eigen[0]), 1e-12)
    major_width = float(eigen[1])
    elongation = math.sqrt(major_width / minor_width)
    return {
        "support_centroid_xy": np.round(centroid, 5).tolist(),
        "support_major_angle": round(math.degrees(math.atan2(major[1], major[0])) % 180, 3),
        "support_elongation": round(elongation, 4),
    }


def occupied_holes(mask):
    filled = ndimage.binary_fill_holes(mask)
    background = filled & ~mask
    identities, count = ndimage.label(background)
    if not count:
        return []
    areas = np.bincount(identities.ravel())[1:]
    areas = np.sort(areas[areas >= 4])[::-1]
    return [int(value) for value in areas[:24]]


def mask_geometry(frame, selected, requested, supported):
    dimensions, units = calibrated_dimensions(frame)
    area = signed_area(selected["vertices"])
    outer = abs(area) * float(np.prod(dimensions))
    occupied = int(np.count_nonzero(supported))
    if occupied < MIN_MARK_PIXELS:
        raise InputError("The marked area has fewer than 20 supported image pixels.")
    raster = int(np.count_nonzero(requested))
    factor = float(np.prod(frame.scale)) if frame.scale else 1
    axes = proper_axis(supported, frame.scale)
    return {
        "vector_outline_area": round(outer, 5),
        "raster_area": round(raster * factor, 5),
        "raster_supported_area": round(occupied * factor, 5),
        "requested_pixels": raster,
        "supported_pixels": occupied,
        "no_support_percent": round(100 * (1 - occupied / raster), 4),
        "outside_or_excluded_holes_px": occupied_holes(supported),
        "shape_centroid": shape_center(selected["vertices"], area),
        "support_spread": axes,
        "units": f"{units}²",
        **region_outline_bounds(selected["vertices"], dimensions),
    }


def geometry_painter(frame, points, segments, closed=False, support=None, centroid=None):
    image = base_rgba(frame)
    width, height = image.size
    pixels = curve_points(points, image.size)
    if support is not None:
        raw = np.asarray(image, dtype=np.float32).copy()
        raw[support, :3] = raw[support, :3] * 0.72 + np.array(MASK_THROUGH[:3], dtype=np.float32) * 0.28
        from PIL import Image
        image = Image.fromarray(raw.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    edges = [*pixels, pixels[0]] if closed else pixels
    stroke = max(1, round(min(width, height) / 310))
    draw.line(edges, fill=SHADOW, width=stroke + 3, joint="curve")
    draw.line(edges, fill=REGION_OUTLINE if closed else ACCENT, width=stroke + 1, joint="curve")
    for index, (point_a, point_b) in enumerate(zip(edges, edges[1:])):
        x = (point_a[0] + point_b[0]) / 2
        y = (point_a[1] + point_b[1]) / 2
        line = segments[index]
        label = f"{line['length']:.2f} {line['units']}"
        draw.text((x + 4, y - 9), label, fill=INK, stroke_width=2, stroke_fill=SHADOW)
    for index, (x, y) in enumerate(pixels, 1):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=INK, outline=SHADOW, width=1)
        draw.text((x + 6, y + 3), str(index), fill=INK, stroke_width=2, stroke_fill=SHADOW)
    if centroid is not None:
        x, y = curve_points([centroid], image.size)[0]
        draw.line((x - 6, y, x + 6, y), fill=EXCLUDED_OUTLINE, width=2)
        draw.line((x, y - 6, x, y + 6), fill=EXCLUDED_OUTLINE, width=2)
    return image


def trace_geometry(frame, available, points):
    segments, units = path_segments(frame, points)
    cumulative = path_cumulative(segments)
    pieces = sample_support_along(available, frame, points)
    samples = sum(item["samples"] for item in pieces)
    present = sum(item["supported_samples"] for item in pieces)
    detail = {
        "points": points,
        "segments": segments,
        "trace_length": cumulative[-1],
        "cumulative": cumulative,
        "turn_angles": path_turns(segments),
        "support_along_line": pieces,
        "overall_support": round(100 * present / samples, 3),
        "length_unit": units,
    }
    return detail, geometry_painter(frame, points, segments)


def sample_route_source(frame, points, sample_limit=MAX_TRACE_SAMPLES):
    """Walk the traced polyline and record the original frame behind each sample."""
    source_map = frame.source_map
    height, width = source_map.shape
    xy = np.asarray(points, dtype=np.float32) * frame.extent - 0.5
    pieces = []
    for index, (start, stop) in enumerate(zip(xy, xy[1:]), 1):
        length = float(np.linalg.norm(stop - start))
        count = min(sample_limit, max(2, int(np.ceil(length)) + 1))
        samples = np.linspace(start, stop, count)
        row = np.clip(samples[:, 1], 0, height - 1)
        col = np.clip(samples[:, 0], 0, width - 1)
        codes = ndimage.map_coordinates(source_map, (row, col), order=0, mode="nearest")
        fractions = np.linspace(0.0, 1.0, count)
        pieces.append({"index": index, "count": count, "fractions": fractions, "codes": codes.astype(np.int16)})
    return pieces


def runs_of_constant(values):
    """Contiguous [start, stop) runs of equal integer values."""
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(values)]
    return [(int(start), int(stop), int(values[start])) for start, stop in zip(starts, stops)]


def route_provenance(frame, points):
    join = frame.join
    entries = {entry["index"]: entry for entry in join["photos"]}
    pieces = sample_route_source(frame, points)
    segment_rows = []
    gaps = []
    totals = {}
    lengths = {}
    uncovered = 0.0
    for piece in pieces:
        index = piece["index"] - 1
        vector = np.asarray(points[index + 1], dtype=np.float64) - np.asarray(points[index], dtype=np.float64)
        factors = np.asarray(frame.extent, dtype=np.float64) * (
            np.asarray(frame.scale, dtype=np.float64) if frame.scale else np.ones(2)
        )
        physical = float(np.linalg.norm(vector * factors))
        runs = []
        for start, stop, code in runs_of_constant(piece["codes"]):
            entry = entries.get(code)
            length = physical * float(piece["fractions"][stop - 1] - piece["fractions"][start])
            run = {
                "from_fraction": round(float(piece["fractions"][start]), 6),
                "to_fraction": round(float(piece["fractions"][stop - 1]), 6),
                "source_code": entry["code"] if entry else None,
                "photo_id": entry["photo_id"] if entry else None,
                "photo_name": entry["name"] if entry else "Not photographed",
                "samples": int(stop - start),
                "length": round(length, 5),
            }
            runs.append(run)
            if entry:
                totals[entry["code"]] = totals.get(entry["code"], 0) + int(stop - start)
                lengths[entry["code"]] = lengths.get(entry["code"], 0.0) + length
            else:
                gaps.append({"segment": piece["index"], **run})
                uncovered += length
        segment_rows.append({"index": piece["index"], "samples": int(piece["count"]), "runs": runs})
    summary = []
    for entry in join["photos"]:
        if totals.get(entry["code"], 0):
            summary.append({
                "source_code": entry["code"],
                "photo_id": entry["photo_id"],
                "photo_name": entry["name"],
                "samples": totals[entry["code"]],
                "trace_length": round(lengths.get(entry["code"], 0.0), 5),
            })
    if gaps:
        summary.append({
            "source_code": None,
            "photo_id": None,
            "photo_name": "Not photographed",
            "samples": sum(item["samples"] for item in gaps),
            "trace_length": round(uncovered, 5),
        })
    samples = sum(piece["count"] for piece in pieces)
    gap_samples = sum(item["samples"] for item in gaps)
    return {
        "join_id": join["id"],
        "join_name": join["name"],
        "join_source_version": join["source_version"],
        "plane_corners": join["plane_corners"],
        "canvas": join["canvas"],
        "source_map_digest": join["source_map_digest"],
        "photos": [
            {"code": entry["code"], "photo_id": entry["photo_id"], "name": entry["name"], "role": entry["role"]}
            for entry in join["photos"]
        ],
        "segments": segment_rows,
        "sources": summary,
        "gaps": gaps,
        "gap_samples": gap_samples,
        "coverage_percent": round(100 * (samples - gap_samples) / max(samples, 1), 3),
        "overlap_supplier": join["overlap"]["supplier_code"],
        "note": (
            "Every trace sample is read from the frame recorded in the joined route source map. "
            "Unphotographed stretches have no original frame and are reported as gaps."
        ),
    }


def trace_source_at(record, position):
    """Frame behind an arbitrary point on a saved route trace.

    The point is projected onto the traced polyline; the contiguous source run
    that contains the projection answers the frame, or reports a gap.
    """
    shape = record.get("shape") or {}
    provenance = shape.get("provenance")
    if not provenance:
        raise InputError("This geometry record was not traced on a joined route.")
    points = np.asarray(shape["points"], dtype=np.float64)
    query = np.asarray(position, dtype=np.float64)
    best = None
    for index in range(len(points) - 1):
        start, stop = points[index], points[index + 1]
        delta = stop - start
        span = float(np.dot(delta, delta))
        fraction = 0.0 if span <= 0 else float(np.clip(np.dot(query - start, delta) / span, 0.0, 1.0))
        closest = start + fraction * delta
        distance = float(np.linalg.norm((query - closest) * provenance["canvas"]))
        if best is None or distance < best[0]:
            best = (distance, index, fraction, closest)
    distance, index, fraction, closest = best
    piece = provenance["segments"][index]
    run = next(
        (item for item in piece["runs"] if item["from_fraction"] - 1e-9 <= fraction <= item["to_fraction"] + 1e-9),
        piece["runs"][-1],
    )
    return {
        "query": [round(float(value), 6) for value in query],
        "nearest_trace_point": [round(float(value), 6) for value in closest],
        "distance_px": round(distance, 3),
        "segment": piece["index"],
        "segment_fraction": round(fraction, 6),
        "source_code": run["source_code"],
        "photo_id": run["photo_id"],
        "photo_name": run["photo_name"],
        "photographed": run["source_code"] is not None,
        "join_id": provenance["join_id"],
        "join_name": provenance["join_name"],
    }


def route_overlay(frame, points, provenance, colors):
    image = base_rgba(frame)
    width, height = image.size
    pixels = curve_points(points, image.size)
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(min(width, height) / 310))
    for piece in provenance["segments"]:
        index = piece["index"] - 1
        start_point = np.asarray(points[index], dtype=np.float64)
        end_point = np.asarray(points[index + 1], dtype=np.float64)
        for run in piece["runs"]:
            a = start_point + (end_point - start_point) * run["from_fraction"]
            b = start_point + (end_point - start_point) * run["to_fraction"]
            ends = curve_points([tuple(a), tuple(b)], image.size)
            draw.line(ends, fill=SHADOW, width=stroke + 4)
            if run["source_code"] is None:
                draw.line(ends, fill=UNPHOTOGRAPHED, width=stroke + 2)
            else:
                draw.line(ends, fill=colors.get(run["source_code"], (255, 255, 255)), width=stroke + 2)
    for index, (x, y) in enumerate(pixels, 1):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=INK, outline=SHADOW, width=1)
        draw.text((x + 6, y + 3), str(index), fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def trace_route_geometry(store, join, points):
    frame = route_frame(store, join)
    available = frame.valid
    segments, units = path_segments(frame, points)
    cumulative = path_cumulative(segments)
    provenance = route_provenance(frame, points)
    pieces = sample_support_along(available, frame, points)
    samples = sum(item["samples"] for item in pieces)
    present = sum(item["supported_samples"] for item in pieces)
    colors = {
        entry["code"]: source_palette(
            next(row["index"] for row in join["photos"] if row["code"] == entry["code"])
        )
        for entry in provenance["photos"]
    }
    detail = {
        "points": points,
        "segments": segments,
        "trace_length": cumulative[-1],
        "cumulative": cumulative,
        "turn_angles": path_turns(segments),
        "support_along_line": pieces,
        "overall_support": round(100 * present / samples, 3),
        "length_unit": units,
        "provenance": provenance,
    }
    overlay = route_overlay(frame, points, provenance, colors)
    return detail, overlay, frame


def polygon_geometry(tile, frame, selection, available):
    vertices = validate_polygon(selection["vertices"])
    requested = polygon_mask(frame.extent, vertices)
    supported = requested & available
    dimensions, units = calibrated_dimensions(frame)
    geometry = mask_geometry(frame, selection, requested, supported)
    segments, _ = path_segments(frame, vertices, closed=True)
    perimeter = path_cumulative(segments)[-1]
    geometry.update(
        points=vertices,
        outline_length=perimeter,
        outline_unit=units,
        outline_segments=segments,
        mask_purpose=selection["purpose"],
        contour_support=sample_support_along(available, frame, vertices, closed=True),
    )
    painter = geometry_painter(frame, vertices, segments, True, supported, geometry["shape_centroid"])
    return geometry, painter


def create_metrology(store, specimen_id, payload):
    data = dictionary(payload, "Image measurement")
    title = string(data.get("title"), "Measurement title", maximum=90)
    mode = enumeration(data.get("mode"), ("trace", "area"), "Geometry measurement")
    tile = store.tile(specimen_id)
    join_id = identity(data["join_id"], "Joined route ID") if data.get("join_id") else None
    if join_id is not None:
        if mode != "trace":
            raise InputError("Area measurement uses the specimen plane, not a joined route.")
        join = store.join(join_id)
        if join["tile_id"] != tile["id"]:
            raise InputError("The joined route belongs to another specimen.")
        points = path_points(data.get("points"))
        shape, overlay, frame = trace_route_geometry(store, join, points)
        return store.save_metrology({
            "title": title,
            "mode": "trace",
            "tile_id": tile["id"],
            "specimen": specimen_provenance(tile),
            "source": {
                "id": join["id"],
                "display_name": join["name"],
                "role": "route-join",
                "plane_to_capture": None,
                "canvas": join["canvas"],
                "plane_size": join["plane_size"],
                "source_map_digest": join["source_map_digest"],
                "synthetic": False,
                "kind": "route-join",
            },
            "join": {
                "id": join["id"],
                "name": join["name"],
                "source_version": join["source_version"],
                "photos": shape["provenance"]["photos"],
                "overlap_supplier": shape["provenance"]["overlap_supplier"],
            },
            "region": None,
            "exclusions": [],
            "shape": shape,
            "image": stored_preview(overlay, store.blobs),
        })
    frame = specimen_frame(store.blobs, tile)
    available = region_domain(frame.extent, tile) & frame.valid
    if mode == "trace":
        points = path_points(data.get("points"))
        shape, overlay = trace_geometry(frame, available, points)
        region = None
    else:
        region = active_region(tile, identity(data.get("region_id"), "Region ID"))
        shape, overlay = polygon_geometry(tile, frame, region, available)
    return store.save_metrology({
        "title": title,
        "mode": mode,
        "tile_id": tile["id"],
        "specimen": specimen_provenance(tile),
        "source": capture_provenance(frame),
        "join": None,
        "region": region,
        "exclusions": [candidate for candidate in tile["regions"] if candidate["purpose"] == "ignore"],
        "shape": shape,
        "image": stored_preview(overlay, store.blobs),
    })
