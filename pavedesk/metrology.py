import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .analysis import capture_provenance, specimen_provenance
from .art import ACCENT, INK, SHADOW, base_rgba, curve_points, stored_preview
from .geometry import active_region, polygon_mask, region_domain, signed_area, specimen_frame, validate_polygon
from .model import InputError, collection, dictionary, enumeration, identity, numeric_pair, string


MIN_SEGMENT = 2.0
MIN_MARK_PIXELS = 20
REGION_OUTLINE = (199, 212, 151)
EXCLUDED_OUTLINE = (237, 122, 110)
MASK_THROUGH = (102, 180, 147, 94)
EDGE_LIMIT = "Survey edge"
SUPPORT_LIMIT = "Unphotographed area"
MAX_CLEARANCE_STATIONS = 400


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


def obstacle_label_image(tile, frame):
    """Label blocked plane pixels: 0 is free, 1..N are named exclusions or unsupported area."""
    labels = np.zeros((frame.extent[1], frame.extent[0]), dtype=np.int32)
    names = []
    for region in tile["regions"]:
        if region["purpose"] != "ignore":
            continue
        mask = polygon_mask(frame.extent, region["vertices"])
        if not mask.any():
            continue
        names.append(region["name"])
        labels[mask] = len(names)
    unsupported = ~frame.valid & (labels == 0)
    if unsupported.any():
        names.append(SUPPORT_LIMIT)
        labels[unsupported] = len(names)
    return labels, names


def clearance_ray(labels, x, y, dx, dy, physical_per_pixel):
    """Walk a unit pixel ray until a blocked label or the plane edge; return distance and label."""
    height, width = labels.shape
    limit = int(math.ceil(math.hypot(width, height))) + 2
    cx, cy = x, y
    for step in range(1, limit):
        cx += dx
        cy += dy
        col = int(round(cx))
        row = int(round(cy))
        if not 0 <= col < width or not 0 <= row < height:
            return step * physical_per_pixel, 0
        if labels[row, col]:
            return step * physical_per_pixel, int(labels[row, col])
    return limit * physical_per_pixel, 0


def clearance_profile(frame, tile, points):
    """Perpendicular room on both sides of a traced path, with the obstacle met on each side."""
    factors, units = calibrated_dimensions(frame)
    scale = frame.scale if frame.scale is not None else (1.0, 1.0)
    labels, names = obstacle_label_image(tile, frame)
    physical = np.asarray(points, dtype=np.float64) * factors
    vectors = np.diff(physical, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    total = float(np.sum(lengths))
    if total <= 0:
        return None, None
    spacing = 100.0 if units == "mm" else total / 200
    spacing = max(spacing, total / MAX_CLEARANCE_STATIONS)
    marks = [index * spacing for index in range(int(math.floor(total / spacing)) + 1)]
    if total - marks[-1] > 1e-6:
        marks.append(total)
    stations = []
    tightest = None
    seg_index = 0
    seg_start = 0.0
    for mark in marks:
        while seg_index < len(lengths) - 1 and seg_start + lengths[seg_index] < mark:
            seg_start += lengths[seg_index]
            seg_index += 1
        ratio = 0.0 if lengths[seg_index] == 0 else (mark - seg_start) / lengths[seg_index]
        pos = physical[seg_index] + ratio * vectors[seg_index]
        direction = vectors[seg_index] / lengths[seg_index]
        px = pos[0] / scale[0] - 0.5
        py = pos[1] / scale[1] - 0.5
        left_x = direction[1] / scale[0]
        left_y = -direction[0] / scale[1]
        norm = math.hypot(left_x, left_y) or 1.0
        left_x, left_y = left_x / norm, left_y / norm
        per_pixel = math.hypot(left_x * scale[0], left_y * scale[1])
        left, left_label = clearance_ray(labels, px, py, left_x, left_y, per_pixel)
        right, right_label = clearance_ray(labels, px, py, -left_x, -left_y, per_pixel)
        station = {
            "along": round(mark, 1),
            "left": round(left, 1),
            "left_limit": names[left_label - 1] if left_label else EDGE_LIMIT,
            "right": round(right, 1),
            "right_limit": names[right_label - 1] if right_label else EDGE_LIMIT,
        }
        stations.append(station)
        if tightest is None or left + right < tightest[0]["left"] + tightest[0]["right"]:
            tightest = (station, (px, py), (left_x, left_y), per_pixel)
    narrow, (px, py), (left_x, left_y), per_pixel = tightest
    pinch = {
        "at": [(px + 0.5) / frame.extent[0], (py + 0.5) / frame.extent[1]],
        "bar": [
            [(px + left_x * narrow["left"] / per_pixel + 0.5) / frame.extent[0],
             (py + left_y * narrow["left"] / per_pixel + 0.5) / frame.extent[1]],
            [(px - left_x * narrow["right"] / per_pixel + 0.5) / frame.extent[0],
             (py - left_y * narrow["right"] / per_pixel + 0.5) / frame.extent[1]],
        ],
        "label": f"Tightest {narrow['left'] + narrow['right']:.0f} {units}",
    }
    min_left = min(stations, key=lambda item: item["left"])
    min_right = min(stations, key=lambda item: item["right"])
    profile = {
        "units": units,
        "station_spacing": round(spacing, 1),
        "stations": stations,
        "tightest": {**narrow, "width": round(narrow["left"] + narrow["right"], 1)},
        "min_left": {"along": min_left["along"], "value": min_left["left"], "limit": min_left["left_limit"]},
        "min_right": {"along": min_right["along"], "value": min_right["right"], "limit": min_right["right_limit"]},
    }
    return profile, pinch


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


def geometry_painter(frame, points, segments, closed=False, support=None, centroid=None, regions=None, pinch=None):
    image = base_rgba(frame)
    width, height = image.size
    pixels = curve_points(points, image.size)
    if support is not None:
        raw = np.asarray(image, dtype=np.float32).copy()
        raw[support, :3] = raw[support, :3] * 0.72 + np.array(MASK_THROUGH[:3], dtype=np.float32) * 0.28
        from PIL import Image
        image = Image.fromarray(raw.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    if regions:
        region_stroke = max(1, round(min(width, height) / 360))
        for region in regions:
            color = EXCLUDED_OUTLINE if region["purpose"] == "ignore" else MASK_THROUGH[:3]
            outline = curve_points(region["vertices"], image.size)
            draw.line([*outline, outline[0]], width=region_stroke + 2, fill=SHADOW, joint="curve")
            draw.line([*outline, outline[0]], width=region_stroke, fill=color, joint="curve")
            x, y = outline[0]
            draw.text((x + 5, y + 4), region["name"][:36], fill=color, stroke_width=2, stroke_fill=SHADOW)
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
    if pinch is not None:
        bar = curve_points(pinch["bar"], image.size)
        draw.line(bar, fill=SHADOW, width=stroke + 5)
        draw.line(bar, fill=EXCLUDED_OUTLINE, width=stroke + 2)
        for bx, by in bar:
            draw.ellipse((bx - 4, by - 4, bx + 4, by + 4), fill=EXCLUDED_OUTLINE, outline=SHADOW, width=1)
        x, y = curve_points([pinch["at"]], image.size)[0]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=EXCLUDED_OUTLINE, width=2)
        mid_x = (bar[0][0] + bar[1][0]) / 2
        mid_y = (bar[0][1] + bar[1][1]) / 2
        draw.text((mid_x + 8, mid_y + 8), pinch["label"], fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def trace_geometry(frame, available, points, tile=None):
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
    regions = None
    pinch = None
    if tile is not None:
        clearance, pinch = clearance_profile(frame, tile, points)
        if clearance is not None:
            detail["clearance"] = clearance
        regions = tile["regions"]
    return detail, geometry_painter(frame, points, segments, regions=regions, pinch=pinch)


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
    frame = specimen_frame(store.blobs, tile)
    available = region_domain(frame.extent, tile) & frame.valid
    if mode == "trace":
        points = path_points(data.get("points"))
        shape, overlay = trace_geometry(frame, available, points, tile)
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
        "region": region,
        "exclusions": [candidate for candidate in tile["regions"] if candidate["purpose"] == "ignore"],
        "shape": shape,
        "image": stored_preview(overlay, store.blobs),
    })
