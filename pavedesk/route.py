"""Join two overlapping rectified captures into one traceable route image.

The joined canvas lives in the registered specimen plane.  Every output pixel
is copied from exactly one original capture; overlap pixels are never blended.
The whole shared overlap is therefore supplied by a single frame, and a hard
source index is stored for every pixel.  That index lets a traced line answer
which original frame supplied any point, and which parts of the drawn line were
never photographed.
"""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import INK, SHADOW, font, stored_preview
from .defects import gaussian_supported
from .geometry import DEFAULT_CORNERS, capture_plane_corners, homography, warp_capture
from .model import InputError, collection, enumeration, identity, string
from .photos import linear_to_srgb, luminance, open_working, png_bytes, srgb_to_linear, uint8_image
from .store import make_id


MAX_JOIN_DIM = 1800
MAX_JOIN_PIXELS = MAX_JOIN_DIM * MAX_JOIN_DIM
MIN_OVERLAP_PIXELS = 240
MAX_ROUTE_PHOTOS = 2
SOURCE_COLORS = (
    (76, 190, 178),
    (240, 158, 100),
    (168, 141, 222),
    (196, 210, 116),
)
UNCOVERED_DARK = (33, 37, 39)
UNCOVERED_LIGHT = (112, 86, 84)
FEATURES = (
    ("fissures", "Surface fissures"),
    ("worn_edge", "Worn marking edge"),
)


@dataclass(frozen=True)
class RouteFrame:
    linear: np.ndarray
    valid: np.ndarray
    clipped: np.ndarray
    source_map: np.ndarray
    photo: dict
    scale: tuple | None
    join: dict

    @property
    def extent(self):
        height, width = self.valid.shape
        return width, height

    def display(self):
        return linear_to_srgb(self.linear)


def apply_homography(matrix, x, y):
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if abs(denominator) < 1e-12:
        raise InputError("A capture projects outside the registered plane.")
    return (
        (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator,
        (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator,
    )


def photo_quad_in_plane(matrix, plane_size):
    """Full-frame extent of one registered capture, in plane pixels."""
    width, height = plane_size
    inverse = np.linalg.inv(matrix)
    points = []
    for corner in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        u, v = apply_homography(inverse, *corner)
        points.append((u * width, v * height))
    return np.asarray(points, dtype=np.float64)


def route_transform(plane_matrix, render_scale, origin, canvas, plane_size):
    """Plane-normalized -> joined-canvas-normalized homography."""
    width, height = plane_size
    canvas_width, canvas_height = canvas
    origin_x, origin_y = origin
    to_canvas = np.array(
        [
            [canvas_width / (render_scale * width), 0.0, origin_x / width],
            [0.0, canvas_height / (render_scale * height), origin_y / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return plane_matrix @ to_canvas


def photo_matrix(tile, photo_id):
    return homography(DEFAULT_CORNERS, capture_plane_corners(tile, photo_id))


def ordered_photos(tile, payload):
    raw = collection(payload.get("photo_ids"), "Route captures", minimum=2, maximum=MAX_ROUTE_PHOTOS)
    identifiers = [identity(item, "Route capture") for item in raw]
    if len(set(identifiers)) != len(identifiers):
        raise InputError("A joined route needs two distinct captures.")
    plane = tile.get("plane")
    if plane is None:
        raise InputError("Set a base photo and mark the specimen plane before joining a route.")
    registered = {plane["photo_id"], *tile["registrations"]}
    photos = []
    for item in identifiers:
        found = next((photo for photo in tile["photos"] if photo["id"] == item), None)
        if found is None:
            raise InputError("A route capture is no longer attached to this specimen.")
        if item not in registered:
            raise InputError("Register every route capture in the specimen plane before joining.")
        photos.append(found)
    return photos


def canvas_layout(quads):
    stacked = np.vstack(quads)
    if not np.all(np.isfinite(stacked)):
        raise InputError("A registered capture does not project into the specimen plane.")
    left, top = float(stacked[:, 0].min()), float(stacked[:, 1].min())
    right, bottom = float(stacked[:, 0].max()), float(stacked[:, 1].max())
    span_x, span_y = right - left, bottom - top
    if span_x < 16 or span_y < 16:
        raise InputError("The registered captures do not cover a usable route.")
    render_scale = min(1.0, MAX_JOIN_DIM / max(span_x, span_y))
    while True:
        width = max(16, int(round(span_x * render_scale)))
        height = max(16, int(round(span_y * render_scale)))
        if width * height <= MAX_JOIN_PIXELS or render_scale <= 0.01:
            break
        render_scale *= 0.94
    return (left, top), (width, height), render_scale


def overlap_features(linear, support):
    """Diagnostic strength for the two features the field notes call out."""
    support = support & np.all(np.isfinite(linear), axis=2)
    if np.count_nonzero(support) < MIN_OVERLAP_PIXELS:
        return None
    light = luminance(linear)
    smooth, weight = gaussian_supported(light, support, 1.6)
    usable = support & (weight > 0.85)
    if np.count_nonzero(usable) < MIN_OVERLAP_PIXELS:
        return None
    detail = np.abs(light - smooth)
    dark_line = np.maximum(smooth - light, 0.0)
    paint = 0.5 * linear[:, :, 0] + 0.5 * linear[:, :, 1] - linear[:, :, 2]
    paint_smooth, paint_weight = gaussian_supported(paint, usable, 1.3)
    paint_usable = usable & (paint_weight > 0.85)
    ragged = np.abs(paint - paint_smooth)
    population = paint_usable if np.count_nonzero(paint_usable) >= MIN_OVERLAP_PIXELS else usable
    return {
        "sharpness": round(float(np.quantile(detail[usable], 0.90)), 7),
        "fissures": round(float(np.quantile(dark_line[usable], 0.99)), 7),
        "worn_edge": round(float(np.quantile(ragged[population], 0.95)), 7),
        "paint_level": round(float(np.quantile(paint[population], 0.95)), 7),
        "pixels": int(np.count_nonzero(support)),
    }


def feature_choice(scores, order_ids):
    """One supplier that keeps the weaker of the two named features strongest."""
    present = [photo_id for photo_id in order_ids if scores.get(photo_id)]
    if len(present) < 2:
        raise InputError("The overlap is too small to compare both captures.")
    relative = {}
    for key, _ in FEATURES:
        best = max(scores[photo_id][key] for photo_id in present)
        relative[key] = {
            photo_id: (scores[photo_id][key] / best if best > 0 else 0.0) for photo_id in present
        }
    balanced = {
        photo_id: min(relative[key][photo_id] for key, _ in FEATURES) for photo_id in present
    }
    chosen = max(
        present,
        key=lambda photo_id: (balanced[photo_id], relative["fissures"][photo_id], -order_ids.index(photo_id)),
    )
    return chosen, balanced, relative


def gap_summary(uncovered, scale):
    labels, count = ndimage.label(uncovered)
    height, width = uncovered.shape
    total = int(np.count_nonzero(uncovered))
    factor = scale[0] * scale[1] if scale else 1.0
    gaps = []
    if count:
        for index, bounds in enumerate(ndimage.find_objects(labels), 1):
            if bounds is None:
                continue
            size = int(np.count_nonzero(labels[bounds] == index))
            if size < 24:
                continue
            y_slice, x_slice = bounds
            gaps.append({
                "pixels": size,
                "area": round(size * factor, 3),
                "bounds": [
                    round(x_slice.start / width, 5), round(y_slice.start / height, 5),
                    round(x_slice.stop / width, 5), round(y_slice.stop / height, 5),
                ],
                "touches_border": bool(
                    x_slice.start == 0 or y_slice.start == 0
                    or x_slice.stop == width or y_slice.stop == height
                ),
            })
    gaps.sort(key=lambda item: -item["pixels"])
    return {
        "pixels": total,
        "percent": round(100 * total / uncovered.size, 3),
        "area": round(total * factor, 3),
        "components": gaps[:24],
        "component_count": len(gaps),
    }


def hatch_uncovered(rgb, covered):
    uncovered = ~covered
    if not np.any(uncovered):
        return rgb
    rgb = rgb.copy()
    rgb[uncovered] = np.asarray(UNCOVERED_DARK, dtype=np.float32) / 255
    rows, cols = np.indices(covered.shape)
    stripe = ((rows + cols) % 16 < 3) & uncovered
    rgb[stripe] = np.asarray(UNCOVERED_LIGHT, dtype=np.float32) / 255
    return rgb


def source_palette(index):
    if index <= 0:
        return (85, 85, 85)
    return SOURCE_COLORS[(index - 1) % len(SOURCE_COLORS)]


def palette_hex(index):
    return "#" + "".join(f"{value:02x}" for value in source_palette(index))


def largest_component_centroid(mask):
    labels, count = ndimage.label(mask)
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    rows, cols = np.nonzero(labels == int(np.argmax(sizes)))
    height, width = mask.shape
    return float(np.median(cols)) / width, float(np.median(rows)) / height


def provenance_picture(display, source_map, entries, gaps):
    """Faded composite with a distinct colour, frame code and label per source."""
    rgb = display * 0.54
    for entry in entries:
        selected = source_map == entry["index"]
        rgb[selected] = rgb[selected] * 0.40 + np.asarray(source_palette(entry["index"]), dtype=np.float32) / 255 * 0.60
    rgb = hatch_uncovered(rgb, source_map > 0)
    picture = uint8_image(np.clip(rgb, 0, 1), np.ones(rgb.shape[:2], dtype=np.float32))
    strip = 16 + 21 * (len(entries) + 1)
    width, height = picture.size
    canvas = Image.new("RGB", (width, height + strip), (24, 33, 35))
    canvas.paste(picture.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    size = max(11, min(19, width // 48))
    typeface = font(size)
    x = max(8, width // 70)
    y = height + 7
    draw.text((x, y), "Frame supplying each pixel", fill=INK, font=typeface)
    y += size + 6
    for entry in entries:
        color = source_palette(entry["index"])
        draw.rectangle((x, y, x + size, y + size), fill=color)
        draw.text((x + size + 6, y - 1), f"{entry['code']} · {entry['name']}"[:86], fill=INK, font=typeface)
        centre = largest_component_centroid(source_map == entry["index"])
        if centre is not None:
            tx = min(max(int(centre[0] * width) + 5, 3), width - 44)
            ty = min(max(int(centre[1] * height) + 4, 3), height - size - 3)
            draw.text((tx, ty), entry["code"], fill=INK, font=typeface, stroke_width=3, stroke_fill=SHADOW)
        y += size + 5
    draw.rectangle((x, y, x + size, y + size), fill=UNCOVERED_DARK)
    draw.text((x + size + 6, y - 1), f"Not photographed · {gaps['percent']:.2f}% of canvas", fill=INK, font=typeface)
    return canvas


def source_sections(source_map, entries, scale):
    total = int(np.count_nonzero(source_map))
    factor = scale[0] * scale[1] if scale else 1.0
    height, width = source_map.shape
    rows = []
    for entry in entries:
        selected = source_map == entry["index"]
        count = int(np.count_nonzero(selected))
        row = {
            **entry,
            "pixels": count,
            "percent": round(100 * count / max(total, 1), 3),
            "area": round(count * factor, 3),
        }
        if count:
            row_idx = np.nonzero(selected)
            row["bounds"] = [
                round(float(row_idx[1].min()) / width, 5), round(float(row_idx[0].min()) / height, 5),
                round(float(row_idx[1].max() + 1) / width, 5), round(float(row_idx[0].max() + 1) / height, 5),
            ]
        rows.append(row)
    return rows


def resolution_scale(tile, plane, render_scale):
    """Millimetres per joined-canvas pixel, from the registered plane density."""
    plane_width, plane_height = plane["size"]
    if not (tile.get("width_mm") and tile.get("height_mm")):
        return None
    return (
        tile["width_mm"] / plane_width / render_scale,
        tile["height_mm"] / plane_height / render_scale,
    )


def create_join(store, specimen_id, payload):
    name = string(payload.get("name", "Joined route"), "Joined route name", maximum=90)
    tile = store.tile(specimen_id)
    plane = tile.get("plane")
    if plane is None:
        raise InputError("Set a base photo and mark the specimen plane before joining a route.")
    photos = ordered_photos(tile, payload)
    overlap_mode = enumeration(payload.get("overlap_source", "auto"), ("auto", "head", "tail"), "Overlap source")

    plane_size = tuple(plane["size"])
    matrices = [photo_matrix(tile, photo["id"]) for photo in photos]
    quads = [photo_quad_in_plane(matrix, plane_size) for matrix in matrices]
    origin, canvas, render_scale = canvas_layout(quads)

    warped, valid_masks, clipped_masks = [], [], []
    for photo, matrix in zip(photos, matrices):
        rgb, alpha = open_working(store.blobs, photo)
        transform = route_transform(matrix, render_scale, origin, canvas, plane_size)
        linear, valid, clipped = warp_capture(rgb, alpha, canvas, transform)
        warped.append(linear)
        valid_masks.append(valid)
        clipped_masks.append(clipped)
    canvas_height, canvas_width = valid_masks[0].shape

    overlap_area = np.logical_and.reduce(valid_masks)
    overlap_pixels = int(np.count_nonzero(overlap_area))
    overlap_percent = round(100 * overlap_pixels / float(canvas_width * canvas_height), 3)

    entries = [
        {
            "index": index,
            "code": f"F{index}",
            "photo_id": photo["id"],
            "name": photo["name"],
            "role": photo["role"],
            "order": index,
        }
        for index, photo in enumerate(photos, 1)
    ]
    order_ids = [entry["photo_id"] for entry in entries]

    if overlap_pixels >= MIN_OVERLAP_PIXELS:
        analysis = ndimage.binary_erosion(overlap_area, iterations=1)
        if np.count_nonzero(analysis) < MIN_OVERLAP_PIXELS:
            analysis = overlap_area
        scores = {}
        for entry, linear, clipped in zip(entries, warped, clipped_masks):
            usable = analysis & ~clipped
            if np.count_nonzero(usable) < MIN_OVERLAP_PIXELS:
                usable = analysis
            scores[entry["photo_id"]] = overlap_features(linear, usable)
        if all(score is None for score in scores.values()):
            raise InputError("The overlap has too little continuous image support to compare.")
        if overlap_mode == "head":
            chosen, balanced, relative = order_ids[0], None, None
            reason = "The whole overlap was assigned to the head capture by your choice."
        elif overlap_mode == "tail":
            chosen, balanced, relative = order_ids[-1], None, None
            reason = "The whole overlap was assigned to the tail capture by your choice."
        else:
            chosen, balanced, relative = feature_choice(scores, order_ids)
            reason = (
                "Both captures cover the whole overlap and no pixels were blended. The automatic "
                "choice keeps the weaker of the two labelled features (surface fissures and the "
                "worn marking edge) strongest, so the joined trace stays continuous."
            )
        chosen_entry = next(entry for entry in entries if entry["photo_id"] == chosen)
    else:
        chosen_entry = entries[0]
        scores = {entry["photo_id"]: None for entry in entries}
        balanced, relative = None, None
        reason = "The captures do not visibly overlap, so each one supplies only its own extent."

    # Unique-only regions keep their sole owner; every shared pixel goes to the
    # single chosen frame.  No pixel is ever averaged or blended.
    source_map = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    for entry in [chosen_entry, *[item for item in entries if item is not chosen_entry]]:
        source_map[(source_map == 0) & valid_masks[entry["index"] - 1]] = entry["index"]
    covered = source_map > 0
    if overlap_pixels >= MIN_OVERLAP_PIXELS:
        split_overlap = int(np.count_nonzero(overlap_area & (source_map != chosen_entry["index"])))
    else:
        split_overlap = 0

    composite_linear = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    composite_clipped = np.zeros((canvas_height, canvas_width), dtype=bool)
    for entry, linear, clipped in zip(entries, warped, clipped_masks):
        selected = source_map == entry["index"]
        composite_linear[selected] = linear[selected]
        composite_clipped |= selected & clipped
    display = np.clip(linear_to_srgb(composite_linear), 0, 1)
    display = hatch_uncovered(display, covered)
    rgba = np.dstack((np.rint(display * 255).astype(np.uint8), np.full(covered.shape, 255, dtype=np.uint8)))
    composite_image = Image.fromarray(rgba, "RGBA")

    scale = resolution_scale(tile, plane, render_scale)
    unit = "mm" if scale else "px"
    gaps = gap_summary(~covered, scale)
    sections = source_sections(source_map, entries, scale)

    features = []
    for key, label in FEATURES:
        if scores.get(chosen_entry["photo_id"]) is None:
            continue
        chosen_score = scores[chosen_entry["photo_id"]][key]
        best_code, best_value = chosen_entry["code"], chosen_score
        for entry in entries:
            score = scores.get(entry["photo_id"])
            if score and score[key] > best_value:
                best_code, best_value = entry["code"], score[key]
        features.append({
            "key": key,
            "label": label,
            "chosen_code": chosen_entry["code"],
            "chosen_score": chosen_score,
            "best_code": best_code,
            "best_score": best_value,
            "retained": best_code == chosen_entry["code"],
        })
    retained = [item["label"] for item in features if item["retained"]]
    sacrificed = [
        {
            "feature": item["label"],
            "better_code": item["best_code"],
            "better_name": next(entry["name"] for entry in entries if entry["code"] == item["best_code"]),
            "chosen_score": item["chosen_score"],
            "better_score": item["best_score"],
        }
        for item in features if not item["retained"]
    ]

    lightness_step = None
    if overlap_pixels >= MIN_OVERLAP_PIXELS:
        chosen_light = luminance(warped[chosen_entry["index"] - 1])[overlap_area & ~composite_clipped]
        others = [
            luminance(warped[entry["index"] - 1])[overlap_area & ~clipped_masks[entry["index"] - 1]]
            for entry in entries if entry["index"] != chosen_entry["index"]
        ]
        others = [values for values in others if len(values)]
        if len(chosen_light) and others:
            lightness_step = round(float(np.median(chosen_light) - np.median(others[0])), 6)

    image_digest = store.blobs.put(png_bytes(composite_image, compression=6))
    palette = [0, 0, 0] + [channel for entry in entries for channel in source_palette(entry["index"])]
    palette += [0, 0, 0] * (256 - len(palette) // 3)
    source_image = Image.fromarray(source_map, mode="P")
    source_image.putpalette(palette[:768])
    source_digest = store.blobs.put(png_bytes(source_image, compression=9))
    provenance_digest = stored_preview(provenance_picture(display, source_map, entries, gaps), store.blobs)

    join = {
        "id": make_id(),
        "tile_id": tile["id"],
        "name": name,
        "source_version": tile["version"],
        "kind": "route-join",
        "unit": unit,
        "canvas": [canvas_width, canvas_height],
        "plane_size": list(plane_size),
        "plane_corners": plane["corners"],
        "render_scale": round(render_scale, 8),
        "plane_origin": [round(origin[0], 5), round(origin[1], 5)],
        "scale": [round(value, 10) for value in scale] if scale else None,
        "photos": entries,
        "image_digest": image_digest,
        "source_map_digest": source_digest,
        "provenance_digest": provenance_digest,
        "coverage": {
            "covered_pixels": int(np.count_nonzero(covered)),
            "covered_percent": round(100 * np.count_nonzero(covered) / covered.size, 3),
            "unphotographed": gaps,
        },
        "overlap": {
            "pixels": overlap_pixels,
            "percent_of_route": overlap_percent,
            "supplier_photo_id": chosen_entry["photo_id"],
            "supplier_code": chosen_entry["code"],
            "supplier_name": chosen_entry["name"],
            "supplier_pixels": overlap_pixels - split_overlap,
            "split_pixels": split_overlap,
            "single_source": split_overlap == 0,
            "lightness_step": lightness_step,
        },
        "sections": sections,
        "decision": {
            "mode": overlap_mode,
            "chosen_photo_id": chosen_entry["photo_id"],
            "chosen_code": chosen_entry["code"],
            "chosen_name": chosen_entry["name"],
            "reason": reason,
            "blend_mode": "none",
            "mixed_pixels": 0,
            "overlap_compared": overlap_pixels >= MIN_OVERLAP_PIXELS,
            "balanced_ratios": {
                next(entry["code"] for entry in entries if entry["photo_id"] == photo_id): round(value, 4)
                for photo_id, value in (balanced or {}).items()
            },
            "features": features,
            "retained": retained,
            "sacrificed": sacrificed,
        },
    }
    return store.save_join(join)


def point_source(join, source_map, position):
    """Which original capture supplies one canvas point, and whether it was photographed."""
    width, height = join["canvas"]
    x = min(max(float(position[0]), 0.0), 1.0) * width
    y = min(max(float(position[1]), 0.0), 1.0) * height
    col = min(max(int(math.floor(x)), 0), width - 1)
    row = min(max(int(math.floor(y)), 0), height - 1)
    code = int(source_map[row, col])
    entry = next((item for item in join["photos"] if item["index"] == code), None)
    return {
        "normalized_position": [round(float(position[0]), 6), round(float(position[1]), 6)],
        "canvas_pixel": [col, row],
        "source_code": entry["code"] if entry else None,
        "photo_id": entry["photo_id"] if entry else None,
        "photo_name": entry["name"] if entry else None,
        "photographed": entry is not None,
    }


def load_source_map(store, join):
    with Image.open(store.blobs.require(join["source_map_digest"])) as image:
        return np.asarray(image.convert("P"), dtype=np.uint8).copy()


def route_frame(store, join):
    with Image.open(store.blobs.require(join["image_digest"])) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
    source_map = load_source_map(store, join)
    valid = source_map > 0
    rgb = rgba[:, :, :3]
    linear = srgb_to_linear(rgb)
    linear[~valid] = 0
    clipped = valid & ((np.max(rgb, axis=2) >= 253 / 255) | (np.max(rgb, axis=2) <= 5 / 255))
    photo = {"id": join["id"], "name": join["name"], "role": "route-join", "synthetic": False}
    scale = tuple(join["scale"]) if join.get("scale") else None
    return RouteFrame(linear, valid, clipped, source_map, photo, scale, join)


def summarize_join(join):
    return {
        key: join[key] for key in (
            "id", "tile_id", "name", "source_version", "unit", "canvas",
            "coverage", "overlap", "decision", "sections", "kind",
        )
    } | {
        "created_at": join.get("created_at"),
        "image_url": f"/api/joins/{join['id']}/image.png",
        "provenance_url": f"/api/joins/{join['id']}/provenance.png",
        "data_url": f"/api/joins/{join['id']}/data.json",
        "palette": [
            {"code": entry["code"], "name": entry["name"], "color": palette_hex(entry["index"])}
            for entry in join["photos"]
        ],
    }
