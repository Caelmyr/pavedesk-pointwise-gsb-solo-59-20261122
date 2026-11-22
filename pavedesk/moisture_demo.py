"""Synthetic two-visit moisture recheck specimen.

Renders one pavement section photographed twice. Each visit keeps a parallel
and a cross polarization capture registered to the same plane. The later visit
misses the right corner (its registered plane stops short), the morning view
holds drying wet marks there, and an old painted patch produces specular glare
in parallel light but reads as paint color so it must not be dispatched.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .demo import (
    PHOTO_SIZE,
    PLANE_SIZE,
    random_field,
    surface_palette,
    warp_to_photo,
)
from .geometry import canvas_dimensions
from .photos import png_bytes
from .store import make_id
from .photos import store_capture


MOISTURE_KEY = "pavedesk.moisture-demo.v1"
# Later-visit plane stops before the right corner, so that corner has no
# later-visit support while the morning visit covers the full section.
LATER_CORNERS = [[0.085, 0.105], [0.802, 0.088], [0.788, 0.902], [0.092, 0.918]]
# Fraction of the shared registered plane that the later capture actually
# covers; everything to the right is recorded as missed (no later evidence).
LATER_PLANE_CLIP = [0.0, 0.0, 0.795, 1.0]
MORNING_POSES = {
    "parallel": [[0.122, 0.107], [0.910, 0.084], [0.925, 0.871], [0.101, 0.916]],
    "cross": [[0.079, 0.101], [0.908, 0.119], [0.877, 0.916], [0.081, 0.909]],
}


@dataclass(frozen=True)
class MoistureField:
    diffuse: np.ndarray
    height: np.ndarray
    wet_am: np.ndarray
    wet_pm: np.ndarray
    graphite_marks: np.ndarray
    roughness: np.ndarray
    old_paint: np.ndarray
    u: np.ndarray
    v: np.ndarray


def gaussian_blob(y, x, center_y, center_x, sigma_y, sigma_x):
    return np.exp(-(((x - center_x) / sigma_x) ** 2 + ((y - center_y) / sigma_y) ** 2))


def bake_moisture_field(seed=20261122):
    width, height = PLANE_SIZE
    shape = height, width
    generator = np.random.default_rng(seed)
    y, x = np.indices(shape, dtype=np.float32)
    y /= height
    x /= width
    broad = random_field(generator, shape, 64, 82)
    medium = random_field(generator, shape, 20, 29)
    pebble = random_field(generator, shape, 2.8, 3.2)
    small = generator.normal(size=shape).astype(np.float32)
    tones = .32 * broad + .28 * medium + .29 * pebble + .19 * small
    palette = ((58, 56, 52), (92, 89, 82), (138, 132, 118))
    diffuse = surface_palette(palette, tones)

    # Persistent damp patch visible in both visits (left-centre).
    persistent = gaussian_blob(y, x, 0.46, 0.40, 0.13, 0.15)
    # Morning-only wet marks, one inside the corner the later visit misses.
    corner_damp = gaussian_blob(y, x, 0.30, 0.86, 0.10, 0.085)
    dried = gaussian_blob(y, x, 0.72, 0.58, 0.09, 0.10)
    wet_am = np.clip(0.95 * persistent + 0.92 * corner_damp + 0.85 * dried, 0, 1)
    # The dried patch is gone later; corner damp is outside the later frame anyway.
    wet_pm = np.clip(0.9 * persistent + 0.1 * dried, 0, 1)

    # Old paint patch: chromatic yellow-white marking kept inside the later
    # visit's coverage, so it is a genuine both-visit false positive to reject.
    paint = (
        (np.abs(y - (0.58 + 0.006 * np.sin(18 * x))) < 0.028)
        & (x > 0.42) & (x < 0.70)
    )
    paint = ndimage.binary_dilation(paint, iterations=2)
    paint_tint = np.array([0.86, 0.73, 0.34], dtype=np.float32)
    diffuse[paint] = diffuse[paint] * 0.25 + paint_tint[None, :] * 0.75
    old_paint = paint.astype(np.float32)

    graphite_marks = np.clip(.68 * persistent + .6 * corner_damp, 0, .88)
    roughness = np.clip(.7 + pebble * .12 - wet_am * .22, .15, .99)
    heights = .15 * broad + .08 * medium + .05 * pebble + .008 * small
    return MoistureField(
        np.clip(diffuse, 0, 1), heights, wet_am.astype(np.float32),
        wet_pm.astype(np.float32), graphite_marks.astype(np.float32),
        roughness.astype(np.float32), old_paint.astype(np.float32), x, y,
    )


def moisture_gloss(field, role, wet):
    x, y = field.u, field.v
    # Glare is governed by the wet film: a damp surface returns the parallel
    # source strongly while cross polarization nearly removes it. There is no
    # broad geometry hotspot, so dry pavement stays below the wet threshold.
    low = (1 - field.roughness) * wet * (0.62 + 0.38 * np.sin(9 * x + 13 * y) ** 2)
    # Old paint keeps a strong parallel specular lobe in both visits.
    paint_glint = field.old_paint * (0.32 + 0.18 * np.sin(20 * x) ** 2)
    if role == "cross":
        intensity = 0.02 * low + 0.015 * field.old_paint
    else:
        intensity = 0.42 * low + paint_glint
    return intensity * (1 - field.graphite_marks * 0.4)


def moisture_light(field, role, wet):
    y, x = np.indices(field.diffuse.shape[:2], dtype=np.float32)
    dy, dx = np.gradient(0.92 * field.height)
    broad = 0.93 + 0.13 * (1 - y / y.max()) - 0.08 * (x / x.max() - 0.5) ** 2
    side = np.clip(0.98 + dx * 0.38 - dy * 0.31, 0.64, 1.2)
    exposure = 0.88 if role == "cross" else 1.09
    rgb = field.diffuse * (broad * side * exposure)[:, :, None]
    reflection = moisture_gloss(field, role, wet)
    rgb += reflection[:, :, None] * np.array([0.69, 0.79, 0.82], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def render_moisture_photo(field, role, wet, corners, photo_size, seed):
    linear = moisture_light(field, role, wet)
    image = warp_to_photo(linear, corners, photo_size, seed)
    array = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


def moisture_regions():
    from .demo import rectangle
    return [
        rectangle("Full recheck area", "inspection", 0.05, 0.06, 0.95, 0.92),
        rectangle("Bare pavement sample", "bare", 0.12, 0.12, 0.26, 0.24),
        rectangle("Old paint sample", "glazed", 0.46, 0.545, 0.66, 0.615),
    ]


def moisture_specimen(store):
    field = bake_moisture_field()
    photos = []
    plan = (
        ("morning", "parallel", field.wet_am, MORNING_POSES["parallel"]),
        ("morning", "cross", field.wet_am, MORNING_POSES["cross"]),
        ("later", "parallel", field.wet_pm, LATER_CORNERS),
        ("later", "cross", field.wet_pm, LATER_CORNERS),
    )
    for visit, role, wet, corners in plan:
        image = render_moisture_photo(field, role, wet, corners, PHOTO_SIZE, 20261122)
        photo = store_capture(
            store.blobs, png_bytes(image), role,
            f"forecourt-{visit}-{role}.png",
            f"Synthetic {visit} {role}-polarized capture.",
            make_id(), synthetic=True, visit=visit,
        )
        photos.append(photo)
    base = photos[0]
    size = canvas_dimensions(base, MORNING_POSES["parallel"], 3600, 2400)
    registrations = {}
    for photo, (_, _, _, corners) in zip(photos[1:], plan[1:]):
        registrations[photo["id"]] = {
            "base_photo_id": base["id"],
            "corners": corners,
            "support_percent": 100,
            "plane_clip": LATER_PLANE_CLIP if photo["visit"] == "later" else None,
        }
    return {
        "name": "Forecourt moisture recheck",
        "batch": "North loop",
        "body": "Concrete",
        "glaze": "Aged lane paint",
        "firing": "Two-visit polarization survey",
        "notes": (
            "Synthetic two-visit survey. Morning and later parallel/cross pairs share one plane; "
            "the later visit misses the right corner. The aged paint strip is a glare false "
            "positive that the color step keeps out of the dispatch range."
        ),
        "width_mm": 3600,
        "height_mm": 2400,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": MORNING_POSES["parallel"], "size": list(size)},
        "regions": moisture_regions(),
        "registrations": registrations,
    }


def load_moisture_demo(store):
    existing = store.seeded_tiles(MOISTURE_KEY)
    if existing:
        return existing, False
    rows = [moisture_specimen(store)]
    return store.seed_tiles(MOISTURE_KEY, rows), True
