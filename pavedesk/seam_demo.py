"""Synthetic repair seam training specimen.

One repair patch is photographed twice: the existing-paving end under a warm
lamp and the fresh fill end under window daylight.  Each view carries a grey
card from the same physical batch.  The warm lamp adds enough red cast that
the old end's calibrated card still exceeds the 5% red-to-green check, while
the daylight end passes both channel checks.  A genuine color step between
the old paving and the new fill is baked into the surface so it survives the
shared calibration.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .geometry import DEFAULT_CORNERS, canvas_dimensions, homography
from .photos import linear_to_srgb, png_bytes, srgb_to_linear, store_capture
from .store import make_id


SEAM_DEMO_KEY = "pavedesk.seam.training.v1"
SEAM_PLANE = (560, 430)
SEAM_PHOTO = (700, 540)
SEAM_SEED = 217

# Linear-light illumination multipliers applied per capture.
WARM_LIGHT = np.array([1.24, 1.0, 0.78], dtype=np.float32)
DAYLIGHT = np.array([0.99, 1.0, 1.035], dtype=np.float32)

# Neutral grey card reflectance (same physical batch in both views).
CARD_REFLECTANCE = np.array([0.19, 0.19, 0.19], dtype=np.float32)

SEAM_X = 0.42
PATCH_LEFT, PATCH_RIGHT = 0.42, 0.74
PATCH_TOP, PATCH_BOTTOM = 0.16, 0.86


@dataclass(frozen=True)
class SeamView:
    role: str
    light: np.ndarray
    label: str


VIEWS = (
    SeamView("warm", WARM_LIGHT, "Warm lamp"),
    SeamView("daylight", DAYLIGHT, "Window daylight"),
)

# Slight pose offsets so registration is exercised, close to the unit plane.
VIEW_CORNERS = {
    "warm": [[0.078, 0.072], [0.928, 0.098], [0.912, 0.928], [0.070, 0.902]],
    "daylight": [[0.088, 0.088], [0.918, 0.080], [0.922, 0.918], [0.082, 0.912]],
}


def _texture_field(generator, shape):
    height, width = shape
    rows, cols = int(np.ceil(height / 58)) + 4, int(np.ceil(width / 74)) + 4
    coarse = generator.normal(size=(rows, cols)).astype(np.float32)
    y, x = np.indices(shape, dtype=np.float32)
    broad = ndimage.map_coordinates(coarse, (y / 58 + 1, x / 74 + 1), order=3)
    pebble = generator.normal(size=shape).astype(np.float32)
    return (0.86 + 0.07 * broad + 0.035 * pebble).astype(np.float32)


def _card_rect(x, y):
    return (x >= 0.075) & (x <= 0.158) & (y >= 0.802) & (y <= 0.925)


def bake_seam_field():
    width, height = SEAM_PLANE
    shape = height, width
    generator = np.random.default_rng(SEAM_SEED)
    y, x = np.indices(shape, dtype=np.float32)
    y /= height
    x /= width
    texture = _texture_field(generator, shape)

    # Existing paving: cool mid-grey aggregate.
    old_base = srgb_to_linear(np.array([112, 108, 102], dtype=np.float32) / 255)
    # Fresh repair fill: warmer and slightly lighter — a real color difference.
    new_base = srgb_to_linear(np.array([138, 126, 104], dtype=np.float32) / 255)

    field = np.broadcast_to(old_base, shape + (3,)).copy()
    patch = (x >= PATCH_LEFT) & (x <= PATCH_RIGHT) & (y >= PATCH_TOP) & (y <= PATCH_BOTTOM)
    field[patch] = new_base
    field *= texture[..., None]

    # The seam itself is a thin darker joint.
    seam = np.abs(x - SEAM_X) < 0.0045
    field[seam] *= 0.55

    # Weathering gradient on the old material right up to the seam, so the
    # old edge is not a flat color and any real edge change stays visible.
    near_old = (x < SEAM_X) & (x > SEAM_X - 0.16) & (y >= PATCH_TOP) & (y <= PATCH_BOTTOM)
    weather = np.clip((SEAM_X - x) / 0.16, 0, 1)
    field[near_old] *= (0.86 + 0.12 * weather[near_old])[..., None]

    # Both grey cards come from the same neutral batch.
    card = _card_rect(x, y)
    field[card] = CARD_REFLECTANCE
    border = (
        (x >= 0.066) & (x <= 0.167) & (y >= 0.793) & (y <= 0.934) & ~card
    )
    field[border] *= 0.62
    return np.clip(field, 0, 1), card


def render_seam_photo(surface, view):
    height, width = SEAM_PHOTO
    lit = surface * view.light[None, None, :]
    corners = VIEW_CORNERS[view.role]
    photo_to_tile = np.linalg.inv(homography(DEFAULT_CORNERS, corners))
    photo_y, photo_x = np.indices((height, width), dtype=np.float32)
    photo_y = (photo_y + 0.5) / height
    photo_x = (photo_x + 0.5) / width
    denominator = photo_to_tile[2, 0] * photo_x + photo_to_tile[2, 1] * photo_y + photo_to_tile[2, 2]
    tile_u = (photo_to_tile[0, 0] * photo_x + photo_to_tile[0, 1] * photo_y + photo_to_tile[0, 2]) / denominator
    tile_v = (photo_to_tile[1, 0] * photo_x + photo_to_tile[1, 1] * photo_y + photo_to_tile[1, 2]) / denominator
    tile_h, tile_w = surface.shape[:2]
    coords = (tile_v * tile_h - 0.5, tile_u * tile_w - 0.5)
    warped = np.stack(
        [
            ndimage.map_coordinates(lit[..., channel], coords, order=1, mode="constant", cval=0.0)
            for channel in range(3)
        ],
        axis=2,
    )
    alpha = ((tile_u >= 0) & (tile_u <= 1) & (tile_v >= 0) & (tile_v <= 1)).astype(np.float32)
    background = np.array([0.045, 0.047, 0.05], dtype=np.float32)
    mixed = warped * alpha[..., None] + background * (1 - alpha[..., None])
    image = np.rint(np.clip(linear_to_srgb(mixed), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(image)


def rectangle(name, purpose, left, top, right, bottom):
    return {
        "id": make_id(),
        "name": name,
        "purpose": purpose,
        "vertices": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


def seam_regions():
    return [
        rectangle("Old-material seam edge", "inspection", 0.255, 0.30, 0.405, 0.70),
        rectangle("New-fill seam edge", "inspection", 0.435, 0.30, 0.600, 0.70),
        rectangle("Warm-view grey card", "neutral", 0.085, 0.812, 0.150, 0.916),
        rectangle("Daylight-view grey card", "neutral", 0.085, 0.812, 0.150, 0.916),
    ]


def seam_training_specimen(store):
    surface, _ = bake_seam_field()
    photos = []
    for view in VIEWS:
        image = render_seam_photo(surface, view)
        photo = store_capture(
            store.blobs,
            png_bytes(image),
            view.role,
            f"harbour-walk-repair-{view.role}.png",
            f"Procedurally rendered repair seam view under {view.label.lower()}.",
            make_id(),
            synthetic=True,
        )
        photos.append(photo)
    base = photos[0]
    size = canvas_dimensions(base, VIEW_CORNERS["warm"], 2200, 1700)
    return {
        "name": "Harbour walk repair",
        "batch": "Quay east",
        "body": "Concrete paving",
        "glaze": "Repair fill",
        "firing": "November survey",
        "notes": (
            "Synthetic repair seam. The warm-lamp view of the existing paving and the window-light "
            "view of the new fill each carry the same neutral grey card batch."
        ),
        "width_mm": 2200,
        "height_mm": 1700,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": VIEW_CORNERS["warm"], "size": list(size)},
        "regions": seam_regions(),
        "registrations": {
            photo["id"]: {
                "base_photo_id": base["id"],
                "corners": VIEW_CORNERS[photo["role"]],
                "support_percent": 100,
            }
            for photo in photos[1:]
        },
    }


def load_seam_training(store):
    existing = store.seeded_tiles(SEAM_DEMO_KEY)
    if existing:
        return existing, False
    row = seam_training_specimen(store)
    return store.seed_tiles(SEAM_DEMO_KEY, [row]), True
