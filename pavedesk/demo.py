from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage

from .geometry import DEFAULT_CORNERS, canvas_dimensions, homography
from .photos import linear_to_srgb, png_bytes, srgb_to_linear
from .store import make_id
from .photos import store_capture


DEMO_KEY = "pavedesk.training.v1"
PLANE_SIZE = (620, 472)
PHOTO_SIZE = (766, 598)
DEMO_ROLES = ("diffuse", "parallel", "cross", "rake-left", "rake-right")
POSES = {
    "diffuse": [[0.092, 0.086], [0.911, 0.112], [0.894, 0.912], [0.081, 0.889]],
    "parallel": [[0.122, 0.107], [0.910, 0.084], [0.925, 0.871], [0.101, 0.916]],
    "cross": [[0.079, 0.101], [0.908, 0.119], [0.877, 0.916], [0.081, 0.909]],
    "rake-left": [[0.089, 0.094], [0.894, 0.088], [0.902, 0.905], [0.064, 0.911]],
    "rake-right": [[0.094, 0.095], [0.913, 0.094], [0.926, 0.906], [0.112, 0.893]],
}


@dataclass(frozen=True)
class DemoSpecimen:
    seed: int
    name: str
    body: str
    glaze: str
    firing: str
    batch: str
    colors: tuple


SERIES = (
    DemoSpecimen(81, "Juniper crossing", "Asphalt", "Yellow lane paint", "April survey", "North loop",
                ((55, 53, 49), (89, 86, 80), (136, 130, 116))),
    DemoSpecimen(114, "Depot apron", "Concrete", "White lane paint", "April survey", "North loop",
                ((72, 70, 63), (123, 119, 107), (165, 159, 142))),
    DemoSpecimen(129, "Sawmill link", "Asphalt", "Yellow lane paint", "May survey", "West route",
                ((43, 46, 47), (81, 85, 84), (120, 121, 117))),
    DemoSpecimen(94, "Cedar forecourt", "Concrete", "White lane paint", "May survey", "West route",
                ((61, 62, 61), (105, 107, 104), (151, 152, 148))),
)

def random_field(generator, size, y_space, x_space):
    height, width = size
    rows, cols = int(np.ceil(height / y_space)) + 4, int(np.ceil(width / x_space)) + 4
    coarse = generator.normal(size=(rows, cols)).astype(np.float32)
    y, x = np.indices(size, dtype=np.float32)
    field = ndimage.map_coordinates(coarse, (y / y_space + 1, x / x_space + 1), order=3)
    return field.astype(np.float32)


@dataclass(frozen=True)
class RoadField:
    diffuse: np.ndarray
    height: np.ndarray
    wet: np.ndarray
    graphite_marks: np.ndarray
    roughness: np.ndarray
    u: np.ndarray
    v: np.ndarray


def surface_palette(colors, values):
    linear_colors = srgb_to_linear(np.array(colors, dtype=np.float32) / 255)
    stops = (-1.2, 0.05, 1.25)
    return np.stack(
        [np.interp(values, stops, linear_colors[:, channel]) for channel in range(3)],
        axis=2,
    ).astype(np.float32)


def neutral_card(diffuse, roughness, x, y):
    sample = (x >= 0.07) & (x <= 0.165) & (y >= 0.844) & (y <= 0.961)
    border = (x >= 0.063) & (x <= 0.172) & (y >= 0.835) & (y <= 0.968) & ~sample
    diffuse[sample] = np.array([0.194, 0.172, 0.155])
    diffuse[border] *= 0.70
    roughness[sample] = 0.76


def jagged_groove(x, y, origin, phase, lower, upper):
    centre = origin + .045*np.sin(x*12+phase) + .009*np.sin(x*71+phase) + .003*np.sin(x*185)
    along = np.clip((x-lower)/.06,0,1) * np.clip((upper-x)/.04,0,1)
    cross = np.exp(-np.square((y-centre)/(.002+.002*along)))
    lip = np.exp(-np.square((y-centre-.004)/.003))
    return cross*along, lip*along


def patch_and_pits(x,y,generator):
    edge = .36 + .005*np.sin(25*y)+.002*np.sin(98*y)
    patch = (x>=edge) & (x<.67) & (y>.23) & (y<.43)
    pits = np.zeros_like(x)
    for _ in range(25):
        px,py = generator.uniform(.19,.88), generator.uniform(.15,.81)
        rad_x,rad_y = generator.uniform(.004,.032),generator.uniform(.005,.024)
        score = ((x-px)/rad_x)**2 + ((y-py)/rad_y)**2
        pits = np.maximum(pits,np.clip(1-score,0,1)*generator.uniform(.38,.95))
    hollows = np.exp(-(((x-.75)/.052)**2 + ((y-.71)/.043)**2))
    return patch,pits,np.maximum(.33*hollows,pits)


def bake_field(specimen):
    width, height = PLANE_SIZE
    shape = height,width
    generator = np.random.default_rng(specimen.seed)
    y,x = np.indices(shape,dtype=np.float32)
    y/=height
    x/=width
    broad = random_field(generator,shape,64,82)
    medium = random_field(generator,shape,20,29)
    pebble = random_field(generator,shape,2.8,3.2)
    small = generator.normal(size=shape).astype(np.float32)
    tones = .32*broad + .28*medium + .29*pebble + .19*small
    diffuse = surface_palette(specimen.colors,tones)
    patch,pits,cups = patch_and_pits(x,y,generator)
    diffuse[patch] *= [.71,.75,.80]
    main,lip = jagged_groove(x,y,.33,.65,.09,.84)
    second,branch_lip = jagged_groove(y,x,.69,1.21,.21,.84)
    fracture = np.maximum(main,second*.75)
    strip = .485+.004*np.sin(23*x+specimen.seed)
    paint = np.clip((.022-np.abs(y-strip))/.004,0,1)
    gap = (.33<x) & (x<.415)
    weather = np.clip(.81*np.maximum(0,.75+.22*pebble+.28*medium+.14*small)-.33*gap,0,.85)
    stamp = paint * weather
    tint = srgb_to_linear(np.array([220,186,72] if specimen.seed in (81,129) else [229,224,205],dtype=np.float32)/255)
    diffuse = diffuse*(1-stamp[...,None])+tint[None,None,:]*stamp[...,None]
    dark=np.clip(.68*fracture+.61*pits,0,.88)
    diffuse *= (1-dark)[...,None]
    wet = np.exp(-(((x-.66)/.23)**2+((y-.60)/.21)**2))
    roughness=np.clip(.68+pebble*.13-wet*.19,.15,.99)
    neutral_card(diffuse,roughness,x,y)
    heights=.15*broad+.08*medium+.05*pebble+.008*small-cups*.38-fracture*.22+(lip+branch_lip)*.14
    return RoadField(np.clip(diffuse,0,1),heights,wet,dark,roughness,x,y)


def gloss_reflection(field, role):
    x, y = field.u, field.v
    hotspot = np.exp(-(((x - 0.64 + 0.17 * y) / 0.21) ** 2 + ((y - 0.38) / 0.25) ** 2))
    elongated = np.exp(-(((x - 0.64 - 0.38 * (y - 0.3)) / 0.075) ** 2 + ((y - 0.49) / 0.31) ** 2))
    low = (1 - field.roughness) * field.wet * (0.5 + 0.5 * np.sin(8 * x + 13 * y) ** 2)
    intensity = 0.079 * low if role == "diffuse" else 0.34 * low + 0.24 * hotspot + 0.12 * elongated
    if role == "cross":
        intensity = (0.009 + hotspot * 0.018) * low
    return intensity * (1 - field.graphite_marks * 0.58)


def light_field(field, role):
    x, y = field.u, field.v
    dy, dx = np.gradient(0.92 * field.height)
    broad = 0.93 + 0.13 * (1 - y) - 0.08 * (x - 0.5) ** 2
    if role in ("rake-left", "rake-right"):
        direction = 1 if role == "rake-left" else -1
        side = 0.91 + direction * 0.13 * (0.5 - x)
        shading = np.clip(0.66 + (dx * direction - dy * 0.23) * 3.9, 0.26, 1.32)
        rgb = field.diffuse * (broad * side * shading)[:, :, None]
    else:
        side = np.clip(0.98 + dx * 0.38 - dy * 0.31, 0.64, 1.2)
        reflection = gloss_reflection(field, role)
        exposure = 0.88 if role == "cross" else 1.09 if role == "parallel" else 1.0
        rgb = field.diffuse * (broad * side * exposure)[:, :, None]
        rgb += reflection[:, :, None] * np.array([0.69, 0.79, 0.82], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def desk_background(width, height, seed):
    generator = np.random.default_rng(seed + 500)
    noise = random_field(generator, (height, width), 21, 30) * 0.018
    y, x = np.indices((height, width), dtype=np.float32)
    grad = 0.12 + 0.19 * np.exp(-((x / width - 0.24) ** 2 + (y / height - 0.10) ** 2) * 2.4)
    grain = generator.normal(size=(height, width)).astype(np.float32) * 0.008
    neutral = np.maximum(grad + grain + noise, 0.035)
    return neutral[:, :, None] * np.array([0.50, 0.57, 0.58], dtype=np.float32)


def surface_alpha(x, y):
    horizontal = np.abs(x - 0.5) - 0.461
    vertical = np.abs(y - 0.5) - 0.452
    corner = np.hypot(np.maximum(horizontal, 0), np.maximum(vertical, 0))
    nearest = np.minimum(np.maximum(horizontal, vertical), 0)
    sdf = corner + nearest - 0.035
    return np.clip(0.5 - sdf * 410, 0, 1)


def warp_to_photo(array, corners, photo_size, seed):
    width, height = photo_size
    photo_to_tile = np.linalg.inv(homography(DEFAULT_CORNERS, corners))
    photo_y, photo_x = np.indices((height, width), dtype=np.float32)
    photo_y = (photo_y + 0.5) / height
    photo_x = (photo_x + 0.5) / width
    denominator = photo_to_tile[2, 0] * photo_x + photo_to_tile[2, 1] * photo_y + photo_to_tile[2, 2]
    tile_u = (photo_to_tile[0, 0] * photo_x + photo_to_tile[0, 1] * photo_y + photo_to_tile[0, 2]) / denominator
    tile_v = (photo_to_tile[1, 0] * photo_x + photo_to_tile[1, 1] * photo_y + photo_to_tile[1, 2]) / denominator
    tile_h, tile_w = array.shape[:2]
    coords = (tile_v * tile_h - 0.5, tile_u * tile_w - 0.5)
    tile_image = np.stack(
        [ndimage.map_coordinates(array[:, :, component], coords, order=1, mode="nearest") for component in range(3)],
        axis=2,
    )
    alpha = surface_alpha(tile_u, tile_v)
    shadow = ndimage.shift(ndimage.gaussian_filter(alpha, 7), (9, 5), order=1, mode="constant")
    stage = desk_background(width, height, seed)
    stage *= (1 - 0.61 * shadow)[:, :, None]
    rim = np.clip(alpha - ndimage.minimum_filter(alpha, size=8), 0, 1)
    tile_image *= 1 - 0.24 * rim[:, :, None]
    mixed = stage * (1 - alpha[:, :, None]) + tile_image * alpha[:, :, None]
    return linear_to_srgb(mixed)


def render_photo(field, role, specimen):
    linear = light_field(field, role)
    image = warp_to_photo(linear, POSES[role], PHOTO_SIZE, specimen.seed)
    array = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


def rectangle(name, purpose, left, top, right, bottom):
    return {
        "id": make_id(), "name": name, "purpose": purpose,
        "vertices": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


def training_regions(specimen):
    return [
        rectangle("Main survey", "inspection", 0.045, 0.050, 0.952, 0.809),
        rectangle("Observed paint", "glazed", 0.18, 0.473, 0.31, 0.490),
        rectangle("Adjacent pavement", "bare", 0.45, 0.67, 0.72, 0.77),
        rectangle("Neutral card", "neutral", 0.086, 0.858, 0.15, 0.943),
        rectangle("Survey marker", "ignore", 0.22, 0.931, 0.39, 0.981),
    ]


def training_specimen(store, specimen):
    field = bake_field(specimen)
    photos = []
    for role in DEMO_ROLES:
        image = render_photo(field, role, specimen)
        photo = store_capture(
            store.blobs, png_bytes(image), role,
            f"{specimen.name.lower().replace(' ', '-')}-{role}.png",
            "Procedurally rendered pavement training photograph.",
            make_id(),
            synthetic=True,
        )
        photos.append(photo)
    base = photos[0]
    size = canvas_dimensions(base, POSES["diffuse"], 3600, 2400)
    return {
        "name": specimen.name,
        "batch": specimen.batch,
        "body": specimen.body,
        "glaze": specimen.glaze,
        "firing": specimen.firing,
        "notes": "Synthetic road survey. Markings, pavement, lighting and depressions are rendered locally.",
        "width_mm": 3600,
        "height_mm": 2400,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": POSES["diffuse"], "size": list(size)},
        "regions": training_regions(specimen),
        "registrations": {
            photo["id"]: {"base_photo_id": base["id"], "corners": POSES[photo["role"]], "support_percent": 100}
            for photo in photos[1:]
        },
    }


def load_training(store):
    existing = store.seeded_tiles(DEMO_KEY)
    if existing:
        return existing, False
    rows = [training_specimen(store, specimen) for specimen in SERIES]
    return store.seed_tiles(DEMO_KEY, rows), True


@lru_cache(maxsize=1)
def preview_photo():
    specimen = SERIES[0]
    rendered = render_photo(bake_field(specimen), "diffuse", specimen)
    return png_bytes(rendered)
