import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import numpy as np

from pavedesk.seam_demo import (
    DAYLIGHT,
    WARM_LIGHT,
    bake_seam_field,
    render_seam_photo,
    seam_regions,
    VIEWS,
)
from pavedesk.seam import build_seam_calibration
from pavedesk.store import Store, make_id
from pavedesk.photos import png_bytes, store_capture
from pavedesk.geometry import canvas_dimensions


def make_tile(store, warm_light, daylight_light, name):
    surface, _ = bake_seam_field()
    photos = []
    for view in VIEWS:
        light = warm_light if view.role == "warm" else daylight_light
        custom = type(view)(view.role, light, view.label)
        image = render_seam_photo(surface, custom)
        photo = store_capture(
            store.blobs, png_bytes(image), view.role,
            f"{name}-{view.role}.png", "synthetic", make_id(), synthetic=True,
        )
        photos.append(photo)
    from pavedesk.seam_demo import VIEW_CORNERS
    base = photos[0]
    size = canvas_dimensions(base, VIEW_CORNERS["warm"], 2200, 1700)
    return {
        "name": name, "batch": "q", "body": "concrete", "glaze": "fill", "firing": "v",
        "notes": "", "width_mm": 2200, "height_mm": 1700, "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": VIEW_CORNERS["warm"], "size": list(size)},
        "regions": seam_regions(),
        "registrations": {
            p["id"]: {"base_photo_id": base["id"], "corners": VIEW_CORNERS[p["role"]], "support_percent": 100}
            for p in photos[1:]
        },
    }


def payload_for(tile):
    regions = {r["name"]: r["id"] for r in tile["regions"]}
    photos = {p["role"]: p["id"] for p in tile["photos"]}
    return {
        "target": 0.18,
        "old_photo_id": photos["warm"],
        "old_card_region_id": regions["Warm-view grey card"],
        "old_edge_region_id": regions["Old-material seam edge"],
        "new_photo_id": photos["daylight"],
        "new_card_region_id": regions["Daylight-view grey card"],
        "new_edge_region_id": regions["New-fill seam edge"],
    }


def edge_colors(assembled):
    out = {}
    for item in assembled:
        out[item["end"].key] = np.asarray(item["edge"]["fingerprint"]["median_lab"])
    return out


def main():
    workdir = Path(tempfile.mkdtemp(prefix="pavedesk-sem-"))
    store = Store(workdir / "data")

    # Build tiles through the public create path.
    row_a = make_tile(store, WARM_LIGHT, DAYLIGHT, "specimen-a")
    tile_a = store.create_tile(row_a)
    _, assembled_a = build_seam_calibration(store, tile_a, payload_for(tile_a))
    colors_a = edge_colors(assembled_a)
    gains_a = {item["end"].key: list(item["card"]["gains"]) for item in assembled_a}

    # Same physical surface, but make the warm lamp even warmer. The daylight
    # capture is unchanged, so its calibrated edge fingerprint must be identical.
    hotter_warm = np.array([1.42, 1.0, 0.62], dtype=np.float32)
    row_b = make_tile(store, hotter_warm, DAYLIGHT, "specimen-b")
    tile_b = store.create_tile(row_b)
    _, assembled_b = build_seam_calibration(store, tile_b, payload_for(tile_b))
    colors_b = edge_colors(assembled_b)
    gains_b = {item["end"].key: list(item["card"]["gains"]) for item in assembled_b}

    print("warm gains A:", np.round(gains_a["old"], 4), " B:", np.round(gains_b["old"], 4))
    print("daylight gains A:", np.round(gains_a["new"], 4), " B:", np.round(gains_b["new"], 4))

    # Daylight end is untouched across the two specimens: gains and edge match.
    assert np.allclose(gains_a["new"], gains_b["new"], atol=1e-6), "the other end's gains must not change"
    assert np.allclose(colors_a["new"], colors_b["new"], atol=1e-6), "the other end's edge color must not change"

    # The warm end's own gains respond to its own light only.
    assert not np.allclose(gains_a["old"], gains_b["old"], atol=1e-4)

    # Within one calibration, old and new edge colors are distinct: the shared
    # target does not pull the two materials toward the same color.
    separation = float(np.linalg.norm(colors_a["old"] - colors_a["new"]))
    print("old vs new edge Lab separation after shared calibration:", round(separation, 3))
    assert separation > 5.0, "the genuine material color change must survive calibration"

    # The two card records share exactly one neutral target, never two targets.
    targets = {item["card"]["target"] for item in assembled_a}
    assert targets == {0.18}

    print("SHARED-CALIBRATION SEMANTICS PASSED")


if __name__ == "__main__":
    main()
