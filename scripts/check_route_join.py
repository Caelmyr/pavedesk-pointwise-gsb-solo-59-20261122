"""End-to-end checks for joined routes.

Run from the repository root:  python3 scripts/check_route_join.py
Builds two overlapping synthetic frames in a temporary directory, registers the
shared joint, joins the route and traces a line across the seam.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedesk.web import create_app  # noqa: E402


def world_frame(width=1600, height=420, seed=5):
    rng = np.random.default_rng(seed)
    y, x = np.indices((height, width))
    base = 0.34 + 0.04 * rng.normal(size=(height, width))
    stripe = HIGHLIGHT = np.exp(-(((y - height * 0.5) / (24 + 3 * np.sin(x * 0.03))) ** 2))
    crack = np.exp(-(((y - (height * 0.5 + 34 + 12 * np.sin(x * 0.02))) / 1.4) ** 2))
    image = np.stack([base] * 3, axis=2)
    image = image * (1 - stripe[:, :, None]) + stripe[:, :, None] * np.array([0.68, 0.58, 0.20])
    image = image * (1 - 0.75 * crack[:, :, None])
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        world = world_frame()
        left = world[:, :1100]
        right = world[:, 500:1600]
        Image.fromarray(left).save(root / "left.png")
        Image.fromarray(right).save(root / "right.png")
        app = create_app(str(root / "data"))
        client = app.test_client()
        tile = client.post("/api/specimens", json={
            "name": "Route check", "width_mm": 1600, "height_mm": 420,
        }).json
        tile_id = tile["id"]

        def upload(path, version):
            with open(path, "rb") as stream:
                response = client.post(
                    f"/api/specimens/{tile_id}/photos",
                    data={"photo": (stream, path.name), "version": str(version), "role": "diffuse", "note": ""},
                    content_type="multipart/form-data",
                )
            assert response.status_code < 400, response.json
            return response.json

        state = upload(root / "left.png", 1)
        state = upload(root / "right.png", state["version"])
        head_id, tail_id = [photo["id"] for photo in state["photos"]]
        # Shared joint: world x 500..1000 is left frame x .4545..0.909 and right frame x 0..0.4545.
        joint = [[0.4545, 0.0], [0.909, 0.0], [0.909, 1.0], [0.4545, 1.0]]
        state = client.put(f"/api/specimens/{tile_id}/plane", json={
            "version": state["version"], "photo_id": head_id, "corners": joint,
            "reset_regions": False, "respect_scale": False,
        }).json
        registration = [[0.0, 0.0], [0.4545, 0.0], [0.4545, 1.0], [0.0, 1.0]]
        state = client.put(f"/api/specimens/{tile_id}/registrations/{tail_id}", json={
            "version": state["version"], "corners": registration,
        }).json

        response = client.post(f"/api/specimens/{tile_id}/joins", json={
            "version": state["version"], "name": "Joint route",
            "photo_ids": [head_id, tail_id], "overlap_source": "auto",
        })
        assert response.status_code == 201, response.json
        join = response.json
        assert join["overlap"]["single_source"], "the overlap must come from one frame"
        assert join["overlap"]["split_pixels"] == 0
        assert join["decision"]["mixed_pixels"] == 0
        covered_percent = join["coverage"]["covered_percent"]
        assert covered_percent > 90, covered_percent

        trace = client.post(f"/api/specimens/{tile_id}/metrology", json={
            "title": "Seam trace", "mode": "trace", "join_id": join["id"],
            "points": [[0.02, 0.5], [0.5, 0.5], [0.98, 0.5]],
        })
        assert trace.status_code == 201, trace.json
        record = trace.json
        provenance = record["shape"]["provenance"]
        codes = {row["source_code"] for row in provenance["sources"]}
        assert {"F1", "F2"}.issubset(codes), codes
        for piece in provenance["segments"]:
            for run in piece["runs"]:
                assert run["source_code"] is not None, "this trace should stay on photographed ground"
        # Any point on the line answers with a frame or an honest gap.
        point = client.get(f"/api/metrology/{record['id']}/source-at?x=0.5&y=0.5").json
        assert point["photographed"] and point["source_code"] in {"F1", "F2"}
        assert client.get(f"/api/metrology/{record['id']}/provenance.csv").status_code == 200
        # Every point answers honestly: a frame code, or an explicit "not photographed".
        for x, y in ((0.0, 0.0), (0.5, 0.5), (0.99, 0.99)):
            answer = client.get(f"/api/joins/{join['id']}/source-at?x={x}&y={y}").json
            assert (answer["source_code"] is None) == (not answer["photographed"])
        print(json.dumps({
            "overlap_percent": join["overlap"]["percent_of_route"],
            "supplier": join["overlap"]["supplier_code"],
            "retained": join["decision"]["retained"],
            "covered_percent": covered_percent,
            "trace_sources": sorted(codes),
            "point_at_middle": point["source_code"],
        }, indent=1))


if __name__ == "__main__":
    main()
