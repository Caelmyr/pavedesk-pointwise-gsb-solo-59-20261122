import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

from pavedesk.web import create_app

workdir = Path(tempfile.mkdtemp(prefix="pavedesk-seam-api-"))
app = create_app(workdir / "data")
client = app.test_client()

# Load the seam training specimen.
resp = client.post("/api/training/seam", json={})
assert resp.status_code == 201, resp.data
tile = resp.get_json()["items"][0]
tile_id = tile["id"]

full = client.get(f"/api/specimens/{tile_id}").get_json()
regions = {r["name"]: r["id"] for r in full["regions"]}
photos = {p["role"]: p["id"] for p in full["photos"]}

approval = {
    "tile_id": tile_id,
    "title": "Harbour walk approved seam",
    "family": "Quay east",
    "notes": "approval via api",
    "target": 0.18,
    "old_photo_id": photos["warm"],
    "old_card_region_id": regions["Warm-view grey card"],
    "old_edge_region_id": regions["Old-material seam edge"],
    "new_photo_id": photos["daylight"],
    "new_card_region_id": regions["Daylight-view grey card"],
    "new_edge_region_id": regions["New-fill seam edge"],
}
resp = client.post("/api/library/seam", json=approval)
assert resp.status_code == 201, resp.data
reference = resp.get_json()
print("approved reference kind:", reference["kind"], "ends:", len(reference["ends"]))

# Library listing contains the seam reference.
lib = client.get("/api/library").get_json()["items"]
assert any(item["id"] == reference["id"] and item["kind"] == "seam" for item in lib)

# Reference image endpoint serves the panel.
resp = client.get(f"/api/library/{reference['id']}/image")
assert resp.status_code == 200 and resp.mimetype == "image/png"
print("panel image bytes:", len(resp.data))

# Run the seam reference match measurement.
measurement_payload = {
    "kind": "seam",
    "options": {
        "reference_id": reference["id"],
        "target": 0.18,
        "old_photo_id": photos["warm"],
        "old_card_region_id": regions["Warm-view grey card"],
        "old_edge_region_id": regions["Old-material seam edge"],
        "new_photo_id": photos["daylight"],
        "new_card_region_id": regions["Daylight-view grey card"],
        "new_edge_region_id": regions["New-fill seam edge"],
    },
}
resp = client.post(f"/api/specimens/{tile_id}/results", json=measurement_payload)
assert resp.status_code == 201, resp.data
result = resp.get_json()
print("result kind:", result["kind"], "| comparable ends:", result["metrics"]["comparable_ends"], "/", result["metrics"]["total_ends"])
for end in result["metrics"]["ends"]:
    print(" ", end["edge_key"], end["status"], "deltaE=", end["delta_e76"])
old = next(e for e in result["metrics"]["ends"] if e["edge_key"] == "old")
new = next(e for e in result["metrics"]["ends"] if e["edge_key"] == "new")
assert old["comparable"] is False and old["delta_e76"] is None
assert new["comparable"] is True and new["delta_e76"] == 0.0

# Overlay image exists.
resp = client.get(f"/api/results/{result['id']}/image/overlay")
assert resp.status_code == 200 and resp.mimetype == "image/png"
print("result overlay bytes:", len(resp.data))

# Measurement without a reference is rejected.
bad = {**measurement_payload, "options": {**measurement_payload["options"], "reference_id": None}}
resp = client.post(f"/api/specimens/{tile_id}/results", json=bad)
assert resp.status_code == 422, resp.data
print("no-reference rejected:", resp.get_json()["error"])

# A normal color reference cannot be used as a seam reference.
color_ref = client.post(
    "/api/library",
    json={
        "tile_id": tile_id,
        "title": "plain color ref",
        "region_id": regions["Old-material seam edge"],
    },
)
assert color_ref.status_code == 201, color_ref.data
wrong_kind = {
    **measurement_payload,
    "options": {**measurement_payload["options"], "reference_id": color_ref.get_json()["id"]},
}
resp = client.post(f"/api/specimens/{tile_id}/results", json=wrong_kind)
assert resp.status_code == 422
print("wrong reference kind rejected:", resp.get_json()["error"])

# Cached second run returns the same id.
resp2 = client.post(f"/api/specimens/{tile_id}/results", json=measurement_payload)
assert resp2.get_json()["id"] == result["id"]
print("cache hit OK")
print("ALL API CHECKS PASSED")
