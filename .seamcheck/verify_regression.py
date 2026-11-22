import json
import sys
import urllib.request

BASE = "http://127.0.0.1:19970/api"


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        return resp.status, (json.loads(raw) if raw and resp.headers.get_content_type() == "application/json" else raw)


# Original four synthetic training specimens still load and measure.
status, training = call("/training", {})
assert status in (200, 201)
assert len(training["items"]) == 4
tile_id = training["items"][0]["id"]
_, full = call(f"/specimens/{tile_id}")
regions = {r["name"]: r["id"] for r in full["regions"]}
base_photo = next(p["id"] for p in full["photos"] if p["registration_state"] == "base")

# A classic color variation measurement still works end to end.
measurement = {
    "kind": "color",
    "photo_id": base_photo,
    "region_id": regions["Main survey"],
    "apply_neutral": True,
    "options": {"grid": 5},
}
status, result = call(f"/specimens/{tile_id}/results", measurement)
assert status == 201
assert result["kind"] == "color" and result["metrics"]["median_lab"]
print("legacy color measurement OK: median L* =", round(result["metrics"]["median_lab"][0], 2))

# A classic reference approval and reference match still works.
approval = {
    "tile_id": tile_id,
    "title": "Legacy color reference",
    "region_id": regions["Adjacent pavement"],
}
status, ref = call("/library", approval)
assert status == 201 and ref["kind"] == "color"
print("legacy reference kind:", ref["kind"])

match = {
    "kind": "reference",
    "photo_id": base_photo,
    "region_id": regions["Adjacent pavement"],
    "apply_neutral": True,
    "options": {"reference_id": ref["id"]},
}
status, result = call(f"/specimens/{tile_id}/results", match)
assert status == 201 and result["kind"] == "reference"
print("legacy reference match deltaE76:", result["metrics"]["delta_e76"])

# Configuration exposes all original tools plus the seam tool.
_, config = call("/configuration")
for name in ("color", "reference", "rim", "joint", "seam"):
    assert name in config["tools"], name
print("all expected tools present:", sorted(config["tools"]))
print("LEGACY REGRESSION CHECKS PASSED")
