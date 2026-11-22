import json
import sys
import urllib.request

BASE = "http://127.0.0.1:19970/api"


def call(path, body=None, method=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method or ("POST" if body is not None else "GET"))
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        return resp.status, (json.loads(raw) if raw and resp.headers.get_content_type() == "application/json" else raw)


status, created = call("/training/seam", {})
assert status == 201
tile = created["items"][0]
tile_id = tile["id"]
_, full = call(f"/specimens/{tile_id}")
regions = {r["name"]: r["id"] for r in full["regions"]}
photos = {p["role"]: p["id"] for p in full["photos"]}
print("loaded tile:", full["name"], "| photos:", list(photos), "| regions:", list(regions))

approval = {
    "tile_id": tile_id,
    "title": "Harbour walk approved seam",
    "family": "Quay east",
    "notes": "live server approval",
    "target": 0.18,
    "old_photo_id": photos["warm"],
    "old_card_region_id": regions["Warm-view grey card"],
    "old_edge_region_id": regions["Old-material seam edge"],
    "new_photo_id": photos["daylight"],
    "new_card_region_id": regions["Daylight-view grey card"],
    "new_edge_region_id": regions["New-fill seam edge"],
}
status, reference = call("/library/seam", approval)
assert status == 201
print("approved seam reference:", reference["id"], "| ends:", len(reference["ends"]))

status, library = call("/library")
assert any(item["id"] == reference["id"] and item["kind"] == "seam" for item in library["items"])

status, panel = call(f"/library/{reference['id']}/image")
assert status == 200 and isinstance(panel, bytes) and panel[:8] == b"\x89PNG\r\n\x1a\n"
print("reference panel PNG bytes:", len(panel))

measurement = {
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
status, result = call(f"/specimens/{tile_id}/results", measurement)
assert status == 201, status
for end in result["metrics"]["ends"]:
    print(" match end:", end["edge_key"], "|", end["status"], "| deltaE:", end["delta_e76"])
old = next(e for e in result["metrics"]["ends"] if e["edge_key"] == "old")
new = next(e for e in result["metrics"]["ends"] if e["edge_key"] == "new")
assert old["comparable"] is False and old["delta_e76"] is None and old["delta_lab"] is None
assert new["comparable"] is True and isinstance(new["delta_e76"], float)
assert result["metrics"]["comparable_ends"] == 1

status, overlay = call(f"/results/{result['id']}/image/overlay")
assert status == 200 and overlay[:8] == b"\x89PNG\r\n\x1a\n"
print("measurement overlay PNG bytes:", len(overlay))

# Measurement summary used by the specimen list must not crash.
status, results = call(f"/specimens/{tile_id}/results")
seam_summary = next(item for item in results["items"] if item["kind"] == "seam")
print("summary brief:", seam_summary["brief"])

# Evidence pinning must accept the seam measurement.
evidence = {
    "title": "Seam evidence packet",
    "result_ids": [result["id"]],
    "notes": "packet",
    "tags": ["seam"],
    "view": "overlay",
    "include_sources": False,
}
status, packet = call(f"/specimens/{tile_id}/evidence", evidence)
assert status == 201, status
print("evidence packet pinned:", packet["id"], "| sheets:", packet["sheet_count"])

# CSV export for the seam measurement.
status, csv_bytes = call(f"/results/{result['id']}/metrics.csv")
assert status == 200
print("CSV first line:", csv_bytes.decode("utf-8-sig").splitlines()[0])
print("LIVE SERVER CHECKS PASSED")
