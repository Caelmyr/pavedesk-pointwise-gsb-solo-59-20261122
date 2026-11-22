import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:19970/api"


def call(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


req = urllib.request.Request(BASE + "/training/seam", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req) as training:
    data = json.loads(training.read())
tile_id = data["items"][0]["id"]
full = json.loads(urllib.request.urlopen(BASE + f"/specimens/{tile_id}").read())
regions = {r["name"]: r["id"] for r in full["regions"]}
photos = {p["role"]: p["id"] for p in full["photos"]}

base = {
    "target": 0.18,
    "old_photo_id": photos["warm"],
    "old_card_region_id": regions["Warm-view grey card"],
    "old_edge_region_id": regions["Old-material seam edge"],
    "new_photo_id": photos["daylight"],
    "new_card_region_id": regions["Daylight-view grey card"],
    "new_edge_region_id": regions["New-fill seam edge"],
}

# 1. Non-neutral region used as a card is rejected with 422.
bad_card = {**base, "old_card_region_id": regions["Old-material seam edge"]}
status, payload = call("/library/seam", {"tile_id": tile_id, "title": "bad card", **bad_card})
assert status == 422, (status, payload)
print("non-neutral card rejected:", payload["error"])

# 2. A malformed (too short / non-id) region id is rejected.
bad_id = {**base, "new_photo_id": "not-an-id"}
status, payload = call("/library/seam", {"tile_id": tile_id, "title": "bad id", **bad_id})
assert status == 422
print("bad id rejected:", payload["error"])

# 3. Missing required field is rejected.
missing = dict(base)
del missing["new_edge_region_id"]
status, payload = call("/library/seam", {"tile_id": tile_id, "title": "missing", **missing})
assert status == 422
print("missing field rejected:", payload["error"])

# 4. Out-of-range target is rejected.
bad_target = {**base, "target": 2.5}
status, payload = call("/library/seam", {"tile_id": tile_id, "title": "bad target", **bad_target})
assert status == 422
print("bad target rejected:", payload["error"])

print("ERROR-HANDLING CHECKS PASSED")
