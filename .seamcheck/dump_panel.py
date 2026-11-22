import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

from pavedesk.seam import approve_seam_reference, build_seam_calibration, seam_panel
from pavedesk.seam_demo import load_seam_training
from pavedesk.store import Store

workdir = Path(tempfile.mkdtemp(prefix="pavedesk-seam-img-"))
store = Store(workdir / "data")
tiles, _ = load_seam_training(store)
tile = store.tile(tiles[0]["id"])
regions = {r["name"]: r for r in tile["regions"]}
photos = {p["role"]: p for p in tile["photos"]}

payload = {
    "target": 0.18,
    "old_photo_id": photos["warm"]["id"],
    "old_card_region_id": regions["Warm-view grey card"]["id"],
    "old_edge_region_id": regions["Old-material seam edge"]["id"],
    "new_photo_id": photos["daylight"]["id"],
    "new_card_region_id": regions["Daylight-view grey card"]["id"],
    "new_edge_region_id": regions["New-fill seam edge"]["id"],
}
reference = approve_seam_reference(
    store, tile, {**payload, "title": "Harbour walk repair seam", "family": "Quay east", "notes": ""}
)
panel_bytes = store.blobs.get(reference["annotated_digest"])
out = Path("/workspace/.seamcheck/seam_approval_panel.png")
out.write_bytes(panel_bytes)
print("wrote", out, len(panel_bytes), "bytes")

# Also dump the two calibrated source frames separately for inspection.
target, assembled = build_seam_calibration(store, tile, payload)
for item in assembled:
    metrics2, panel2, _ = None, None, None
panel = seam_panel(assembled, "Calibration check", "shared target 0.180")
panel.save("/workspace/.seamcheck/seam_calibration_check.png")
print("wrote calibration check panel")
