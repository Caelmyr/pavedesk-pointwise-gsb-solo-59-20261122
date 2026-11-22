"""End-to-end checks for the repair seam color match feature.

Run with: python3 -I verify_seam.py  (paths are injected below).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import numpy as np

from pavedesk.model import InputError
from pavedesk.seam import (
    CARD_CHANNEL_TOLERANCE,
    approve_seam_reference,
    build_seam_calibration,
    seam_match,
)
from pavedesk.seam_demo import load_seam_training
from pavedesk.store import Store


def main():
    workdir = Path(tempfile.mkdtemp(prefix="pavedesk-seam-"))
    store = Store(workdir / "data")
    tiles, created = load_seam_training(store)
    assert created, "seam demo should be created once"
    tile = store.tile(tiles[0]["id"])

    region_by_name = {region["name"]: region for region in tile["regions"]}
    photo_by_role = {photo["role"]: photo for photo in tile["photos"]}
    warm = photo_by_role["warm"]
    daylight = photo_by_role["daylight"]
    old_edge = region_by_name["Old-material seam edge"]
    new_edge = region_by_name["New-fill seam edge"]
    warm_card = region_by_name["Warm-view grey card"]
    daylight_card = region_by_name["Daylight-view grey card"]

    payload = {
        "target": 0.18,
        "old_photo_id": warm["id"],
        "old_card_region_id": warm_card["id"],
        "old_edge_region_id": old_edge["id"],
        "new_photo_id": daylight["id"],
        "new_card_region_id": daylight_card["id"],
        "new_edge_region_id": new_edge["id"],
    }

    target, assembled = build_seam_calibration(store, tile, payload)
    by_key = {item["end"].key: item for item in assembled}

    print("== shared calibration ==")
    for key, item in by_key.items():
        check = item["card"]["card_neutrality"]
        print(
            key,
            "gains", np.round(item["card"]["gains"], 4),
            "R/G%", check["red_deviation_percent"],
            "B/G%", check["blue_deviation_percent"],
            "comparable", check["comparable"],
        )

    # 1. Same neutral target drives both ends.
    assert len({item["card"]["target"] for item in assembled}) == 1
    # 2. Warm (old) end fails the 5% check; daylight (new) end passes.
    old_check = by_key["old"]["card"]["card_neutrality"]
    new_check = by_key["new"]["card"]["card_neutrality"]
    assert not old_check["comparable"], "warm lamp end must be non-comparable"
    assert max(old_check["red_deviation_percent"], old_check["blue_deviation_percent"]) > 5.0
    assert new_check["comparable"], "daylight end must be comparable"
    assert max(new_check["red_deviation_percent"], new_check["blue_deviation_percent"]) <= 5.0
    # 3. The gains are per-photo but the card is forced to one target.
    assert by_key["old"]["card"]["target"] == by_key["new"]["card"]["target"] == 0.18
    # 4. Edge fingerprints differ — the genuine color step is preserved.
    old_lab = np.asarray(by_key["old"]["edge"]["fingerprint"]["median_lab"])
    new_lab = np.asarray(by_key["new"]["edge"]["fingerprint"]["median_lab"])
    edge_step = float(np.linalg.norm(old_lab - new_lab))
    print("edge median delta E (visible color change):", round(edge_step, 3))
    assert edge_step > 1.0

    # 5. Approve the reference; panel image stored.
    reference = approve_seam_reference(
        store,
        tile,
        {
            **payload,
            "title": "Harbour walk repair seam",
            "family": "Quay east",
            "notes": "approval test",
        },
    )
    assert reference["kind"] == "seam"
    assert reference["annotated_digest"]
    stored_ref = store.reference(reference["id"])
    assert len(stored_ref["ends"]) == 2
    print("reference approved:", reference["id"], "panel digest", reference["annotated_digest"][:12])

    # 6. Reference match: old end non-comparable -> no delta E; new end gets one.
    metrics, panel, assembled_again = seam_match(store, tile, stored_ref, payload)
    end_metrics = {entry["edge_key"]: entry for entry in metrics["ends"]}
    print("== reference match ==")
    for entry in metrics["ends"]:
        print(entry["edge_key"], entry["status"], "deltaE", entry["delta_e76"])
    assert end_metrics["old"]["comparable"] is False
    assert end_metrics["old"]["delta_e76"] is None
    assert end_metrics["old"]["delta_lab"] is None
    assert end_metrics["new"]["comparable"] is True
    assert isinstance(end_metrics["new"]["delta_e76"], float)
    assert metrics["comparable_ends"] == 1

    # 7. A second run with a neutral daylight-like warm card still carries the
    #    stored failure: build a reference whose old end was non-comparable and
    #    confirm the reference itself keeps that flag (not hidden as a delta E).
    ref_old = next(end for end in stored_ref["ends"] if end["edge_key"] == "old")
    assert ref_old["card"]["card_neutrality"]["comparable"] is False
    print("reference old-end status preserved:", ref_old["card"]["card_neutrality"]["status"])

    # 8. Wrong shared target is rejected.
    bad = {**payload, "target": 0.22}
    try:
        seam_match(store, tile, stored_ref, bad)
    except InputError as error:
        print("target mismatch rejected:", error)
    else:
        raise AssertionError("mismatched neutral target should be rejected")

    # 9. Reusing the same photo for both ends is rejected.
    same = {**payload, "new_photo_id": warm["id"]}
    try:
        build_seam_calibration(store, tile, same)
    except InputError as error:
        print("same-photo rejected:", error)
    else:
        raise AssertionError("two ends must use different captures")

    print("tolerance constant:", CARD_CHANNEL_TOLERANCE)
    print("ALL SEAM CHECKS PASSED")


if __name__ == "__main__":
    main()
