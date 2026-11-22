from .calibration import calibration_fields, derive_calibration, present_calibrations
from .color import robust_neutral
from .geometry import (
    DEFAULT_CORNERS,
    canvas_dimensions,
    corners_parameter,
    homography,
    region_domain,
    specimen_frame,
    validate_polygon,
)
from .model import (
    ConflictError,
    InputError,
    MAX_CALIBRATIONS,
    MAX_REGIONS,
    MAX_TILE_IMAGES,
    ROLE_LABELS,
    dictionary,
    enumeration,
    flag,
    identity,
    match_slot,
    model_version,
    physical_scale,
    region_fields,
    scalar,
    string,
    tile_fields,
)
from .photos import diagnostic_flags, store_capture
from .store import make_id


def describe_photo(photo, plane, registrations):
    info = {**photo}
    info["role_label"] = ROLE_LABELS[photo["role"]]
    info["diagnostics"] = diagnostic_flags(photo)
    is_base = plane is not None and plane["photo_id"] == photo["id"]
    info["registration_state"] = "base" if is_base else "registered" if photo["id"] in registrations else "unregistered"
    return info


def summarize_tile(tile):
    base = tile.get("plane")
    photos = tile["photos"]
    cover = next((item for item in photos if base and item["id"] == base["photo_id"]), None)
    if cover is None and photos:
        cover = photos[0]
    summary = {
        key: tile[key] for key in (
            "id", "name", "batch", "body", "glaze", "firing", "width_mm", "height_mm", "version", "synthetic",
        )
    }
    return {
        **summary,
        "photo_count": len(photos),
        "region_count": len(tile["regions"]),
        "calibration_count": len(tile.get("calibrations", [])),
        "cover_url": f"/api/specimens/{tile['id']}/photos/{cover['id']}/thumbnail" if cover else None,
        "registered_count": int(bool(base)) + len(tile["registrations"]),
        "updated_at": tile["updated_at"],
    }


def present_tile(tile):
    registrations = tile["registrations"]
    photos = []
    for record in tile["photos"]:
        data = describe_photo(record, tile["plane"], registrations)
        data["preview_url"] = f"/api/specimens/{tile['id']}/photos/{record['id']}/preview"
        data["thumbnail_url"] = f"/api/specimens/{tile['id']}/photos/{record['id']}/thumbnail"
        data["original_url"] = f"/api/specimens/{tile['id']}/photos/{record['id']}/original"
        if data["registration_state"] != "unregistered":
            data["registered_url"] = f"/api/specimens/{tile['id']}/viewer/{record['id']}?v={tile['version']}"
        photos.append(data)
    native = tile.get("plane")
    return {
        **tile,
        "photos": photos,
        "calibrations": present_calibrations(tile),
        "canvas_size": native["size"] if native else None,
    }


def edit_specimen(store, identifier, payload):
    version = model_version(payload)

    def change(tile):
        updated = tile_fields(payload, tile)
        tile.update(updated)

    return store.edit_tile(identifier, version, change, "metadata", "Updated specimen details")


def import_photo(store, tile_id, expected, binary, filename, role, note):
    validated_role = enumeration(role, ROLE_LABELS, "Lighting role")
    validated_note = string(note, "Capture notes", maximum=500, required=False)
    current = store.tile(tile_id)
    store.require_version(current, expected)
    if len(current["photos"]) >= MAX_TILE_IMAGES:
        raise InputError(f"A specimen supports at most {MAX_TILE_IMAGES} photographs.")
    record = store_capture(store.blobs, binary, validated_role, filename, validated_note, make_id())
    if any(photo["source_digest"] == record["source_digest"] and photo["role"] == validated_role for photo in current["photos"]):
        raise InputError("This image is already on the specimen with the same lighting role.")

    def change(tile):
        tile["photos"].append(record)
        if tile["plane"] is None and len(tile["photos"]) == 1:
            size = canvas_dimensions(record, DEFAULT_CORNERS)
            tile["plane"] = {"photo_id": record["id"], "corners": DEFAULT_CORNERS, "size": list(size)}

    return store.edit_tile(tile_id, expected, change, "capture", f"Imported {ROLE_LABELS[validated_role]} · {record['name']}")


def edit_photo(store, tile_id, capture_id, payload):
    expected = model_version(payload)
    capture_id = identity(capture_id, "Photo ID")

    def change(tile):
        source = match_slot(tile["photos"], capture_id, "photo")
        revised = enumeration(payload.get("role", source["role"]), ROLE_LABELS, "Lighting role")
        if any(
            item["id"] != capture_id
            and item["source_digest"] == source["source_digest"]
            and item["role"] == revised
            for item in tile["photos"]
        ):
            raise InputError("The same image already has this role on the specimen.")
        source["role"] = revised
        source["note"] = string(
            payload.get("note", source["note"]),
            "Capture notes",
            maximum=500,
            required=False,
        )

    return store.edit_tile(tile_id, expected, change, "capture", "Updated photo notes and lighting label")


def remove_photo(store, tile_id, capture_id, payload):
    expected = model_version(payload)
    capture_id = identity(capture_id, "Photo ID")
    reset = flag(payload.get("reset_plane", False), "Reset specimen plane")

    def change(tile):
        photo = match_slot(tile["photos"], capture_id, "photo")
        if tile["plane"] and tile["plane"]["photo_id"] == capture_id:
            if (tile["registrations"] or tile["regions"]) and not reset:
                raise ConflictError("Removing the base photo requires resetting its current sampling plane.")
            tile["plane"] = None
            tile["calibration"] = None
            tile["calibrations"] = []
            tile["registrations"] = {}
            tile["regions"] = []
        tile["registrations"].pop(capture_id, None)
        tile["photos"].remove(photo)

    return store.edit_tile(tile_id, expected, change, "capture", "Removed a photo from the live specimen")


def preview_transform(store, tile_id, photo_id, payload, mode):
    current = store.tile(tile_id)
    photo_id = identity(photo_id, "Photo ID")
    marked = corners_parameter(payload.get("corners"))
    photo = match_slot(current["photos"], photo_id, "photo")
    if mode == "base":
        width = current["width_mm"] if payload.get("respect_scale", True) else None
        height = current["height_mm"] if payload.get("respect_scale", True) else None
        dims = canvas_dimensions(photo, marked, width, height)
        updated = {**current, "plane": {"photo_id": photo_id, "corners": marked, "size": list(dims)}}
    else:
        if current["plane"] is None or current["plane"]["photo_id"] == photo_id:
            raise InputError("Set another photo as the base before checking registration.")
        change = {"base_photo_id": current["plane"]["photo_id"], "corners": marked}
        updated = {**current, "registrations": {**current["registrations"], photo_id: change}}
    return specimen_frame(store.blobs, updated, photo_id)


def save_plane(store, tile_id, payload):
    dictionary(payload, "Specimen plane")
    expected = model_version(payload)
    photo_id = identity(payload.get("photo_id"), "Base photo ID")
    corners = corners_parameter(payload.get("corners"))
    reset = flag(payload.get("reset_regions", False), "Reset specimen plane")
    respect_scale = flag(payload.get("respect_scale", True), "Respect specimen dimensions")
    homography(DEFAULT_CORNERS, corners)

    def change(tile):
        photo = match_slot(tile["photos"], photo_id, "photo")
        size = canvas_dimensions(
            photo, corners,
            tile["width_mm"] if respect_scale else None,
            tile["height_mm"] if respect_scale else None,
        )
        updated = {"photo_id": photo_id, "corners": corners, "size": list(size)}
        if tile["plane"] == updated:
            return
        if (tile["registrations"] or tile["regions"]) and not reset:
            raise ConflictError("Changing the specimen plane requires confirming reset of its live regions and registrations.")
        tile["plane"] = updated
        tile["registrations"].clear()
        tile["regions"].clear()
        tile["calibration"] = None
        tile["calibrations"] = []

    return store.edit_tile(tile_id, expected, change, "plane", "Saved the photographed tile plane")


def save_registration(store, identifier, photo_id, payload):
    expected = model_version(payload)
    photo_id = identity(photo_id, "Photo ID")
    corners = corners_parameter(payload.get("corners"))
    homography(DEFAULT_CORNERS, corners)
    current = store.tile(identifier)
    store.require_version(current, expected)
    frame = preview_transform(store, identifier, photo_id, payload, "registration")
    supported = float(frame.valid.mean())
    if supported < 0.15:
        raise InputError("Mark a region with at least 15% visible support.")

    def change(tile):
        tile["registrations"][photo_id] = {
            "base_photo_id": tile["plane"]["photo_id"],
            "corners": corners,
            "support_percent": round(100 * supported, 2),
        }

    return store.edit_tile(identifier, expected, change, "registration", "Registered a cross-light capture")


def save_region(store, identifier, payload, region_id=None):
    expected = model_version(payload)
    region = region_fields(payload)
    validate_polygon(region["vertices"])
    chosen = identity(region_id, "Region ID") if region_id is not None else make_id()
    region["id"] = chosen

    def change(tile):
        if not tile["plane"]:
            raise InputError("Set a base photo before defining a sampling region.")
        if region_id is None:
            if len(tile["regions"]) >= MAX_REGIONS:
                raise InputError(f"A specimen supports at most {MAX_REGIONS} regions.")
            tile["regions"].append(region)
        else:
            previous = match_slot(tile["regions"], chosen, "sampling region")
            index = tile["regions"].index(previous)
            tile["regions"][index] = region
        if tile["calibration"]:
            reasons = {region["purpose"], previous["purpose"]} if region_id is not None else {region["purpose"]}
            if tile["calibration"]["region_id"] == chosen or "ignore" in reasons:
                tile["calibration"] = None

    return store.edit_tile(identifier, expected, change, "region", f"Saved sampling region {region['name']}")


def remove_region(store, identifier, region_id, payload):
    expected = model_version(payload)
    chosen = identity(region_id, "Region ID")

    def change(tile):
        region = match_slot(tile["regions"], chosen, "sampling region")
        tile["regions"].remove(region)
        if tile["calibration"] and (tile["calibration"]["region_id"] == chosen or region["purpose"] == "ignore"):
            tile["calibration"] = None

    return store.edit_tile(identifier, expected, change, "region", "Removed a live sampling region")


def save_calibration(store, identifier, payload, calibration_id=None):
    expected = model_version(payload)
    fields = calibration_fields(payload)
    chosen = identity(calibration_id, "Calibration ID") if calibration_id is not None else make_id()
    fields["id"] = chosen

    def change(tile):
        plane = tile.get("plane")
        if plane is None:
            raise InputError("Set a base photo before marking old stakes.")
        if tile.get("width_mm") is None or tile.get("height_mm") is None:
            raise InputError(
                "Enter the specimen width and height before using a local distance calibration."
            )
        derive_calibration(fields, tuple(plane["size"]), physical_scale(tile, *plane["size"]))
        records = tile.setdefault("calibrations", [])
        if calibration_id is None:
            if len(records) >= MAX_CALIBRATIONS:
                raise InputError(f"A specimen supports at most {MAX_CALIBRATIONS} local distance calibrations.")
            records.append(fields)
        else:
            previous = match_slot(records, chosen, "local distance calibration")
            records[records.index(previous)] = fields

    verb = "Saved" if calibration_id is None else "Updated"
    return store.edit_tile(
        identifier, expected, change, "calibration",
        f"{verb} local distance calibration {fields['title']}",
    )


def remove_calibration(store, identifier, calibration_id, payload):
    expected = model_version(payload)
    chosen = identity(calibration_id, "Calibration ID")

    def change(tile):
        records = tile.setdefault("calibrations", [])
        record = match_slot(records, chosen, "local distance calibration")
        records.remove(record)

    return store.edit_tile(
        identifier, expected, change, "calibration", "Removed a local distance calibration",
    )


def calibrate(store, identifier, payload):
    expected = model_version(payload)
    region_id = identity(payload.get("region_id"), "Neutral card region")
    target = scalar(payload.get("target", 0.18), "Neutral reflectance target", lower=0.08, upper=0.8)
    before = store.tile(identifier)
    store.require_version(before, expected)
    selected = match_slot(before["regions"], region_id, "neutral card")
    if selected["purpose"] != "neutral":
        raise InputError("Select a region marked as a neutral color card.")
    frame = specimen_frame(store.blobs, before)
    area = region_domain(frame.extent, before)
    record = robust_neutral(frame, selected, area, target)
    record["plane_size"] = before["plane"]["size"]
    record["plane_corners"] = before["plane"]["corners"]
    record["region_vertices"] = selected["vertices"]

    def change(tile):
        tile["calibration"] = record

    return store.edit_tile(identifier, expected, change, "calibration", "Saved neutral-card image gains")


def clear_calibration(store, identifier, payload):
    expected = model_version(payload)

    def change(tile):
        tile["calibration"] = None

    return store.edit_tile(identifier, expected, change, "calibration", "Cleared neutral-card image gains")
