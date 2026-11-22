"""Local distance calibration for one span with conflicting recorded distances.

The specimen plane carries one global millimetre scale.  A local calibration
does not change it.  Instead it anchors two points on the plane (the two old
stakes), records both distances copied for that span (the old inspection
ledger and the newly pasted tape) and derives a scale factor that is applied
only to traced lengths between the two anchors.  The reading that the chosen
resolution cannot honour is reported explicitly so the drawing and the saved
measurement can name the mismatch.
"""

import math

import numpy as np

from .model import (
    InputError,
    dictionary,
    enumeration,
    MissingError,
    numeric_pair,
    physical_scale,
    scalar,
    string,
)


LEDGER = "ledger"
TAPE = "tape"
BALANCED = "balanced"
RESOLUTIONS = (BALANCED, LEDGER, TAPE)
RESOLUTION_LABELS = {
    LEDGER: "Hold the old ledger distance",
    TAPE: "Follow the new tape reading",
    BALANCED: "Split the difference",
}
UNMATCHED_LABELS = {LEDGER: "Old ledger reading", TAPE: "New tape reading", "both": "Both readings"}

# Distances closer than this are treated as agreeing; the conflict is still
# reported, but no reading is called unmatched.
CONFLICT_TOLERANCE_MM = 0.5
MIN_STAKE_SPAN_PX = 4.0


def calibration_fields(payload):
    data = dictionary(payload, "Local distance calibration")
    start = numeric_pair(data.get("start"), "Stake A", lower=0, upper=1)
    end = numeric_pair(data.get("end"), "Stake B", lower=0, upper=1)
    if start == end:
        raise InputError("Mark two different stake positions.")
    return {
        "title": string(data.get("title"), "Calibration title", maximum=90),
        "start": start,
        "end": end,
        "ledger_mm": scalar(data.get("ledger_mm"), "Ledger distance", lower=0.01, upper=100000),
        "tape_mm": scalar(data.get("tape_mm"), "Tape distance", lower=0.01, upper=100000),
        "ledger_source": string(
            data.get("ledger_source"), "Ledger note", maximum=200, required=False,
        ),
        "tape_source": string(
            data.get("tape_source"), "Tape note", maximum=200, required=False,
        ),
        "resolution": enumeration(
            data.get("resolution", BALANCED), RESOLUTIONS, "Calibration resolution",
        ),
    }


def stake_span(start, end, extent, scale):
    width, height = extent
    delta_x = (end[0] - start[0]) * width
    delta_y = (end[1] - start[1]) * height
    pixels = math.hypot(delta_x, delta_y)
    physical = math.hypot(delta_x * scale[0], delta_y * scale[1])
    return pixels, physical


def derive_calibration(record, extent, scale):
    """Attach the adopted span, local factor and residuals to a stored record."""
    if scale is None:
        raise InputError(
            "Enter the specimen width and height before using a local distance calibration."
        )
    pixels, plane_mm = stake_span(record["start"], record["end"], extent, scale)
    if pixels < MIN_STAKE_SPAN_PX:
        raise InputError("Mark two stakes at least four registered pixels apart.")
    if not math.isfinite(plane_mm) or plane_mm <= 0:
        raise InputError("The marked stakes do not span a usable plane distance.")
    ledger = float(record["ledger_mm"])
    tape = float(record["tape_mm"])
    resolution = record["resolution"]
    conflict = abs(tape - ledger)
    if resolution == LEDGER:
        adopted = ledger
    elif resolution == TAPE:
        adopted = tape
    else:
        adopted = (ledger + tape) / 2
    factor = adopted / plane_mm
    ledger_residual = abs(adopted - ledger)
    tape_residual = abs(adopted - tape)
    if conflict <= CONFLICT_TOLERANCE_MM:
        unmatched = None
    elif resolution == LEDGER:
        unmatched = TAPE
    elif resolution == TAPE:
        unmatched = LEDGER
    else:
        unmatched = "both"
    unmatched_mm = {None: 0.0, LEDGER: ledger_residual, TAPE: tape_residual}.get(
        unmatched, max(ledger_residual, tape_residual),
    )
    return {
        **record,
        "plane_span_px": round(pixels, 5),
        "plane_span_mm": round(plane_mm, 5),
        "global_mm_per_px": round(plane_mm / pixels, 8),
        "adopted_span_mm": round(adopted, 5),
        "local_factor": round(factor, 10),
        "local_mm_per_px": round(adopted / pixels, 8),
        "conflict_mm": round(conflict, 5),
        "ledger_residual_mm": round(ledger_residual, 5),
        "tape_residual_mm": round(tape_residual, 5),
        "unmatched": unmatched,
        "unmatched_mm": round(unmatched_mm, 5),
        "tolerance_mm": CONFLICT_TOLERANCE_MM,
        "invalid": False,
    }


def present_calibrations(tile):
    plane = tile.get("plane")
    records = tile.get("calibrations", [])
    if not plane:
        return []
    extent = tuple(plane["size"])
    scale = physical_scale(tile, *extent)
    presented = []
    for record in records:
        try:
            presented.append(derive_calibration(record, extent, scale))
        except InputError as error:
            presented.append({**record, "invalid": True, "invalid_reason": str(error)})
    return presented


def find_calibration(tile, identifier):
    record = next(
        (item for item in tile.get("calibrations", []) if item["id"] == identifier),
        None,
    )
    if record is None:
        raise MissingError("The chosen local distance calibration is no longer on this specimen.")
    return record


def applied_calibration(tile, frame, identifier):
    record = find_calibration(tile, identifier)
    return derive_calibration(record, frame.extent, frame.scale)


def station_of(point, extent, calibration):
    width, height = extent
    anchors = np.asarray([calibration["start"], calibration["end"]], dtype=np.float64)
    anchors = anchors * np.array([width, height], dtype=np.float64)
    axis = anchors[1] - anchors[0]
    denominator = float(axis @ axis)
    if denominator <= 0:
        raise InputError("The local distance calibration has no usable stake axis.")
    position = np.asarray(point, dtype=np.float64) * np.array([width, height], dtype=np.float64)
    return float((position - anchors[0]) @ axis / denominator)


def expand_for_stakes(points, extent, calibration):
    """Insert a vertex wherever the trace enters or leaves the stake span.

    Each resulting piece then lies wholly on one side of both stakes, so a
    single distance basis can be named for every drawn segment.
    """
    expanded = [list(points[0])]
    for first, second in zip(points[:-1], points[1:]):
        start = station_of(first, extent, calibration)
        end = station_of(second, extent, calibration)
        crossings = []
        for boundary in (0.0, 1.0):
            if (start - boundary) * (end - boundary) < 0:
                crossings.append((boundary - start) / (end - start))
        crossings.sort()
        anchor_a = np.asarray(first, dtype=np.float64)
        anchor_b = np.asarray(second, dtype=np.float64)
        for fraction in crossings:
            if fraction <= 1e-9 or fraction >= 1 - 1e-9:
                continue
            point = anchor_a + fraction * (anchor_b - anchor_a)
            expanded.append([
                round(min(1.0, max(0.0, float(point[0]))), 9),
                round(min(1.0, max(0.0, float(point[1]))), 9),
            ])
        expanded.append(list(second))
    return expanded


def local_fractions(points, extent, calibration):
    """Fraction of each trace segment whose stake-axis station lies in [0, 1]."""
    fractions = []
    for first, second in zip(points[:-1], points[1:]):
        start = station_of(first, extent, calibration)
        end = station_of(second, extent, calibration)
        low, high = sorted((start, end))
        if high - low <= 1e-12:
            fractions.append(1.0 if 0.0 <= low <= 1.0 else 0.0)
        else:
            inside = min(high, 1.0) - max(low, 0.0)
            fractions.append(min(1.0, max(0.0, inside / (high - low))))
    return fractions


def segment_basis(first, second, extent, calibration):
    """Name one distance basis for a drawn piece.

    A piece wholly before stake A or wholly past stake B uses the plane
    scale; a piece whose stations both fall between the stakes uses the local
    tape scale.
    """
    start = station_of(first, extent, calibration)
    end = station_of(second, extent, calibration)
    low, high = sorted((start, end))
    tolerance = 1e-7
    if high <= tolerance or low >= 1.0 - tolerance:
        return PLANE
    if low >= -tolerance and high <= 1.0 + tolerance:
        return LOCAL
    return MIXED


PLANE = "plane"
LOCAL = "local"
MIXED = "mixed"
