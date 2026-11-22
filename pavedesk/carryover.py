"""Carry a defect observation number across repeat visits of one section.

When the same road section is measured again with the same "Visible
defects" settings, an observation that appears again can keep the number
it had before.  This module ranks earlier observations by how much their
stored outline overlaps the new observation's outline.  It only reads
records; applying a chosen continuation is a separate, confirmed step.
"""
from PIL import Image, ImageDraw
import numpy as np

from .model import InputError, MissingError


def find_observation(record, key):
    for observation in record["metrics"]["observations"]:
        if observation["key"] == key:
            return observation
    raise MissingError(f"Observation {key} is not in measurement {record['id']}.")


def hull_points(points):
    """Monotone-chain convex hull of (x, y) points."""
    ordered = sorted(set(points))
    if len(ordered) <= 2:
        return ordered

    def turn(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def outline_mask(observation, extent):
    """Rasterize one stored observation outline in the registered plane.

    Stored outlines are edge points sorted by angle around the centroid.
    For thin elongated marks that ordering zig-zags across the mark, so a
    direct polygon fill cancels itself; the convex hull of the same edge
    points is the stable footprint for near-convex spots and marks.
    """
    width, height = extent
    points = [(x * width - 0.5, y * height - 0.5) for x, y in observation["outline"]]
    if len(points) < 3:
        left, top, right, bottom = observation["bounds"]
        points = [
            (left * width - 0.5, top * height - 0.5),
            (right * width - 0.5, top * height - 0.5),
            (right * width - 0.5, bottom * height - 0.5),
            (left * width - 0.5, bottom * height - 0.5),
        ]
    grid = Image.new("1", (width, height))
    ImageDraw.Draw(grid).polygon(hull_points(points), fill=1)
    return np.asarray(grid, dtype=bool)


def decision_snapshot(record, key):
    mark = record["decisions"].get(key)
    if mark is None:
        return {"decision": "pending", "notes": ""}
    return {"decision": mark["decision"], "notes": mark.get("notes", "")}


def overlap_scores(previous, current, current_key):
    """Rank previous observations by outline overlap with one current observation.

    Both measurements must share the registered plane size, which aligned
    repeat visits of one section do.  Returns rows sorted by IoU, best
    first, restricted to observations that share at least one pixel.
    """
    extent = tuple(current["support"]["registered_image_extent"])
    if tuple(previous["support"]["registered_image_extent"]) != extent:
        raise InputError(
            "The two measurements use different registered plane sizes; "
            "align both visits to the same plane before carrying numbers."
        )
    merged = find_observation(current, current_key)
    merged_mask = outline_mask(merged, extent)
    merged_area = int(merged_mask.sum())
    rows = []
    for observation in previous["metrics"]["observations"]:
        mask = outline_mask(observation, extent)
        area = int(mask.sum())
        shared = int(np.count_nonzero(mask & merged_mask))
        if not shared:
            continue
        union = area + merged_area - shared
        rows.append({
            "key": observation["key"],
            "type": observation["type"],
            **decision_snapshot(previous, observation["key"]),
            "area_pixels": area,
            "shared_pixels": shared,
            "iou": round(shared / union, 4),
            "share_of_new": round(shared / merged_area, 4) if merged_area else 0.0,
            "share_of_old": round(shared / area, 4) if area else 0.0,
        })
    rows.sort(key=lambda row: (-row["iou"], row["key"]))
    return rows
