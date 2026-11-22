# PaveDesk

PaveDesk is a local field-photo workbench for pavement and paved forecourts. It retains the original camera file, lets a surveyor register a measured surface and compare differently lit views, and saves the measurements with the geometry used to make them.

## Start

```sh
docker compose up --build --wait
```

Visit `http://127.0.0.1:19964`. The standard Compose service exposes a loopback address and keeps the original images, data index and reports in one named volume. Set `PAVEDESK_PORT` for another local port. `docker compose stop` keeps the volume. For native use install the pinned packages into a Python 3.11 environment and run `./start.sh`.

An existing image with these exact standard wheel versions can be used for the dependency stage when a package index is unavailable. That image contributes only its Python site packages. Use `PAVEDESK_DEPS_IMAGE` to select it. On the authoring host `solo51-glazebench:1.0` contains the exact set.

## Survey a surface

- **Load training surveys** opens four synthetic sections. Every section has a diffuse photograph, two polarization views and two raking-light views. Lane paint, surface flaws, stone highlights and the reference card are generated locally. Their dimensions are demonstration values.
- **Captures** keeps the actual image, metadata and clipped/unsupported pixels. Mark the same four physical points clockwise on each image. Registered images and the split-view comparison show how the points align.
- **Sampling regions** marks inspection polygons, a bare sample, a paint sample and a neutral reference. Exclusion can mark an object that obscures the surface. These regions stay in the registered plane.
- **Measure** maps paint coverage, color variation, visible pits, polarization contrast, opposite-light response, texture direction, fissures, margin color and line profiles. The marking wear scan reads a straight painted band compared to its nearby pavement. Repair seam scan compares observed levels on both sides. Exposed aggregate returns the extent and position of local bright clusters.
- **Geometry** measures lengths and areas in millimeters when both dimensions are supplied. A traced path also reports the perpendicular room left and right of the line along its length, naming the marked exclusion or plane edge that limits each side, and marks the tightest station on the overlay. Missing regions are recorded separately. **Reference library** keeps an approved visual sample for comparison. **Survey comparisons** groups retained measurements by material, visit, marking or route.
- **Evidence** pins dated results and exports a printable sheet, local ZIP and numerical table. **Received evidence** verifies the internal file manifest and opens a local copy. Original photographs can be attached explicitly when pinning.

## Limits and interpretation

PNG, JPEG, WebP and single-frame TIFF are accepted up to 24 MiB and 12 megapixels. The registered plane is limited to 1,800 pixels on its longest edge; an individual survey can hold ten photographs and 24 regions. Source bytes are retained while the working image applies embedded ICC conversion and EXIF orientation. High-bit grayscale is converted to a display-referred working image. An image without an embedded profile is interpreted as sRGB. The normal optics do not provide absolute surface depth, certified pigment readings or vehicle safety advice. A dark or bright feature is an image candidate. Use the actual photo, physical measurements and support labels to confirm it. A received ZIP accepts up to 24 MiB compressed, 72 MiB examined and twelve declared measurements.

The workbench derives its image loading, plane registration and evidence libraries from the local GlazeBench normal-product code. The pavement rendering, marking scan, repair seam scan and aggregate measurement are PaveDesk code. The `pavedesk` package contains the normal Python service and `static` contains the local browser interface.
