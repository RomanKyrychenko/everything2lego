"""Name the parts of a segmented mesh with a vision model (optional).

A part segmenter (SegviGen, the built-in one, or a person with a paint
bucket) hands the pipeline parts without saying what they are.  The
builders only know what to do with a part when it has a name: a "window"
is cut out of the wall and framed, a "roof" is packed from slopes, a
"foundation" from plates.  Geometry can guess a house's parts, but the
input may be a car, a chair or a dragon.

This module renders the mesh with each part highlighted, shows the
pictures to Claude and asks, for every part, what it is and how a LEGO
builder should treat it.  The answer fills `PartLabels.names` with the
vocabulary `regions_from_labels` understands and `PartLabels.descriptions`
with the free-form names ("left front wheel", "chimney").

Nothing here is required: it needs the `anthropic` package and an API key
(`pip install everything2lego[vlm]`, `ANTHROPIC_API_KEY`), costs a few
cents per mesh, and when either is missing the pipeline goes on with
unnamed parts.  Results are cached by mesh and labels, so a second run on
the same input is free.  The pictures are drawn by a software rasteriser
in PIL: no GPU, no OpenGL.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from .parts import GROUND_NAMES, OPENING_NAMES, ROOF_NAMES, WALL_NAMES, PartLabels

DEFAULT_MODEL = "claude-opus-5-5"
# `fallbacks="default"` re-runs a request the model declines on Anthropic's
# recommended fallback model, inside the same call.  It is a beta feature.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CACHE_DIR = Path(os.environ.get("EVERYTHING2LEGO_CACHE",
                                Path.home() / ".cache" / "everything2lego")) / "labeller"

# How a LEGO builder treats a part.  These are the names the builders
# read; everything else about a part is description.
KINDS = ("wall", "roof", "window", "door", "opening", "foundation", "other")
assert set(KINDS) - {"other"} <= WALL_NAMES | ROOF_NAMES | OPENING_NAMES | GROUND_NAMES
PARTS_PER_REQUEST = 16
HIGHLIGHT = (255, 0, 190)
MUTED = (200, 200, 200)

SYSTEM_PROMPT = """You help turn 3D models into LEGO builds. A mesh has been
split into parts by an automatic segmenter; the parts have no names. You
are shown the whole object, then each part highlighted in bright magenta
against the rest of the object in grey, each from the viewpoint that shows
it best.

For every part give:
- name: a short, specific name of what the part is on this kind of object
  ("left front wheel", "chimney", "seat cushion", "front wall").
- kind: how a LEGO builder should treat it, one of
  wall (a large, flat, vertical surface: a building's wall, a cabinet side,
  a van's body panel), roof (a sloped or flat top surface that covers the
  object: a house roof, a car roof, a lid), window (a window pane or glazed
  opening set in a wall), door (a door or hatch set in a wall), opening (any
  other hole or recess in a wall), foundation (the flat underside or base
  the object stands on), other (everything else: wheels, trim, figures,
  organic shapes, anything curved).
  Use window/door/opening only for a part that sits in the plane of a larger
  wall; a part that is itself a whole wall or a whole roof is a wall or a
  roof even if it has openings painted on it.
- confidence: 0 to 1.

Also name the object as a whole. The segmenter is imperfect: a part may be
a fragment of a larger thing or two things merged; name what the highlight
covers."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "description": "What the whole object is."},
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "name", "kind", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["object", "parts"],
    "additionalProperties": False,
}


class LabellerError(RuntimeError):
    """The vision model could not be asked (no package, no key, no network,
    a refusal); the caller decides whether that is fatal."""


@dataclass
class PartName:
    id: int
    name: str
    kind: str
    confidence: float


@dataclass
class Naming:
    """What the model said about an object and its parts."""
    object: str
    parts: list[PartName]
    model: str = ""
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    images: dict[str, bytes] = field(default_factory=dict, repr=False)

    def by_id(self) -> dict[int, PartName]:
        return {p.id: p for p in self.parts}


# -- rendering --------------------------------------------------------------

def _view_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Rotation taking world (y up) to camera space (x right, y up, z
    towards the viewer) for a camera at the given azimuth around y and
    elevation above the horizon."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return rx @ ry


# Front-left, front-right, back-right and back-left three-quarter views
# (the front is +z, as the mesh fixtures and the viewer have it), the top,
# the front straight on, and the underside for parts seen from nowhere else.
VIEWS = {
    "front-left": _view_matrix(-35, 28), "front-right": _view_matrix(35, 28),
    "back-right": _view_matrix(145, 28), "back-left": _view_matrix(-145, 28),
    "top": _view_matrix(0, 89), "front": _view_matrix(0, 5), "bottom": _view_matrix(0, -89),
}
LIGHT = np.array([0.3, 0.8, 0.55]) / np.linalg.norm([0.3, 0.8, 0.55])


def render(mesh: trimesh.Trimesh, view: np.ndarray, colours: np.ndarray, *,
           size: int = 512, margin: float = 0.06) -> Image.Image:
    """An orthographic picture of `mesh` with one RGB colour per face,
    painted back to front with normal shading.  Pure PIL."""
    verts = np.asarray(mesh.vertices, float) @ view.T
    centre = (verts.min(axis=0) + verts.max(axis=0)) / 2
    extent = max(1e-9, float(np.ptp(verts[:, :2], axis=0).max()))
    scale = size * (1 - 2 * margin) / extent
    xy = (verts[:, :2] - centre[:2]) * scale
    xy[:, 1] = -xy[:, 1]
    xy += size / 2
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals, float) @ view.T
    facing = normals[:, 2] > 0
    depth = verts[:, 2][faces].mean(axis=1)
    order = np.argsort(depth)                 # far first
    shaded = (colours.astype(float) * (0.45 + 0.55 * np.clip(normals @ LIGHT, 0, 1))[:, None])
    shaded = np.clip(shaded, 0, 255).astype(np.uint8)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pts = xy[faces]                           # (F, 3, 2)
    for f in order:
        if not facing[f]:
            continue
        draw.polygon([tuple(p) for p in pts[f]], fill=tuple(int(c) for c in shaded[f]))
    return img


def _visible_area(mesh: trimesh.Trimesh, faces: np.ndarray, view: np.ndarray) -> float:
    """Projected area of the front-facing faces of a part in a view (no
    occlusion test: good enough to pick the view that shows it best)."""
    n = np.asarray(mesh.face_normals)[faces] @ view.T
    a = np.asarray(mesh.area_faces)[faces]
    return float((a * np.clip(n[:, 2], 0, None)).sum())


def _distinct_colours(n: int, *, min_gap: float = 72.0) -> np.ndarray:
    """`n` colours a GLB round trip keeps apart: `load_part_labels` merges
    colours closer than its merge distance (L1 60) and the colours it is
    told about in a sidecar are parts outright, so each colour is chosen,
    farthest-point style, to be far from every colour before it."""
    grid = np.arange(16, 256, 24, dtype=float)
    candidates = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    chosen = [np.array([230.0, 40.0, 40.0])]
    while len(chosen) < n:
        score = np.min([np.abs(candidates - c).sum(axis=1) for c in chosen], axis=0)
        best = int(np.argmax(score))
        if score[best] < min_gap:
            warnings.warn(f"only {len(chosen)} of {n} part colours can be told apart after a GLB "
                          "round trip; some parts will merge", RuntimeWarning, stacklevel=3)
        chosen.append(candidates[best])
    return np.asarray(chosen[:n], dtype=np.uint8).reshape(-1, 3)


def render_parts(mesh: trimesh.Trimesh, labels: PartLabels, *, size: int = 512,
                 overview_size: int = 768) -> dict[str, Image.Image]:
    """The pictures the model is shown: two overviews of the object with
    every part in its own colour, and each part highlighted in magenta from
    the viewpoint that shows the most of it.  Keys are "overview-<view>"
    and "part-<id>"."""
    F = len(mesh.faces)
    ids = labels.ids
    palette = _distinct_colours(len(ids))
    index = {k: i for i, k in enumerate(ids)}
    overview = np.full((F, 3), 160, dtype=np.uint8)
    for k in ids:
        overview[labels.labels == k] = palette[index[k]]
    images: dict[str, Image.Image] = {}
    for name in ("front-left", "back-right"):
        images[f"overview-{name}"] = render(mesh, VIEWS[name], overview, size=overview_size)
    for k in ids:
        faces = np.flatnonzero(labels.labels == k)
        best = max(VIEWS, key=lambda v: _visible_area(mesh, faces, VIEWS[v]))
        colours = np.tile(np.asarray(MUTED, np.uint8), (F, 1))
        colours[faces] = HIGHLIGHT
        images[f"part-{k}"] = render(mesh, VIEWS[best], colours, size=size)
    return images


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# -- the request --------------------------------------------------------------

def _image_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(png).decode()}}


def _messages(pngs: dict[str, bytes], ids: list[int], hint: str) -> list[dict]:
    content: list[dict] = []
    if hint:
        content.append({"type": "text", "text": f"The user says the object is: {hint}."})
    content.append({"type": "text", "text": "The whole object, each part in its own colour, "
                                            "from the front-left and from the back-right:"})
    content += [_image_block(pngs["overview-front-left"]), _image_block(pngs["overview-back-right"])]
    for k in ids:
        content.append({"type": "text", "text": f"Part {k}, highlighted in magenta:"})
        content.append(_image_block(pngs[f"part-{k}"]))
    content.append({"type": "text", "text": f"Name the object and parts {', '.join(map(str, ids))}."})
    return [{"role": "user", "content": content}]


def make_client() -> Any:
    """A zero-argument Anthropic client (reads ANTHROPIC_API_KEY and the
    other standard variables); a LabellerError when the SDK is missing."""
    try:
        import anthropic
    except ImportError as exc:
        raise LabellerError("the anthropic package is not installed; "
                            "pip install 'everything2lego[vlm]'") from exc
    return anthropic.Anthropic()


def ask(client: Any, pngs: dict[str, bytes], ids: list[int], *, model: str,
        hint: str = "", max_tokens: int = 4000) -> tuple[dict, Any]:
    """One request naming `ids`; the parsed JSON and the raw response."""
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=_messages(pngs, ids, hint),
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            fallbacks="default",
            betas=[FALLBACK_BETA],
        )
    except LabellerError:
        raise
    except Exception as exc:        # auth, network, rate limit: all mean "no names today"
        raise LabellerError(f"{type(exc).__name__}: {exc}") from exc
    if getattr(response, "stop_reason", None) == "refusal":
        raise LabellerError("the model declined to describe these pictures")
    text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), None)
    if text is None:
        raise LabellerError("the model returned no text")
    try:
        return json.loads(text), response
    except json.JSONDecodeError as exc:
        raise LabellerError(f"the model returned invalid JSON: {exc}") from exc


def _cache_key(mesh: trimesh.Trimesh, labels: PartLabels, model: str, hint: str) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(labels.labels, dtype=np.int32).tobytes())
    h.update(json.dumps({"model": model, "hint": hint, "prompt": SYSTEM_PROMPT}).encode())
    return h.hexdigest()[:24]


def name_parts(mesh: trimesh.Trimesh, labels: PartLabels, *, client: Any = None,
               model: str = DEFAULT_MODEL, hint: str = "", cache: bool = True,
               debug_dir: str | Path | None = None) -> Naming:
    """Ask the vision model what each part of `mesh` is.  `client` is any
    object with the Anthropic SDK's `beta.messages.create` (the real one by
    default; a fake in tests).  Raises LabellerError when it cannot ask."""
    ids = labels.ids
    if not ids:
        return Naming(object="", parts=[], model=model)
    key = _cache_key(mesh, labels, model, hint)
    cached = CACHE_DIR / f"{key}.json"
    if cache and cached.exists():
        data = json.loads(cached.read_text())
        naming = Naming(data["object"], [PartName(**p) for p in data["parts"]],
                        model=data.get("model", model), cached=True)
        if set(naming.by_id()) >= set(ids):
            return naming
    images = render_parts(mesh, labels)
    pngs = {k: _png(v) for k, v in images.items()}
    if debug_dir is not None:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        for k, png in pngs.items():
            (Path(debug_dir) / f"{k}.png").write_bytes(png)
    if client is None:
        client = make_client()
    naming = Naming(object="", parts=[], model=model, images=pngs)
    for start in range(0, len(ids), PARTS_PER_REQUEST):
        chunk = ids[start:start + PARTS_PER_REQUEST]
        data, response = ask(client, pngs, chunk, model=model, hint=hint)
        naming.object = naming.object or str(data.get("object", "")).strip()
        for p in data.get("parts", []):
            try:
                pid = int(p["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if pid in chunk:
                naming.parts.append(PartName(pid, str(p.get("name", "")).strip(),
                                             _normalise_kind(p.get("kind")),
                                             float(p.get("confidence", 0.0))))
        usage = getattr(response, "usage", None)
        naming.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        naming.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        served = getattr(response, "model", None)
        if served:
            naming.model = str(served)
    missing = set(ids) - set(naming.by_id())
    if missing:
        warnings.warn(f"the model did not name parts {sorted(missing)}", RuntimeWarning, stacklevel=2)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"object": naming.object, "model": naming.model,
                                      "parts": [p.__dict__ for p in naming.parts]}, indent=1))
    return naming


def _normalise_kind(kind: Any) -> str:
    k = str(kind or "other").strip().lower()
    return k if k in KINDS else "other"


def apply(labels: PartLabels, naming: Naming, *, min_confidence: float = 0.0,
          keep_existing: bool = True) -> PartLabels:
    """Write the naming into `labels`: `names` get the builder vocabulary
    (nothing for "other"), `descriptions` the free-form names.  Names
    already present (from a sidecar) are kept unless `keep_existing` is
    off."""
    for p in naming.parts:
        if p.confidence < min_confidence:
            continue
        labels.descriptions[p.id] = p.name
        if keep_existing and p.id in labels.names:
            continue
        if p.kind != "other":
            labels.names[p.id] = p.kind
        else:
            labels.names.pop(p.id, None)
    return labels


def export_labelled(mesh: trimesh.Trimesh, labels: PartLabels, path: str | Path) -> dict[str, str]:
    """Write `mesh` with one flat colour per part to `path` and the names
    to the JSON sidecar `load_part_labels` reads next to it; returns the
    sidecar mapping.  This is the pair `semantic --parts` takes, so a
    naming is paid for once and reused."""
    path = Path(path)
    ids = labels.ids
    palette = _distinct_colours(len(ids))
    out = mesh.copy()
    fc = np.full((len(mesh.faces), 4), 255, dtype=np.uint8)
    fc[:, :3] = 160
    for i, k in enumerate(ids):
        fc[labels.labels == k, :3] = palette[i]
    out.visual = trimesh.visual.ColorVisuals(out, face_colors=fc)
    out.export(path)
    colours = {k: tuple(int(c) for c in palette[i]) for i, k in enumerate(ids)}
    names = sidecar(PartLabels(labels.labels, names=labels.names, colours=colours,
                               descriptions=labels.descriptions))
    path.with_suffix(".json").write_text(json.dumps(names, indent=1))
    return names


def sidecar(labels: PartLabels) -> dict[str, str]:
    """The `{"#rrggbb": name}` mapping `load_part_labels` reads, for the
    labels' colours; descriptions go in when a part has no builder name."""
    out: dict[str, str] = {}
    for k, rgb in labels.colours.items():
        name = labels.names.get(k) or labels.descriptions.get(k)
        if name:
            out["#%02x%02x%02x" % tuple(int(c) for c in rgb[:3])] = name
    return out
