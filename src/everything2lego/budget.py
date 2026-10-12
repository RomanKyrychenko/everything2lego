"""Pick resolution, hollowing and colour count for a target part count.

The part count of a packed model is predicted from one cheap measurement
(a pack at a low resolution, 16 studs by default) scaled by a power law in
the stud resolution, with correction factors for hollowing and for the
number of colour clusters.  The exponents and factors are calibrated on
the README's watermill data (38,196 parts at 80 studs solid with 10
colours; 26,724 hollow; 12,598 at 56 studs hollow; 5,690 at 40 studs with
5 colours and aggressive hollowing) and on the repo's carve example
(632 / 1,601 / 3,550 parts at 16 / 24 / 32 studs, solid):

* shells grow like resolution^2.3 (the watermill: 56 -> 80 studs hollow
  gives an exponent of 2.1-2.3),
* solids grow like resolution^2.5 (the carve example gives 2.3-2.5; the
  packer fills interiors with big bricks, so solids stay well below the
  naive ^3),
* hollowing saves 30% at 80 studs (26,724 / 38,196) and almost nothing at
  16 studs, which the two exponents plus a constant of 0.966 reproduce,
* halving the colour-cluster count saves about 10% (5 colours + aggressive
  hollowing at 40 studs ends 2% below the pure resolution scaling).

`choose` then walks resolutions from the largest down and, at each one,
tries solid first, then hollow, then fewer colour clusters, so the result
is the highest resolution at which some setting fits, with the gentlest
setting at that resolution.  Everything is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

SHELL_EXPONENT = 2.3
SOLID_EXPONENT = 2.5
# parts_hollow(s) = HOLLOW_FACTOR * parts_solid(16) * (s/16)^SHELL_EXPONENT,
# so that the saving is 30% at 80 studs.
HOLLOW_FACTOR = 0.70 * (80 / 16) ** (SOLID_EXPONENT - SHELL_EXPONENT)
COLOR_EXPONENT = 0.15
MEASURE_STUDS = 16
STUD_CANDIDATES = (8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 112, 128)
COLOR_CANDIDATES = (10, 8, 6, 5, 4, 3, 2)


@dataclass(frozen=True)
class Measurement:
    """One cheap pack: `parts` placed for a grid at `max_studs` studs."""
    max_studs: int
    hollow: bool
    parts: int
    colors: int


@dataclass(frozen=True)
class Choice:
    max_studs: int
    hollow: bool
    color_clusters: int
    predicted: int
    fits: bool
    measurement: Measurement

    def describe(self) -> str:
        knobs = [f"{self.max_studs} studs", "hollow" if self.hollow else "solid",
                 f"{self.color_clusters} colours"]
        tail = "" if self.fits else " (over budget even at the smallest setting)"
        return (f"{', '.join(knobs)}: about {self.predicted} parts predicted from "
                f"{self.measurement.parts} at {self.measurement.max_studs} studs{tail}")


def predict(measurement: Measurement, max_studs: int, hollow: bool,
            color_clusters: Optional[int] = None) -> int:
    """Predicted part count at `max_studs` with `hollow` and
    `color_clusters` colours, from one measurement."""
    ratio = max_studs / measurement.max_studs
    if measurement.hollow:
        parts = measurement.parts * ratio ** SHELL_EXPONENT
        if not hollow:
            # Un-hollowing a measurement: invert the hollow model.
            parts = parts / HOLLOW_FACTOR * ratio ** (SOLID_EXPONENT - SHELL_EXPONENT)
            parts = max(parts, measurement.parts * ratio ** SOLID_EXPONENT)
    else:
        if hollow:
            parts = HOLLOW_FACTOR * measurement.parts * ratio ** SHELL_EXPONENT
            # Hollowing never adds parts.
            parts = min(parts, measurement.parts * ratio ** SOLID_EXPONENT)
        else:
            parts = measurement.parts * ratio ** SOLID_EXPONENT
    if color_clusters is not None and measurement.colors > 0:
        k = min(color_clusters, measurement.colors)
        parts *= (k / measurement.colors) ** COLOR_EXPONENT
    return max(1, int(round(parts)))


def color_candidates(measured_colors: int) -> list[int]:
    """Colour-cluster counts to try, from the measured count downwards."""
    out = [measured_colors]
    out += [k for k in COLOR_CANDIDATES if k < measured_colors]
    return out


def choose(max_parts: int, measure: Callable[[int, bool], Measurement], *,
           max_studs_cap: int = 128, hollow: Optional[bool] = None,
           allow_colors: bool = True, measure_studs: int = MEASURE_STUDS,
           candidates: Sequence[int] = STUD_CANDIDATES) -> Choice:
    """Pick the settings whose predicted part count fits `max_parts`.

    `measure(max_studs, hollow)` packs the model once at a low resolution
    and returns a `Measurement`.  `hollow=None` lets the chooser decide;
    True or False pins it.  `allow_colors=False` keeps the measured colour
    count.  Resolutions above `max_studs_cap` are never chosen.
    """
    if max_parts < 1:
        raise ValueError("max_parts must be positive")
    studs = sorted({s for s in candidates if s <= max_studs_cap} | {min(max_studs_cap, measure_studs)})
    measure_at = min(measure_studs, max_studs_cap)
    measurement = measure(measure_at, bool(hollow))
    colors = color_candidates(measurement.colors) if allow_colors else [measurement.colors]
    # Knob order at one resolution: as measured, then hollow, then (hollow
    # with) fewer colour clusters.
    if hollow is None:
        settings = [(False, colors[0])] + [(True, k) for k in colors]
    else:
        settings = [(hollow, k) for k in colors]
    last: Choice | None = None
    for s in reversed(studs):
        for h, k in settings:
            predicted = predict(measurement, s, h, k)
            last = Choice(s, h, k, predicted, predicted <= max_parts, measurement)
            if last.fits:
                return last
    assert last is not None
    return last
