"""Slope courses: a slope on a plate of the same footprint, offered to the sampler as one option.

Real roofs rarely match LEGO's slope angles; builders lay rows of slopes on a plate layer so each row rises one plate
more: a 33 deg slope rises 3 plates over its 2-stud run, so slope + plate rows climb 2 plates per stud (~39 deg), close
to the watermill's 35-40 deg roof.  As one option the engine needs nothing new (cells, fit, support, statistics: the
slope's token); the placement expands back into the plate and the slope on it.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .sampler import SUB, Orient, _halo, pack, popcount, unpack


def _fine(o: Orient) -> np.ndarray:
    g = np.zeros(o.size, np.uint64); g[tuple(o.cells.T)] = o.bits
    return unpack(g)


def course_orient(slope: Orient, plate_part, plate: Orient) -> Orient:
    """The slope one plate up, on the plate (same footprint): one Orient, its `stack` naming the plate."""
    h, W, D = slope.size
    fine = np.zeros(((h + 1) * SUB, W * SUB, D * SUB), bool)
    fine[:SUB] = _fine(plate)                                     # the plate
    fine[SUB:] = _fine(slope)                                     # the slope on it
    bits = pack(fine, (h + 1, W, D)); cells = np.argwhere(bits != 0); cb = bits[tuple(cells.T)]
    sides = tuple((hx, hy + 2, hz, dx, dz, k) for hx, hy, hz, dx, dz, k in slope.sides)   # half plates: one plate = 2
    pix = (np.argwhere(fine.any(axis=1)), np.argwhere(fine.any(axis=0)), np.argwhere(fine.any(axis=2)))
    return Orient(2000 + slope.rot, (h + 1, W, D), cells, cb, popcount(cb) / SUB ** 3, tuple(int(v) for v in cells[0]),
                  slope.top, plate.bottom, sides, _halo(cells, (h + 1, W, D)), pix, None,
                  (slope.rot, plate_part, plate.rot))


def with_courses(vocab: tuple) -> tuple:
    """The vocabulary with a slope-on-plate option for every upright slope whose footprint a plain plate has."""
    plates = {}
    for v in sorted((v for v in vocab if v.box == "plate"), key=lambda v: -v.n_sets):
        for o in v.orients:
            plates.setdefault((o.size[1], o.size[2]), (v, o))       # the most used plate of each footprint
    out = []
    for v in vocab:
        extra = []
        if v.category == "Bricks Sloped" and "Inverted" not in v.part.name:
            for o in v.orients:
                if o.snot is None and o.stack is None and o.bottom and (o.size[1], o.size[2]) in plates:
                    pv, po = plates[(o.size[1], o.size[2])]
                    extra.append(course_orient(o, pv.part, po))
        out.append(dataclasses.replace(v, orients=v.orients + tuple(extra)) if extra else v)
    return tuple(out)


def expand(part, colour: int, x: int, z: int, y: int, o: Orient):
    """Placements of one placed option: the slope and its plate for a course, else the part itself."""
    from .pack import Placement
    if o.stack is None:
        return [Placement(part, colour, x, z, y, o.size[1], o.size[2], o.rot, snot=o.snot)]
    srot, ppart, prot = o.stack
    return [Placement(ppart, colour, x, z, y, o.size[1], o.size[2], prot),
            Placement(part, colour, x, z, y + 1, o.size[1], o.size[2], srot)]
