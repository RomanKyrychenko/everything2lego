"""Embeddings of what every part depicts (part_objects.py's phrases), for matching a region or an object's named
parts to parts later: in the labeller's image-text space (SigLIP 2's text side), so an image crop and a phrase like
"water wheel" are scored against them alike.

A phrase listed by many parts ("platform", 330 parts) says little about which part: each phrase gets
idf = log(parts / parts listing it or a near-duplicate of it (cos >= --dup)), so "window"/"windows" or "post"/"pole"
share one count.  Per part: its phrases' embeddings and weights, and one vector, their idf-weighted mean (unit).

Writes <out>/<model>.objects.npz: ldraw (n,), part_vec (n, d) float16; phrase (m,), phrase_part (m,) index into
ldraw, phrase_vec (m, d) float16, idf (m,).

    tools/timed.sh objects-embed .venv-arm/bin/python tools/part_objects_embed.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
OBJ = Path("/Volumes/PRO-G40/lego_data/analysis/part_objects")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--template", default="a photo of a {}")
    ap.add_argument("--dup", type=float, default=0.95, help="phrases this close count as one for idf")
    a = ap.parse_args()
    import semantic_labels as S
    S.MODEL = a.model
    L = [json.loads(l) for l in open(OBJ / "objects.jsonl")]
    ldraw = np.array([d["ldraw"] for d in L])
    uniq = sorted({o for d in L for o in d["objects"]})
    U = S.embed_texts([a.template.format(o) for o in uniq]).astype(np.float32)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    row = {o: i for i, o in enumerate(uniq)}
    has = np.zeros((len(L), len(uniq)), bool)
    for p, d in enumerate(L):
        has[p, [row[o] for o in d["objects"]]] = True
    near = (U @ U.T) >= a.dup                                # a phrase and its near-duplicates
    df = (has.astype(np.float32) @ near.astype(np.float32) > 0).sum(0)   # parts listing it or a near-duplicate
    idf_u = np.log(len(L) / np.maximum(df, 1))
    pp, pv, ph, pi = [], [], [], []
    for p, d in enumerate(L):
        j = np.array([row[o] for o in d["objects"]], int)
        w = idf_u[j]
        v = (U[j] * w[:, None]).sum(0); pv.append(v / (np.linalg.norm(v) + 1e-9))
        ph += d["objects"]; pp += [p] * len(j); pi += list(w)
    f = OBJ / (a.model.replace("/", "__") + ".objects.npz")
    np.savez_compressed(f, ldraw=ldraw, part_vec=np.array(pv, np.float16), phrase=np.array(ph),
                        phrase_part=np.array(pp), phrase_vec=U[[row[o] for o in ph]].astype(np.float16),
                        idf=np.array(pi, np.float32))
    print(f"{len(L)} parts, {len(ph)} phrases ({len(uniq)} distinct) -> {f}")
    lo = np.argsort(idf_u)[:8]; hi = np.argsort(-idf_u)[:5]
    print("least telling:", ", ".join(f"{uniq[i]} {idf_u[i]:.2f}" for i in lo))
    print("most telling: ", ", ".join(f"{uniq[i]} {idf_u[i]:.2f}" for i in hi))
    # self-check: each part's own phrases rank it high among all parts
    q = S.embed_texts([a.template.format(x) for x in ("water wheel", "chimney", "stone wall", "thatched roof", "lamp post")]).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    P = np.array(pv, np.float32); names = {d["ldraw"]: d.get("name", "") for d in L}
    for t, s in zip(("water wheel", "chimney", "stone wall", "thatched roof", "lamp post"), q @ P.T):
        print(f"  {t:14s} -> " + "; ".join(names[ldraw[k]][:30] for k in np.argsort(-s)[:4]))


if __name__ == "__main__":
    main()
