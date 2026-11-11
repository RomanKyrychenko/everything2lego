"""Pairwise VLM judge for build renders, BrickBench-style (arXiv 2610.12452):
a local Ollama vision model sees two renders and answers
  alignment -- which assembly matches the reference better (needs --reference)
  design    -- which assembly is better designed, judged by how pieces are used
              (reference withheld).
Each pair is asked in both orders (A/B, B/A) to cancel position bias; answers
are fitted to Bradley-Terry ratings, printed on an Elo scale (mean 1500).
Answers are cached in <out>.cache.json, keyed by images+mtimes, question, model.

    python tools/design_judge.py BUILD_DIR... [--reference mesh_view.png]
        [--model qwen3.5:9b] [--out design_judge.json]
    python tools/design_judge.py --self-check
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "http://localhost:11434/api/chat"
PROMPTS = {
    "alignment": ("Both assemblies are LEGO models built to reproduce the reference object. "
                  "Which assembly matches the reference better (overall shape, proportions, "
                  "silhouette and colours)? Reply with only the letter A or B, nothing else."),
    "design": ("Both images show LEGO assemblies, each from three viewpoints. Which assembly is "
               "better designed? Judge how the pieces are used: clean, purposeful part choices "
               "and building techniques like an official LEGO set, versus noisy, random or "
               "ill-fitting parts. Reply with only the letter A or B, nothing else."),
}


def bt_fit(wins: np.ndarray, iters: int = 2000, prior: float = 0.5) -> np.ndarray:
    """Bradley-Terry strengths by MM (Hunter 2004). wins[i, j] = times i beat j (ties as 0.5
    each); `prior` virtual wins per pair keep undefeated/winless builds finite."""
    w = wins + prior * (1 - np.eye(len(wins)))
    n = w + w.T
    p = np.ones(len(w))
    for _ in range(iters):
        p = w.sum(1) / (n / (p[:, None] + p[None, :])).sum(1)
        p /= np.exp(np.log(p).mean())
    return 1500 + 400 * np.log10(p)


def self_check() -> None:
    rng = np.random.default_rng(0)
    true = np.array([0.0, 0.5, 1.0, 2.0])                  # log-strengths
    wins = np.zeros((4, 4))
    for i, j in itertools.permutations(range(4), 2):
        wins[i, j] = rng.binomial(200, 1 / (1 + np.exp(true[j] - true[i])))
    elo = bt_fit(wins, prior=0.0)
    got = (elo - elo.mean()) * np.log(10) / 400
    assert np.allclose(got, true - true.mean(), atol=0.25), got
    assert np.all(np.isfinite(bt_fit(np.array([[0, 5], [0, 0.]]))))  # undefeated stays finite
    print("bt self-check ok", np.round(got, 2))


def parse(text: str) -> str | None:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = (re.fullmatch(r"\W*([AB])\W*", t) or re.search(r"answer\W*(?:is\W*)?(?:assembly\s*)?([AB])\b", t, re.I)
         or re.search(r"\bassembly\s+([AB])\b", t, re.I))
    if m:
        return m.group(1).upper()
    letters = set(re.findall(r"\b([AB])\b", t))
    return letters.pop() if len(letters) == 1 else None


def ask(model: str, images: list[tuple[str, Path]], question: str) -> str:
    # all images in one message: per-image messages gave 100 % "A" (pure position bias)
    order = ", ".join(f"image {k + 1} is the {label}" for k, (label, _) in enumerate(images))
    msgs = [{"role": "user", "content": f"{order[0].upper()}{order[1:]}. {question}",
             "images": [base64.b64encode(p.read_bytes()).decode() for _, p in images]}]
    body = dict(model=model, messages=msgs, stream=False, think=False,
                options=dict(temperature=0, seed=0, num_ctx=8192, num_predict=32))
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("builds", nargs="*", type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--out", type=Path, default=Path("design_judge.json"))
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    renders = [b / "render.png" for b in a.builds]
    for r in renders + ([a.reference] if a.reference else []):
        assert r.is_file(), f"missing {r}"
    names = ["/".join(b.resolve().parts[-2:]) for b in a.builds]
    cache_path = a.out.with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    tag = lambda p: f"{p.resolve()}@{p.stat().st_mtime_ns}"
    questions = ["design"] + (["alignment"] if a.reference else [])

    n = len(renders)
    res, times = {}, []
    for q in questions:
        wins, comps = np.zeros((n, n)), np.zeros(n)
        agree = both = unparsed = 0
        for i, j in itertools.combinations(range(n), 2):
            picks = []
            for x, y in ((i, j), (j, i)):
                imgs = ([("reference", a.reference)] if q == "alignment" else []) + \
                       [("Assembly A", renders[x]), ("Assembly B", renders[y])]
                key = "|".join([a.model, PROMPTS[q]] + [f"{l}={tag(p)}" for l, p in imgs])
                if key not in cache:
                    t = time.time()
                    cache[key] = ask(a.model, imgs, PROMPTS[q])
                    times.append(time.time() - t)
                    cache_path.write_text(json.dumps(cache, indent=1))
                ans = parse(cache[key])
                picks.append({"A": x, "B": y}.get(ans))
                print(f"  {q:9s} A={names[x]:12s} B={names[y]:12s} -> {cache[key]!r}")
            for p in picks:
                comps[[i, j]] += 1
                if p is None:
                    unparsed += 1
                    wins[i, j] += 0.5; wins[j, i] += 0.5        # unparseable counts as a tie
                else:
                    wins[p, i + j - p] += 1
            if None not in picks:
                both += 1; agree += picks[0] == picks[1]
        res[q] = dict(elo=bt_fit(wins).tolist(), wins=wins.sum(1).tolist(), comps=comps.tolist(),
                      consistency=agree / both if both else float("nan"), unparsed=unparsed)

    print(f"\n{'build':14s}" + "".join(f"{q + ' elo':>15s}{'W/N':>9s}" for q in questions))
    for k in range(n):
        print(f"{names[k]:14s}" + "".join(
            f"{res[q]['elo'][k]:15.0f}{res[q]['wins'][k]:5.1f}/{res[q]['comps'][k]:<3.0f}" for q in questions))
    for q in questions:
        print(f"{q}: position consistency {res[q]['consistency']:.2f}, unparsed {res[q]['unparsed']}")
    if times:
        print(f"{len(times)} new queries, {np.mean(times):.1f} s each")
    a.out.write_text(json.dumps(dict(model=a.model, builds=names, prompts=PROMPTS, results=res), indent=1))


if __name__ == "__main__":
    main()
