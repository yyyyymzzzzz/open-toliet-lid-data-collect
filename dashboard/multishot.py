#!/usr/bin/env python3
"""Screenshot the hero and every section of a built page, cut into readable chunks.

$DPVIZ_ROOT/venv/bin/python multishot.py index.html --out $DPVIZ_ROOT/shots/r1 [--width 1440] [--chunk 1500]

Writes hero_<w>.png and <section>_<w>_p<N>.png (N = 0, 1, ...), prints each section's height and any
console errors. Read the chunks with the Read tool: a whole 4,000 px section shrinks too much to judge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shot import open_page, sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--chunk", type=int, default=1500)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        browser, page = open_page(p, args.page, args.width, 900, errors)
        page.screenshot(path=str(out / f"hero_{args.width}.png"))
        boxes = page.evaluate(
            "[...document.querySelectorAll('main > section.sec')].map(s => {"
            "const r = s.getBoundingClientRect(); return [s.id, r.top + scrollY, r.height]; })"
        )
        for sid, top, height in boxes:
            n = 0
            for y in range(0, int(height), args.chunk):
                h = min(args.chunk, int(height) - y)
                page.screenshot(path=str(out / f"{sid}_{args.width}_p{n}.png"), full_page=True,
                                clip={"x": 0, "y": top + y, "width": args.width, "height": h})
                n += 1
            print(f"{sid}: {int(height)} px, {n} chunk(s)")
        browser.close()
    for e in errors:
        print(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
