#!/usr/bin/env python3
"""Screenshot a built page, or one section of it, with headless Chromium.

$DPVIZ_ROOT/venv/bin/python shot.py preview/20-efficiency.html --section efficiency --out $DPVIZ_ROOT/efficiency/s1440.png
$DPVIZ_ROOT/venv/bin/python shot.py preview/20-efficiency.html --section efficiency --width 1000 --out .../s1000.png
$DPVIZ_ROOT/venv/bin/python shot.py index.html --full --out .../full.png
$DPVIZ_ROOT/venv/bin/python shot.py preview/x.html --section x --hover "#x .dot" --out .../hover.png

Animations are shown in their final state (window.DP_STATIC) and the sticky nav is made static, so the
shot shows the page as a reader sees it after scrolling. Console errors are printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = os.environ.get("DPVIZ_ROOT", "/var/tmp/dpviz")
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", f"{ROOT}/browsers")
LIBS = f"{ROOT}/libs/usr/lib/x86_64-linux-gnu"

from playwright.sync_api import sync_playwright  # noqa: E402

SHOT_CSS = ".reveal{opacity:1!important;transform:none!important} .topnav{position:static!important}"


def open_page(p, page_path: str, width: int, height: int, errors: list[str]):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(x for x in (LIBS, env.get("LD_LIBRARY_PATH", "")) if x)
    browser = p.chromium.launch(executable_path=os.environ.get("DPVIZ_CHROME") or None, args=["--no-sandbox", "--disable-dev-shm-usage"], env=env)
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.add_init_script("window.DP_STATIC = true;")
    page.goto(page_path if page_path.startswith(("http://", "https://")) else Path(page_path).resolve().as_uri(), wait_until="load")
    page.add_style_tag(content=SHOT_CSS)
    page.wait_for_timeout(600)
    return browser, page


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--out", required=True)
    ap.add_argument("--section", help="section id to capture (element screenshot)")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--full", action="store_true", help="full-page screenshot")
    ap.add_argument("--hover", help="CSS selector to hover before the shot (tooltip check)")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        browser, page = open_page(p, args.page, args.width, args.height, errors)
        if args.hover:
            page.hover(args.hover)
            page.wait_for_timeout(250)
            page.screenshot(path=args.out)  # viewport shot, so the fixed tooltip is in the picture
        elif args.section:
            page.locator(f"#{args.section}").screenshot(path=args.out)
        else:
            page.screenshot(path=args.out, full_page=args.full)
        browser.close()
    for e in errors:
        print(e)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
