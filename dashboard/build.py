#!/usr/bin/env python3
"""Assemble index.html from shell.html, theme.css, dp.js, hero.html, footer.html and sections/*.html.

python3 build.py                                  # whole page -> index.html
python3 build.py --copy-to ../<name>.html         # also copy the single file somewhere to share
python3 build.py --only 20-efficiency [--only ...] # preview/<names>.html with just those sections
python3 build.py --check                          # validate fragments, write nothing
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTIONS = HERE / "sections"
SECTION_OPEN = re.compile(r"<section\b([^>]*)>", re.I)
ATTR = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')
EXTERNAL = [
    (re.compile(r"<link\b", re.I), "<link> tag (no external stylesheets)"),
    (re.compile(r"""\bsrc\s*=\s*["']https?://""", re.I), "external src="),
    (re.compile(r"@import", re.I), "@import"),
    (re.compile(r"url\(\s*['\"]?https?://", re.I), "external url()"),
]
SECRETS = [
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"\bAKLT[A-Za-z0-9+/=]{16,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][A-Za-z0-9+/=_\-]{16,}['\"]"),
]
PLACEHOLDER = re.compile(r"\{\{[^}]{1,60}\}\}")


def read(name: str) -> str:
    path = HERE / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def check_fragment(path: Path) -> tuple[list[str], list[str], dict]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []
    roots = [dict(ATTR.findall(attrs)) for attrs in SECTION_OPEN.findall(text)]
    roots = [r for r in roots if "sec" in r.get("class", "").split()]
    if len(roots) != 1:
        errors.append(f"needs exactly one <section class=\"sec\" id=... data-nav=...> root, found {len(roots)}")
        return errors, warnings, {}
    root = roots[0]
    sid = root.get("id", "")
    if not sid:
        errors.append("section root has no id")
    if not root.get("data-nav"):
        errors.append("section root has no data-nav (short nav label)")
    for pattern, what in EXTERNAL:
        if pattern.search(text):
            errors.append(f"external resource: {what}")
    for pattern in SECRETS:
        if pattern.search(text):
            errors.append(f"looks like a credential: {pattern.pattern}")
    if PLACEHOLDER.search(text):
        warnings.append(f"unfilled placeholder: {PLACEHOLDER.search(text).group(0)}")
    for style in re.findall(r"<style[^>]*>(.*?)</style>", text, re.S | re.I):
        body = re.sub(r"/\*.*?\*/", "", style, flags=re.S)
        for block in re.findall(r"([^{}]+)\{", body):
            sel = block.strip()
            if not sel or sel.startswith("@") or sel in {"from", "to"} or re.fullmatch(r"[\d.%\s,]+", sel):
                continue
            for part in sel.split(","):
                part = part.strip()
                if part and not part.startswith(f"#{sid}"):
                    warnings.append(f"CSS selector not scoped under #{sid}: {part[:80]}")
    for script in re.findall(r"<script[^>]*>(.*?)</script>", text, re.S | re.I):
        s = script.strip()
        if s and not (s.startswith("(") or s.startswith("DP.ready") or s.startswith("!function") or s.startswith("{")):
            warnings.append("script should be an IIFE or DP.ready(...) so it adds no globals")
    return errors, warnings, root


def assemble(fragments: list[Path]) -> str:
    nav, bodies = [], []
    for path in fragments:
        _, _, root = check_fragment(path)
        if root:
            nav.append(f'<a href="#{root["id"]}">{root["data-nav"]}</a>')
        bodies.append(f"<!-- ===== {path.name} ===== -->\n{path.read_text(encoding='utf-8')}\n")
    page = read("shell.html")
    page = page.replace("/*THEME*/", read("theme.css"))
    page = page.replace("/*DPJS*/", read("dp.js"))
    page = page.replace("/*APPJS*/", read("app.js"))
    page = page.replace("<!--HERO_KPIS-->", read("hero.html"))
    page = page.replace("<!--FOOTER-->", read("footer.html"))
    page = page.replace("<!--NAV-->", "\n    ".join(nav))
    page = page.replace("<!--SECTIONS-->", "\n".join(bodies))
    return page


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[], help="fragment stem(s) for a preview page")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--copy-to", help="also copy the built single file to this path")
    args = ap.parse_args()

    fragments = sorted(SECTIONS.glob("*.html"))
    if args.only:
        fragments = [p for p in fragments if any(p.stem == o or p.stem.startswith(o) for o in args.only)]
        if not fragments:
            print(f"no fragment matches {args.only}", file=sys.stderr)
            return 2

    failed = False
    ids: dict[str, str] = {}
    for path in fragments:
        errors, warnings, root = check_fragment(path)
        sid = root.get("id") if root else None
        if sid in ids:
            errors.append(f"duplicate section id #{sid} (also in {ids[sid]})")
        if sid:
            ids[sid] = path.name
        status = "ERROR" if errors else ("warn" if warnings else "ok")
        print(f"[{status}] {path.name}  #{sid}  {path.stat().st_size / 1024:.1f} KB")
        for e in errors:
            print(f"    error: {e}")
        for w in warnings[:12]:
            print(f"    warn:  {w}")
        if len(warnings) > 12:
            print(f"    warn:  ... {len(warnings) - 12} more")
        failed |= bool(errors)
    if args.check:
        return 1 if failed else 0
    if failed:
        print("fix the errors above; nothing written", file=sys.stderr)
        return 1

    page = assemble(fragments)
    if args.only:
        out = HERE / "preview" / ("+".join(p.stem for p in fragments) + ".html")
        out.parent.mkdir(exist_ok=True)
        out.write_text(page, encoding="utf-8")
        print(f"wrote {out} ({len(page) / 1024:.0f} KB)")
        return 0
    leftovers = sorted(set(PLACEHOLDER.findall(page)))
    if leftovers:
        print(f"warn: unfilled placeholders in the page: {', '.join(leftovers[:8])}")
    out = HERE / "index.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page) / 1024:.0f} KB)")
    if args.copy_to:
        dest = Path(args.copy_to)
        dest = dest if dest.is_absolute() else (HERE / dest).resolve()
        shutil.copyfile(out, dest)
        print(f"copied to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
