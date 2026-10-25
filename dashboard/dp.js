/* Shared helpers for every section. Loaded in <head>, before any section script.
   EDIT PER PROJECT: DP.ERAS (ordered bands, e.g. versions, phases, data releases) and DP.LANES (themes). */
(function () {
  const SVGNS = "http://www.w3.org/2000/svg";
  const PALETTE = ["#64748b", "#7c3aed", "#2563eb", "#0891b2", "#059669", "#d97706", "#db2777", "#ea580c"]; // = --v0..--v7

  function setAttrs(el, attrs) {
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k === "html") el.innerHTML = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    return el;
  }
  function append(el, children) {
    for (const c of children.flat(Infinity)) {
      if (c === undefined || c === null || c === false) continue;
      el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  const DP = {
    PALETTE,
    /* Ordered bands; colour i = --v<i>. Example (versions of a codebase):
       { id: "v0", name: "V0", title: "Naive 单模型", from: "09-23", to: "09-25", color: PALETTE[0] } */
    ERAS: [],
    /* Theme lanes for swimlane charts, e.g. { id: "arm", name: "模型与合约", color: "#4f46e5" } */
    LANES: [],
    /* Day 0 of DP.day(); set to the first date the page covers. */
    START: Date.UTC(2026, 0, 1),
    YEAR: 2026,

    era(id) { return DP.ERAS.find((e) => e.id === id); },
    lane(id) { return DP.LANES.find((l) => l.id === id); },
    fmt(n, digits = 0) {
      if (n === null || n === undefined || Number.isNaN(n)) return "–";
      return Number(n).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
    },
    pct(a, b, digits = 1) { return b ? (100 * a / b).toFixed(digits) + "%" : "–"; },
    /* DP.h("div", {class: "card"}, "text", child) -> HTMLElement */
    h(tag, attrs, ...children) { return append(setAttrs(document.createElement(tag), attrs), children); },
    /* DP.s("rect", {x: 0, y: 0}) -> SVGElement */
    s(tag, attrs, ...children) { return append(setAttrs(document.createElementNS(SVGNS, tag), attrs), children); },
    /* Linear scale: DP.scale([d0, d1], [r0, r1])(x); .invert(y) */
    scale(domain, range) {
      const [d0, d1] = domain, [r0, r1] = range;
      const f = (x) => r0 + (x - d0) * (r1 - r0) / ((d1 - d0) || 1);
      f.invert = (y) => d0 + (y - r0) * (d1 - d0) / ((r1 - r0) || 1);
      return f;
    },
    /* "10-05" or "10-05 14:00" -> days since DP.START (fractional). */
    day(md) {
      const [d, t] = md.split(" ");
      const [m, dd] = d.split("-").map(Number);
      let ms = Date.UTC(DP.YEAR, m - 1, dd) - DP.START;
      if (t) { const [hh, mm] = t.split(":").map(Number); ms += (hh * 60 + mm) * 60000; }
      return ms / 86400000;
    },
    /* Tooltip: DP.tip(el, html | () => html) */
    tip(el, html) {
      const box = DP._tipBox || (DP._tipBox = document.body.appendChild(DP.h("div", { class: "dp-tip" })));
      const move = (e) => {
        const pad = 14, w = box.offsetWidth, hgt = box.offsetHeight;
        let x = e.clientX + pad, y = e.clientY + pad;
        if (x + w > innerWidth - 8) x = e.clientX - w - pad;
        if (y + hgt > innerHeight - 8) y = e.clientY - hgt - pad;
        box.style.left = x + "px"; box.style.top = y + "px";
      };
      el.addEventListener("mouseenter", (e) => { box.innerHTML = typeof html === "function" ? html() : html; box.classList.add("on"); move(e); });
      el.addEventListener("mousemove", move);
      el.addEventListener("mouseleave", () => box.classList.remove("on"));
      return el;
    },
    ready(fn) {
      if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
      else fn();
    },
  };
  window.DP = DP;

  /* Page behaviour: reveal on scroll, count-up numbers, active nav link.
     window.DP_STATIC = true (set by the screenshot scripts) shows everything in its final state. */
  DP.ready(() => {
    const finalText = (el) => DP.fmt(parseFloat(el.dataset.to), parseInt(el.dataset.digits || "0", 10));
    if (window.DP_STATIC) {
      document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in"));
      document.querySelectorAll(".count[data-to]").forEach((el) => { el.dataset.done = "1"; el.textContent = finalText(el); });
    }
    const io = new IntersectionObserver((entries) => {
      for (const en of entries) {
        if (!en.isIntersecting) continue;
        en.target.classList.add("in");
        en.target.querySelectorAll(".count[data-to]").forEach(countUp);
        if (en.target.matches(".count[data-to]")) countUp(en.target);
        io.unobserve(en.target);
      }
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    document.querySelectorAll(".reveal, .count[data-to]").forEach((el) => io.observe(el));

    function countUp(el) {
      if (el.dataset.done) return; el.dataset.done = "1";
      const to = parseFloat(el.dataset.to), digits = parseInt(el.dataset.digits || "0", 10);
      const dur = 1100, t0 = performance.now();
      const step = (t) => {
        const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
        el.textContent = DP.fmt(to * e, digits);
        if (k < 1) requestAnimationFrame(step); else el.textContent = DP.fmt(to, digits);
      };
      requestAnimationFrame(step);
    }

    const links = [...document.querySelectorAll(".topnav a[href^='#']")];
    const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const navIo = new IntersectionObserver((entries) => {
      for (const en of entries) {
        if (!en.isIntersecting) continue;
        links.forEach((a) => a.classList.remove("active"));
        const a = byId.get(en.target.id); if (a) a.classList.add("active");
      }
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("main > section.sec[id]").forEach((s) => navIo.observe(s));
  });
})();
