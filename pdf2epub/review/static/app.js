"use strict";
// pdf2epub review screen: vanilla JS, no build step.
const $ = (id) => document.getElementById(id);
let book = null, current = null, view = null, linking = null;

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  if (!res.ok) { const msg = await res.text(); alert(msg); throw new Error(msg); }
  return res.json();
}

function visiblePages() {
  const todo = $("filter").value === "todo";
  return book.pages.filter((p) => !todo || (p.status !== "clean" && p.status !== "reviewed"));
}

function renderList() {
  const nav = $("pages"); nav.innerHTML = "";
  const counts = {};
  for (const p of book.pages) counts[p.status] = (counts[p.status] || 0) + 1;
  $("counts").textContent = Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join(" · ");
  for (const p of visiblePages()) {
    const a = document.createElement("a");
    a.href = "#" + p.index; a.textContent = `${p.index}${p.printed_page ? " (" + p.printed_page + ")" : ""}`;
    a.className = p.status + (p.index === current ? " current" : "");
    a.title = p.codes.join(", ");
    a.onclick = (e) => { e.preventDefault(); load(p.index); };
    nav.appendChild(a);
  }
}

async function load(i) {
  current = i; location.hash = i;
  view = await api(`/api/page/${i}`);
  show();
  renderList();
}

function show() {
  $("page-label").textContent = `${view.index}${view.printed_page ? " / " + view.printed_page : ""} · ${view.status}`;
  $("page-image").src = view.image + "?t=" + Date.now();
  $("main").innerHTML = view.html.main;
  $("notes").innerHTML = view.html.notes;
  $("flags").innerHTML = view.flags.map((f) => `<div class="${f.severity}">${f.code}: ${escapeHtml(f.message)}</div>`).join("");
  $("status").textContent = view.reviewed_file ? "محفوظة في المراجعة" : "";
  $("editor").hidden = true; $("text").hidden = false;
  drawBoxes(); wireLinks();
}

function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function drawBoxes() {
  const img = $("page-image"), boxes = $("boxes");
  const draw = () => {
    boxes.innerHTML = "";
    if (!view.width) return;
    const s = img.clientWidth / view.width;
    for (const r of view.regions) {
      if (!r.bbox) continue;
      const d = document.createElement("div");
      d.className = r.kind;
      Object.assign(d.style, { left: r.bbox[0] * s + "px", top: r.bbox[1] * s + "px", width: (r.bbox[2] - r.bbox[0]) * s + "px", height: (r.bbox[3] - r.bbox[1]) * s + "px" });
      d.innerHTML = `<span>${r.kind}${r.pos ? " · " + r.pos : ""}</span>`;
      boxes.appendChild(d);
    }
  };
  img.onload = draw; if (img.complete) draw();
}

function line(a, b) {
  const svg = $("lines"), box = $("text").getBoundingClientRect();
  const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
  svg.innerHTML = `<line x1="${ra.left - box.left + ra.width / 2}" y1="${ra.top - box.top + ra.height / 2}" x2="${rb.left - box.left + rb.width / 2}" y2="${rb.top - box.top + rb.height / 2}"/>`;
}

function wireLinks() {
  document.querySelectorAll(".note").forEach((n) => {
    n.onmouseenter = () => {
      const ref = document.querySelector(`sup.ref[data-ref="${n.dataset.ref}"]`);
      n.classList.add("hl");
      if (ref) { ref.classList.add("hl"); line(n.querySelector(".note-head"), ref); }
    };
    n.onmouseleave = () => { document.querySelectorAll(".hl").forEach((e) => e.classList.remove("hl")); $("lines").innerHTML = ""; };
  });
  document.querySelectorAll("sup.ref").forEach((r) => {
    r.onmouseenter = () => {
      const n = document.querySelector(`.note[data-note="${r.dataset.note}"]`);
      r.classList.add("hl");
      if (n) { n.classList.add("hl"); line(r, n.querySelector(".note-head")); }
    };
    r.onmouseleave = () => { document.querySelectorAll(".hl").forEach((e) => e.classList.remove("hl")); $("lines").innerHTML = ""; };
  });
  document.querySelectorAll(".linkbtn").forEach((b) => { b.onclick = () => { linking = b.dataset.note; $("hint").hidden = false; }; });
}

function selectionOffsets() {
  const sel = window.getSelection();
  if (!sel.rangeCount) return null;
  const range = sel.getRangeAt(0);
  const pos = (node, offset) => {
    const span = (node.nodeType === 3 ? node.parentElement : node).closest("[data-off]");
    if (!span) return null;
    return { block: span.closest("[data-block]").dataset.block, off: Number(span.dataset.off) + (node.nodeType === 3 ? offset : 0) };
  };
  const a = pos(range.startContainer, range.startOffset), b = pos(range.endContainer, range.endOffset);
  if (!a || !b || a.block !== b.block) return null;
  return { block: a.block, start: Math.min(a.off, b.off), end: Math.max(a.off, b.off) };
}

$("link-ok").onclick = async () => {
  const s = selectionOffsets();
  if (!linking || !s || s.end <= s.start) { alert("اختر كلمات من فقرة واحدة في النص"); return; }
  view = await api(`/api/page/${current}/link`, { note: linking, block: s.block, start: s.start, end: s.end });
  linking = null; $("hint").hidden = true; show(); refreshBook();
};
$("link-cancel").onclick = () => { linking = null; $("hint").hidden = true; };
$("approve").onclick = async () => { view = await api(`/api/page/${current}/approve`, {}); show(); await refreshBook(); next(); };
$("revert").onclick = async () => { if (confirm("حذف تعديلات المراجعة لهذه الصفحة؟")) { view = await api(`/api/page/${current}/revert`, {}); show(); refreshBook(); } };
$("edit").onclick = () => { $("tags").value = view.tags; $("editor").hidden = false; $("text").hidden = true; };
$("cancel").onclick = () => { $("editor").hidden = true; $("text").hidden = false; };
$("save").onclick = async () => { view = await api(`/api/page/${current}/tags`, { tags: $("tags").value }); show(); refreshBook(); };
$("filter").onchange = renderList;
$("prev").onclick = () => step(-1);
$("next").onclick = () => step(1);
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowLeft") step(1);
  if (e.key === "ArrowRight") step(-1);
});

function step(d) {
  const list = visiblePages().map((p) => p.index);
  const k = list.indexOf(current);
  const target = list[k + d] ?? list[d > 0 ? 0 : list.length - 1];
  if (target !== undefined) load(target);
}
function next() { step(1); }
async function refreshBook() { book = await api("/api/book"); renderList(); }

(async () => {
  book = await api("/api/book");
  $("book-title").textContent = book.title;
  const first = Number(location.hash.slice(1)) || (visiblePages()[0] || book.pages[0] || {}).index;
  if (first) await load(first); else renderList();
})();
