// Nova's chat page.
// The browser keeps the conversation in `convo` and sends ALL of it with
// every message, because the server (like the model) remembers nothing.
// Attached files are turned into text and placed inside the user's message.

const $ = id => document.getElementById(id);
const root = document.documentElement;
const chatEl = $("chat");
const messagesEl = $("messages");
const welcomeEl = $("welcome");
const form = $("composer");
const fieldEl = $("field");
const input = $("input");
const sendBtn = $("send");
const jumpBtn = $("jump");
const modelEl = $("model");
const markEl = $("mark");
const accessBox = $("access");
const accessInput = $("access-code");
const attachBtn = $("attach-btn");
const fileInput = $("file-input");
const attachmentsEl = $("attachments");
const dropzone = $("dropzone");
const toastEl = $("toast");
const themeBtn = $("theme-toggle");

const STORE_KEY = "nova-history-v2";
const CODE_KEY = "nova-access-code";
const THEME_KEY = "nova-theme";
const LIB_KEY = "nova-library-v1";
const AGENTS_KEY = "nova-agents";
const MAX_LIB_DOCS = 20;
const MAX_LIB_CHUNKS = 400;            // same limit as the server
const MAX_FILES = 5;
const MAX_BYTES = 4 * 1024 * 1024;     // same limit as the server
const MAX_FILE_CHARS = 15000;          // about 4,000 tokens per file

const SERVER_EXT = new Set(["pdf", "docx"]);   // binary: the server extracts the text
const IMAGE_EXT = new Set(["png", "jpg", "jpeg", "gif", "webp", "heic", "bmp"]);
const TEXT_EXT = new Set(("txt md markdown csv tsv json xml yaml yml html htm css js mjs ts jsx tsx py ipynb " +
  "java c h cpp hpp cs go rs rb php sql sh ps1 bat ini toml cfg log tex r kt swift dart lua vue svelte").split(" "));

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const finePointer = matchMedia("(hover: hover) and (pointer: fine)").matches;

const MARK = '<svg class="mark" viewBox="0 0 24 24" aria-hidden="true"><circle class="m-ring" cx="12" cy="12" r="8.5"/><circle class="m-core" cx="12" cy="12" r="2.8"/><g class="m-sat"><circle cx="20.5" cy="12" r="1.7"/></g></svg>';
const ICON_X = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
const ICON_CHEVRON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>';
const ICON_COPY = '<svg class="i-copy" viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2.5"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>';
const ICON_CHECK = '<svg class="i-check" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';

let convo = load(STORE_KEY, []);     // [{role, content, text?, files?}]
// The user's RAG library: [{name, chars, embedded, chunks: [{section, text, vec}]}].
// The server can't keep files (it's serverless), so the browser keeps them.
let library = load(LIB_KEY, []).filter(d => d && d.name && Array.isArray(d.chunks));
let attachments = [];                // files waiting to be sent
let controller = null;               // lets the Stop button cancel a request
let activeStream = null;             // the reply being revealed right now
let pinned = true;                   // is the chat scrolled to the bottom?
let nextId = 1;
let chatId = 0;                      // changes when a new chat starts

// ---------- helpers ----------
function load(key, fallback) {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
  catch { return fallback; }
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage full or blocked */ }
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function extOf(name) {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}
function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
function scrollDown() { chatEl.scrollTop = chatEl.scrollHeight; }

let toastTimer;
function toast(message) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 3200);
}

// ---------- Markdown ----------
// A small Markdown renderer: headings, lists, tables, quotes, code blocks,
// `code`, **bold**, *italic* and links. All text is HTML-escaped before any
// tags are added, so a reply can never inject HTML into the page.
const RE = {
  heading: /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/,
  hr: /^\s{0,3}([-*_])(\s*\1){2,}\s*$/,
  ul: /^(\s*)[-*+•]\s+(.*)$/,
  ol: /^(\s*)(\d{1,9})[.)]\s+(.*)$/,
  quote: /^\s{0,3}>\s?(.*)$/,
  tableSep: /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/,
};

function inline(raw) {
  const codes = [];
  let s = escapeHtml(raw).replace(/`([^`\n]+)`/g, (_, code) => {
    codes.push(code);                       // park code spans so bold/italic can't touch them
    return `\u0001${codes.length - 1}\u0001`;
  });
  s = s
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*\n]*)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    // Citations become source tags: [D3] -> D3, [Live data] -> Live data
    .replace(/\[(D\d{1,3}|Live data)\]/g, '<span class="ref">$1</span>');
  return s
    .replace(/\u0001(\d+)\u0001/g, (_, i) => `<code>${codes[i] ?? ""}</code>`)
    .replace(/\n/g, "<br>");
}

const cells = line => line.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
const isTableStart = (lines, i) =>
  lines[i].includes("|") && i + 1 < lines.length && lines[i + 1].includes("-") && RE.tableSep.test(lines[i + 1]);

function blocks(text) {
  const lines = text.split("\n");
  const out = [];
  let i = 0, m;

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    if ((m = line.match(RE.heading))) {
      const tag = m[1].length <= 2 ? "h3" : m[1].length === 3 ? "h4" : "h5";
      out.push(`<${tag}>${inline(m[2])}</${tag}>`);
      i++; continue;
    }
    if (RE.hr.test(line)) { out.push("<hr>"); i++; continue; }

    if (isTableStart(lines, i)) {
      const head = cells(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].trim() && lines[i].includes("|")) rows.push(cells(lines[i++]));
      out.push('<div class="table-wrap"><table><thead><tr>' +
        head.map(c => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map(r => "<tr>" + head.map((_, k) => `<td>${inline(r[k] ?? "")}</td>`).join("") + "</tr>").join("") +
        "</tbody></table></div>");
      continue;
    }

    if (RE.ul.test(line) || RE.ol.test(line)) {
      const ordered = RE.ol.test(line);
      const own = ordered ? RE.ol : RE.ul;
      const start = ordered ? Number(line.match(RE.ol)[2]) : 1;
      const items = [];
      while (i < lines.length) {
        const l = lines[i];
        if (!l.trim()) {
          // a blank line ends the list unless more of the list follows it
          let j = i + 1;
          while (j < lines.length && !lines[j].trim()) j++;
          const next = lines[j] ?? "";
          const more = (m = next.match(own)) ? true : /^\s{2,}\S/.test(next);
          if (j < lines.length && more) { i = j; continue; }
          break;
        }
        const any = l.match(RE.ul) || l.match(RE.ol);
        const mine = l.match(own);
        if (!items.length) {
          items.push({ text: mine[mine.length - 1], sub: false });
        } else if (mine && mine[1].length < 2) {
          items.push({ text: mine[mine.length - 1], sub: false });
        } else if (any && any[1].length >= 2) {
          items.push({ text: any[any.length - 1], sub: true });
        } else if (/^\s{2,}\S/.test(l)) {
          items[items.length - 1].text += "\n" + l.trim();     // continuation line
        } else {
          break;
        }
        i++;
      }
      const open = ordered ? `<ol${start !== 1 ? ` start="${start}"` : ""}>` : "<ul>";
      out.push(open + items.map(it => `<li${it.sub ? ' class="sub"' : ""}>${inline(it.text)}</li>`).join("") +
        (ordered ? "</ol>" : "</ul>"));
      continue;
    }

    if (RE.quote.test(line)) {
      const quoted = [];
      while (i < lines.length && (m = lines[i].match(RE.quote))) { quoted.push(m[1]); i++; }
      out.push(`<blockquote>${inline(quoted.join("\n"))}</blockquote>`);
      continue;
    }

    // Anything else is a paragraph: collect lines until a blank line or another block.
    const para = [line];
    i++;
    while (i < lines.length && lines[i].trim() &&
           !RE.heading.test(lines[i]) && !RE.hr.test(lines[i]) && !RE.ul.test(lines[i]) &&
           !RE.ol.test(lines[i]) && !RE.quote.test(lines[i]) && !isTableStart(lines, i)) {
      para.push(lines[i++]);
    }
    out.push(`<p>${inline(para.join("\n"))}</p>`);
  }
  return out.join("");
}

function render(text) {
  return text.replace(/\u0001/g, "").split("```").map((part, i) => {
    if (i % 2 === 0) return blocks(part);
    // Inside a ``` code block. The first line may name the language.
    const nl = part.indexOf("\n");
    const first = (nl === -1 ? part : part.slice(0, nl)).trim();
    const named = /^[\w+#.-]{0,20}$/.test(first);
    const lang = named ? first : "";
    const code = named ? (nl === -1 ? "" : part.slice(nl + 1)) : part;
    return `<div class="code"><div class="code-head"><span>${escapeHtml(lang || "code")}</span>` +
      `<button type="button" class="code-copy">Copy</button></div>` +
      `<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre></div>`;
  }).join("");
}

const renderUser = text => escapeHtml(text).replace(/`([^`\n]+)`/g, "<code>$1</code>");

// ---------- file chips ----------
function chipHTML(a, removable) {
  const label = (a.ext || "file").slice(0, 4).toUpperCase();
  let meta = formatSize(a.size);
  if (a.status === "loading") meta = "Reading…";
  if (a.status === "error") meta = a.error;
  if (a.status === "ready" && a.truncated) meta += a.inLibrary ? " · full text in library" : " · first 15k chars";
  return `<span class="ext">${escapeHtml(label)}</span>
    <span class="info"><span class="fname" title="${escapeHtml(a.name)}">${escapeHtml(a.name)}</span>
    <span class="fmeta">${escapeHtml(meta)}</span></span>` +
    (removable ? `<button type="button" class="remove" aria-label="Remove ${escapeHtml(a.name)}">${ICON_X}</button>` : "");
}

function upsertChip(a) {
  let el = attachmentsEl.querySelector(`[data-id="${a.id}"]`);
  if (!el) {
    el = document.createElement("div");
    el.dataset.id = a.id;
    attachmentsEl.appendChild(el);
  }
  el.className = `file-chip ${a.status} ${a.ext}`;
  el.innerHTML = chipHTML(a, true);
  el.querySelector(".remove").addEventListener("click", () => removeAttachment(a.id));
  updateSendState();
}

function removeAttachment(id) {
  attachments = attachments.filter(a => a.id !== id);
  const el = attachmentsEl.querySelector(`[data-id="${id}"]`);
  if (el) {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 200);
  }
  updateSendState();
}

// ---------- reading files ----------
async function readFile(file, a) {
  if (file.size > MAX_BYTES) throw new Error("Too big (max 4 MB)");
  if (file.size === 0) throw new Error("File is empty");

  let text;
  if (SERVER_EXT.has(a.ext)) {
    // PDF / Word: send the raw bytes to the server, get text back.
    const res = await fetch(`/api/extract?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Access-Code": accessInput.value },
      body: file,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      if (res.status === 401) accessBox.classList.remove("hidden");
      throw new Error(typeof data.detail === "string" ? data.detail : `Error ${res.status}`);
    }
    text = data.text;
    a.truncated = data.truncated;
  } else if (TEXT_EXT.has(a.ext) || file.type.startsWith("text/")) {
    text = await file.text();                       // the browser reads it directly
    if (text.includes("\u0000")) throw new Error("Not a text file");
  } else if (IMAGE_EXT.has(a.ext) || file.type.startsWith("image/")) {
    throw new Error("Images aren't supported yet");
  } else {
    throw new Error("Unsupported file type");
  }

  text = text.trim();
  if (!text) throw new Error("No readable text");
  if (text.length > MAX_FILE_CHARS) {
    text = text.slice(0, MAX_FILE_CHARS);
    a.truncated = true;
  }
  a.text = text;
  // Too long to fit in one message? Index the WHOLE file for RAG, so Nova's
  // Researcher can search all of it instead of only the first 15k characters.
  if (a.truncated) {
    a.inLibrary = true;
    addToLibrary(file);
  }
}

function addFiles(fileList) {
  for (const file of fileList) {
    if (attachments.length >= MAX_FILES) {
      toast(`Up to ${MAX_FILES} files per message.`);
      break;
    }
    const a = { id: nextId++, name: file.name, size: file.size, ext: extOf(file.name),
                status: "loading", text: "", truncated: false };
    attachments.push(a);
    upsertChip(a);
    readFile(file, a)
      .then(() => { a.status = "ready"; })
      .catch(err => {
        a.status = "error";
        a.error = err.message;
        setTimeout(() => removeAttachment(a.id), 5000);   // error chips fade away
      })
      .finally(() => { if (attachments.includes(a)) upsertChip(a); });
  }
  input.focus();
}

// The text the MODEL sees: each file wrapped in tags, then the question.
function buildContent(text, files) {
  if (!files.length) return text;
  const blocksOfText = files.map(f => {
    const note = f.inLibrary ? ' note="first 15,000 characters only; the full document is in the library, searchable with search_documents"'
      : f.truncated ? ' note="only the first 15,000 characters"' : "";
    return `<file name="${f.name.replace(/"/g, "'")}"${note}>\n${f.text}\n</file>`;
  });
  const ask = text || (files.length > 1
    ? "Please read these files and give me a short summary of each."
    : "Please read this file and give me a short summary.");
  return `${blocksOfText.join("\n\n")}\n\n${ask}`;
}

// ---------- drawing messages ----------
function hideWelcome() { welcomeEl.classList.add("hidden"); }

function addUser(item) {
  hideWelcome();
  const el = document.createElement("article");
  el.className = "msg user";
  if (item.files?.length) {
    const files = document.createElement("div");
    files.className = "files";
    item.files.forEach(f => {
      const chip = document.createElement("div");
      chip.className = `file-chip ready ${f.ext}`;
      chip.innerHTML = chipHTML({ ...f, status: "ready" }, false);
      files.appendChild(chip);
    });
    el.appendChild(files);
  }
  if (item.text) {
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = renderUser(item.text);
    el.appendChild(bubble);
  }
  messagesEl.appendChild(el);
  pinned = true;
  scrollDown();
}

function addBot(text = "") {
  hideWelcome();
  const el = document.createElement("article");
  el.className = "msg bot";
  el.innerHTML = `<span class="avatar">${MARK}</span>
    <div class="content"><div class="who">Nova</div><div class="tools"></div><div class="body"></div>
    <div class="actions"><button type="button" class="copy" aria-label="Copy reply">
      <span class="icons">${ICON_COPY}${ICON_CHECK}</span><span class="label">Copy</span></button></div></div>`;
  const body = el.querySelector(".body");
  let raw = text;
  body.innerHTML = render(raw);

  const copyBtn = el.querySelector(".copy");
  copyBtn.addEventListener("click", () => copyText(raw, copyBtn, copyBtn.querySelector(".label")));

  messagesEl.appendChild(el);
  scrollDown();
  return {
    el, body,
    thinking() { body.innerHTML = '<span class="thinking">Thinking…</span>'; },
    // Work log: one numbered step per agent action ("Analyst → calculating 12000 * 12").
    tool(label) {
      const tools = el.querySelector(".tools");
      let log = tools.querySelector(".log");
      if (!log) {
        tools.innerHTML = `<button type="button" class="log-toggle" aria-expanded="false">${ICON_CHEVRON}<span></span></button><ol class="log"></ol>`;
        log = tools.querySelector(".log");
        tools.querySelector(".log-toggle").addEventListener("click", () => {
          const open = tools.classList.toggle("open");
          tools.querySelector(".log-toggle").setAttribute("aria-expanded", String(open));
        });
      }
      log.querySelectorAll(".step.running").forEach(s => s.classList.remove("running"));
      const m = label.match(/^(Planner|Analyst|Researcher|Writer|Reviewer)\s*(?:→|:)\s*(.*)$/);
      const step = document.createElement("li");
      step.className = "step running";
      const num = document.createElement("span");
      num.className = "ref";
      num.textContent = String(log.children.length + 1);
      step.append(num);
      if (m) {
        const agent = document.createElement("span");
        agent.className = "agent";
        agent.textContent = m[1];
        step.append(agent);
      }
      const detail = document.createElement("span");
      detail.className = "detail";
      detail.textContent = m ? m[2] : label;  // textContent: never parsed as HTML
      detail.title = label;
      step.append(detail);
      log.append(step);
      if (pinned) scrollDown();
    },
    toolsDone() { el.querySelectorAll(".step.running").forEach(s => s.classList.remove("running")); },
    // When the answer is finished, fold the log into one line ("6 steps").
    finishLog() {
      const tools = el.querySelector(".tools");
      const n = tools.querySelectorAll(".step").length;
      if (!n) return;
      tools.querySelector(".log-toggle span").textContent = `${n} step${n > 1 ? "s" : ""}`;
      tools.classList.add("done");
    },
    set(text) { raw = text; body.innerHTML = render(text); },
    fail(message) { el.classList.add("error"); body.textContent = message; },
  };
}

async function copyText(text, button, label) {
  try {
    await navigator.clipboard.writeText(text);
    const original = label.textContent;
    button.classList.add("done");
    label.textContent = "Copied";
    setTimeout(() => { button.classList.remove("done"); label.textContent = original; }, 1600);
  } catch { toast("Couldn't copy. Select the text instead."); }
}

// Copy buttons inside code blocks (the blocks are redrawn while streaming, so listen on the parent).
messagesEl.addEventListener("click", e => {
  const btn = e.target.closest(".code-copy");
  if (btn) copyText(btn.closest(".code").querySelector("code").textContent, btn, btn);
});

// ---------- smooth streaming ----------
// Groq sends text in fast, uneven bursts. Instead of dumping each burst on the
// page, we reveal the text at a steady pace and fade each new piece in.
function createStreamer(bot) {
  const FADE = 380;                 // ms each new piece takes to fade in
  let target = "";                  // everything received so far
  let shown = 0;                    // how many characters are on screen
  let ended = false;
  let waiting = false, rafId = 0, timer = 0;
  let lastFrame = performance.now();
  let lastTotal = 0, base = 0;
  let stamps = [];                  // [{end, t}] which characters appeared when
  let resolve;
  const finished = new Promise(r => { resolve = r; });

  function paint(now) {
    const body = bot.body;
    body.innerHTML = render(target.slice(0, shown));
    if (reduceMotion) return;

    // Find every piece of visible text and its position.
    const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT, {
      acceptNode: n => (n.parentElement.closest(".code-head") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    const nodes = [];
    let total = 0;
    while (walker.nextNode()) {
      nodes.push([walker.currentNode, total]);
      total += walker.currentNode.data.length;
    }
    if (total > lastTotal) stamps.push({ end: total, t: now });
    lastTotal = total;
    while (stamps.length && now - stamps[0].t > FADE) base = stamps.shift().end;
    base = Math.min(base, total);
    if (!stamps.length) return;

    // Wrap the recently revealed text in spans that are part-way through fading in.
    for (const [node, start] of nodes) {
      const end = start + node.data.length;
      if (end <= base) continue;
      const frag = document.createDocumentFragment();
      let pos = start;
      if (base > start) { frag.append(node.data.slice(0, base - start)); pos = base; }
      let prev = base;
      for (const s of stamps) {
        const a = Math.max(prev, pos), b = Math.min(s.end, end);
        if (b > a) {
          const span = document.createElement("span");
          span.className = "fresh";
          span.style.animationDelay = `${-(now - s.t)}ms`;
          span.textContent = node.data.slice(a - start, b - start);
          frag.append(span);
          pos = b;
        }
        prev = Math.max(prev, s.end);
      }
      if (pos < end) frag.append(node.data.slice(pos - start));
      node.replaceWith(frag);
    }
  }

  function frame() {
    if (!waiting) return;
    waiting = false;
    cancelAnimationFrame(rafId);
    clearTimeout(timer);

    const now = performance.now();
    const ticks = Math.min(30, Math.max(1, (now - lastFrame) / 16.7));   // 1 tick = one 60 fps frame
    lastFrame = now;
    const backlog = target.length - shown;
    if (backlog > 0) {
      // Reveal faster when a lot is waiting, slower near the end. Based on time
      // passed, so the pace is the same whether frames come fast or slowly.
      const step = Math.max(2 * ticks, backlog * (1 - Math.pow(0.93, ticks)));
      shown += reduceMotion ? backlog : Math.ceil(step);
      shown = Math.min(shown, target.length);
      const last = target.charCodeAt(shown - 1);
      if (last >= 0xD800 && last <= 0xDBFF && shown < target.length) shown++;   // don't split an emoji
    }
    if (backlog > 0 || stamps.length) {
      paint(now);
      if (pinned) scrollDown();
    }
    if (shown < target.length || stamps.length || !ended) schedule();
    else { bot.set(target); if (pinned) scrollDown(); resolve(); }
  }

  function schedule() {
    if (waiting) return;
    waiting = true;
    // Browsers pause requestAnimationFrame when the window is hidden or minimized,
    // so a timer runs as a backup. Whichever fires first draws the next frame.
    rafId = requestAnimationFrame(frame);
    timer = setTimeout(frame, 120);
  }

  return {
    finished,
    push(chunk) {
      target += chunk;
      if (!waiting) lastFrame = performance.now();   // start the clock at the first piece
      schedule();
    },
    end() { ended = true; schedule(); },
    // Throw away everything shown so far (the draft) and start again.
    reset() {
      target = ""; shown = 0; stamps = []; base = 0; lastTotal = 0;
      bot.body.innerHTML = '<span class="thinking">Revising…</span>';
    },
    stop() { target = target.slice(0, shown); ended = true; schedule(); },
    text: () => target,
  };
}

// ---------- sending ----------
function setBusy(busy) {
  sendBtn.classList.toggle("stop", busy);
  sendBtn.setAttribute("aria-label", busy ? "Stop" : "Send");
  markEl.classList.toggle("busy", busy);
  fieldEl.classList.toggle("busy", busy);
  updateSendState();
}

function updateSendState() {
  const hasText = input.value.trim().length > 0;
  const hasFiles = attachments.some(a => a.status === "ready");
  sendBtn.disabled = !controller && !hasText && !hasFiles;
}

async function send(text) {
  if (controller) return;
  if (attachments.some(a => a.status === "loading")) { toast("Still reading your files…"); return; }
  const files = attachments.filter(a => a.status === "ready");
  if (!text && !files.length) return;

  const item = {
    role: "user",
    content: buildContent(text, files),
    text,
    files: files.map(({ name, size, ext, truncated }) => ({ name, size, ext, truncated })),
  };
  convo.push(item);
  attachments.forEach(a => removeAttachment(a.id));
  addUser(item);

  const thisChat = chatId;
  const bot = addBot();
  bot.thinking();
  const stream = createStreamer(bot);
  activeStream = stream;
  controller = new AbortController();
  setBusy(true);

  let failure = null;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Access-Code": accessInput.value },
      body: JSON.stringify({
        messages: convo.map(({ role, content }) => ({ role, content })),
        library: library.filter(d => d.chunks.length).map(({ name, chunks }) => ({ name, chunks })),
        agents: agentsToggle.checked,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      if (res.status === 401) accessBox.classList.remove("hidden");
      throw new Error(typeof err.detail === "string" ? err.detail : `Something went wrong (error ${res.status}).`);
    }

    // Read the streamed reply piece by piece, like the for-loop in chatbot.py.
    // The server marks tool activity as "\x02label\x03" inside the text stream.
    // Those markers become status chips; everything else is the reply.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!bot.el.classList.contains("streaming")) bot.el.classList.add("streaming");
      pending += decoder.decode(value, { stream: true });
      // "\x04" means the Reviewer asked for a revision: the new answer replaces the draft.
      let out = "";
      while (true) {
        const start = pending.search(/[\x02\x04]/);
        if (start === -1) { out += pending; pending = ""; break; }
        out += pending.slice(0, start);
        if (pending[start] === "\x04") {
          if (out) { stream.push(out); out = ""; }
          stream.reset();
          pending = pending.slice(start + 1);
          continue;
        }
        const end = pending.indexOf("\x03", start);
        if (end === -1) { pending = pending.slice(start); break; }   // marker split across chunks
        bot.tool(pending.slice(start + 1, end));
        pending = pending.slice(end + 1);
      }
      if (out) { bot.toolsDone(); stream.push(out); }
    }
    bot.toolsDone();
    stream.end();
  } catch (err) {
    stream.stop();
    if (err.name !== "AbortError") failure = err.message;
  }

  await stream.finished;                  // wait until everything received is on screen
  const reply = stream.text();
  bot.el.classList.remove("streaming");
  bot.finishLog();

  if (thisChat !== chatId) {
    // "New chat" was pressed while Nova was replying: this answer belongs to the old chat.
  } else if (reply.trim() && !failure) {
    convo.push({ role: "assistant", content: reply });   // remember Nova's answer (or the part shown)
    save(STORE_KEY, convo);
  } else {
    convo.pop();                          // the question was never answered, like messages.pop()
    bot.fail(failure || (stream.stopped ? "Stopped" : "Nova returned an empty reply. Try again."));
  }

  controller = null;
  activeStream = null;
  setBusy(false);
  if (finePointer) input.focus();
}

function stopReply() {
  if (activeStream) { activeStream.stopped = true; activeStream.stop(); }
  if (controller) controller.abort();
}

// ---------- library (RAG over the user's documents) ----------
const libraryEl = $("library");
const libraryScrim = $("library-scrim");
const libraryList = $("library-list");
const libraryInput = $("library-input");
const libraryCount = $("library-count");
const agentsToggle = $("agents-toggle");
const ICON_TRASH = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/></svg>';
const pendingDocs = new Map();   // name -> "Indexing…" or an error, while uploading

function libraryChunkCount() { return library.reduce((n, d) => n + d.chunks.length, 0); }

function saveLibrary() {
  try { localStorage.setItem(LIB_KEY, JSON.stringify(library)); }
  catch { toast("Your browser's storage is full. Remove a document from the library."); }
}

function renderLibrary() {
  const items = [
    ...library.map(d => ({ name: d.name, state: "ready",
      sub: `${d.chunks.length} chunks · ${d.chars >= 1000 ? Math.round(d.chars / 1000) + "k" : d.chars} characters` +
           (d.embedded ? "" : " · keyword search only") + (d.truncated ? " · trimmed" : "") })),
    ...[...pendingDocs].map(([name, s]) => ({ name, state: s === "busy" ? "busy" : "error",
      sub: s === "busy" ? "Indexing…" : s })),
  ];
  libraryList.innerHTML = items.map(it =>
    `<li class="lib-item ${it.state}"><span class="meta"><span class="nm" title="${escapeHtml(it.name)}">${escapeHtml(it.name)}</span>` +
    `<span class="sub">${escapeHtml(it.sub)}</span></span>` +
    (it.state === "ready" ? `<button type="button" data-name="${escapeHtml(it.name)}" aria-label="Remove ${escapeHtml(it.name)}">${ICON_TRASH}</button>` : "") +
    `</li>`).join("");
  $("library-empty").classList.toggle("hidden", items.length > 0);
  libraryCount.textContent = String(library.length);
  libraryCount.classList.toggle("hidden", library.length === 0);
}

async function addToLibrary(file) {
  if (library.length >= MAX_LIB_DOCS) { toast(`The library holds up to ${MAX_LIB_DOCS} documents.`); return; }
  if (file.size > MAX_BYTES) { toast(`${file.name}: too big (max 4 MB)`); return; }
  pendingDocs.set(file.name, "busy");
  renderLibrary();
  try {
    // The server reads the file, splits it into chunks and embeds them.
    const res = await fetch(`/api/library?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Access-Code": accessInput.value },
      body: file,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      if (res.status === 401) accessBox.classList.remove("hidden");
      throw new Error(typeof data.detail === "string" ? data.detail : `Error ${res.status}`);
    }
    const room = MAX_LIB_CHUNKS - libraryChunkCount() + (library.find(d => d.name === data.name)?.chunks.length || 0);
    if (room <= 0) throw new Error("Library is full. Remove a document first.");
    library = library.filter(d => d.name !== data.name);          // re-adding replaces
    library.push({ name: data.name, chars: data.chars, embedded: data.embedded,
                   truncated: data.truncated || data.chunks.length > room, chunks: data.chunks.slice(0, room) });
    saveLibrary();
    pendingDocs.delete(file.name);
    toast(`Added ${file.name} to your library`);
  } catch (err) {
    pendingDocs.set(file.name, err.message);
    setTimeout(() => { pendingDocs.delete(file.name); renderLibrary(); }, 6000);
  }
  renderLibrary();
}

function openLibrary() {
  libraryEl.classList.remove("hidden");
  libraryScrim.classList.remove("hidden");
  $("library-close").focus();
}
function closeLibrary() {
  libraryEl.classList.add("hidden");
  libraryScrim.classList.add("hidden");
  $("library-btn").focus();
}

$("library-btn").addEventListener("click", openLibrary);
$("library-close").addEventListener("click", closeLibrary);
libraryScrim.addEventListener("click", closeLibrary);
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !libraryEl.classList.contains("hidden")) closeLibrary();
});
$("library-add").addEventListener("click", () => libraryInput.click());
libraryInput.addEventListener("change", () => {
  [...libraryInput.files].forEach(addToLibrary);
  libraryInput.value = "";
});
libraryList.addEventListener("click", e => {
  const btn = e.target.closest("button[data-name]");
  if (!btn) return;
  library = library.filter(d => d.name !== btn.dataset.name);
  saveLibrary();
  renderLibrary();
});

agentsToggle.checked = load(AGENTS_KEY, true) !== false;
agentsToggle.addEventListener("change", () => save(AGENTS_KEY, agentsToggle.checked));
renderLibrary();

// ---------- events ----------
form.addEventListener("submit", e => {
  e.preventDefault();
  if (controller) { stopReply(); return; }          // the button is "Stop" while busy
  const text = input.value.trim();
  if (!text && !attachments.some(a => a.status === "ready")) return;
  input.value = "";
  autoGrow();
  send(text);
});

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
  updateSendState();
}
input.addEventListener("input", autoGrow);
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!controller) form.requestSubmit();
  }
});

// Paste a file straight into the message box
input.addEventListener("paste", e => {
  const files = [...(e.clipboardData?.files || [])];
  if (files.length) { e.preventDefault(); addFiles(files); }
});

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { addFiles([...fileInput.files]); fileInput.value = ""; });

// Drag and drop anywhere on the page
let dragDepth = 0;
const hasFiles = e => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", e => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  dropzone.classList.add("show");
});
window.addEventListener("dragover", e => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("dragleave", e => {
  if (!hasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) dropzone.classList.remove("show");
});
window.addEventListener("drop", e => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  dropzone.classList.remove("show");
  addFiles([...e.dataTransfer.files]);
});

document.querySelectorAll(".suggestion").forEach(btn =>
  btn.addEventListener("click", () => send(btn.dataset.prompt))
);

$("new-chat").addEventListener("click", () => {
  stopReply();
  chatId++;
  convo = [];
  save(STORE_KEY, convo);
  const old = [...messagesEl.children];
  old.forEach(el => el.classList.add("leaving"));
  setTimeout(() => {
    old.forEach(el => el.remove());
    welcomeEl.classList.remove("hidden", "play");
    void welcomeEl.offsetWidth;          // restart the welcome animation
    welcomeEl.classList.add("play");
    jumpBtn.classList.remove("show");
    pinned = true;
    if (finePointer) input.focus();
  }, old.length ? 280 : 0);
});

// Follow new text only while the reader is at the bottom of the chat.
chatEl.addEventListener("scroll", () => {
  pinned = chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 90;
  jumpBtn.classList.toggle("show", !pinned && messagesEl.children.length > 0);
}, { passive: true });
jumpBtn.addEventListener("click", () => { pinned = true; scrollDown(); });

accessInput.value = load(CODE_KEY, "");
accessInput.addEventListener("input", () => save(CODE_KEY, accessInput.value));

// ---------- theme ----------
function paintThemeColor() {
  $("theme-color").content = getComputedStyle(root).getPropertyValue("--bg").trim();
}
themeBtn.addEventListener("click", () => {
  const next = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  save(THEME_KEY, next);
  paintThemeColor();
});
// Follow the system theme until the visitor picks one themselves.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", e => {
  if (load(THEME_KEY, null)) return;
  root.dataset.theme = e.matches ? "dark" : "light";
  paintThemeColor();
});
paintThemeColor();

// ---------- start-up ----------
if (convo.length) {
  convo.forEach(m => (m.role === "user" ? addUser(m) : addBot(m.content)));
  // A restored chat should simply be there, not animate in again.
  [...messagesEl.children].forEach(el => { el.style.animation = "none"; });
  scrollDown();
}

fetch("/api/health")
  .then(r => r.json())
  .then(info => {
    modelEl.textContent = info.ok ? info.model.split("/").pop() : "missing API key";
    if (info.access_code_required) accessBox.classList.remove("hidden");
  })
  .catch(() => { modelEl.textContent = "offline"; });

updateSendState();
if (finePointer) input.focus();
