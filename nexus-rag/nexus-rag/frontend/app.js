/* app.js — Nexus frontend logic */

const API = "http://localhost:7860/api";

// ─── State ─────────────────────────────────────────────────────────────
let conversationHistory = [];
let isStreaming         = false;

// ─── Panel Navigation ─────────────────────────────────────────────────
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const panel = btn.dataset.panel;
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`panel-${panel}`).classList.add("active");

    const titles = {
      chat:   "💬 AI Chat",
      search: "🔍 Semantic Search",
      ingest: "📥 Ingest Documents",
      admin:  "⚙️ Admin",
    };
    document.getElementById("topbar-title").textContent = titles[panel] || panel;

    if (panel === "admin") loadAdminStats();
  });
});

// ─── Health Check ──────────────────────────────────────────────────────
async function checkHealth() {
  const dot  = document.getElementById("endee-status");
  const text = document.getElementById("endee-status-text");
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      dot.className  = "status-dot online";
      text.textContent = "Nexus API • Online";
    } else throw new Error();
  } catch {
    dot.className  = "status-dot";
    text.textContent = "Nexus API • Offline";
  }
}

checkHealth();
setInterval(checkHealth, 15000);

// ─── Suggestion Chips ─────────────────────────────────────────────────
document.querySelectorAll(".suggestion-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    document.getElementById("chat-input").value = chip.textContent.trim();
    sendChat();
  });
});

// ─── Chat ──────────────────────────────────────────────────────────────
const chatInput = document.getElementById("chat-input");
const sendBtn   = document.getElementById("send-btn");
const chatMsgs  = document.getElementById("chat-messages");

chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

// Auto-resize textarea
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
});

sendBtn.addEventListener("click", sendChat);

async function sendChat() {
  const text = chatInput.value.trim();
  if (!text || isStreaming) return;

  // Clear welcome screen
  document.getElementById("welcome").style.display = "none";

  // User bubble
  appendMessage("user", text);
  conversationHistory.push({ role: "user", content: text });
  chatInput.value = "";
  chatInput.style.height = "auto";

  // Assistant bubble (streaming)
  const { bubble, searchArea, sourcesArea } = appendAssistantBubble();

  isStreaming = true;
  sendBtn.disabled = true;

  const category = document.getElementById("filter-category").value;
  const topK     = parseInt(document.getElementById("filter-topk").value) || 5;

  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message:              text,
        conversation_history: conversationHistory.slice(-10).slice(0, -1),
        filters:              category ? { category } : null,
        top_k:                topK,
        hybrid_alpha:         0.7,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const dec    = new TextDecoder();
    let fullText = "";
    let sources  = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const raw = dec.decode(value, { stream: true });
      for (const line of raw.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (!data) continue;

        let evt;
        try { evt = JSON.parse(data); } catch { continue; }

        if (evt.type === "search_start") {
          searchArea.innerHTML = `<span class="spinner"></span> Searching: <em>${escapeHtml(evt.query)}</em>`;
          searchArea.style.display = "flex";
        }

        else if (evt.type === "chunk_found") {
          searchArea.style.display = "none";
          sources = sources.concat(evt.chunks || []);
          renderSources(sourcesArea, sources);
        }

        else if (evt.type === "text") {
          fullText += evt.delta;
          bubble.innerHTML = renderMarkdown(fullText) + '<span class="typing-cursor">▌</span>';
          chatMsgs.scrollTop = chatMsgs.scrollHeight;
        }

        else if (evt.type === "done") {
          bubble.innerHTML = renderMarkdown(fullText);
          searchArea.style.display = "none";
          conversationHistory.push({ role: "assistant", content: fullText });
        }

        else if (evt.type === "error") {
          bubble.innerHTML = `<span style="color:var(--red)">⚠ Error: ${escapeHtml(evt.message)}</span>`;
        }
      }
    }
  } catch (err) {
    bubble.innerHTML = `<span style="color:var(--red)">⚠ Failed to connect to Nexus API. Is the backend running?</span>`;
    console.error(err);
  } finally {
    isStreaming     = false;
    sendBtn.disabled = false;
    chatMsgs.scrollTop = chatMsgs.scrollHeight;
  }
}

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="avatar">${role === "user" ? "👤" : "🧠"}</div>
    <div class="message-body">
      <div class="message-meta">${role === "user" ? "You" : "Nexus AI"} · ${now()}</div>
      <div class="bubble">${renderMarkdown(text)}</div>
    </div>
  `;
  chatMsgs.appendChild(div);
  chatMsgs.scrollTop = chatMsgs.scrollHeight;
}

function appendAssistantBubble() {
  const div = document.createElement("div");
  div.className = "message assistant";
  div.innerHTML = `
    <div class="avatar">🧠</div>
    <div class="message-body">
      <div class="message-meta">Nexus AI · ${now()}</div>
      <div class="search-activity" style="display:none"></div>
      <div class="bubble"><span class="spinner"></span></div>
      <div class="sources"></div>
    </div>
  `;
  chatMsgs.appendChild(div);
  chatMsgs.scrollTop = chatMsgs.scrollHeight;
  return {
    bubble:      div.querySelector(".bubble"),
    searchArea:  div.querySelector(".search-activity"),
    sourcesArea: div.querySelector(".sources"),
  };
}

function renderSources(container, chunks) {
  if (!chunks.length) return;
  const unique = [...new Map(chunks.map(c => [c.title, c])).values()];
  container.innerHTML = unique.slice(0, 4).map(c => `
    <span class="source-chip" title="${escapeHtml(c.title)} | Score: ${c.similarity}">
      📄 ${escapeHtml(c.title.length > 30 ? c.title.slice(0, 30) + "…" : c.title)}
    </span>
  `).join("");
}

// ─── Search Panel ──────────────────────────────────────────────────────
document.getElementById("search-btn").addEventListener("click", doSearch);
document.getElementById("search-input").addEventListener("keydown", e => {
  if (e.key === "Enter") doSearch();
});

async function doSearch() {
  const query = document.getElementById("search-input").value.trim();
  if (!query) return;

  const category    = document.getElementById("search-category").value;
  const topK        = parseInt(document.getElementById("search-topk").value) || 8;
  const resultsDiv  = document.getElementById("search-results");

  resultsDiv.innerHTML = `<div class="empty-state"><div class="spinner"></div><p>Searching…</p></div>`;

  try {
    const res = await fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k:   topK,
        filters: category ? { category } : null,
      }),
    });

    const data = await res.json();

    if (!data.results || data.results.length === 0) {
      resultsDiv.innerHTML = `<div class="empty-state"><div class="ei">🔍</div><p>No results found for "<strong>${escapeHtml(query)}</strong>"</p></div>`;
      return;
    }

    resultsDiv.innerHTML = data.results.map(r => `
      <div class="result-card">
        <div class="result-header">
          <div class="result-title">${escapeHtml(r.title)}</div>
          <div class="result-score">${(r.similarity * 100).toFixed(1)}%</div>
        </div>
        <div class="result-meta">
          <span class="tag cat">📁 ${escapeHtml(r.category)}</span>
          <span class="tag src">🔗 ${escapeHtml(r.source)}</span>
          ${r.author ? `<span class="tag">✍️ ${escapeHtml(r.author)}</span>` : ""}
        </div>
        <div class="result-excerpt">${escapeHtml(r.excerpt)}</div>
      </div>
    `).join("");

  } catch (err) {
    resultsDiv.innerHTML = `<div class="empty-state" style="color:var(--red)"><div class="ei">⚠️</div><p>Search failed: ${escapeHtml(err.message)}</p></div>`;
  }
}

// ─── Ingest Panel ──────────────────────────────────────────────────────
document.getElementById("ingest-form-el").addEventListener("submit", async e => {
  e.preventDefault();
  const result  = document.getElementById("ingest-result");
  const text     = document.getElementById("doc-text").value.trim();
  const title    = document.getElementById("doc-title").value.trim();
  const category = document.getElementById("doc-category").value;
  const source   = document.getElementById("doc-source").value.trim() || "manual";
  const author   = document.getElementById("doc-author").value.trim();

  if (!text || !title) { alert("Title and text are required."); return; }

  const btn = e.target.querySelector(".btn.primary");
  btn.disabled = true;
  btn.textContent = "Ingesting…";
  result.style.display = "none";

  try {
    const res = await fetch(`${API}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        documents: [{ text, title, category, source, author }],
      }),
    });
    const data = await res.json();
    if (res.ok) {
      result.className = "ingest-result success";
      result.textContent = `✅ Ingested "${title}" → ${data.total_chunks} chunks stored in Endee.`;
      e.target.reset();
    } else {
      throw new Error(data.detail || "Unknown error");
    }
  } catch (err) {
    result.className = "ingest-result error";
    result.textContent = `❌ Error: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "⚡ Ingest Document";
  }
});

// ─── Admin Panel ──────────────────────────────────────────────────────
async function loadAdminStats() {
  try {
    const res  = await fetch(`${API}/index/stats`);
    const data = await res.json();
    const info = data.index || {};

    document.getElementById("stat-name").textContent  = info.name  || "nexus_knowledge_base";
    document.getElementById("stat-dim").textContent   = info.dimension || "384";
    document.getElementById("stat-docs").textContent  = info.vector_count ?? "—";
    document.getElementById("stat-space").textContent = info.space_type  || "cosine";
    document.getElementById("stat-prec").textContent  = info.precision   || "INT8";
    document.getElementById("stat-sparse").textContent = info.sparse_model || "endee_bm25";
  } catch {
    /* silently ignore if API not reachable */
  }
}

document.getElementById("reset-btn").addEventListener("click", async () => {
  if (!confirm("⚠️ This will delete ALL indexed documents. Continue?")) return;
  try {
    const res  = await fetch(`${API}/index/reset`, { method: "DELETE" });
    const data = await res.json();
    alert(data.message || "Index reset.");
    loadAdminStats();
  } catch (err) {
    alert("Error: " + err.message);
  }
});

document.getElementById("sample-btn").addEventListener("click", async () => {
  const btn = document.getElementById("sample-btn");
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    const res  = await fetch(`${API}/health`);
    alert(res.ok
      ? "✅ Run: python scripts/ingest_sample_data.py to load the sample knowledge base."
      : "⚠️ Backend not reachable.");
  } finally {
    btn.disabled = false;
    btn.textContent = "📥 Load Sample Data";
  }
});

// ─── Utilities ────────────────────────────────────────────────────────

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(str) {
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// Simple markdown-to-HTML (no deps)
function renderMarkdown(md) {
  return md
    // Code blocks
    .replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="lang-${lang}">${escapeHtml(code.trim())}</code></pre>`)
    // Inline code
    .replace(/`([^`]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`)
    // Bold
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    // Italic
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    // Headers
    .replace(/^### (.+)$/gm, "<h4>$1</h4>")
    .replace(/^## (.+)$/gm,  "<h3>$1</h3>")
    // Bullet lists
    .replace(/^[\*\-] (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
    // Numbered lists
    .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
    // Blockquotes
    .replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>")
    // Paragraphs (double newline → <p>)
    .replace(/\n\n/g, "</p><p>")
    // Single newlines
    .replace(/\n/g, "<br>")
    // Wrap in paragraph
    .replace(/^/, "<p>")
    .replace(/$/, "</p>")
    // Clean up empty <p></p>
    .replace(/<p><\/p>/g, "")
    .replace(/<p>(<[hul])/g, "$1")
    .replace(/<\/[hul][^>]*>(<\/p>)/g, match => match.replace("</p>", ""));
}
