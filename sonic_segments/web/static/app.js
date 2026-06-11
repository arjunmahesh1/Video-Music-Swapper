/* Sonic Segments intake page logic */

const state = {
  mode: "variants",
  moods: new Set(),
  demographics: new Set(),
  sources: new Set(),
  meta: null,
  file: null,
  spotify: { connected: false, user: null },
};

const $ = (id) => document.getElementById(id);
const MAX_DEMOS = 5;

// Sources relevant per mode (order = display order).
const MODE_SOURCES = {
  demo: ["spotify", "reference", "musicgen", "library"],
  variants: ["library", "jamendo", "uploaded", "musicgen", "reference"],
};
const MODE_DEFAULTS = { demo: ["spotify"], variants: ["library", "musicgen"] };

async function init() {
  startWaves();
  const [meta, spotify] = await Promise.all([
    fetch("/api/meta").then((r) => r.json()),
    fetch("/api/spotify/status").then((r) => r.json()).catch(() => ({ connected: false })),
  ]);
  state.meta = meta;
  state.spotify = spotify;
  renderMoods(meta.moods);
  renderDemographics(meta.segments);
  setMode("variants");
}

function renderMoods(moods) {
  const box = $("moodChips");
  box.innerHTML = "";
  moods.forEach((m) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = m.label;
    chip.onclick = () => {
      state.moods.has(m.id) ? state.moods.delete(m.id) : state.moods.add(m.id);
      chip.classList.toggle("active", state.moods.has(m.id));
    };
    box.appendChild(chip);
  });
}

function renderDemographics(segments) {
  const grid = $("demoGrid");
  grid.innerHTML = "";
  segments.forEach((s) => {
    const card = document.createElement("div");
    card.className = "demo-card";
    card.innerHTML = `<b>${s.label}</b><span>${s.region} · ${s.core_genres.slice(0, 3).join(", ")}</span>`;
    card.title = s.description;
    card.onclick = () => {
      if (state.demographics.has(s.id)) {
        state.demographics.delete(s.id);
      } else if (state.demographics.size < MAX_DEMOS) {
        state.demographics.add(s.id);
      }
      card.classList.toggle("active", state.demographics.has(s.id));
      $("capNote").textContent = `${state.demographics.size} of ${MAX_DEMOS} selected`;
    };
    grid.appendChild(card);
  });
}

function renderSources() {
  const list = $("srcList");
  list.innerHTML = "";
  const order = MODE_SOURCES[state.mode];
  const infos = Object.fromEntries(state.meta.sources.map((s) => [s.id, s]));

  order.forEach((id) => {
    const s = infos[id];
    if (!s) return;
    const isSpotify = id === "spotify";
    const spotifyReady = state.spotify.connected;
    const usable = isSpotify ? spotifyReady : s.available;

    const item = document.createElement("div");
    item.className = "src-item" + (usable ? "" : " disabled");
    const badge = s.demo_only
      ? '<span class="badge demo">DEMO ONLY</span>'
      : '<span class="badge legal">CLEARED</span>';

    let extra = "";
    if (isSpotify) {
      extra = spotifyReady
        ? `<span class="spotify-connected">✓ Connected as ${state.spotify.user}</span>
           <a href="/spotify/login" style="font-size:12px;margin-left:8px">switch account</a>`
        : `<a class="spotify-connect" href="/spotify/login">Connect Spotify</a>`;
    }
    const reason = !usable && !isSpotify ? " — " + s.reason : "";

    item.innerHTML = `<input class="src-check" type="checkbox" ${usable ? "" : "disabled"}>
      <div><b>${s.label}</b> ${badge}<p>${s.description}${reason}</p>${extra}</div>`;

    const cb = item.querySelector("input");
    const sync = () => {
      item.classList.toggle("active", cb.checked);
      cb.checked ? state.sources.add(id) : state.sources.delete(id);
      $("refQueryWrap").style.display = state.sources.has("reference") ? "block" : "none";
      $("ownTrackWrap").style.display = state.sources.has("uploaded") ? "block" : "none";
    };
    cb.checked = usable && state.sources.has(id);
    item.onclick = (e) => {
      if (e.target.closest("a")) return; // let auth/switch links navigate
      if (!usable) return;
      if (e.target !== cb) cb.checked = !cb.checked;
      sync();
    };
    sync();
    list.appendChild(item);
  });
}

function setMode(mode) {
  state.mode = mode;
  $("modeDemo").classList.toggle("active", mode === "demo");
  $("modeVariants").classList.toggle("active", mode === "variants");
  $("demoCard").style.display = mode === "variants" ? "block" : "none";
  $("srcStep").textContent = mode === "variants" ? "5" : "4";
  state.sources = new Set(
    MODE_DEFAULTS[mode].filter((id) => {
      if (id === "spotify") return state.spotify.connected;
      return state.meta.sources.find((s) => s.id === id && s.available);
    })
  );
  renderSources();
}

$("modeDemo").onclick = () => setMode("demo");
$("modeVariants").onclick = () => setMode("variants");

/* dropzone */
const dz = $("dropzone");
dz.onclick = () => $("videoFile").click();
$("videoFile").onchange = () => setFile($("videoFile").files[0]);
["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]); });

function setFile(file) {
  state.file = file || null;
  dz.classList.toggle("has-file", !!file);
  $("dropLabel").innerHTML = file
    ? `<b>${file.name}</b><br><span style="font-size:13px">${(file.size / 1e6).toFixed(1)} MB — click to change</span>`
    : `<b>Drop your ad file here</b> or click to browse<br><span style="font-size:13px">MP4 preferred — full quality scores best</span>`;
}

/* submit */
$("intakeForm").onsubmit = async (e) => {
  e.preventDefault();
  const err = $("err");
  err.textContent = "";

  if (!state.file && !$("adUrl").value.trim()) return (err.textContent = "Add your ad: drop a file or paste a link.");
  if (state.moods.size === 0) return (err.textContent = "Pick at least one mood.");
  if (state.mode === "variants" && state.demographics.size === 0) return (err.textContent = "Pick at least one audience.");
  if (state.sources.size === 0) return (err.textContent = "Pick at least one music source.");

  const fd = new FormData();
  fd.append("mode", state.mode);
  fd.append("brand", $("brand").value);
  fd.append("vibe", $("vibe").value);
  fd.append("url", $("adUrl").value.trim());
  fd.append("moods", [...state.moods].join(","));
  fd.append("demographics", [...state.demographics].join(","));
  fd.append("sources", [...state.sources].join(","));
  fd.append("reference_query", $("refQuery").value);
  if (state.file) fd.append("video", state.file);
  if ($("ownTrack").files[0]) fd.append("own_track", $("ownTrack").files[0]);

  const btn = $("submitBtn");
  btn.disabled = true;
  btn.textContent = "Uploading…";
  try {
    const resp = await fetch("/api/campaigns", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Something went wrong.");
    window.location.href = data.status_url;
  } catch (ex) {
    err.textContent = ex.message;
    btn.disabled = false;
    btn.textContent = "Re-score my ad →";
  }
};

/* ---- animated soundwave hero ---- */
function startWaves() {
  const canvas = $("waveCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let w = 0, h = 0, t = 0;

  const resize = () => {
    w = canvas.offsetWidth * dpr;
    h = canvas.offsetHeight * dpr;
    canvas.width = w;
    canvas.height = h;
  };
  window.addEventListener("resize", resize);
  resize();

  // layered waves: coral, teal, ink — quiet on the left (under the copy), louder right
  const waves = [
    { amp: 0.10, freq: 2.2, speed: 0.55, phase: 0.0, color: "255,90,54",  alpha: 0.30, width: 1.8 },
    { amp: 0.16, freq: 1.4, speed: 0.34, phase: 2.1, color: "13,148,136", alpha: 0.22, width: 1.6 },
    { amp: 0.07, freq: 3.1, speed: 0.85, phase: 4.0, color: "16,24,40",   alpha: 0.10, width: 1.2 },
    { amp: 0.22, freq: 0.9, speed: 0.22, phase: 1.0, color: "255,90,54",  alpha: 0.10, width: 2.4 },
    { amp: 0.13, freq: 1.8, speed: 0.45, phase: 5.2, color: "13,148,136", alpha: 0.10, width: 1.2 },
  ];

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function frame() {
    ctx.clearRect(0, 0, w, h);
    const mid = h * 0.58;
    waves.forEach((wv) => {
      ctx.beginPath();
      const step = Math.max(3 * dpr, w / 320);
      for (let x = 0; x <= w; x += step) {
        const p = x / w;
        const loudness = 0.25 + 0.75 * p * p;            // grows to the right
        const envelope = Math.sin(Math.PI * Math.min(1, p * 1.06)); // taper edges
        const y =
          mid +
          Math.sin(p * Math.PI * 2 * wv.freq + t * wv.speed + wv.phase) *
            wv.amp * h * envelope * loudness;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(${wv.color},${wv.alpha})`;
      ctx.lineWidth = wv.width * dpr;
      ctx.stroke();
    });
    t += 0.016;
    if (!reduced) requestAnimationFrame(frame);
  }
  frame();
}

init();
