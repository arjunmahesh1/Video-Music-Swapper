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
    fetch("/api/spotify/status", { cache: "no-store" }).then((r) => r.json()).catch(() => ({ connected: false })),
  ]);
  state.meta = meta;
  state.spotify = spotify;
  renderMoods(meta.moods);
  renderDemographics(meta.segments);

  const flag = new URLSearchParams(location.search).get("spotify");
  if (flag) {
    setMode("demo"); // full-page auth fallback: land them back in the demo flow
    restoreDraft(); //  ...with everything they had typed (file needs re-attaching)
    if (flag !== "connected") {
      $("err").textContent =
        flag === "denied"
          ? "Spotify authorization was cancelled."
          : "Spotify connection failed — try again, and check the redirect URI in your Spotify dashboard.";
    } else {
      $("err").textContent = state.file ? "" : "Connected — re-attach your video file and you're set.";
    }
    history.replaceState(null, "", "/#intake");
  } else {
    setMode("variants");
  }
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
           <a href="#" class="spotify-switch" style="font-size:12px;margin-left:8px">switch account</a>`
        : `<a class="spotify-connect" href="#">Connect Spotify</a>`;
    }
    const reason = !usable
      ? " — " + (isSpotify ? "connect Spotify below to enable" : s.reason)
      : "";

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
      const connectLink = e.target.closest(".spotify-connect, .spotify-switch");
      if (connectLink) {
        e.preventDefault();
        spotifyAuth(connectLink.classList.contains("spotify-switch"));
        return;
      }
      if (e.target.closest("a")) return;
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

/* ---- Spotify auth in a popup: the form never reloads ---- */
function spotifyAuth(switchAccount) {
  saveDraft();
  const pop = window.open(
    switchAccount ? "https://accounts.spotify.com/logout" : "/spotify/login",
    "spotify-auth",
    "width=520,height=720"
  );
  if (switchAccount && pop) {
    // give the logout a beat, then start a fresh authorization in the popup
    setTimeout(() => { try { pop.location = "/spotify/login"; } catch (e) {} }, 1100);
  }
}

window.addEventListener("message", async (e) => {
  if (e.origin !== window.location.origin || typeof e.data !== "string" || !e.data.startsWith("spotify:")) return;
  const status = e.data.split(":")[1];
  state.spotify = await fetch("/api/spotify/status", { cache: "no-store" })
    .then((r) => r.json())
    .catch(() => ({ connected: false }));
  renderSources(); // re-paints "Connected as <name>" with the fresh account
  $("err").textContent =
    status === "connected" ? "" :
    status === "denied" ? "Spotify authorization was cancelled." :
    "Spotify connection failed — check the redirect URI in your Spotify dashboard.";
});

/* ---- draft persistence: survives any full-page auth fallback ---- */
const DRAFT_KEY = "sonic_draft";
function saveDraft() {
  localStorage.setItem(DRAFT_KEY, JSON.stringify({
    ts: Date.now(), mode: state.mode,
    brand: $("brand").value, vibe: $("vibe").value, url: $("adUrl").value,
    transcript: $("transcript").value, refQuery: $("refQuery").value,
    moods: [...state.moods], demographics: [...state.demographics], sources: [...state.sources],
  }));
}
function restoreDraft() {
  try {
    const d = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
    if (!d || Date.now() - d.ts > 3600e3) return;
    $("brand").value = d.brand || ""; $("vibe").value = d.vibe || "";
    $("adUrl").value = d.url || ""; $("transcript").value = d.transcript || "";
    $("refQuery").value = d.refQuery || "";
    state.moods = new Set(d.moods || []);
    state.demographics = new Set(d.demographics || []);
    [...$("moodChips").children].forEach((chip, i) =>
      chip.classList.toggle("active", state.moods.has(state.meta.moods[i].id)));
    [...$("demoGrid").children].forEach((card, i) =>
      card.classList.toggle("active", state.demographics.has(state.meta.segments[i].id)));
    $("capNote").textContent = `${state.demographics.size} of ${MAX_DEMOS} selected`;
    setMode(d.mode || "variants");
    state.sources = new Set(d.sources || []);
    renderSources();
  } catch (e) { /* corrupt draft: ignore */ }
}

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
  fd.append("transcript", $("transcript").value);
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

  // Audio-style waveform: a dense carrier oscillation whose amplitude is
  // modulated by drifting pseudo-noise, so peaks burst and decay irregularly
  // like a real recording instead of a uniform sine.
  const layers = [
    { cycles: 46, speed: 0.45, color: "139,92,246", alpha: 0.36, width: 1.7, amp: 0.26, seed: 1.0, glow: 8 },  // hyper purple
    { cycles: 64, speed: 0.65, color: "56,189,248", alpha: 0.28, width: 1.3, amp: 0.20, seed: 7.3, glow: 6 },  // light blue
    { cycles: 34, speed: 0.30, color: "13,148,136", alpha: 0.26, width: 1.6, amp: 0.23, seed: 13.7, glow: 0 }, // teal
    { cycles: 88, speed: 0.85, color: "16,24,40",   alpha: 0.10, width: 1.0, amp: 0.14, seed: 23.1, glow: 0 }, // ink detail
    { cycles: 24, speed: 0.22, color: "139,92,246", alpha: 0.10, width: 2.6, amp: 0.32, seed: 31.9, glow: 0 }, // purple halo
  ];

  // smooth drifting noise in [-1, 1] built from incommensurate sines
  const noise = (p, time, seed) =>
    Math.sin(p * 5.3 + time * 0.50 + seed) * 0.5 +
    Math.sin(p * 11.7 - time * 0.31 + seed * 2.1) * 0.3 +
    Math.sin(p * 23.1 + time * 0.73 + seed * 3.7) * 0.2;

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function frame() {
    ctx.clearRect(0, 0, w, h);
    const mid = h * 0.84; // ride low in the band, well under the headline copy
    layers.forEach((ly) => {
      ctx.beginPath();
      const step = Math.max(2 * dpr, w / 560);
      for (let x = 0; x <= w; x += step) {
        const p = x / w;
        // bursty envelope: noise mapped to 0..1 then sharpened, edges tapered
        let env = noise(p * 2.2, t, ly.seed) * 0.5 + 0.5;
        env = Math.pow(env, 1.7);
        const taper = Math.sin(Math.PI * Math.min(1, Math.max(0, p * 1.04)));
        const quietLeft = 0.45 + 0.55 * p; // calmer under the text column
        const carrier = Math.sin(p * Math.PI * 2 * ly.cycles + t * ly.speed * 2.2 + ly.seed);
        const y = mid + carrier * env * taper * quietLeft * ly.amp * h;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(${ly.color},${ly.alpha})`;
      ctx.lineWidth = ly.width * dpr;
      ctx.shadowColor = ly.glow ? `rgba(${ly.color},0.55)` : "transparent";
      ctx.shadowBlur = ly.glow * dpr;
      ctx.stroke();
    });
    ctx.shadowBlur = 0;
    t += 0.016;
    if (!reduced) requestAnimationFrame(frame);
  }
  frame();
}

init();
