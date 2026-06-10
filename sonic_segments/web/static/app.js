/* Sonic Segments intake page logic */

const state = {
  mode: "variants",
  moods: new Set(),
  demographics: new Set(),
  sources: new Set(),
  meta: null,
  file: null,
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
  const meta = await (await fetch("/api/meta")).json();
  state.meta = meta;
  renderMoods(meta.moods);
  renderDemographics(meta.segments);
  setMode("variants");
  checkSpotify();
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
    const item = document.createElement("div");
    item.className = "src-item" + (s.available ? "" : " disabled");
    const badge = s.demo_only ? '<span class="badge demo">DEMO ONLY</span>' : '<span class="badge legal">CLEARED</span>';
    item.innerHTML = `<input class="src-check" type="checkbox" ${s.available ? "" : "disabled"}>
      <div><b>${s.label}</b> ${badge}<p>${s.description} ${s.available ? "" : "— " + s.reason}</p></div>`;
    const cb = item.querySelector("input");
    const sync = () => {
      item.classList.toggle("active", cb.checked);
      cb.checked ? state.sources.add(id) : state.sources.delete(id);
      $("refQueryWrap").style.display = state.sources.has("reference") ? "block" : "none";
      $("ownTrackWrap").style.display = state.sources.has("uploaded") ? "block" : "none";
    };
    cb.checked = s.available && state.sources.has(id);
    item.onclick = (e) => {
      if (!s.available) return;
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
    MODE_DEFAULTS[mode].filter((id) => state.meta.sources.find((s) => s.id === id && s.available))
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

/* spotify chip */
async function checkSpotify() {
  const chip = $("spotifyChip");
  try {
    const st = await (await fetch("/api/spotify/status")).json();
    if (st.connected) {
      chip.textContent = `Spotify: ${st.user}`;
      chip.classList.add("connected");
      chip.removeAttribute("href");
    } else {
      chip.textContent = "Connect Spotify";
    }
  } catch { chip.textContent = "Connect Spotify"; }
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

init();
