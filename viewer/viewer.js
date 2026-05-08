// Hockey AI 3D viewer entry point.
// Errors surface in #data-status via the inline handler in index.html.
import * as THREE from "./lib/three.module.js";
import { buildRink, RINK_LENGTH_FT, RINK_WIDTH_FT, CENTER_X, CENTER_Y } from "./rink.js";
import { createAvatar, updateAvatar, createPuck } from "./avatar.js";
import {
  makeTopDownCamera, makeBroadcastCamera, makePOVCamera,
  updatePOVCamera, resizeCamera,
} from "./camera.js";
import { loadFromUrl, loadFromFile } from "./data.js";
import { Playback } from "./playback.js";

// ── DOM
const container = document.getElementById("canvas-container");
const filePicker = document.getElementById("file-picker");
const dataStatus = document.getElementById("data-status");
const camButtons = document.querySelectorAll(".camera-controls button");
const povSelect = document.getElementById("pov-player");
const playBtn = document.getElementById("play-btn");
const scrubber = document.getElementById("scrubber");
const frameCounter = document.getElementById("frame-counter");
const timeCounter = document.getElementById("time-counter");
const speedSelect = document.getElementById("speed");
const eventMarkersEl = document.getElementById("event-markers");
const eventTooltipEl = document.getElementById("event-tooltip");
const eventBannerEl = document.getElementById("active-event-banner");
const povStatusEl = document.getElementById("pov-status");

// ── Three.js scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101418);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.6));
const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(80, 80, 40);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = -120;
sun.shadow.camera.right = 120;
sun.shadow.camera.top = 80;
sun.shadow.camera.bottom = -80;
scene.add(sun);

scene.add(buildRink());
const puck = createPuck();
puck.visible = false;
scene.add(puck);

// ── Decision-recommendation arrow (C1).
// A thin glowing arrow rendered from the carrier toward the appropriate
// goal end during high-conf shot / missed-shot events. Hidden otherwise.
const decisionArrow = (() => {
  const len = 30;  // 30 ft default; rescaled per event
  const shaftGeom = new THREE.CylinderGeometry(0.18, 0.18, len, 8);
  const tipGeom = new THREE.ConeGeometry(0.55, 1.6, 10);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x3ddc84, transparent: true, opacity: 0.85,
  });
  const shaft = new THREE.Mesh(shaftGeom, mat);
  const tip = new THREE.Mesh(tipGeom, mat);
  // Cylinder default axis = Y; we'll rotate the whole group to point.
  // Origin of the group is the carrier's foot position.
  shaft.position.set(0, 0.6, len / 2);  // start at origin, extend +Z
  shaft.rotation.x = Math.PI / 2;
  tip.position.set(0, 0.6, len + 0.8);
  tip.rotation.x = Math.PI / 2;
  const group = new THREE.Group();
  group.add(shaft);
  group.add(tip);
  group.visible = false;
  return { group, shaft, tip, mat, len };
})();
scene.add(decisionArrow.group);

// ── Cameras
let cameras = null; // { topdown, broadcast, pov }
let activeCamMode = "topdown";

function buildCameras() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  cameras = {
    topdown: makeTopDownCamera(w / h),
    broadcast: makeBroadcastCamera(w / h),
    pov: makePOVCamera(w / h),
  };
}
buildCameras();

// ── Avatar registry
const avatars = new Map(); // track_id -> { mesh, lastPos, lastFrameIdx }

function getOrCreateAvatar(trackId, team, isGoalie = false) {
  if (avatars.has(trackId)) return avatars.get(trackId);
  const mesh = createAvatar(team || "unknown", isGoalie);
  scene.add(mesh);
  const entry = { mesh, lastPos: null, lastFrameIdx: -1 };
  avatars.set(trackId, entry);
  return entry;
}

function clearAvatars() {
  for (const { mesh } of avatars.values()) scene.remove(mesh);
  avatars.clear();
}

// ── Data + playback state
let data = null;
let playback = null;

async function loadData(payloadOrUrl, fromFile = false) {
  try {
    data = fromFile ? await loadFromFile(payloadOrUrl) : await loadFromUrl(payloadOrUrl);
    playback = new Playback(data);
    scrubber.max = Math.max(0, data.totalFrames - 1);
    scrubber.value = 0;
    populatePOVPicker();
    renderEventMarkers();
    clearAvatars();

    // Display file name (from File obj or URL) + frames + calibration quality
    let fileName;
    if (fromFile) {
      fileName = payloadOrUrl.name;
    } else {
      try { fileName = decodeURIComponent(payloadOrUrl.split("/").pop()); }
      catch { fileName = payloadOrUrl; }
    }
    const q = data.calibrationQuality;
    const qStr = q
      ? ` · ${q.in_rink_pct}% in rink`
      : " · (no calibration_quality field — re-run pipeline for the metric)";
    dataStatus.textContent = `${fileName} · ${data.totalFrames} frames @ ${data.fps}fps${qStr}`;
    updateFrameDisplay();
  } catch (e) {
    dataStatus.textContent = `Load failed: ${e.message}`;
    console.error(e);
  }
}

function renderEventMarkers() {
  eventMarkersEl.innerHTML = "";
  if (!data || data.totalFrames <= 1) return;
  const total = data.totalFrames - 1;

  // Calibration coverage bands first (drawn under the markers so markers
  // remain crisp). Each contiguous calibrated range becomes a thin green
  // bar; the rest of the scrubber implicitly shows as "no data."
  for (const [s, e] of data.calibratedRanges()) {
    const left = (s / total) * 100;
    const width = ((e - s + 1) / total) * 100;
    const band = document.createElement("div");
    band.className = "calib-band";
    band.style.left = `${left}%`;
    band.style.width = `${Math.max(0.2, width)}%`;
    eventMarkersEl.appendChild(band);
  }

  for (const ev of data.events) {
    const pct = (ev.frame_idx / total) * 100;
    const marker = document.createElement("div");
    marker.className = `event-marker rating-${ev.rating || "neutral"}`;
    marker.style.left = `${pct}%`;
    marker.title = formatEventLabel(ev);
    marker.addEventListener("click", () => jumpToFrame(ev.frame_idx));
    marker.addEventListener("mouseenter", (e) => showTooltip(ev, e));
    marker.addEventListener("mouseleave", hideTooltip);
    eventMarkersEl.appendChild(marker);
  }
}

function formatEventLabel(ev) {
  const t = ev.event_type.replace(/_/g, " ").toUpperCase();
  const d = ev.decision_made || "";
  const r = ev.rating ? ` (${ev.rating})` : "";
  const team = ev.team ? ` · ${ev.team.replace("team_", "Team ").toUpperCase()}` : "";
  return `${t}: ${d}${r}${team} · conf ${ev.confidence.toFixed(2)} · ${ev.timestamp_sec.toFixed(1)}s`;
}

function showTooltip(ev) {
  eventTooltipEl.textContent = formatEventLabel(ev);
  eventTooltipEl.classList.remove("hidden");
}

function hideTooltip() {
  eventTooltipEl.classList.add("hidden");
}

function jumpToFrame(idx) {
  if (!playback) return;
  playback.setFrame(idx);
  applyFrame(playback.currentFrame());
  updateFrameDisplay();
}

function populatePOVPicker() {
  povSelect.innerHTML = '<option value="">Pick player...</option>';
  if (!data) return;
  // Show only tracks with real presence — ByteTrack assigns lots of
  // 1-frame transient IDs on broadcast footage and they're useless POVs.
  const candidates = data.povCandidates({ minFrames: 15, max: 30 });
  for (const c of candidates) {
    const opt = document.createElement("option");
    opt.value = String(c.trackId);
    const teamLabel = c.team === "team_a" ? "A" : c.team === "team_b" ? "B" : "?";
    opt.textContent = `#${c.trackId} (${teamLabel}, ${c.frames} frames)`;
    povSelect.appendChild(opt);
  }
  povSelect.disabled = candidates.length === 0;
}

function updateFrameDisplay() {
  if (!playback) return;
  const idx = Math.floor(playback.frameIdx);
  scrubber.value = idx;
  const fr = data.frameAt(idx);
  const calibTag = fr && !fr.calibrated ? " · NO POSITION DATA" : "";
  frameCounter.textContent = `${idx} / ${data.totalFrames - 1}${calibTag}`;
  timeCounter.textContent = `${playback.currentTimestamp().toFixed(2)}s`;
  // Banner update too, in case scrubbing landed inside an event window
  // even though applyFrame returned early (uncalibrated frame).
  updateActiveEventOverlays(idx);
}

function applyFrame(fr) {
  if (!fr) return;

  // When calibration is missing this frame, hold the previous positions
  // for everyone instead of hiding the scene. Calibration drops in/out
  // frequently in broadcast follow-cam, so erasing the world every dropout
  // makes the playback unwatchable.
  if (!fr.calibrated) {
    // Silently hold the last known avatar positions and bail. Banner /
    // arrow updates still flow through updateFrameDisplay().
    return;
  }

  const fps = data.fps;
  const seenIds = new Set();
  const currentFrameIdx = Math.floor(playback.frameIdx);

  function applyOne(p, isGoalie) {
    const team = p.team || data.dominantTeamFor(p.track_id);
    const entry = getOrCreateAvatar(p.track_id, team, isGoalie);
    const pos = { x: p.ice_x, y: p.ice_y };
    // Real dt is derived from elapsed frames since this avatar was last
    // updated. When the avatar reappears after a long absence
    // (track-ID drop / calibration gap / scrub), treat it as a teleport
    // so we don't compute a 1000 ft/s "speed" and spin the limbs.
    const frameDelta = entry.lastFrameIdx >= 0
      ? Math.max(1, currentFrameIdx - entry.lastFrameIdx)
      : 1;
    const TELEPORT_FRAMES = Math.round(fps * 0.5); // >0.5s gap = treat as teleport
    if (frameDelta > TELEPORT_FRAMES || !entry.lastPos) {
      // Snap to new position; reset velocity history and cycle phase so
      // the skating animation doesn't jolt.
      entry.mesh.position.set(pos.x, 0, pos.y);
      entry.mesh.userData.cyclePhase = 0;
      entry.lastPos = pos;
    } else {
      const dt = frameDelta / fps;
      updateAvatar(entry.mesh, entry.lastPos, pos, dt);
      entry.lastPos = pos;
    }
    entry.mesh.visible = true;
    entry.lastFrameIdx = currentFrameIdx;
    seenIds.add(p.track_id);
  }

  for (const p of fr.players || []) applyOne(p, false);
  for (const g of fr.goalies || []) applyOne(g, true);
  // Avatars not seen this calibrated frame: hide only if they've been
  // missing for a while (>STALE_FRAMES). Calibration drops in/out across
  // 70-90% of frames in broadcast follow-cam, so we hold avatars for
  // ~5s of real time before hiding them.
  const STALE_FRAMES = Math.max(60, Math.round(fps * 5.0));
  for (const [tid, entry] of avatars) {
    if (seenIds.has(tid)) continue;
    if (entry.lastFrameIdx >= 0 && currentFrameIdx - entry.lastFrameIdx > STALE_FRAMES) {
      entry.mesh.visible = false;
    }
  }

  if (fr.puck) {
    puck.visible = true;
    puck.position.set(fr.puck.ice_x, 0.15, fr.puck.ice_y);
  }

  updateActiveEventOverlays(currentFrameIdx);
}

// Picks the active high-conf event (highest confidence wins on overlap),
// drives the active-event banner + the 3D shot-recommendation arrow.
function updateActiveEventOverlays(frameIdx) {
  if (!data) return;
  const active = data.eventsAtFrame(frameIdx);
  if (!active.length) {
    eventBannerEl.classList.add("hidden");
    decisionArrow.group.visible = false;
    return;
  }
  // Prefer the highest-confidence event when multiple overlap.
  active.sort((a, b) => b.confidence - a.confidence);
  const ev = active[0];

  eventBannerEl.textContent = formatEventLabel(ev);
  eventBannerEl.className = `rating-${ev.rating || "neutral"}`;

  // 3D arrow only for shot-shaped decisions where we know the carrier.
  const isShotShaped = (
    (ev.event_type === "shot_vs_pass" && ev.decision_made === "shot") ||
    (ev.event_type === "missed_opportunity" && ev.decision_made === "missed_shot") ||
    (ev.event_type === "odd_man_rush" && ev.decision_made === "shot")
  );
  if (!isShotShaped || ev.player_id === null) {
    decisionArrow.group.visible = false;
    return;
  }
  // Find the carrier's current ice position
  let carrierPos = null;
  for (const p of (data.frameAt(frameIdx).players || [])) {
    if (p.track_id === ev.player_id) { carrierPos = p; break; }
  }
  if (!carrierPos) {
    decisionArrow.group.visible = false;
    return;
  }
  // Target: the goal end farthest along x from the carrier (shooting away
  // from where they came, toward the offensive net). RINK_LENGTH_FT and
  // CENTER_X are imported from rink.js.
  const isLeftSide = carrierPos.ice_x < CENTER_X;
  const goalIceX = isLeftSide ? RINK_LENGTH_FT - 11 : 11;
  const goalIceY = RINK_WIDTH_FT / 2;
  const dx = goalIceX - carrierPos.ice_x;
  const dz = goalIceY - carrierPos.ice_y;
  const dist = Math.sqrt(dx * dx + dz * dz);

  decisionArrow.group.position.set(carrierPos.ice_x, 0, carrierPos.ice_y);
  decisionArrow.group.rotation.y = -Math.atan2(dz, dx) - Math.PI / 2;
  // Scale arrow length to ~80% of the actual distance to the goal so it
  // visually "points at" the net without overshooting.
  const targetLen = Math.max(8, dist * 0.8);
  decisionArrow.group.scale.set(1, 1, targetLen / decisionArrow.len);

  // Color matches event rating
  const colors = { good: 0x3ddc84, warning: 0xf5c542, poor: 0xe5484d };
  decisionArrow.mat.color.setHex(colors[ev.rating] ?? 0x888888);
  decisionArrow.group.visible = true;
}

function activeCamera() {
  if (activeCamMode === "pov") {
    const tid = parseInt(povSelect.value);
    const entry = isNaN(tid) ? null : avatars.get(tid);
    const isLive = entry && entry.mesh.visible;
    if (isLive) updatePOVCamera(cameras.pov, entry.mesh);
    // POV-target hint: surface "OFF SCREEN" when the picked avatar isn't
    // currently rendered (calibration gap, ID drop, or hasn't appeared yet).
    if (!isNaN(tid)) {
      povStatusEl.classList.toggle("hidden", isLive || !povSelect.value);
    } else {
      povStatusEl.classList.add("hidden");
    }
    return cameras.pov;
  }
  povStatusEl.classList.add("hidden");
  return cameras[activeCamMode];
}

// ── Render loop
function tick(nowMs) {
  if (playback) {
    playback.tick(nowMs);
    // Auto-skip uncalibrated gaps during playback so the user doesn't
    // stare at empty ice for stretches where the model lost the rink.
    // Only kicks in while playing — manual scrubbing still lands wherever
    // the user dragged to (with a visible "no positions" hint).
    if (playback.playing && data) {
      const idx = Math.floor(playback.frameIdx);
      const fr = data.frameAt(idx);
      if (fr && !fr.calibrated) {
        const next = data.nextCalibratedFrom(idx);
        if (next > idx) playback.setFrame(next);
        else if (next < 0) playback.pause();
      }
    }
    applyFrame(playback.currentFrame());
    updateFrameDisplay();
  }
  renderer.render(scene, activeCamera());
  requestAnimationFrame(tick);
}

function resize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h, false);
  if (cameras) {
    resizeCamera(cameras.topdown, w, h);
    resizeCamera(cameras.broadcast, w, h);
    resizeCamera(cameras.pov, w, h);
  }
}
window.addEventListener("resize", resize);
resize();

// ── UI wiring
camButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    activeCamMode = btn.dataset.cam;
    camButtons.forEach((b) => b.classList.toggle("active", b === btn));
  });
});

filePicker.addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) loadData(f, true);
});

playBtn.addEventListener("click", () => {
  if (!playback) return;
  playback.togglePlay();
  playBtn.textContent = playback.playing ? "Pause" : "Play";
});

scrubber.addEventListener("input", (e) => {
  if (!playback) return;
  playback.setFrame(parseInt(e.target.value));
  applyFrame(playback.currentFrame());
  updateFrameDisplay();
});

speedSelect.addEventListener("change", (e) => {
  if (playback) playback.setSpeed(parseFloat(e.target.value));
});

povSelect.addEventListener("change", (e) => {
  // Auto-jump to the first frame this track is visible — otherwise the
  // POV camera stares into empty space until the player happens to appear.
  if (!data || !playback) return;
  const tid = parseInt(e.target.value);
  if (isNaN(tid)) return;
  const first = data.firstFrameOfTrack(tid);
  if (first >= 0) {
    // Land on the first calibrated frame at or after the player's debut so
    // we have positions to render the avatar from.
    const target = data.nextCalibratedFrom(first);
    jumpToFrame(target >= 0 ? target : first);
  }
});

// Auto-load via ?data=URL query param so scripts/serve_viewer.py can deep-link.
const params = new URLSearchParams(window.location.search);
const dataParam = params.get("data");
if (dataParam) loadData(dataParam, false);

requestAnimationFrame(tick);
