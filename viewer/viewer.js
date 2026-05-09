// Hockey AI 3D viewer entry point.
// Errors surface in #data-status via the inline handler in index.html.
import * as THREE from "./lib/three.module.js";
import { buildRink, RINK_LENGTH_FT, RINK_WIDTH_FT, CENTER_X, CENTER_Y } from "./rink.js";
import { createAvatar, updateAvatar, createPuck } from "./avatar.js";
import {
  makeTopDownCamera, makeBroadcastCamera, makePOVCamera,
  makeDecisionCamera, updatePOVCamera, updateDecisionCamera, resizeCamera,
} from "./camera.js";
import { loadFromUrl, loadFromFile } from "./data.js";
import { Playback } from "./playback.js";
import { Quiz, QUIZ_OPTIONS } from "./quiz.js";

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
const hudCounterEl = document.getElementById("hud-counter");
const quizToggleBtn = document.getElementById("quiz-toggle");
const quizScoreEl = document.getElementById("quiz-score");
const quizOverlayEl = document.getElementById("quiz-overlay");
const quizQuestionEl = document.getElementById("quiz-question");
const quizChoicesEl = document.getElementById("quiz-choices");
const quizRevealEl = document.getElementById("quiz-reveal");
const quizRevealResultEl = document.getElementById("quiz-reveal-result");
const quizRevealDetailEl = document.getElementById("quiz-reveal-detail");
const quizReplayBtn = document.getElementById("quiz-replay-btn");
const quizViewTopdownBtn = document.getElementById("quiz-view-topdown");
const quizViewPovBtn = document.getElementById("quiz-view-pov");
const quizScorecardEl = document.getElementById("quiz-scorecard");
const quizScorecardScoreEl = document.getElementById("quiz-scorecard-score");
const quizScorecardBreakdownEl = document.getElementById("quiz-scorecard-breakdown");
const quizScorecardRestartBtn = document.getElementById("quiz-scorecard-restart");
const quizScorecardCloseBtn = document.getElementById("quiz-scorecard-close");

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
    decision: makeDecisionCamera(w / h),
  };
}
buildCameras();

// ── Avatar registry
const avatars = new Map(); // track_id -> { mesh, lastPos, lastFrameIdx }

// Quiz-pause view preference. "topdown" reads tactically; "pov" puts the
// user at the actor's eye level looking forward — the END-GOAL training
// experience for real-time decision-making feel.
let quizViewMode = "topdown";

// Yellow halo ring placed under the quiz actor's avatar so the user
// can spot them at a glance. Positioned each frame in activeCamera().
const actorHalo = (() => {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(2.4, 3.0, 32),
    new THREE.MeshBasicMaterial({
      color: 0xffe54a, side: THREE.DoubleSide,
      transparent: true, opacity: 0.85,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.06;
  ring.visible = false;
  return ring;
})();
scene.add(actorHalo);

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

// ── Quiz state machine (Phase D)
const quiz = new Quiz();

quiz.onPause = (event) => {
  if (!playback) return;
  playback.pause();
  playBtn.textContent = "Play";
  document.body.classList.add("quiz-paused");
  // Camera switch is handled inside activeCamera() — it sees
  // quiz.phase === "paused" and uses top-down (coach's-tape view).
  quizQuestionEl.textContent =
    `${event.event_type.replace(/_/g, " ").toUpperCase()} — what would you do?`;
  quizChoicesEl.innerHTML = "";
  const opts = QUIZ_OPTIONS[event.event_type] || [];
  opts.forEach((opt, i) => {
    const btn = document.createElement("button");
    btn.className = "quiz-choice-btn";
    const hotkey = String(i + 1);
    btn.dataset.hotkey = hotkey;
    btn.dataset.choice = opt;
    btn.innerHTML = `${opt.replace(/_/g, " ").toUpperCase()}<span class="hotkey">${hotkey}</span>`;
    btn.addEventListener("click", () => quiz.commit(opt));
    quizChoicesEl.appendChild(btn);
  });
  quizOverlayEl.classList.remove("hidden");
  quizRevealEl.classList.add("hidden");
};

quiz.onReveal = (event, result) => {
  quizOverlayEl.classList.add("hidden");
  quizRevealEl.classList.remove("hidden");
  quizRevealEl.className = result.matched ? "matched" : "diverged";
  quizRevealResultEl.textContent = result.matched
    ? "✓ You agreed with the player"
    : "✗ You'd have done it differently";
  const ratingClass = `rating-${result.rating || "neutral"}`;
  quizRevealDetailEl.innerHTML = `
    <span class="label">Your pick:</span> <span class="actual">${result.picked.toUpperCase()}</span>
    &nbsp;·&nbsp;
    <span class="label">Actual:</span> <span class="actual">${result.actual.toUpperCase()}</span>
    &nbsp;·&nbsp;
    <span class="label">AI:</span> <span class="${ratingClass}">${(result.rating || "neutral").toUpperCase()}</span>
  `;
};

quiz.onResume = () => {
  quizOverlayEl.classList.add("hidden");
  quizRevealEl.classList.add("hidden");
  document.body.classList.remove("quiz-paused");
  if (playback && quiz.active) {
    playback.play();
    playBtn.textContent = "Pause";
  }
};

quiz.onScoreChange = (score) => {
  quizScoreEl.textContent = `${score.matched} / ${score.total}`;
};

quiz.onComplete = (summary) => {
  // End-of-clip scorecard
  if (playback) {
    playback.pause();
    playBtn.textContent = "Play";
  }
  quizOverlayEl.classList.add("hidden");
  quizRevealEl.classList.add("hidden");
  quizScorecardScoreEl.textContent = `${summary.matched} / ${summary.total}`;
  quizScorecardBreakdownEl.innerHTML = "";
  for (const item of summary.breakdown) {
    const row = document.createElement("div");
    row.className = "row " + (item.result.matched ? "matched" : "diverged");
    const left = document.createElement("span");
    left.textContent =
      `${item.event.event_type.replace(/_/g, " ")} @ ${item.event.timestamp_sec.toFixed(1)}s`;
    const right = document.createElement("span");
    const pickActual = item.result.matched
      ? `✓ ${item.result.picked.toUpperCase()}`
      : `✗ ${item.result.picked.toUpperCase()} (actual: ${item.result.actual.toUpperCase()})`;
    right.textContent = pickActual;
    row.appendChild(left);
    row.appendChild(right);
    quizScorecardBreakdownEl.appendChild(row);
  }
  quizScorecardEl.classList.remove("hidden");
};

async function loadData(payloadOrUrl, fromFile = false) {
  try {
    data = fromFile ? await loadFromFile(payloadOrUrl) : await loadFromUrl(payloadOrUrl);
    playback = new Playback(data);
    scrubber.max = Math.max(0, data.totalFrames - 1);
    scrubber.value = 0;
    populatePOVPicker();
    renderEventMarkers();
    quiz.loadFromData(data);
    quiz.deactivate();
    quizToggleBtn.classList.remove("active");
    quizScoreEl.classList.add("hidden");
    quizScoreEl.textContent = "0 / 0";
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

  // Avatars not seen this calibrated frame: hide if they've been
  // missing for >STALE_FRAMES (~1.5s). Tighter than the prior 5s window
  // because ByteTrack on these clips emits hundreds of ghost track IDs
  // per real player; long staleness lets ghosts pile up into a crowd.
  const STALE_FRAMES = Math.max(20, Math.round(fps * 1.5));
  for (const [tid, entry] of avatars) {
    if (seenIds.has(tid)) continue;
    if (entry.lastFrameIdx >= 0 && currentFrameIdx - entry.lastFrameIdx > STALE_FRAMES) {
      entry.mesh.visible = false;
    }
  }

  // Hard cap on visible avatars. With 5v5 + 2 goalies = 12 actual players
  // on the ice, anything beyond ~14 visible is ghost-track noise. Keep
  // the most-recently-updated ones; hide the rest.
  const MAX_VISIBLE_AVATARS = 14;
  const visible = [];
  for (const [tid, entry] of avatars) {
    if (entry.mesh.visible) visible.push([tid, entry]);
  }
  if (visible.length > MAX_VISIBLE_AVATARS) {
    visible.sort((a, b) => b[1].lastFrameIdx - a[1].lastFrameIdx);
    for (let i = MAX_VISIBLE_AVATARS; i < visible.length; i++) {
      visible[i][1].mesh.visible = false;
    }
  }

  // HUD: how many avatars are currently rendered vs the per-frame raw
  // detection count. Helps diagnose "where are my players" — if rawN is
  // low the data is sparse; if rawN is high but visibleN is low something
  // is hiding them. Reads AFTER stale-hide + cap so it reflects what's
  // actually on screen this frame.
  const rawN = (fr.players?.length || 0) + (fr.goalies?.length || 0);
  let visibleN = 0;
  for (const e of avatars.values()) if (e.mesh.visible) visibleN++;
  hudCounterEl.textContent = `${visibleN} avatars · ${rawN} this frame · ${avatars.size} total tracks`;

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
  // Quiz takes priority: while paused on a decision, use either the
  // TOP-DOWN camera (default — best for tactical pattern-reading) or
  // the PLAYER-POV camera (the END-GOAL training experience: see what
  // the carrier sees at the moment of decision). User toggles via the
  // buttons in the quiz overlay or the V key.
  if (quiz.active && quiz.phase === "paused" && quiz.currentEvent) {
    const tid = quiz.currentEvent.player_id;
    const entry = (tid !== null && tid !== undefined) ? avatars.get(tid) : null;
    if (entry && entry.mesh.visible) {
      actorHalo.position.set(entry.mesh.position.x, 0.06, entry.mesh.position.z);
      actorHalo.visible = true;
      povStatusEl.classList.add("hidden");
      if (quizViewMode === "pov") {
        updatePOVCamera(cameras.pov, entry.mesh);
        return cameras.pov;
      }
      return cameras.topdown;
    } else {
      // Actor has no current avatar (track-id discontinuity). The
      // renderable filter in quiz.loadFromData should normally prevent
      // this, but keep the hint surfaced in case it slips through.
      actorHalo.visible = false;
      povStatusEl.classList.toggle("hidden", false);
      // POV with no actor = nowhere to put the camera; force top-down.
      return cameras.topdown;
    }
  }
  actorHalo.visible = false;

  if (activeCamMode === "pov") {
    const tid = parseInt(povSelect.value);
    const entry = isNaN(tid) ? null : avatars.get(tid);
    const isLive = entry && entry.mesh.visible;
    if (isLive) updatePOVCamera(cameras.pov, entry.mesh);
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
    const idx = Math.floor(playback.frameIdx);

    // Quiz check FIRST so the auto-skip below doesn't leap over an
    // event that's sitting in an uncalibrated stretch. On the rush /
    // 1v1 clips, ~80% of frames are uncalibrated and most events
    // happen there — without this ordering, auto-skip jumped from
    // (event - 100) to (event + 200) in one step and the trigger
    // window check never fired.
    quiz.tick(idx, playback.playing);
    // End-of-clip detection — fires the scorecard once per session.
    quiz.maybeComplete(idx);

    // Auto-skip uncalibrated gaps during playback — but only when
    // quiz isn't currently paused at a decision moment AND the
    // upcoming gap doesn't contain an unconsumed quiz event.
    if (playback.playing && data && quiz.phase === "idle") {
      const fr = data.frameAt(idx);
      if (fr && !fr.calibrated) {
        const next = data.nextCalibratedFrom(idx);
        if (next > idx) {
          // If there's a pending quiz event between idx and next,
          // clamp the skip so we don't jump past it.
          const clampTarget = quiz.active
            ? quiz.firstUnconsumedTriggerInRange(idx, next - 1) ?? next
            : next;
          playback.setFrame(clampTarget);
        } else if (next < 0) {
          playback.pause();
        }
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
    resizeCamera(cameras.decision, w, h);
  }
}
window.addEventListener("resize", resize);
resize();

// ── UI wiring
camButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    // Skip non-camera buttons that happen to live in .camera-controls
    // (e.g. the Quiz Mode toggle). Without this guard the click would
    // set activeCamMode = undefined → cameras[undefined] → renderer
    // crashes on the next frame with "camera.parent of undefined".
    if (!btn.dataset.cam) return;
    activeCamMode = btn.dataset.cam;
    camButtons.forEach((b) => {
      if (!b.dataset.cam) return;
      b.classList.toggle("active", b === btn);
    });
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
  const idx = parseInt(e.target.value);
  playback.setFrame(idx);
  applyFrame(playback.currentFrame());
  updateFrameDisplay();
  quiz.onScrubTo(idx);
});

quizToggleBtn.addEventListener("click", () => {
  if (!data) return;
  quiz.toggle();
  if (quiz.active) {
    quizToggleBtn.classList.add("active");
    quizScoreEl.classList.remove("hidden");
    if (playback && !playback.playing) {
      playback.play();
      playBtn.textContent = "Pause";
    }
  } else {
    quizToggleBtn.classList.remove("active");
    quizScoreEl.classList.add("hidden");
    quizOverlayEl.classList.add("hidden");
    quizRevealEl.classList.add("hidden");
  }
});

// Keyboard shortcuts: 1-4 commit a quiz choice; Esc skips the current
// question; Space toggles play/pause (out of quiz only).
window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;
  if (quiz.active && quiz.phase === "paused") {
    if (e.key === "Escape") {
      quiz.skip();
      return;
    }
    if (e.key === "v" || e.key === "V") {
      // Toggle the quiz view between top-down and player POV
      setQuizView(quizViewMode === "pov" ? "topdown" : "pov");
      return;
    }
    const num = parseInt(e.key);
    if (!isNaN(num) && num >= 1 && num <= 9) {
      const btn = quizChoicesEl.querySelector(`[data-hotkey="${num}"]`);
      if (btn) {
        const choice = btn.dataset.choice;
        quiz.commit(choice);
      }
    }
    return;
  }
  if (e.key === " " && playback) {
    e.preventDefault();
    playback.togglePlay();
    playBtn.textContent = playback.playing ? "Pause" : "Play";
  }
});

speedSelect.addEventListener("change", (e) => {
  if (playback) playback.setSpeed(parseFloat(e.target.value));
});

function setQuizView(mode) {
  quizViewMode = mode;
  quizViewTopdownBtn.classList.toggle("active", mode === "topdown");
  quizViewPovBtn.classList.toggle("active", mode === "pov");
}
quizViewTopdownBtn.addEventListener("click", () => setQuizView("topdown"));
quizViewPovBtn.addEventListener("click", () => setQuizView("pov"));

quizReplayBtn.addEventListener("click", () => {
  // Scrub the playback back to the start of the just-revealed event
  // so the user can rewatch the actual play in real time.
  if (!quiz.currentEvent && quiz.choices.size === 0) return;
  // Find the most recent committed event
  let target = null;
  for (const ev of quiz.summary().breakdown) target = ev.event; // last wins
  if (!target) return;
  jumpToFrame(target.frame_start);
  quizRevealEl.classList.add("hidden");
});

quizScorecardRestartBtn.addEventListener("click", () => {
  quizScorecardEl.classList.add("hidden");
  if (!data) return;
  // Reset quiz state and scrub to start
  quiz.deactivate();
  quiz.activate();
  quizScoreEl.textContent = "0 / 0";
  jumpToFrame(0);
  if (playback) {
    playback.play();
    playBtn.textContent = "Pause";
  }
});

quizScorecardCloseBtn.addEventListener("click", () => {
  quizScorecardEl.classList.add("hidden");
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
