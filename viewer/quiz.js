// Phase D Quiz: turn passive review into active decision training.
//
// Each high-confidence event becomes a quiz item. Playback auto-pauses
// ~1s before the actor's puck release, the camera snaps to that player's
// POV, and the user picks what they would have done. After commit, we
// reveal the actual decision + AI rating + recommended option, and
// accumulate a score across the clip.

// Per-event-type choice menus. Only events whose decision_made matches
// one of the menu options are quiz-eligible — keeps the experience
// coherent (no "guess the forecheck pressure level" trivia).
export const QUIZ_OPTIONS = {
  shot_vs_pass:        ["shot", "pass", "dump"],
  odd_man_rush:        ["shot", "pass", "deke"],
  zone_entry:          ["carry", "dump", "pass_in"],
  breakout:            ["carry", "rim", "direct_pass", "chip"],
  missed_opportunity:  ["shot", "pass", "hold"],
};

const REVEAL_HOLD_MS = 4000;
const PAUSE_LEAD_FRAMES = 30;  // ~1 second before the decision frame at 30fps

export class Quiz {
  constructor() {
    this.active = false;
    this.phase = "idle";       // "idle" | "paused" | "revealed"
    this.currentEvent = null;
    this.choices = new Map();  // event_index -> { picked, actual, matched, rating }
    this.score = { matched: 0, total: 0 };
    // Cached map: only events whose type has a quiz menu
    this._eligibleEvents = [];
    // Track which events we've already prompted for so we don't re-trigger
    this._consumedEventIdxs = new Set();
    this._revealTimerId = null;
    // Wall-clock target for resuming after reveal
    this._revealUntilMs = 0;
    // Callbacks set by the viewer
    this.onPause = null;     // (event) => void  — viewer pauses + shows choice UI
    this.onReveal = null;    // (event, result) => void
    this.onResume = null;    // () => void  — viewer hides quiz UI, resumes playback
    this.onScoreChange = null;
  }

  loadFromData(data) {
    if (!data || !data.events) {
      this._eligibleEvents = [];
      return;
    }
    // Quality filter: only keep events the viewer can render meaningfully.
    // We pre-build a frame-index → calibrated-frame index lookup so that
    // for each event we can check whether the actor's track_id has been
    // calibrated recently AND whether enough other players have positions
    // to form a coachable scene. Without this, the quiz fires prompts
    // pointing at empty rinks.
    const frames = data.frames || [];
    // Per-track last-calibrated-frame map, built once in a forward sweep.
    const lastCalibFrame = new Map();   // track_id -> last frame_idx
    let lastFrameAnyPlayers = -1;
    let recentPlayerCount = new Map();  // frame_idx -> distinct active tracks
    for (let i = 0; i < frames.length; i++) {
      const fr = frames[i];
      if (!fr.calibrated) continue;
      lastFrameAnyPlayers = i;
      const seen = new Set();
      for (const p of fr.players || []) {
        if (p.track_id !== undefined && p.track_id >= 0) {
          lastCalibFrame.set(p.track_id, i);
          seen.add(p.track_id);
        }
      }
      for (const g of fr.goalies || []) {
        if (g.track_id !== undefined && g.track_id >= 0) {
          lastCalibFrame.set(g.track_id, i);
          seen.add(g.track_id);
        }
      }
      recentPlayerCount.set(i, seen.size);
    }

    // Renderable threshold: actor must have a calibrated position within
    // RECENCY_FRAMES of the event, AND there must be ≥4 distinct tracks
    // with recent positions for the scene to read as hockey.
    const RECENCY_FRAMES = 90;  // ~3 sec @ 30fps
    const fps = data.fps || 30;

    function recentTrackCount(eventIdx) {
      let count = 0;
      const cutoff = eventIdx - RECENCY_FRAMES;
      for (const [tid, lastI] of lastCalibFrame) {
        if (lastI >= cutoff && lastI <= eventIdx) count++;
      }
      return count;
    }

    this._eligibleEvents = data.events
      .filter((e) => QUIZ_OPTIONS[e.event_type])
      .filter((e) => {
        if (e.player_id === null || e.player_id === undefined) return false;
        const lastSeen = lastCalibFrame.get(e.player_id);
        if (lastSeen === undefined) return false;
        if (e.frame_idx - lastSeen > RECENCY_FRAMES) return false;
        if (recentTrackCount(e.frame_idx) < 4) return false;
        return true;
      })
      .sort((a, b) => a.frame_idx - b.frame_idx);

    // Diagnostic to help triage why quiz might be silent on a given clip.
    const total = data.events.filter((e) => QUIZ_OPTIONS[e.event_type]).length;
    console.log(
      `Quiz: ${this._eligibleEvents.length}/${total} quiz-eligible events ` +
      `pass the renderable filter (actor has calibrated history + >=4 players in scene)`
    );
  }

  toggle() {
    this.active ? this.deactivate() : this.activate();
  }

  activate() {
    this.active = true;
    this.phase = "idle";
    this._consumedEventIdxs.clear();
    this.choices.clear();
    this.score = { matched: 0, total: 0 };
    if (this.onScoreChange) this.onScoreChange(this.score);
  }

  deactivate() {
    this.active = false;
    this.phase = "idle";
    this.currentEvent = null;
    if (this._revealTimerId !== null) {
      clearTimeout(this._revealTimerId);
      this._revealTimerId = null;
    }
    if (this.onResume) this.onResume();
  }

  /** Per-tick check: if quiz is active and we've reached a pre-event window
   *  for an unconsumed event, fire onPause and transition to "paused" state.
   *  Called from the viewer's render loop. `currentFrameIdx` is the playback
   *  position; `playing` is whether playback is advancing. */
  tick(currentFrameIdx, playing) {
    if (!this.active || this.phase !== "idle" || !playing) return;
    for (const ev of this._eligibleEvents) {
      const ekey = ev.frame_idx;  // unique enough for our purposes
      if (this._consumedEventIdxs.has(ekey)) continue;
      const triggerFrame = ev.frame_idx - PAUSE_LEAD_FRAMES;
      if (currentFrameIdx >= triggerFrame && currentFrameIdx < ev.frame_idx + 5) {
        // Fire pause for this event
        this._consumedEventIdxs.add(ekey);
        this.currentEvent = ev;
        this.phase = "paused";
        if (this.onPause) this.onPause(ev);
        return;
      }
    }
  }

  /** Called by the viewer when the user picks a choice. */
  commit(picked) {
    if (this.phase !== "paused" || !this.currentEvent) return;
    const ev = this.currentEvent;
    const actual = (ev.decision_made || "").toLowerCase();
    const pickedLower = (picked || "").toLowerCase();
    // Missed opportunity: the actual player DIDN'T act, so "hold" = match
    let matched;
    if (ev.event_type === "missed_opportunity") {
      matched = pickedLower === "hold";
    } else {
      matched = pickedLower === actual;
    }
    const result = {
      picked: pickedLower,
      actual,
      matched,
      rating: ev.rating || "neutral",
    };
    this.choices.set(ev.frame_idx, result);
    this.score.total += 1;
    if (matched) this.score.matched += 1;
    if (this.onScoreChange) this.onScoreChange(this.score);

    this.phase = "revealed";
    if (this.onReveal) this.onReveal(ev, result);
    this._revealUntilMs = performance.now() + REVEAL_HOLD_MS;
    this._revealTimerId = setTimeout(() => this._afterReveal(), REVEAL_HOLD_MS);
  }

  /** Skip the current question without recording a choice — useful for
   *  scrubbing past a moment the user doesn't want to engage with. */
  skip() {
    if (this.phase !== "paused" || !this.currentEvent) return;
    this.phase = "idle";
    this.currentEvent = null;
    if (this.onResume) this.onResume();
  }

  _afterReveal() {
    this._revealTimerId = null;
    this.currentEvent = null;
    this.phase = "idle";
    if (this.onResume) this.onResume();
  }

  /** Return the lowest trigger frame for an unconsumed quiz event in
   *  [fromFrame, toFrame], or null if none. Used by the viewer's
   *  auto-skip-uncalibrated logic to clamp the skip and not leap past
   *  a quiz event sitting in an uncalibrated gap. */
  firstUnconsumedTriggerInRange(fromFrame, toFrame) {
    let best = null;
    for (const ev of this._eligibleEvents) {
      if (this._consumedEventIdxs.has(ev.frame_idx)) continue;
      const trigger = ev.frame_idx - PAUSE_LEAD_FRAMES;
      if (trigger >= fromFrame && trigger <= toFrame) {
        if (best === null || trigger < best) best = trigger;
      }
    }
    return best;
  }

  /** Reset quiz progress when the user scrubs backward. We only re-prompt
   *  for events the user crossed back over. */
  onScrubTo(frameIdx) {
    if (!this.active) return;
    // Drop consumed flags for events whose trigger frame is now in the future
    for (const ev of this._eligibleEvents) {
      const triggerFrame = ev.frame_idx - PAUSE_LEAD_FRAMES;
      if (frameIdx < triggerFrame) {
        this._consumedEventIdxs.delete(ev.frame_idx);
      }
    }
    // If we're mid-pause and the user scrubbed past the event, dismiss
    if (this.phase !== "idle" && this.currentEvent) {
      const ev = this.currentEvent;
      if (frameIdx > ev.frame_idx + 5 || frameIdx < ev.frame_idx - PAUSE_LEAD_FRAMES) {
        this.phase = "idle";
        this.currentEvent = null;
        if (this._revealTimerId !== null) {
          clearTimeout(this._revealTimerId);
          this._revealTimerId = null;
        }
        if (this.onResume) this.onResume();
      }
    }
  }
}
