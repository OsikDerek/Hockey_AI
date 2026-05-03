// Playback controller: drives the current frame index based on real time.

export class Playback {
  constructor(data) {
    this.data = data;
    this.frameIdx = 0;
    this.playing = false;
    this.speed = 1.0;
    this._lastTickMs = null;
  }

  play() { this.playing = true; this._lastTickMs = null; }
  pause() { this.playing = false; }
  togglePlay() { this.playing ? this.pause() : this.play(); }
  setFrame(idx) {
    this.frameIdx = Math.max(0, Math.min(this.data.totalFrames - 1, idx));
  }
  setSpeed(s) { this.speed = s; }

  // Advance frame index by real elapsed time. Returns true if the frame index changed.
  tick(nowMs) {
    if (!this.playing) {
      this._lastTickMs = nowMs;
      return false;
    }
    if (this._lastTickMs === null) {
      this._lastTickMs = nowMs;
      return false;
    }
    const dt = (nowMs - this._lastTickMs) / 1000;
    this._lastTickMs = nowMs;
    const delta = dt * this.data.fps * this.speed;
    const newIdx = Math.min(this.data.totalFrames - 1, this.frameIdx + delta);
    const oldIdx = this.frameIdx;
    this.frameIdx = newIdx;
    if (newIdx >= this.data.totalFrames - 1) {
      this.playing = false; // stop at end
    }
    return Math.floor(newIdx) !== Math.floor(oldIdx);
  }

  currentFrame() {
    return this.data.frameAt(Math.floor(this.frameIdx));
  }

  currentTimestamp() {
    const fr = this.currentFrame();
    return fr ? fr.timestamp_sec : 0;
  }
}
