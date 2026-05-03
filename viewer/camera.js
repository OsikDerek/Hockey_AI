// Camera controllers: top-down, broadcast-side, POV.
import * as THREE from "./lib/three.module.js";
import {
  RINK_LENGTH_FT, RINK_WIDTH_FT, CENTER_X, CENTER_Y,
} from "./rink.js";

export function makeTopDownCamera(aspect) {
  const halfL = RINK_LENGTH_FT / 2 + 12;
  const halfW = RINK_WIDTH_FT / 2 + 12;
  // Frame the rink while preserving aspect
  let viewW, viewH;
  if (aspect >= RINK_LENGTH_FT / RINK_WIDTH_FT) {
    viewH = halfW * 2;
    viewW = viewH * aspect;
  } else {
    viewW = halfL * 2;
    viewH = viewW / aspect;
  }
  const cam = new THREE.OrthographicCamera(
    -viewW / 2, viewW / 2,
    viewH / 2, -viewH / 2,
    1, 500,
  );
  cam.position.set(CENTER_X, 200, CENTER_Y);
  cam.lookAt(CENTER_X, 0, CENTER_Y);
  return cam;
}

export function makeBroadcastCamera(aspect) {
  const cam = new THREE.PerspectiveCamera(35, aspect, 0.1, 1000);
  cam.position.set(CENTER_X, 65, -55);
  cam.lookAt(CENTER_X, 0, CENTER_Y);
  return cam;
}

export function makePOVCamera(aspect) {
  const cam = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
  return cam;
}

// Update the POV camera to follow a player avatar at head height,
// looking forward in their facing direction.
export function updatePOVCamera(cam, avatar) {
  if (!avatar) return;
  const headHeight = 5.0;
  cam.position.set(avatar.position.x, headHeight, avatar.position.z);
  const yaw = avatar.userData.yaw ?? 0;
  const lookAheadFt = 30;
  const tx = avatar.position.x + Math.sin(yaw) * lookAheadFt;
  const tz = avatar.position.z + Math.cos(yaw) * lookAheadFt;
  cam.lookAt(tx, headHeight - 1.0, tz);
}

export function resizeCamera(cam, w, h) {
  if (cam.isOrthographicCamera) {
    const aspect = w / h;
    const halfL = RINK_LENGTH_FT / 2 + 12;
    const halfW = RINK_WIDTH_FT / 2 + 12;
    let viewW, viewH;
    if (aspect >= RINK_LENGTH_FT / RINK_WIDTH_FT) {
      viewH = halfW * 2;
      viewW = viewH * aspect;
    } else {
      viewW = halfL * 2;
      viewH = viewW / aspect;
    }
    cam.left = -viewW / 2;
    cam.right = viewW / 2;
    cam.top = viewH / 2;
    cam.bottom = -viewH / 2;
    cam.updateProjectionMatrix();
  } else if (cam.isPerspectiveCamera) {
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
  }
}
