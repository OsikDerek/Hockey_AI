// NHL rink geometry (feet). Z=0 is the ice surface, Y=0 is one side board,
// Y=85 is the other; X=0 to X=200 runs goal-line-to-goal-line.
import * as THREE from "./lib/three.module.js";

export const RINK_LENGTH_FT = 200;
export const RINK_WIDTH_FT = 85;
export const LEFT_GOAL_X = 11;
export const RIGHT_GOAL_X = 189;
export const CENTER_X = 100;
export const CENTER_Y = 42.5;
export const BLUE_LEFT_X = 75;
export const BLUE_RIGHT_X = 125;

const ICE_COLOR = 0xf3f7fa;
const RED = 0xd62828;
const BLUE = 0x2188ff;
const TAN = 0xc8a25a;

export function buildRink() {
  const group = new THREE.Group();

  // Ice surface — slight Y offset (0.01) so paint planes don't z-fight
  const iceGeo = new THREE.PlaneGeometry(RINK_LENGTH_FT, RINK_WIDTH_FT);
  const iceMat = new THREE.MeshStandardMaterial({ color: ICE_COLOR, roughness: 0.4 });
  const ice = new THREE.Mesh(iceGeo, iceMat);
  ice.rotation.x = -Math.PI / 2;
  ice.position.set(CENTER_X, 0, CENTER_Y);
  ice.receiveShadow = true;
  group.add(ice);

  // Center red line
  const centerLine = paintRect(2, RINK_WIDTH_FT, RED);
  centerLine.position.set(CENTER_X, 0.02, CENTER_Y);
  group.add(centerLine);

  // Blue lines
  for (const bx of [BLUE_LEFT_X, BLUE_RIGHT_X]) {
    const line = paintRect(2, RINK_WIDTH_FT, BLUE);
    line.position.set(bx, 0.02, CENTER_Y);
    group.add(line);
  }

  // Center ice circle (just a stylized ring)
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(14, 15, 64),
    new THREE.MeshBasicMaterial({ color: BLUE, side: THREE.DoubleSide })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(CENTER_X, 0.03, CENTER_Y);
  group.add(ring);

  // Center ice dot
  group.add(paintDot(CENTER_X, CENTER_Y, 1.0, BLUE));

  // Faceoff dots (4 end-zone, 4 neutral-zone)
  for (const x of [LEFT_GOAL_X + 20, RIGHT_GOAL_X - 20]) {
    for (const y of [22, RINK_WIDTH_FT - 22]) {
      group.add(paintDot(x, y, 1.0, RED));
    }
  }
  for (const x of [BLUE_LEFT_X + 5, BLUE_RIGHT_X - 5]) {
    for (const y of [22, RINK_WIDTH_FT - 22]) {
      group.add(paintDot(x, y, 0.7, RED));
    }
  }

  // Goal creases (blue half-discs)
  for (const [gx, dir] of [[LEFT_GOAL_X, 1], [RIGHT_GOAL_X, -1]]) {
    const crease = new THREE.Mesh(
      new THREE.CircleGeometry(6, 32, dir > 0 ? -Math.PI / 2 : Math.PI / 2, Math.PI),
      new THREE.MeshBasicMaterial({ color: BLUE, transparent: true, opacity: 0.35 })
    );
    crease.rotation.x = -Math.PI / 2;
    crease.position.set(gx, 0.02, CENTER_Y);
    group.add(crease);
  }

  // Goal line
  for (const gx of [LEFT_GOAL_X, RIGHT_GOAL_X]) {
    const goalLine = paintRect(0.5, RINK_WIDTH_FT, RED);
    goalLine.position.set(gx, 0.02, CENTER_Y);
    group.add(goalLine);
  }

  // Goal nets (simple wireframe boxes 6 ft wide x 4 ft tall x 4 ft deep)
  for (const [gx, dir] of [[LEFT_GOAL_X, -1], [RIGHT_GOAL_X, 1]]) {
    const net = new THREE.Mesh(
      new THREE.BoxGeometry(4, 4, 6),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true })
    );
    net.position.set(gx + dir * 2, 2, CENTER_Y);
    group.add(net);
  }

  // Boards — 4 thin walls around the rink
  const boardHeight = 3.5;
  const boardThick = 0.5;
  const longBoard = new THREE.BoxGeometry(RINK_LENGTH_FT, boardHeight, boardThick);
  const shortBoard = new THREE.BoxGeometry(boardThick, boardHeight, RINK_WIDTH_FT);
  const boardMat = new THREE.MeshStandardMaterial({ color: TAN, roughness: 0.7 });
  const placements = [
    [longBoard, CENTER_X, boardHeight / 2, -boardThick / 2],
    [longBoard, CENTER_X, boardHeight / 2, RINK_WIDTH_FT + boardThick / 2],
    [shortBoard, -boardThick / 2, boardHeight / 2, CENTER_Y],
    [shortBoard, RINK_LENGTH_FT + boardThick / 2, boardHeight / 2, CENTER_Y],
  ];
  for (const [geo, x, y, z] of placements) {
    const m = new THREE.Mesh(geo, boardMat);
    m.position.set(x, y, z);
    group.add(m);
  }

  return group;
}

function paintRect(w, h, color) {
  const geo = new THREE.PlaneGeometry(w, h);
  const mat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.x = -Math.PI / 2;
  return m;
}

function paintDot(x, y, radius, color) {
  const geo = new THREE.CircleGeometry(radius, 24);
  const mat = new THREE.MeshBasicMaterial({ color });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.x = -Math.PI / 2;
  m.position.set(x, 0.025, y);
  return m;
}
