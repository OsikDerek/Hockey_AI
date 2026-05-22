// NHL rink geometry (feet). Z=0 is the ice surface, Y=0 is one side board,
// Y=85 is the other; X=0 to X=200 runs goal-line-to-goal-line.
//
// The rink is NOT a rectangle: the corners are quarter-circle arcs of a
// 28 ft radius, so the ice sheet and the boards both follow a rounded
// rectangle. Painted lines that would run into a corner (the goal lines)
// are clipped to the actual board-to-board chord at their x.
import * as THREE from "./lib/three.module.js";

export const RINK_LENGTH_FT = 200;
export const RINK_WIDTH_FT = 85;
export const CORNER_RADIUS_FT = 28; // NHL regulation corner arc radius
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

// A rounded rectangle THREE.Shape centred on the origin: x in [-L/2, L/2],
// y in [-W/2, W/2], with quarter-circle corners of the given radius.
function roundedRectShape(length, width, radius) {
  const hl = length / 2;
  const hw = width / 2;
  const r = Math.min(radius, hl, hw);
  const s = new THREE.Shape();
  s.moveTo(-hl + r, -hw);
  s.lineTo(hl - r, -hw);
  s.absarc(hl - r, -hw + r, r, -Math.PI / 2, 0, false);
  s.lineTo(hl, hw - r);
  s.absarc(hl - r, hw - r, r, 0, Math.PI / 2, false);
  s.lineTo(-hl + r, hw);
  s.absarc(-hl + r, hw - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(-hl, -hw + r);
  s.absarc(-hl + r, -hw + r, r, Math.PI, Math.PI * 1.5, false);
  s.closePath();
  return s;
}

// Half-width of the ice at a given x (feet from the rink centre line of
// width). Inside a corner arc the rink narrows; elsewhere it is the full
// half-width. Used to clip the goal lines to the board-to-board chord.
export function rinkHalfWidthAt(x) {
  const hw = RINK_WIDTH_FT / 2;
  const r = CORNER_RADIUS_FT;
  const cornerX = Math.min(x, RINK_LENGTH_FT - x); // distance into the end
  if (cornerX >= r) return hw;                     // straight side board
  // Within the corner arc: arc centre is r in from each board.
  const dx = r - cornerX;
  return hw - r + Math.sqrt(Math.max(0, r * r - dx * dx));
}

export function buildRink() {
  const group = new THREE.Group();

  // Ice surface — a rounded-rectangle sheet, not a plain plane.
  const iceShape = roundedRectShape(RINK_LENGTH_FT, RINK_WIDTH_FT,
                                    CORNER_RADIUS_FT);
  const iceGeo = new THREE.ShapeGeometry(iceShape);
  const iceMat = new THREE.MeshStandardMaterial({
    color: ICE_COLOR, roughness: 0.4, side: THREE.DoubleSide,
  });
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

  // Goal lines — clipped to the board-to-board chord, since at x=11/189
  // the line runs into the rounded corners and is shorter than full width.
  for (const gx of [LEFT_GOAL_X, RIGHT_GOAL_X]) {
    const goalLine = paintRect(0.5, 2 * rinkHalfWidthAt(gx), RED);
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

  // Boards — a continuous wall following the rounded-rectangle perimeter.
  // Built by extruding a thin frame (the rink outline with a slightly
  // smaller rink outline punched out as a hole) upward by the board height.
  const boardHeight = 3.5;
  const boardThick = 0.6;
  const outer = roundedRectShape(
    RINK_LENGTH_FT + 2 * boardThick, RINK_WIDTH_FT + 2 * boardThick,
    CORNER_RADIUS_FT + boardThick);
  outer.holes.push(
    roundedRectShape(RINK_LENGTH_FT, RINK_WIDTH_FT, CORNER_RADIUS_FT));
  const boardGeo = new THREE.ExtrudeGeometry(outer, {
    depth: boardHeight, bevelEnabled: false,
  });
  const boardMat = new THREE.MeshStandardMaterial({ color: TAN, roughness: 0.7 });
  const boards = new THREE.Mesh(boardGeo, boardMat);
  boards.rotation.x = -Math.PI / 2;     // extrude direction -> world +Y
  boards.position.set(CENTER_X, 0, CENTER_Y);
  group.add(boards);

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
