// Stick-figure player avatar: capsule body + head + stick + procedural
// skating limbs. Pose is faked from velocity since we don't have joint data.
import * as THREE from "./lib/three.module.js";

const TEAM_COLORS = {
  team_a: 0x28dc28,
  team_b: 0x2828dc,
  unknown: 0x9aa3ad,
};

const SKIN_COLOR = 0xe2bfa1;
const STICK_COLOR = 0x6a4220;
// Blade was 0x121212 (near-black) — read as a puck on every player from
// top-down. Mid-tone tape-gray + heel/toe shape now reads as "stick blade".
const BLADE_COLOR = 0x4a5560;

export function createAvatar(team, isGoalie = false) {
  const color = TEAM_COLORS[team] || TEAM_COLORS.unknown;
  const root = new THREE.Group();
  const heightFt = isGoalie ? 5.0 : 5.5;
  const radius = isGoalie ? 1.4 : 0.9;

  // Foot disc — a flat team-colored disc on the ice at the avatar's feet.
  // Crucial for top-down readability; the body capsule is invisible from
  // straight overhead.
  const discRadius = isGoalie ? 2.4 : 1.6;
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(discRadius, 24),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85 }),
  );
  disc.rotation.x = -Math.PI / 2;
  disc.position.y = 0.04;
  root.add(disc);
  const discRing = new THREE.Mesh(
    new THREE.RingGeometry(discRadius * 0.95, discRadius, 24),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }),
  );
  discRing.rotation.x = -Math.PI / 2;
  discRing.position.y = 0.05;
  root.add(discRing);

  // Body: capsule from waist to chest
  const bodyMat = new THREE.MeshStandardMaterial({ color, roughness: 0.5 });
  const body = new THREE.Mesh(
    new THREE.CapsuleGeometry(radius, heightFt - 2.5, 4, 12),
    bodyMat,
  );
  body.position.y = heightFt - 2.0;
  body.castShadow = true;
  root.add(body);

  // Head
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.55, 12, 12),
    new THREE.MeshStandardMaterial({ color: SKIN_COLOR, roughness: 0.6 }),
  );
  head.position.y = heightFt - 0.4;
  head.castShadow = true;
  root.add(head);

  // Legs (two cylinders pivoted at the hip)
  const legGeo = new THREE.CylinderGeometry(0.25, 0.25, 2.4, 8);
  const legMat = new THREE.MeshStandardMaterial({ color, roughness: 0.5 });
  const leftHip = new THREE.Group();
  leftHip.position.set(-radius * 0.6, heightFt - 2.7, 0);
  const leftLeg = new THREE.Mesh(legGeo, legMat);
  leftLeg.position.y = -1.2;
  leftLeg.castShadow = true;
  leftHip.add(leftLeg);
  root.add(leftHip);

  const rightHip = new THREE.Group();
  rightHip.position.set(radius * 0.6, heightFt - 2.7, 0);
  const rightLeg = new THREE.Mesh(legGeo, legMat);
  rightLeg.position.y = -1.2;
  rightLeg.castShadow = true;
  rightHip.add(rightLeg);
  root.add(rightHip);

  // Arms (two cylinders pivoted at the shoulder, both reaching forward to grip stick)
  const armGeo = new THREE.CylinderGeometry(0.18, 0.18, 1.8, 8);
  const armMat = new THREE.MeshStandardMaterial({ color, roughness: 0.5 });
  const leftShoulder = new THREE.Group();
  leftShoulder.position.set(-radius * 0.9, heightFt - 1.0, 0);
  const leftArm = new THREE.Mesh(armGeo, armMat);
  leftArm.position.y = -0.9;
  leftShoulder.add(leftArm);
  leftShoulder.rotation.x = -Math.PI / 2.6;
  root.add(leftShoulder);

  const rightShoulder = new THREE.Group();
  rightShoulder.position.set(radius * 0.9, heightFt - 1.0, 0);
  const rightArm = new THREE.Mesh(armGeo, armMat);
  rightArm.position.y = -0.9;
  rightShoulder.add(rightArm);
  rightShoulder.rotation.x = -Math.PI / 2.2;
  root.add(rightShoulder);

  // Stick: shaft + blade. Held forward and slightly off to the side.
  if (!isGoalie) {
    const stick = new THREE.Group();
    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.08, 5.5, 6),
      new THREE.MeshStandardMaterial({ color: STICK_COLOR }),
    );
    shaft.rotation.z = Math.PI / 2;
    shaft.rotation.y = -Math.PI / 12;
    shaft.position.set(2.0, 0.4, 0.4);
    stick.add(shaft);
    // Slim, slightly-raised, slightly-yawed blade. Old shape (1.2 × 0.2 × 0.4
    // flat at y=0.05) read as a puck on every player from top-down.
    const blade = new THREE.Mesh(
      new THREE.BoxGeometry(1.4, 0.08, 0.18),
      new THREE.MeshStandardMaterial({ color: BLADE_COLOR, roughness: 0.7 }),
    );
    blade.position.set(4.6, 0.18, 0.55);
    blade.rotation.y = -Math.PI / 16; // slight toe-out so it reads as a blade
    blade.name = "stickBlade";
    stick.add(blade);
    root.add(stick);
    // Stash a reference for puck-snap lookup.
    root.userData.stickBlade = blade;
  } else {
    // Goalie pad/glove rough cue: a wider, lower colored block in front
    const pad = new THREE.Mesh(
      new THREE.BoxGeometry(2.4, 1.4, 0.8),
      new THREE.MeshStandardMaterial({ color, roughness: 0.5 }),
    );
    pad.position.set(0, heightFt - 3.4, 0.7);
    root.add(pad);

    // Goalie stick: wide paddle on the ice + short angled shaft up to glove.
    // Distinct from player stick — paddle is roughly 2x as wide as a player
    // blade, sits centered in front of the pad rather than off to the side,
    // and the shaft is shorter + tilted more upright (goalies stand in a
    // crouch with hands at hip height, not extended out).
    const goalieStick = new THREE.Group();
    const paddle = new THREE.Mesh(
      new THREE.BoxGeometry(3.0, 0.18, 0.6),
      new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.55 }),
    );
    paddle.position.set(0, 0.10, 1.9);
    paddle.castShadow = true;
    paddle.name = "goaliePaddle";
    goalieStick.add(paddle);
    const goalieShaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.13, 0.13, 3.4, 6),
      new THREE.MeshStandardMaterial({ color: STICK_COLOR }),
    );
    // Angle the shaft forward-up from the pad to the paddle: ~45° tilt
    // so it visually connects the goalie's body to the paddle on the ice.
    goalieShaft.rotation.x = Math.PI / 4;
    goalieShaft.position.set(0, 1.3, 0.85);
    goalieShaft.castShadow = true;
    goalieStick.add(goalieShaft);
    root.add(goalieStick);
    root.userData.goalieStick = goalieStick;
  }

  // Stash limb refs for skating animation
  root.userData.limbs = { leftHip, rightHip };
  root.userData.cyclePhase = Math.random() * Math.PI * 2;
  root.userData.team = team;
  root.userData.isGoalie = isGoalie;

  return root;
}

export function updateAvatar(avatar, prevPos, currPos, dtSeconds) {
  // Position (Y is up; ice is at Y=0)
  avatar.position.set(currPos.x, 0, currPos.y);

  // Facing: look toward velocity direction. Use a small EMA on the
  // facing angle so jitter doesn't spin the avatar.
  const dx = currPos.x - prevPos.x;
  const dy = currPos.y - prevPos.y;
  const speed = Math.hypot(dx, dy) / Math.max(1e-3, dtSeconds);
  if (speed > 0.5) {
    const targetYaw = Math.atan2(dx, dy);
    const prev = avatar.userData.yaw ?? targetYaw;
    const diff = wrapAngle(targetYaw - prev);
    const newYaw = prev + diff * 0.25;
    avatar.userData.yaw = newYaw;
    avatar.rotation.y = newYaw;
  }

  // Skating cycle: only animate when moving
  const limbs = avatar.userData.limbs;
  if (limbs && !avatar.userData.isGoalie) {
    if (speed > 0.5) {
      avatar.userData.cyclePhase += dtSeconds * Math.min(8, speed * 0.6);
      const swing = Math.sin(avatar.userData.cyclePhase) * 0.6;
      limbs.leftHip.rotation.x = swing;
      limbs.rightHip.rotation.x = -swing;
    } else {
      // Decay toward neutral standing pose
      limbs.leftHip.rotation.x *= 0.9;
      limbs.rightHip.rotation.x *= 0.9;
    }
  }
}

function wrapAngle(a) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

export function createPuck() {
  // Group: puck disc + bright outline ring so the one-and-only puck is
  // instantly distinguishable from each player's stick blade at any zoom.
  const group = new THREE.Group();
  const disc = new THREE.Mesh(
    new THREE.CylinderGeometry(0.5, 0.5, 0.25, 20),
    new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 0.45 }),
  );
  disc.castShadow = true;
  group.add(disc);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.55, 0.85, 24),
    new THREE.MeshBasicMaterial({
      color: 0xffb347, side: THREE.DoubleSide,
      transparent: true, opacity: 0.9,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.13; // just above the disc top
  group.add(ring);
  group.userData.ring = ring;
  return group;
}
