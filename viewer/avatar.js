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
const BLADE_COLOR = 0x121212;

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
    const blade = new THREE.Mesh(
      new THREE.BoxGeometry(1.2, 0.2, 0.4),
      new THREE.MeshStandardMaterial({ color: BLADE_COLOR }),
    );
    blade.position.set(4.5, 0.05, 0.5);
    stick.add(blade);
    root.add(stick);
  } else {
    // Goalie pad/glove rough cue: a wider, lower colored block in front
    const pad = new THREE.Mesh(
      new THREE.BoxGeometry(2.4, 1.4, 0.8),
      new THREE.MeshStandardMaterial({ color, roughness: 0.5 }),
    );
    pad.position.set(0, heightFt - 3.4, 0.7);
    root.add(pad);
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
  const puck = new THREE.Mesh(
    new THREE.CylinderGeometry(0.5, 0.5, 0.25, 16),
    new THREE.MeshStandardMaterial({ color: 0x111111 }),
  );
  puck.castShadow = true;
  return puck;
}
