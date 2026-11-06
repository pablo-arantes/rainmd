/**
 * The bonded terms, as a reference implementation.
 *
 * 🔴 THESE ARE NOT PERIODIC, BECAUSE OPENMM'S ARE NOT. HarmonicBondForce,
 * HarmonicAngleForce and PeriodicTorsionForce all default to
 * `usesPeriodicBoundaryConditions = false`, and AmberPrmtopFile does not change
 * it - so a bonded term reads the plain difference of coordinates, and a
 * molecule must therefore stay whole in the coordinate array. This is why
 * nothing in this engine wraps positions into the box during a run: wrapping
 * would split a molecule across the boundary and stretch one of its bonds
 * across the whole cell, which is an immediate explosion rather than a drift.
 * Imaging is applied when a frame is WRITTEN, never to the running state.
 */
import { cross } from "./geometry.js";

/**
 * @param {object} system from buildSystem
 * @param {Float64Array} positions 3N, nm
 * @param {Float64Array} forces 3N, accumulated into, kJ/mol/nm
 * @returns {number} energy in kJ/mol
 */
export function harmonicBondEnergy(system, positions, forces) {
  const { count, atomA, atomB, k, r0 } = system.harmonicBonds;
  let energy = 0;
  for (let index = 0; index < count; index += 1) {
    const a = atomA[index] * 3;
    const b = atomB[index] * 3;
    const dx = positions[a] - positions[b];
    const dy = positions[a + 1] - positions[b + 1];
    const dz = positions[a + 2] - positions[b + 2];
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const stretch = r - r0[index];
    energy += 0.5 * k[index] * stretch * stretch;
    // dE/dr = k * stretch; the force is -dE/dr along the unit vector.
    const scale = r > 0 ? (k[index] * stretch) / r : 0;
    forces[a] -= scale * dx;
    forces[a + 1] -= scale * dy;
    forces[a + 2] -= scale * dz;
    forces[b] += scale * dx;
    forces[b + 1] += scale * dy;
    forces[b + 2] += scale * dz;
  }
  return energy;
}

/**
 * E = (1/2) k (theta - theta0)^2.
 *
 * 🔴 THE GRADIENT IS SINGULAR AT 0 AND 180 DEGREES and a linear arrangement is
 * not hypothetical: a CO2, an azide, or a water whose H-H constraint is being
 * solved can pass through it. `sin(theta)` in the denominator goes to zero and
 * the force goes to infinity, which in f32 on a GPU is a NaN that reaches every
 * atom within two steps. The guard clamps the cross-product magnitude rather
 * than the angle, because that is the quantity actually divided by.
 */
export function harmonicAngleEnergy(system, positions, forces) {
  const { count, atomA, atomB, atomC, k, theta0 } = system.topology.angles;
  const constrainedPairs = constrainedPairSet(system);
  let energy = 0;
  const normal = new Float64Array(3);
  for (let index = 0; index < count; index += 1) {
    const ia = atomA[index];
    const ib = atomB[index];
    const ic = atomC[index];
    const a = ia * 3;
    const b = ib * 3;
    const c = ic * 3;
    const v1x = positions[a] - positions[b];
    const v1y = positions[a + 1] - positions[b + 1];
    const v1z = positions[a + 2] - positions[b + 2];
    const v2x = positions[c] - positions[b];
    const v2y = positions[c + 1] - positions[b + 1];
    const v2z = positions[c + 2] - positions[b + 2];
    const n1 = v1x * v1x + v1y * v1y + v1z * v1z;
    const n2 = v2x * v2x + v2y * v2y + v2z * v2z;
    cross(v1x, v1y, v1z, v2x, v2y, v2z, normal);
    const crossMagnitude = Math.sqrt(
      normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2],
    );
    const dot = v1x * v2x + v1y * v2y + v1z * v2z;
    const theta = Math.atan2(crossMagnitude, dot);
    const deviation = theta - theta0[index];
    energy += 0.5 * k[index] * deviation * deviation;
    const dEdTheta = k[index] * deviation;
    const safeCross = Math.max(crossMagnitude, 1e-12);
    // 🔴 BOTH PREFACTORS ARE NEGATIVE OF WHAT THEY LOOK LIKE THEY SHOULD BE,
    // and getting them backwards leaves the ENERGY exactly right - the energy
    // never reads the gradient. Caught only by differentiating this function
    // numerically (tools/check-force-gradients.mjs), which reported 9.0 against
    // a 1e-5 bar while every angle energy matched OpenMM to 2.3e-15.
    //   F_a = -(dE/dtheta / (|v1|^2 |v1 x v2|)) (v1 x (v1 x v2))
    //   F_c = +(dE/dtheta / (|v2|^2 |v1 x v2|)) (v2 x (v1 x v2))
    const scale1 = -dEdTheta / (n1 * safeCross);
    const scale2 = dEdTheta / (n2 * safeCross);
    const fax = scale1 * (v1y * normal[2] - v1z * normal[1]);
    const fay = scale1 * (v1z * normal[0] - v1x * normal[2]);
    const faz = scale1 * (v1x * normal[1] - v1y * normal[0]);
    const fcx = scale2 * (v2y * normal[2] - v2z * normal[1]);
    const fcy = scale2 * (v2z * normal[0] - v2x * normal[2]);
    const fcz = scale2 * (v2x * normal[1] - v2y * normal[0]);
    forces[a] += fax;
    forces[a + 1] += fay;
    forces[a + 2] += faz;
    forces[c] += fcx;
    forces[c + 1] += fcy;
    forces[c + 2] += fcz;
    forces[b] -= fax + fcx;
    forces[b + 1] -= fay + fcy;
    forces[b + 2] -= faz + fcz;
  }
  return energy;
}

/** Unused by the angle term itself; kept for the HAngles path this port refuses. */
function constrainedPairSet() { return null; }

/**
 * E = k (1 + cos(n*phi - phase)), for proper and improper torsions alike.
 *
 * 🔴 AMBER'S IMPROPERS USE THE SAME FORMULA AS ITS PROPERS. They are marked by
 * a negative fourth atom index in the topology and differ ONLY in that they
 * generate no 1-4 pair - there is no separate harmonic-improper term as in
 * CHARMM. A port that routes them to a different functional form gets the
 * planarity of every aromatic ring and every amide wrong.
 */
export function periodicTorsionEnergy(system, positions, forces) {
  const { count, atomA, atomB, atomC, atomD, k, periodicity, phase } = system.topology.dihedrals;
  let energy = 0;
  const n1 = new Float64Array(3);
  const n2 = new Float64Array(3);
  for (let index = 0; index < count; index += 1) {
    const a = atomA[index] * 3;
    const b = atomB[index] * 3;
    const c = atomC[index] * 3;
    const d = atomD[index] * 3;
    const b1x = positions[b] - positions[a];
    const b1y = positions[b + 1] - positions[a + 1];
    const b1z = positions[b + 2] - positions[a + 2];
    const b2x = positions[c] - positions[b];
    const b2y = positions[c + 1] - positions[b + 1];
    const b2z = positions[c + 2] - positions[b + 2];
    const b3x = positions[d] - positions[c];
    const b3y = positions[d + 1] - positions[c + 1];
    const b3z = positions[d + 2] - positions[c + 2];
    cross(b1x, b1y, b1z, b2x, b2y, b2z, n1);
    cross(b2x, b2y, b2z, b3x, b3y, b3z, n2);
    const n1Squared = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
    const n2Squared = n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2];
    const b2Length = Math.sqrt(b2x * b2x + b2y * b2y + b2z * b2z);
    if (n1Squared < 1e-24 || n2Squared < 1e-24 || b2Length < 1e-12) continue;
    // phi from atan2, so the sign of the torsion is right. Using acos of the
    // normalised dot alone loses it, and a torsion with the wrong sign has the
    // right energy and a mirrored force.
    const dotNormals = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
    const crossNormalsDotB2 = (n1[1] * n2[2] - n1[2] * n2[1]) * b2x
      + (n1[2] * n2[0] - n1[0] * n2[2]) * b2y
      + (n1[0] * n2[1] - n1[1] * n2[0]) * b2z;
    const phi = Math.atan2(crossNormalsDotB2 / b2Length, dotNormals);
    const n = periodicity[index];
    const argument = n * phi - phase[index];
    energy += k[index] * (1 + Math.cos(argument));
    const dEdPhi = -k[index] * n * Math.sin(argument);

    // 🔴 THE TWO END PREFACTORS WERE NEGATED, WHICH NEGATES THE WHOLE TORSION
    // FORCE - the middle two atoms are linear combinations of these, so the
    // combination below was right and the inputs were not. With the energy
    // already exact to the last digit against OpenMM (phases included, which is
    // what proves the phi convention here is right), nothing but a numerical
    // gradient could see it.
    //   dphi/dr_i = -(|b2|/|n1|^2) n1,  F_i = -dE/dphi * dphi/dr_i
    const forceOnA = dEdPhi * b2Length / n1Squared;
    const forceOnD = -dEdPhi * b2Length / n2Squared;
    const fax = forceOnA * n1[0];
    const fay = forceOnA * n1[1];
    const faz = forceOnA * n1[2];
    const fdx = forceOnD * n2[0];
    const fdy = forceOnD * n2[1];
    const fdz = forceOnD * n2[2];
    const b2SquaredInverse = 1 / (b2Length * b2Length);
    const t = (b1x * b2x + b1y * b2y + b1z * b2z) * b2SquaredInverse;
    const u = (b3x * b2x + b3y * b2y + b3z * b2z) * b2SquaredInverse;
    forces[a] += fax;
    forces[a + 1] += fay;
    forces[a + 2] += faz;
    forces[d] += fdx;
    forces[d + 1] += fdy;
    forces[d + 2] += fdz;
    // 🔴 THE MIDDLE TWO ATOMS ARE NOT `-fa + t*fa - u*fd`, WHICH IS THE FORM
    // THIS PORT HAD AND WHICH IS WHAT MOST DERIVATIONS LOOK LIKE AT A GLANCE.
    // The coefficients were obtained by differentiating this file's own `phi`
    // numerically and solving for them, because every remembered form
    // disagreed with it:
    //   dphi/dr_j = -(1+t) dphi/dr_i + u dphi/dr_l
    //   dphi/dr_k =     t  dphi/dr_i - (1+u) dphi/dr_l
    // with t = (b1.b2)/|b2|^2 and u = (b3.b2)/|b2|^2. The four together sum to
    // zero, which is the invariance a torsion force must have and the one cheap
    // check that catches a wrong coefficient here: a form that does not sum to
    // zero puts a net force on the molecule and makes it fly.
    forces[b] += -(1 + t) * fax + u * fdx;
    forces[b + 1] += -(1 + t) * fay + u * fdy;
    forces[b + 2] += -(1 + t) * faz + u * fdz;
    forces[c] += t * fax - (1 + u) * fdx;
    forces[c + 1] += t * fay - (1 + u) * fdy;
    forces[c + 2] += t * faz - (1 + u) * fdz;
  }
  return energy;
}
