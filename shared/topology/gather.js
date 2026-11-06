/**
 * Invert the bonded term lists: for each atom, which terms touch it and in
 * which position.
 *
 * 🔴 THIS EXISTS SO THE GPU NEEDS NO ATOMICS, AND THAT IS A CORRECTNESS
 * DECISION RATHER THAN A PERFORMANCE ONE. The obvious kernel is one thread per
 * bond doing an atomicAdd into three force components - but WGSL has no
 * atomicAdd on f32, so it would have to be fixed point, and fixed point needs a
 * scale chosen in advance against forces that span six orders of magnitude
 * between an equilibrated step and a clashing one at minimisation step zero.
 * Worse, floating-point atomics are not associative, so two runs of the same
 * step give different answers in the last bits and the differential test that
 * is supposed to validate the kernel has no fixed value to compare against.
 *
 * Inverting the lists instead gives one thread per ATOM, which writes its own
 * three components and adds nothing to anyone else's. The same step gives the
 * same bits every time, on every device. The cost is recomputing each term once
 * per atom it touches - twice for a bond, three times for an angle, four for a
 * torsion - which on a system whose work is 99% nonbonded is not measurable.
 *
 * 🔴 AND AN ATOM CAN APPEAR TWICE IN ONE TERM. A four-membered ring closure
 * can give a torsion where atoms 1 and 4 are the same atom, and a water's H-H
 * constraint bond is not the only degenerate case AMBER writes. Each
 * (term, role) pair is recorded separately, so such an atom gets both
 * contributions - deduplicating by term index would silently drop one.
 */

/** Term kinds, as the shader's two-bit tag. */
export const TERM_BOND = 0;
export const TERM_ANGLE = 1;
export const TERM_TORSION = 2;

const MAX_TERM_INDEX = (1 << 27) - 1;

/**
 * @param {object} system from buildSystem
 * @returns {object} flat arrays ready to upload
 */
export function buildBondedGather(system) {
  const atomCount = system.topology.atomCount;
  const bonds = system.harmonicBonds;
  const angles = system.topology.angles;
  const torsions = system.topology.dihedrals;

  if (Math.max(bonds.count, angles.count, torsions.count) > MAX_TERM_INDEX) {
    throw new Error(
      `a bonded term count exceeds ${MAX_TERM_INDEX}, which is what the shader's `
      + "packed (kind, role, index) word can address",
    );
  }

  // Terms are concatenated into one index array and one parameter array so the
  // kernel needs two storage bindings rather than six - the spec guarantees
  // only eight per stage, and the nonbonded kernel needs its share.
  const termIndices = new Uint32Array(bonds.count * 2 + angles.count * 3 + torsions.count * 4);
  const termParams = new Float32Array(bonds.count * 2 + angles.count * 2 + torsions.count * 3);
  const angleIndexBase = bonds.count * 2;
  const torsionIndexBase = angleIndexBase + angles.count * 3;
  const angleParamBase = bonds.count * 2;
  const torsionParamBase = angleParamBase + angles.count * 2;

  for (let index = 0; index < bonds.count; index += 1) {
    termIndices[index * 2] = bonds.atomA[index];
    termIndices[index * 2 + 1] = bonds.atomB[index];
    termParams[index * 2] = bonds.k[index];
    termParams[index * 2 + 1] = bonds.r0[index];
  }
  for (let index = 0; index < angles.count; index += 1) {
    termIndices[angleIndexBase + index * 3] = angles.atomA[index];
    termIndices[angleIndexBase + index * 3 + 1] = angles.atomB[index];
    termIndices[angleIndexBase + index * 3 + 2] = angles.atomC[index];
    termParams[angleParamBase + index * 2] = angles.k[index];
    termParams[angleParamBase + index * 2 + 1] = angles.theta0[index];
  }
  for (let index = 0; index < torsions.count; index += 1) {
    termIndices[torsionIndexBase + index * 4] = torsions.atomA[index];
    termIndices[torsionIndexBase + index * 4 + 1] = torsions.atomB[index];
    termIndices[torsionIndexBase + index * 4 + 2] = torsions.atomC[index];
    termIndices[torsionIndexBase + index * 4 + 3] = torsions.atomD[index];
    termParams[torsionParamBase + index * 3] = torsions.k[index];
    termParams[torsionParamBase + index * 3 + 1] = torsions.periodicity[index];
    termParams[torsionParamBase + index * 3 + 2] = torsions.phase[index];
  }

  // Count, then fill: two passes so the CSR arrays are exactly sized.
  const counts = new Int32Array(atomCount);
  const note = (atom) => { counts[atom] += 1; };
  for (let index = 0; index < bonds.count; index += 1) {
    note(bonds.atomA[index]); note(bonds.atomB[index]);
  }
  for (let index = 0; index < angles.count; index += 1) {
    note(angles.atomA[index]); note(angles.atomB[index]); note(angles.atomC[index]);
  }
  for (let index = 0; index < torsions.count; index += 1) {
    note(torsions.atomA[index]); note(torsions.atomB[index]);
    note(torsions.atomC[index]); note(torsions.atomD[index]);
  }
  const start = new Uint32Array(atomCount + 1);
  for (let atom = 0; atom < atomCount; atom += 1) start[atom + 1] = start[atom] + counts[atom];
  const list = new Uint32Array(start[atomCount]);
  const cursor = start.slice(0, atomCount);
  const pack = (kind, role, index) => (kind << 30) | (role << 27) | index;
  const place = (atom, word) => { list[cursor[atom]] = word; cursor[atom] += 1; };
  for (let index = 0; index < bonds.count; index += 1) {
    place(bonds.atomA[index], pack(TERM_BOND, 0, index));
    place(bonds.atomB[index], pack(TERM_BOND, 1, index));
  }
  for (let index = 0; index < angles.count; index += 1) {
    place(angles.atomA[index], pack(TERM_ANGLE, 0, index));
    place(angles.atomB[index], pack(TERM_ANGLE, 1, index));
    place(angles.atomC[index], pack(TERM_ANGLE, 2, index));
  }
  for (let index = 0; index < torsions.count; index += 1) {
    place(torsions.atomA[index], pack(TERM_TORSION, 0, index));
    place(torsions.atomB[index], pack(TERM_TORSION, 1, index));
    place(torsions.atomC[index], pack(TERM_TORSION, 2, index));
    place(torsions.atomD[index], pack(TERM_TORSION, 3, index));
  }

  let widest = 0;
  for (let atom = 0; atom < atomCount; atom += 1) {
    widest = Math.max(widest, start[atom + 1] - start[atom]);
  }

  return {
    termIndices,
    termParams,
    start,
    list,
    layout: {
      bondCount: bonds.count,
      angleCount: angles.count,
      torsionCount: torsions.count,
      angleIndexBase,
      torsionIndexBase,
      angleParamBase,
      torsionParamBase,
    },
    widestAtom: widest,
  };
}
