/**
 * The analytical Lennard-Jones tail correction.
 *
 * 🔴 OPENMM APPLIES THIS BY DEFAULT AND IT IS NOT A PAIR SUM. Every periodic
 * `NonbondedForce` has `useDispersionCorrection` true unless asked otherwise,
 * so an OpenMM energy carries a single volume-dependent estimate of the
 * dispersion beyond the cutoff that no neighbour loop will ever reproduce.
 * Measured on the solvated fixture it is **-41.537158 kJ/mol**, which is 0.35%
 * of the total energy - small enough to look like a kernel defect and large
 * enough to matter.
 *
 * 🔴 AND IT IS WHY NPT NEEDS IT AT ALL. The correction scales as 1/V, so it
 * contributes to the pressure; a barostat run without it equilibrates to the
 * wrong density. For a fixed-volume run it is a constant offset and could be
 * ignored, which is why it was isolated and excluded from the differential
 * tests rather than implemented, until the barostat arrived.
 *
 * 🔴 THE PAIR COUNTING IS OPENMM'S AND IT IS NOT THE OBVIOUS ONE. The textbook
 * form averages over all N^2 ordered pairs; OpenMM averages over the
 * **N(N+1)/2 unordered interactions, self-pairs included**, and then multiplies
 * by N^2. Measured on this system the two differ by 0.16% - the obvious form
 * gives -41.469645 against OpenMM's -41.537158, and the convention below
 * reproduces it to **ratio 1.000000**. Three candidate conventions were
 * computed and compared against that number; this is the one that matched.
 */

/**
 * The composition-weighted A and B sums, which do not depend on the volume.
 *
 * Computed once per system. The barostat re-evaluates the energy at a trial
 * volume thousands of times, and this is the part that would otherwise be
 * recomputed over an NTYPES^2 table every time.
 *
 * @param {object} system from buildSystem
 * @returns {{a: number, b: number, atomCount: number}} in kJ/mol*nm^12 and nm^6
 */
export function dispersionCoefficients(system) {
  const topology = system.topology;
  const atomCount = topology.atomCount;
  const typeCount = topology.typeCount;
  const counts = new Int32Array(typeCount);
  for (let atom = 0; atom < atomCount; atom += 1) counts[topology.atomTypes[atom]] += 1;

  let orderedA = 0;
  let orderedB = 0;
  let diagonalA = 0;
  let diagonalB = 0;
  for (let i = 0; i < typeCount; i += 1) {
    for (let j = 0; j < typeCount; j += 1) {
      const pair = typeCount * i + j;
      orderedA += counts[i] * counts[j] * topology.ljA[pair];
      orderedB += counts[i] * counts[j] * topology.ljB[pair];
    }
    const self = typeCount * i + i;
    diagonalA += counts[i] * topology.ljA[self];
    diagonalB += counts[i] * topology.ljB[self];
  }
  // The average over N(N+1)/2 unordered interactions, scaled by N^2 - which is
  // (ordered + diagonal) * N / (N + 1). See the note above.
  const scale = atomCount / (atomCount + 1);
  return {
    a: (orderedA + diagonalA) * scale,
    b: (orderedB + diagonalB) * scale,
    atomCount,
  };
}

/**
 * The correction at a given volume.
 *
 * @param {{a: number, b: number}} coefficients from dispersionCoefficients
 * @param {number} volume nm^3
 * @param {number} cutoff nm
 * @returns {number} kJ/mol
 */
export function dispersionEnergy(coefficients, volume, cutoff) {
  if (!(volume > 0) || !(cutoff > 0)) return 0;
  const cutoff3 = cutoff * cutoff * cutoff;
  const cutoff9 = cutoff3 * cutoff3 * cutoff3;
  return (2 * Math.PI / volume)
    * (coefficients.a / (9 * cutoff9) - coefficients.b / (3 * cutoff3));
}
