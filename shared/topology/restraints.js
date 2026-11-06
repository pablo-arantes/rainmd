/**
 * Positional restraints on the solute, as the Making-it-rain notebooks apply
 * them during equilibration.
 *
 * 🔴 THE FUNCTIONAL FORM IS `k * d^2`, WITHOUT THE HALF. The notebooks build
 *
 *     CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2;')
 *
 * so the energy is `k * |r - r0|^2` and the force is `-2k * (r - r0)`. Every
 * other harmonic term in this port carries OpenMM's `1/2 k x^2` convention, and
 * reading this one the same way would make every restraint half as stiff as the
 * number on the slider. The slider says 500 and means 500 kJ/mol/nm^2 of
 * `k * d^2`, which is a force of 1000 kJ/mol/nm at one nanometre of
 * displacement.
 *
 * 🔴 AND IT IS A `periodicdistance`, SO THE MINIMUM IMAGE APPLIES. It almost
 * never matters - a restrained atom sits within a tenth of a nanometre of its
 * reference - but the exception is real: the barostat scales coordinates while
 * the reference stays where it was, so after enough accepted volume moves an
 * atom near a face can wrap relative to its own reference. OpenMM applies the
 * convention, so this does too, and the two agree rather than diverging in a
 * corner nobody tests.
 *
 * 🔴 THE REFERENCE IS ABSOLUTE AND DOES NOT FOLLOW THE BOX. OpenMM captures
 * `x0, y0, z0` once, when the force is added, and the barostat never touches
 * them - so a restraint pulls an atom towards where it was in the box as it
 * was. That looks like an oversight and is the behaviour the notebooks rely on:
 * it is what stops the solute drifting while the water is still finding its
 * density. Re-scaling the references with the box would let the solute ride the
 * volume changes, which is the thing the restraint exists to prevent.
 */

/**
 * The notebooks' own selection, as a pytraj mask:
 *
 *     !(:H*) & !(:WAT) & !(:Na+) & !(:Cl-) & !(:Mg+) & !(:K+)
 *
 * which is every solute atom that is not a hydrogen - the solvent and the
 * counter-ions are left free, because letting them relax around a held solute
 * is the entire purpose of the stage.
 *
 * @param {object} system from buildSystem
 * @param {object} selections from buildSelections
 * @returns {Int32Array} atom indices, ascending
 */
export function soluteHeavyAtoms(system, selections) {
  const heavy = new Set(selections.groups.heavy);
  return Int32Array.from([...selections.groups.solute].filter((atom) => heavy.has(atom)));
}

/**
 * @param {object} options
 * @param {number} options.force `k` in kJ/mol/nm^2, in the `k*d^2` convention
 * @param {ArrayLike<number>} options.atoms which atoms are restrained
 * @param {ArrayLike<number>} options.reference 3N positions to restrain towards;
 *   only the restrained atoms' entries are read
 * @returns {{force: number, atoms: Int32Array, reference: Float32Array}} the
 *   reference packed to three floats per RESTRAINED atom, not per system atom
 */
export function buildRestraints(options) {
  const force = options.force ?? 0;
  const atoms = Int32Array.from(options.atoms ?? []);
  if (!(force > 0) || atoms.length === 0) {
    return { force: 0, atoms: new Int32Array(0), reference: new Float32Array(0) };
  }
  const source = options.reference;
  // 🔴 f64, NOT f32, AND THE GPU CONVERTS AT UPLOAD. Packing the reference to
  // f32 here cost 5.0e-7 of relative error against OpenMM on a term whose
  // every other comparison in this repository sits at 1e-16 - because the
  // reference is a box-spanning coordinate and f32 resolves it to about
  // `span * eps`. The device has to take f32 either way; the host must not.
  const reference = new Float64Array(atoms.length * 3);
  for (let slot = 0; slot < atoms.length; slot += 1) {
    const atom = atoms[slot];
    reference[slot * 3] = source[atom * 3];
    reference[slot * 3 + 1] = source[atom * 3 + 1];
    reference[slot * 3 + 2] = source[atom * 3 + 2];
  }
  return { force, atoms, reference };
}

/**
 * Energy and forces of the restraint term, on the host.
 *
 * @param {object} restraints from buildRestraints
 * @param {ArrayLike<number>} positions 3N, nm
 * @param {Float64Array} forces accumulated into, 3N
 * @param {object} [options]
 * @param {Float64Array|null} [options.boxVectors] for the minimum image; null
 *   for a non-periodic system, where the plain difference is the distance
 * @returns {number} the restraint energy in kJ/mol
 */
export function restraintEnergy(restraints, positions, forces, options = {}) {
  const { force, atoms, reference } = restraints;
  if (force <= 0 || atoms.length === 0) return 0;
  const box = options.boxVectors ?? null;
  let total = 0;
  for (let slot = 0; slot < atoms.length; slot += 1) {
    const atom = atoms[slot];
    let dx = positions[atom * 3] - reference[slot * 3];
    let dy = positions[atom * 3 + 1] - reference[slot * 3 + 1];
    let dz = positions[atom * 3 + 2] - reference[slot * 3 + 2];
    if (box !== null) {
      // 🔴 THE SAME c -> b -> a ORDER THE REST OF THE PORT USES. A triclinic
      // box has to be reduced in that order or the image chosen is not the
      // nearest one; `cpu/forces/geometry.js` states why.
      const cScale = Math.round(dz / box[8]);
      dx -= cScale * box[6];
      dy -= cScale * box[7];
      dz -= cScale * box[8];
      const bScale = Math.round(dy / box[4]);
      dx -= bScale * box[3];
      dy -= bScale * box[4];
      const aScale = Math.round(dx / box[0]);
      dx -= aScale * box[0];
    }
    const squared = dx * dx + dy * dy + dz * dz;
    total += force * squared;
    // F = -dU/dr = -2k * d
    const scale = 2 * force;
    forces[atom * 3] -= scale * dx;
    forces[atom * 3 + 1] -= scale * dy;
    forces[atom * 3 + 2] -= scale * dz;
  }
  return total;
}
