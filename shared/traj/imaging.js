/**
 * Wrap whole molecules into the unit cell, for a frame that is being written.
 *
 * 🔴 THIS WAS MISSING AND THE DOCS CLAIMED IT WAS THERE.
 * `docs/ARCHITECTURE.md` said "imaging is applied when a frame is written,
 * never to the running state" - the second half was true and the first half was
 * never implemented. So every trajectory this port wrote carried unwrapped
 * coordinates, and because water diffuses freely, the solvent spreads further
 * from the solute the longer the run goes. Opened in VMD or PyMOL it looks
 * exactly like the waters are flying away from the protein, which is what a
 * reader reported. The physics was never wrong - the nonbonded and PME paths
 * agree with OpenMM to 1.4e-6 and use the minimum image throughout - but the
 * FILE was not what any other tool writes.
 *
 * 🔴 BY MOLECULE CENTROID, WHICH IS WHAT OPENMM DOES - measured, not assumed.
 * A short run through OpenMM's own `DCDReporter` with the notebooks' exact call
 * puts 393 of 393 molecule centroids inside the cell and only 387 of 393 first
 * atoms, so the reference point is the centroid and individual atoms are allowed
 * to stick out by up to a bond length. Wrapping per ATOM instead would be the
 * obvious implementation and would tear every molecule apart at the boundary:
 * a water with its oxygen at one face and a hydrogen at the other, drawn as a
 * bond across the whole box.
 *
 * 🔴 AND NEVER TO THE RUNNING STATE. The bonded forces are not periodic, so a
 * molecule must stay whole in the integrator's coordinate array; wrapping there
 * would split one across the boundary and stretch a bond across the cell. This
 * writes into a separate buffer for exactly that reason.
 */

/**
 * @param {ArrayLike<number>} positions 3N, nm, unwrapped
 * @param {ArrayLike<number>} boxVectors 9 floats, reduced upper triangular
 * @param {object} molecules
 * @param {ArrayLike<number>} molecules.moleculeOfAtom one molecule id per atom
 * @param {number} molecules.count how many molecules there are
 * @param {Float64Array} [out] written into; allocated if absent
 * @returns {Float64Array} 3N wrapped coordinates
 */
export function imageMolecules(positions, boxVectors, molecules, out) {
  const atomCount = Math.floor(positions.length / 3);
  const wrapped = out ?? new Float64Array(positions.length);
  wrapped.set(positions.length === wrapped.length ? positions
    : positions.subarray(0, wrapped.length));
  if (boxVectors === null || boxVectors === undefined) return wrapped;
  const [ax, , , bx, by, , cx, cy, cz] = boxVectors;
  if (!(ax > 0) || !(by > 0) || !(cz > 0)) return wrapped;

  const { moleculeOfAtom, count } = molecules;
  // Centroids first: one pass to accumulate, so a molecule's atoms need not be
  // contiguous. A prmtop's are, but nothing here depends on that.
  const sums = new Float64Array(count * 3);
  const sizes = new Float64Array(count);
  for (let atom = 0; atom < atomCount; atom += 1) {
    const molecule = moleculeOfAtom[atom];
    sums[molecule * 3] += positions[atom * 3];
    sums[molecule * 3 + 1] += positions[atom * 3 + 1];
    sums[molecule * 3 + 2] += positions[atom * 3 + 2];
    sizes[molecule] += 1;
  }

  // The integer translation per molecule, then applied to every one of its
  // atoms so the molecule arrives whole.
  const shifts = new Float64Array(count * 3);
  for (let molecule = 0; molecule < count; molecule += 1) {
    const size = sizes[molecule];
    if (size === 0) continue;
    const x = sums[molecule * 3] / size;
    const y = sums[molecule * 3 + 1] / size;
    const z = sums[molecule * 3 + 2] / size;
    // 🔴 c THEN b THEN a, THE SAME ORDER THE MINIMUM IMAGE USES. For a sheared
    // cell the axes are not independent: removing `a` before `c` can leave the
    // centroid outside, because `c` carries x and y components that the earlier
    // step did not know about. `cpu/forces/geometry.js` states the same rule for
    // the same reason.
    const nc = Math.floor(z / cz);
    const yAfterC = y - nc * cy;
    const nb = Math.floor(yAfterC / by);
    const xAfterCB = x - nc * cx - nb * bx;
    const na = Math.floor(xAfterCB / ax);
    shifts[molecule * 3] = -(na * ax + nb * bx + nc * cx);
    shifts[molecule * 3 + 1] = -(nb * by + nc * cy);
    shifts[molecule * 3 + 2] = -(nc * cz);
  }
  for (let atom = 0; atom < atomCount; atom += 1) {
    const molecule = moleculeOfAtom[atom];
    wrapped[atom * 3] = positions[atom * 3] + shifts[molecule * 3];
    wrapped[atom * 3 + 1] = positions[atom * 3 + 1] + shifts[molecule * 3 + 1];
    wrapped[atom * 3 + 2] = positions[atom * 3 + 2] + shifts[molecule * 3 + 2];
  }
  return wrapped;
}

/** The molecule partition a writer needs, from a parsed topology. */
export function moleculesOf(topology) {
  return {
    moleculeOfAtom: topology.moleculeOfAtom,
    count: topology.moleculeCount,
  };
}
