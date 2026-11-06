/**
 * Secondary structure from alpha-carbon geometry alone.
 *
 * 🔴 NOT DSSP, AND THE DIFFERENCE IS STATED RATHER THAN HIDDEN. DSSP assigns
 * structure from backbone hydrogen-bond energies - it needs N, H, C and O
 * positions, a Coulomb energy per candidate pair, and a pattern-matching pass
 * over the resulting bond map. What this does instead is the classical
 * CA-only distance criterion: an alpha helix and a beta strand put their
 * i -> i+3 and i -> i+4 alpha carbons at distances that do not overlap, so two
 * numbers per residue separate them. It agrees with DSSP on most residues and
 * disagrees at the ends of elements, where DSSP itself is sensitive to a single
 * hydrogen bond.
 *
 * That is the right trade here for one reason: this drives a COLOUR, not a
 * measurement. Nothing in `metrics.js` reads it, no number in `docs/STATUS.md`
 * depends on it, and a residue at the frayed end of a helix drawn as coil costs
 * a reader nothing. If an assignment ever feeds an analysis, this is not the
 * function to use.
 *
 * 🔴 AND THE GEOMETRY IS IDEAL-STRUCTURE GEOMETRY, DERIVED NOT TUNED. An ideal
 * alpha helix is a 0.23 nm radius, a 0.15 nm rise and 100 degrees of rotation
 * per residue, so
 *
 *   |CA(i) - CA(i+3)| = sqrt((2r sin(150 deg))^2 + (3d)^2) = 0.505 nm
 *   |CA(i) - CA(i+4)| = sqrt((2r sin(20 deg))^2  + (4d)^2) = 0.620 nm
 *
 * and an extended strand at 0.38 nm per residue gives 1.00 nm and 1.33 nm for
 * the same two separations. Those four numbers are the whole method; the windows
 * below are centred on them. `test/interface.test.js` builds both ideal forms
 * analytically and asserts the assignment, so the constants are checked against
 * the geometry they came from rather than against a hand-labelled structure.
 */

export const HELIX = "helix";
export const SHEET = "sheet";
export const COIL = "coil";

/**
 * Windows on |CA(i) - CA(i+3)| and |CA(i) - CA(i+4)|, in nm.
 *
 * 🔴 THE TWO WINDOWS MUST NOT OVERLAP OR THE CRITERION DECIDES NOTHING. Helix
 * is 0.505/0.620 and strand is 1.00/1.33 - a factor of two apart - which is why
 * a method this crude works at all. The widths are generous because a thermal
 * structure is not an ideal one: a 20% window on the helix still cannot reach
 * the strand's range.
 */
const HELIX_THREE = [0.44, 0.60];
const HELIX_FOUR = [0.52, 0.72];
const SHEET_THREE = [0.88, 1.12];
const SHEET_FOUR = [1.15, 1.45];

const inside = (value, [low, high]) => value >= low && value <= high;

function distance(positions, a, b) {
  const dx = positions[a * 3] - positions[b * 3];
  const dy = positions[a * 3 + 1] - positions[b * 3 + 1];
  const dz = positions[a * 3 + 2] - positions[b * 3 + 2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * @param {ArrayLike<number>} positions 3N, nm
 * @param {ArrayLike<number>} trace alpha-carbon atom indices, in chain order
 * @param {ArrayLike<number>} [breaks] indices into `trace` that START a new
 *   chain. Without them two chain ends separated in space are measured as
 *   though they were sequential, which reads as coil - harmless - but two chain
 *   ends that happen to lie 0.5 nm apart read as HELIX, which is not.
 * @returns {string[]} one of HELIX / SHEET / COIL per entry in `trace`
 */
export function assignSecondary(positions, trace, breaks = []) {
  const count = trace.length;
  const assignment = new Array(count).fill(COIL);
  if (count < 5) return assignment;
  const starts = new Set(breaks);
  // A residue is assigned from the window that STARTS at it, then the
  // assignment is spread over the residues that window spans - so a helix is
  // marked along its whole length rather than only at its first turn.
  const vote = new Array(count).fill(null);
  for (let index = 0; index + 4 < count; index += 1) {
    // Refuse a window that crosses a chain break.
    let crosses = false;
    for (let step = 1; step <= 4; step += 1) {
      if (starts.has(index + step)) { crosses = true; break; }
    }
    if (crosses) continue;
    const three = distance(positions, trace[index], trace[index + 3]);
    const four = distance(positions, trace[index], trace[index + 4]);
    let kind = null;
    if (inside(three, HELIX_THREE) && inside(four, HELIX_FOUR)) kind = HELIX;
    else if (inside(three, SHEET_THREE) && inside(four, SHEET_FOUR)) kind = SHEET;
    if (kind === null) continue;
    for (let step = 0; step <= 4; step += 1) {
      // 🔴 FIRST VOTE WINS, NOT LAST. Overlapping windows at a helix-strand
      // junction disagree, and letting the later one overwrite makes the
      // boundary depend on iteration order rather than on geometry.
      if (vote[index + step] === null) vote[index + step] = kind;
    }
  }
  for (let index = 0; index < count; index += 1) {
    if (vote[index] !== null) assignment[index] = vote[index];
  }
  return assignment;
}

/**
 * Per-atom assignment, for a viewer that colours atoms rather than residues.
 *
 * Every atom of a residue takes its residue's assignment; an atom in a residue
 * with no alpha carbon - a ligand, an ion, a water - is COIL, which the viewer
 * draws in its neutral colour.
 */
export function secondaryOfAtoms(system, positions, selections) {
  const topology = system.topology;
  const trace = [];
  const residueOfTrace = [];
  const breaks = [];
  let previousMolecule = -1;
  for (const atom of selections.groups.alphaCarbon) {
    const molecule = topology.moleculeOfAtom[atom];
    if (molecule !== previousMolecule && trace.length > 0) breaks.push(trace.length);
    previousMolecule = molecule;
    trace.push(atom);
    residueOfTrace.push(topology.residueOfAtom[atom]);
  }
  const perTrace = assignSecondary(positions, trace, breaks);
  const ofResidue = new Map();
  for (let index = 0; index < perTrace.length; index += 1) {
    ofResidue.set(residueOfTrace[index], perTrace[index]);
  }
  const perAtom = new Array(topology.atomCount).fill(COIL);
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    perAtom[atom] = ofResidue.get(topology.residueOfAtom[atom]) ?? COIL;
  }
  const counts = { helix: 0, sheet: 0, coil: 0 };
  for (const kind of perTrace) counts[kind] += 1;
  return { perAtom, perResidue: perTrace, trace, breaks, counts };
}
