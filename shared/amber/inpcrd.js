/**
 * AMBER ASCII coordinate/restart files (`.inpcrd`, `.crd`, `.rst7`).
 *
 * 🔴 WHAT IS IN THE FILE IS DECIDED BY COUNTING, NOT BY A HEADER. The format
 * carries coordinates, optionally velocities, and optionally a box, with
 * nothing to say which - so the only way to know is to count the numbers after
 * the atom-count line and compare against 3N:
 *
 *   3N       coordinates only                 (tleap's saveamberparm)
 *   3N + 6   coordinates and a box            (tleap, solvated)
 *   6N       coordinates and velocities       (a restart, no box)
 *   6N + 6   coordinates, velocities, box     (a restart from a periodic run)
 *
 * 🔴 AND 3N + 6 IS AMBIGUOUS AT N = 2. Two atoms give 6 coordinates, so
 * "coordinates and a box" and "coordinates and velocities" are the same count.
 * The box wins, because a two-atom periodic system is a thing people build and
 * a two-atom restart is not - and the ambiguity is reported rather than hidden.
 *
 * NetCDF restarts are NOT read here. They are binary, they are what `ntxo=2`
 * writes, and a browser drop of one is refused by name rather than parsed into
 * nonsense.
 */
import { AMBER_VELOCITY_SCALE, NM_PER_ANGSTROM } from "../units.js";

/**
 * @param {string} text the whole file
 * @param {object} [options]
 * @param {number} [options.expectedAtomCount] the topology's NATOM, checked
 *   against the file's own count. A mismatch is the single most common user
 *   error - a `.crd` from one tleap run beside a `.prmtop` from another - and
 *   it must be caught here, because downstream it is a silently wrong
 *   simulation rather than an error.
 */
export function parseInpcrd(text, options = {}) {
  if (text.charCodeAt(0) === 0x43 && text.slice(0, 3) === "CDF") {
    throw new Error(
      "this is a NetCDF restart (binary), not an ASCII inpcrd. Re-save it as "
      + "ASCII (`ntxo = 1` in AMBER, or ParmEd's `.rst7`), or use the `.crd` "
      + "that tleap's saveamberparm wrote.",
    );
  }
  const lines = text.split(/\r?\n/);
  if (lines.length < 2) throw new Error("inpcrd is too short to hold an atom count");
  const title = lines[0].trim();
  const headerFields = lines[1].trim().split(/\s+/).filter((field) => field !== "");
  const atomCount = Number.parseInt(headerFields[0], 10);
  if (!Number.isInteger(atomCount) || atomCount <= 0) {
    throw new Error(`inpcrd declares an unusable atom count: ${JSON.stringify(lines[1])}`);
  }
  const time = headerFields.length > 1 ? Number(headerFields[1]) : null;
  if (options.expectedAtomCount !== undefined && options.expectedAtomCount !== atomCount) {
    throw new Error(
      `coordinate file has ${atomCount} atoms but the topology has `
      + `${options.expectedAtomCount}. These two files are not from the same `
      + "tleap run - re-save the pair together.",
    );
  }

  // 🔴 SLICED BY WIDTH, NOT SPLIT ON WHITESPACE: the format is 6F12.7, and a
  // coordinate of -1234.5678901 fills its field with no leading space, so two
  // adjacent large negative coordinates run together. A solvated box centred at
  // the origin has plenty of them.
  const numbers = [];
  for (let index = 2; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim() === "") continue;
    for (let field = 0; field < 6; field += 1) {
      const start = field * 12;
      if (start >= line.length) break;
      const text12 = line.slice(start, start + 12).trim();
      if (text12 === "") continue;
      const value = Number(text12);
      if (!Number.isFinite(value)) {
        throw new Error(
          `non-numeric coordinate field ${JSON.stringify(text12)} on line ${index + 1}`,
        );
      }
      numbers.push(value);
    }
  }

  const coordinateCount = atomCount * 3;
  const total = numbers.length;
  let hasVelocities = false;
  let hasBox = false;
  const ambiguous = [];
  if (total === coordinateCount) {
    // nothing more
  } else if (total === coordinateCount + 6) {
    hasBox = true;
    if (atomCount === 2) ambiguous.push("at 2 atoms, +6 numbers could be a box or velocities; read as a box");
  } else if (total === coordinateCount * 2) {
    hasVelocities = true;
  } else if (total === coordinateCount * 2 + 6) {
    hasVelocities = true;
    hasBox = true;
  } else {
    throw new Error(
      `inpcrd holds ${total} numbers, which is none of ${coordinateCount} `
      + `(coordinates), ${coordinateCount + 6} (coordinates+box), `
      + `${coordinateCount * 2} (coordinates+velocities) or `
      + `${coordinateCount * 2 + 6} (all three) for ${atomCount} atoms. `
      + "The file is truncated or does not match this topology.",
    );
  }

  const positions = new Float64Array(coordinateCount);
  for (let index = 0; index < coordinateCount; index += 1) {
    positions[index] = numbers[index] * NM_PER_ANGSTROM;
  }
  let velocities = null;
  if (hasVelocities) {
    velocities = new Float64Array(coordinateCount);
    for (let index = 0; index < coordinateCount; index += 1) {
      velocities[index] = numbers[coordinateCount + index]
        * AMBER_VELOCITY_SCALE * NM_PER_ANGSTROM;
    }
  }
  let box = null;
  if (hasBox) {
    const offset = total - 6;
    box = {
      lengths: [
        numbers[offset] * NM_PER_ANGSTROM,
        numbers[offset + 1] * NM_PER_ANGSTROM,
        numbers[offset + 2] * NM_PER_ANGSTROM,
      ],
      angles: [numbers[offset + 3], numbers[offset + 4], numbers[offset + 5]],
    };
  }
  return { title, atomCount, time, positions, velocities, box, ambiguous };
}

/**
 * Box lengths and angles -> the three periodic vectors, reduced.
 *
 * 🔴 PME NEEDS VECTORS, AND A TRUNCATED OCTAHEDRON IS NOT A BOX OF THREE
 * LENGTHS. `solvateOct` writes all three angles as 109.4712206 degrees, and a
 * reciprocal lattice built from lengths alone is then wrong - not slightly, but
 * by the whole shear. The notebooks use `solvateBox`, which is 90/90/90 and
 * where this reduces to a diagonal matrix, so the octahedral path is the one
 * that will be wrong silently if it is not done properly here.
 *
 * 🔴 AND THE RESULT IS REDUCED THE WAY OPENMM REDUCES IT. A periodic box has
 * infinitely many equivalent vector sets, and a minimum-image convention is
 * only valid for the reduced one: `a` along x, `b` in the xy plane, each
 * off-diagonal no more than half the diagonal below it. Skipping the reduction
 * gives correct energies for a rectangular box and wrong ones for a sheared
 * box, found only by comparing against OpenMM on an octahedral system.
 *
 * @param {{lengths: number[], angles: number[]}} box lengths in nm, angles in degrees
 * @returns {Float64Array} nine components, row-major: ax ay az bx by bz cx cy cz
 */
export function boxVectorsFromLengthsAndAngles(box) {
  const [a, b, c] = box.lengths;
  const toRadians = Math.PI / 180;
  const alpha = box.angles[0] * toRadians;
  const beta = box.angles[1] * toRadians;
  const gamma = box.angles[2] * toRadians;

  const va = [a, 0, 0];
  const vb = [b * Math.cos(gamma), b * Math.sin(gamma), 0];
  const cx = c * Math.cos(beta);
  const cy = c * (Math.cos(alpha) - Math.cos(beta) * Math.cos(gamma)) / Math.sin(gamma);
  const czSquared = c * c - cx * cx - cy * cy;
  if (czSquared <= 0) {
    throw new Error(
      `box lengths ${box.lengths.join("/")} with angles ${box.angles.join("/")} `
      + "do not describe a realisable cell",
    );
  }
  const vc = [cx, cy, Math.sqrt(czSquared)];

  // Clean up the values that should be exactly zero for a rectangular box, so a
  // 90-degree cell gets a diagonal matrix rather than one with 1e-17 shears -
  // the kernels branch on "is this box rectangular" to skip six multiplies per
  // minimum image, and 1e-17 is not zero.
  const vectors = [va, vb, vc];
  for (const vector of vectors) {
    for (let index = 0; index < 3; index += 1) {
      if (Math.abs(vector[index]) < 1e-12 * a) vector[index] = 0;
    }
  }

  // OpenMM's reduction: subtract the row below, nearest integer, twice over.
  const reduce = (target, source, component) => {
    const shift = Math.round(target[component] / source[component]);
    if (shift === 0) return;
    for (let index = 0; index < 3; index += 1) target[index] -= shift * source[index];
  };
  reduce(vc, vb, 1);
  reduce(vc, va, 0);
  reduce(vb, va, 0);

  return Float64Array.from([...va, ...vb, ...vc]);
}
