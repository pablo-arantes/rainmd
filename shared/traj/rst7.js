/**
 * An AMBER ASCII restart file, so a run on this page can be continued anywhere.
 *
 * The output pairs with the `.prmtop` the user already has, which means a
 * simulation started in the browser can be continued by AMBER, by OpenMM or by
 * the notebooks - and vice versa, since shared/amber/inpcrd.js reads exactly
 * this format back.
 *
 * 🔴 VELOCITIES ARE WRITTEN IN AMBER'S TIME UNIT, NOT nm/ps. The file's number
 * is angstroms per AMBER time unit, and one AMBER time unit is 1/20.455 ps -
 * so the conversion out is the exact inverse of the one on the way in. Writing
 * nm/ps instead produces a restart that is silently 20x too cold, which the
 * thermostat then repairs over a few ps: the run does not fail, it just starts
 * wrong.
 */
import { AMBER_VELOCITY_SCALE, NM_PER_ANGSTROM } from "../units.js";

/** `%FORMAT(6F12.7)`, which is fixed width and must stay fixed width. */
function formatColumns(values, perLine = 6) {
  const lines = [];
  for (let start = 0; start < values.length; start += perLine) {
    let line = "";
    for (let index = start; index < Math.min(start + perLine, values.length); index += 1) {
      const text = values[index].toFixed(7);
      // 🔴 A COORDINATE THAT OVERFLOWS TWELVE COLUMNS MUST BE REFUSED, NOT
      // TRUNCATED. The reader slices by width, so a 13-character field shifts
      // every value after it by one column - producing a file that parses and
      // whose coordinates are nonsense from that point on. It takes a position
      // past 99999 angstroms, which means an atom that has flown away.
      if (text.length > 12) {
        throw new Error(
          `coordinate ${values[index]} does not fit the 12-column restart `
          + "format. An atom has left the box, which means the simulation blew "
          + "up rather than that the writer is wrong.",
        );
      }
      line += text.padStart(12);
    }
    lines.push(line);
  }
  return lines;
}

/**
 * @param {object} options
 * @param {ArrayLike<number>} options.positions 3N, nm
 * @param {ArrayLike<number>} [options.velocities] 3N, nm/ps
 * @param {{lengths: number[], angles: number[]}} [options.box] nm and degrees
 * @param {number} [options.time] ps
 * @returns {string}
 */
export function writeRestart(options) {
  const atomCount = Math.floor(options.positions.length / 3);
  const lines = [options.title ?? "RainMD restart"];
  // The atom count and the time, in the `%FORMAT(I5,5E15.7)` header that every
  // reader special-cases. Written with the count first and the time beside it.
  lines.push(`${String(atomCount).padStart(5)}${(options.time ?? 0).toExponential(7).padStart(15)}`);

  const coordinates = new Array(atomCount * 3);
  for (let index = 0; index < atomCount * 3; index += 1) {
    coordinates[index] = options.positions[index] / NM_PER_ANGSTROM;
  }
  lines.push(...formatColumns(coordinates));

  if (options.velocities !== undefined && options.velocities !== null) {
    const velocities = new Array(atomCount * 3);
    for (let index = 0; index < atomCount * 3; index += 1) {
      velocities[index] = options.velocities[index]
        / (AMBER_VELOCITY_SCALE * NM_PER_ANGSTROM);
    }
    lines.push(...formatColumns(velocities));
  }

  if (options.box !== undefined && options.box !== null) {
    lines.push(...formatColumns([
      options.box.lengths[0] / NM_PER_ANGSTROM,
      options.box.lengths[1] / NM_PER_ANGSTROM,
      options.box.lengths[2] / NM_PER_ANGSTROM,
      options.box.angles[0], options.box.angles[1], options.box.angles[2],
    ]));
  }
  return `${lines.join("\n")}\n`;
}
