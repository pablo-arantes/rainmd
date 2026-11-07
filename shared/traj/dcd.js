/**
 * A DCD trajectory writer, byte-compatible with the one OpenMM uses.
 *
 * DCD is the format the Making-it-rain notebooks write and the one cpptraj,
 * MDAnalysis, VMD and PyMOL all read, so a run on this page produces something
 * the rest of a user's toolchain already understands. Paired with the `.prmtop`
 * they already have, the output is directly analysable.
 *
 * 🔴 EVERY FIELD POSITION HERE WAS READ OUT OF A FILE OPENMM WROTE, NOT OUT OF
 * A FORMAT DESCRIPTION. The published descriptions of DCD disagree with each
 * other about the header - whether `delta` is a float or a double, where the
 * periodic flag sits, and above all what order the six unit-cell numbers are
 * in. Guessing produced a file that a reader accepted and then reported a box
 * 90 degrees wrong. The layout below is what `openmm.app.DCDFile` actually
 * emits, decoded field by field with a deliberately triclinic cell so that all
 * six numbers were distinguishable:
 *
 *   0   int    84                 leading Fortran record marker
 *   4   char4  "CORD"
 *   8   int    NSET               number of frames
 *   12  int    ISTART             first step
 *   16  int    NSAVC              steps between frames
 *   20  int    NSTEP              (NSET - 1) * NSAVC, i.e. the last frame's step
 *   24  int[5] 0
 *   44  f32    DELTA              the timestep in AKMA units, NOT picoseconds
 *   48  int    unit-cell flag
 *   52  int[8] 0
 *   84  int    24                 format version
 *   88  int    84                 trailing marker
 *   92  int    164                title record length
 *   96  int    2                  number of 80-byte title lines
 *   100 char80 x2
 *   260 int    164
 *   264 int    4 / NATOM / 4
 *
 * then per frame, with the cell record present only if the flag is set:
 *
 *   int 48; f64[6] a, cos(gamma), b, cos(beta), cos(alpha), c; int 48
 *   int 4N; f32[N] x; int 4N        (and the same for y and z)
 *
 * 🔴 DELTA IS IN AKMA TIME UNITS. One AKMA unit is 0.04888821 ps, so a 2 fs
 * step is stored as 0.0409 - and a reader that trusts it will report a
 * trajectory 20x too fast or too slow if picoseconds are written instead.
 *
 * 🔴 AND THE HEADER IS BUILT LAST. OpenMM seeks back to patch NSET and NSTEP
 * as it goes; a browser assembling a Blob cannot seek, so the frames are kept
 * as chunks and the header is written when the trajectory is finished. That is
 * also why `finish()` rather than a stream.
 */
import { imageMolecules } from "./imaging.js";

/** One AKMA time unit, in picoseconds. */
const AKMA_PICOSECONDS = 0.04888821;

/** Angstroms per nanometre, for a format that predates SI in this field. */
const ANGSTROM_PER_NM = 10;

export class DcdWriter {
  /**
   * @param {object} options
   * @param {number} options.atomCount
   * @param {number} options.timestep ps
   * @param {number} options.interval steps between written frames
   * @param {boolean} options.periodic whether a unit-cell record is written
   * @param {number} [options.firstStep]
   * @param {string} [options.title]
   */
  constructor(options) {
    this.atomCount = options.atomCount;
    this.timestep = options.timestep;
    this.interval = options.interval;
    this.periodic = options.periodic === true;
    // 🔴 THE MOLECULE PARTITION, OR NO IMAGING. A writer handed no partition
    // cannot wrap whole molecules and must not wrap per atom, so it writes raw
    // coordinates rather than tearing every water in half at the boundary.
    // Every caller in this repository supplies it; the option exists so a
    // caller that genuinely wants unwrapped output can ask for it.
    this.molecules = options.molecules ?? null;
    this.imaged = this.molecules === null
      ? null : new Float64Array(this.atomCount * 3);
    this.firstStep = options.firstStep ?? 0;
    this.title = options.title ?? "Created by RainMD";
    this.frameCount = 0;
    /** @type {Uint8Array[]} */
    this.chunks = [];
    this.x = new Float32Array(this.atomCount);
    this.y = new Float32Array(this.atomCount);
    this.z = new Float32Array(this.atomCount);
  }

  /**
   * @param {ArrayLike<number>} positions 3N, nm
   * @param {Float64Array|null} [boxVectors] nine components, nm
   */
  appendFrame(positions, boxVectors = null) {
    if (positions.length < this.atomCount * 3) {
      throw new RangeError(
        `frame has ${positions.length} coordinates, expected ${this.atomCount * 3}`,
      );
    }
    if (this.periodic && boxVectors === null) {
      throw new Error(
        "this trajectory declares a periodic box, so every frame needs its box "
        + "vectors - a reader that finds the flag set and no record reads the "
        + "first coordinates as a unit cell",
      );
    }
    // 🔴 WRAPPED INTO THE CELL BEFORE IT IS WRITTEN - see shared/traj/imaging.js.
    // Without this the file carries the integrator's unwrapped coordinates, and
    // since water diffuses freely the solvent appears to fly away from the
    // solute as the run goes on. OpenMM's own DCDReporter wraps by default;
    // matching it is what makes this file interchangeable with one the
    // notebooks wrote.
    const frameCoordinates = this.molecules === null || boxVectors === null
      ? positions
      : imageMolecules(positions, boxVectors, this.molecules, this.imaged);
    // 🔴 DCD IS STRUCT-OF-ARRAYS: all x, then all y, then all z, each in its
    // own Fortran record. Writing interleaved xyz produces a file every reader
    // accepts and whose every coordinate is wrong.
    for (let atom = 0; atom < this.atomCount; atom += 1) {
      this.x[atom] = frameCoordinates[atom * 3] * ANGSTROM_PER_NM;
      this.y[atom] = frameCoordinates[atom * 3 + 1] * ANGSTROM_PER_NM;
      this.z[atom] = frameCoordinates[atom * 3 + 2] * ANGSTROM_PER_NM;
    }

    const cellBytes = this.periodic ? 56 : 0;
    const axisBytes = this.atomCount * 4 + 8;
    const frame = new ArrayBuffer(cellBytes + axisBytes * 3);
    const view = new DataView(frame);
    let offset = 0;
    if (this.periodic) {
      view.setInt32(offset, 48, true);
      for (const [index, value] of cellFromVectors(boxVectors).entries()) {
        view.setFloat64(offset + 4 + index * 8, value, true);
      }
      view.setInt32(offset + 52, 48, true);
      offset += 56;
    }
    for (const axis of [this.x, this.y, this.z]) {
      view.setInt32(offset, this.atomCount * 4, true);
      for (let atom = 0; atom < this.atomCount; atom += 1) {
        view.setFloat32(offset + 4 + atom * 4, axis[atom], true);
      }
      view.setInt32(offset + 4 + this.atomCount * 4, this.atomCount * 4, true);
      offset += axisBytes;
    }
    this.chunks.push(new Uint8Array(frame));
    this.frameCount += 1;
    return this.frameCount;
  }

  /** The complete file. */
  finish() {
    const header = this.#header();
    let total = header.length;
    for (const chunk of this.chunks) total += chunk.length;
    const out = new Uint8Array(total);
    out.set(header, 0);
    let offset = header.length;
    for (const chunk of this.chunks) { out.set(chunk, offset); offset += chunk.length; }
    return out;
  }

  #header() {
    const buffer = new ArrayBuffer(276);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    view.setInt32(0, 84, true);
    bytes.set([0x43, 0x4f, 0x52, 0x44], 4); // "CORD"
    view.setInt32(8, this.frameCount, true);
    view.setInt32(12, this.firstStep, true);
    view.setInt32(16, this.interval, true);
    // The last frame's step, which is how OpenMM fills this field: zero for a
    // single-frame file, (NSET - 1) * NSAVC otherwise.
    view.setInt32(20, Math.max(0, this.frameCount - 1) * this.interval, true);
    view.setFloat32(44, this.timestep / AKMA_PICOSECONDS, true);
    view.setInt32(48, this.periodic ? 1 : 0, true);
    view.setInt32(84, 24, true);
    view.setInt32(88, 84, true);
    view.setInt32(92, 164, true);
    view.setInt32(96, 2, true);
    writeFixedString(bytes, 100, this.title, 80);
    writeFixedString(bytes, 180, `Created ${new Date().toISOString()}`, 80);
    view.setInt32(260, 164, true);
    view.setInt32(264, 4, true);
    view.setInt32(268, this.atomCount, true);
    view.setInt32(272, 4, true);
    return bytes;
  }
}

/**
 * Box vectors -> the six numbers DCD stores.
 *
 * 🔴 THE ORDER IS [a, cos(gamma), b, cos(beta), cos(alpha), c], WHICH IS NOT
 * ANY OBVIOUS ORDER AND IS NOT WHAT MOST DESCRIPTIONS SAY. It was read off a
 * file OpenMM wrote with a triclinic cell, which is the only way to tell
 * `cos(alpha)` from `cos(gamma)` - on a rectangular box every angle is 90
 * degrees and all three cosines are zero, so a wrong order is invisible on
 * exactly the boxes `solvateBox` produces and wrong on every octahedral one.
 *
 * @param {Float64Array} vectors nine components, row-major, nm
 */
export function cellFromVectors(vectors) {
  const a = [vectors[0], vectors[1], vectors[2]];
  const b = [vectors[3], vectors[4], vectors[5]];
  const c = [vectors[6], vectors[7], vectors[8]];
  const length = (v) => Math.hypot(v[0], v[1], v[2]);
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const la = length(a);
  const lb = length(b);
  const lc = length(c);
  const cosAlpha = lb * lc > 0 ? dot(b, c) / (lb * lc) : 0;
  const cosBeta = la * lc > 0 ? dot(a, c) / (la * lc) : 0;
  const cosGamma = la * lb > 0 ? dot(a, b) / (la * lb) : 0;
  return [
    la * ANGSTROM_PER_NM, cosGamma,
    lb * ANGSTROM_PER_NM, cosBeta,
    cosAlpha, lc * ANGSTROM_PER_NM,
  ];
}

function writeFixedString(bytes, offset, text, width) {
  // 🔴 PADDED WITH NULs, BECAUSE THAT IS WHAT OPENMM WRITES. The first version
  // of this padded with spaces on the reasoning that the records are Fortran
  // CHARACTER*80 - which sounds right and was wrong: `struct.pack('<80s', ...)`
  // pads with zero bytes, and the 70 differing bytes between this writer's
  // output and OpenMM's were all title padding. Matching the reference
  // implementation exactly is worth more than an argument about what readers
  // prefer, and it makes the two files byte-identical outside the timestamp.
  for (let index = 0; index < width; index += 1) {
    const code = index < text.length ? text.charCodeAt(index) : 0;
    bytes[offset + index] = code < 128 ? code : 0x3f;
  }
}
