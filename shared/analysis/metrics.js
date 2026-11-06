/**
 * Live analysis: RMSD, radius of gyration, RMSF and interface distance.
 *
 * Ported from JIGGLE's `core/analysis.py`, whose design decisions are the
 * reason this file exists in this shape. Two of them are the substance:
 *
 * 🔴 RMSD SPLIT FOUR WAYS FOR A TWO-BODY SYSTEM, AND THE FOURTH IS THE ONE
 * THAT MATTERS. A single whole-complex RMSD averages a ligand's drift across
 * every receptor atom that did not move, so a ligand that has come loose barely
 * registers. JIGGLE measured it: a binder displaced by exactly 0.500 nm while
 * the receptor stayed put showed **0.120 nm** in the combined trace. Fitting on
 * the receptor and then measuring the binder recovered the full 0.500 nm. A
 * centroid interface distance misses it too, because it cannot see a rotation
 * or a slide that preserves the centroid separation. So the panel reports:
 *   complex   - everything, fitted on everything
 *   receptor  - is the receptor itself stable?
 *   binder    - is the binder changing shape?
 *   binder on receptor - is the POSE holding? <- this one
 *
 * 🔴 RMSF FROM RUNNING SUMS, NOT FROM STORED FRAMES. RMSF is normally computed
 * after a run from the whole trajectory, which is not available while the run is
 * happening - and keeping every frame's coordinates grows without bound. The
 * identity `RMSF^2 = <|r|^2> - |<r>|^2` is exact, not an approximation, and
 * needs only a sum and a sum of squares per residue: memory is O(residues)
 * whatever the length of the run.
 *
 * 🔴 AND EVERY FRAME IS SUPERPOSED BEFORE IT IS ACCUMULATED. Without that,
 * the whole molecule's tumbling and drift swamp the local fluctuation RMSF is
 * meant to measure, and every residue reports roughly the same large number.
 */
import { kabschTransform, applyTransform, centroid } from "./kabsch.js";

/**
 * RMSD over `measure`, after superposing on `fit`.
 *
 * With `fit === measure` this is an ordinary aligned RMSD.
 *
 * @param {ArrayLike<number>} positions 3N
 * @param {ArrayLike<number>} reference 3N
 * @param {ArrayLike<number>} fit atom indices to superpose on
 * @param {ArrayLike<number>} measure atom indices to measure over
 */
export function rmsdFitOn(positions, reference, fit, measure) {
  if (fit.length === 0 || measure.length === 0) return null;
  const transform = kabschTransform(positions, reference, fit);
  const moved = new Float64Array(3);
  let total = 0;
  for (const atom of measure) {
    applyTransform(transform, positions, atom, moved);
    const dx = moved[0] - reference[atom * 3];
    const dy = moved[1] - reference[atom * 3 + 1];
    const dz = moved[2] - reference[atom * 3 + 2];
    total += dx * dx + dy * dy + dz * dz;
  }
  return Math.sqrt(total / measure.length);
}

/** Mass-weighted radius of gyration over a selection. */
export function radiusOfGyration(positions, masses, indices) {
  if (indices.length === 0) return null;
  const com = centroid(positions, indices, masses);
  let weighted = 0;
  let total = 0;
  for (const atom of indices) {
    const mass = masses[atom];
    const dx = positions[atom * 3] - com[0];
    const dy = positions[atom * 3 + 1] - com[1];
    const dz = positions[atom * 3 + 2] - com[2];
    weighted += mass * (dx * dx + dy * dy + dz * dz);
    total += mass;
  }
  return total > 0 ? Math.sqrt(weighted / total) : null;
}

/**
 * Centroid-to-centroid distance between two selections.
 *
 * 🔴 A CHEAP PROXY, NOT A REACTION COORDINATE. It cannot see a rotation or a
 * slide that preserves the centroid separation, which is exactly what the
 * fit-on-receptor RMSD is for. Reported beside it rather than instead of it.
 */
export function interfaceDistance(positions, first, second) {
  if (first.length === 0 || second.length === 0) return null;
  const a = centroid(positions, first);
  const b = centroid(positions, second);
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** Per-residue RMSF, accumulated one frame at a time. */
export class RunningRmsf {
  /**
   * @param {ArrayLike<number>} reference 3N reference positions
   * @param {ArrayLike<number>} residueAtoms one representative atom per residue
   * @param {ArrayLike<number>} fit atoms to superpose each frame on
   */
  constructor(reference, residueAtoms, fit) {
    this.reference = Float64Array.from(reference);
    this.residueAtoms = Int32Array.from(residueAtoms);
    this.fit = Int32Array.from(fit);
    this.sum = new Float64Array(this.residueAtoms.length * 3);
    this.sumSquares = new Float64Array(this.residueAtoms.length);
    this.frames = 0;
  }

  addFrame(positions) {
    if (this.fit.length === 0) return;
    const transform = kabschTransform(positions, this.reference, this.fit);
    const moved = new Float64Array(3);
    for (let index = 0; index < this.residueAtoms.length; index += 1) {
      applyTransform(transform, positions, this.residueAtoms[index], moved);
      this.sum[index * 3] += moved[0];
      this.sum[index * 3 + 1] += moved[1];
      this.sum[index * 3 + 2] += moved[2];
      this.sumSquares[index] += moved[0] * moved[0] + moved[1] * moved[1]
        + moved[2] * moved[2];
    }
    this.frames += 1;
  }

  /**
   * 🔴 ZEROS UNTIL TWO FRAMES HAVE ARRIVED. One frame has no fluctuation to
   * measure, and reporting a number there would be meaningless rather than
   * merely imprecise.
   */
  values() {
    const out = new Float64Array(this.residueAtoms.length);
    if (this.frames < 2) return out;
    for (let index = 0; index < out.length; index += 1) {
      const mx = this.sum[index * 3] / this.frames;
      const my = this.sum[index * 3 + 1] / this.frames;
      const mz = this.sum[index * 3 + 2] / this.frames;
      const meanSquare = this.sumSquares[index] / this.frames;
      // 🔴 CLAMPED, BECAUSE CANCELLATION CAN MAKE THIS SLIGHTLY NEGATIVE for a
      // residue that has barely moved - and sqrt of that is NaN, which
      // propagates into the plot and stays there.
      const variance = meanSquare - (mx * mx + my * my + mz * mz);
      out[index] = Math.sqrt(Math.max(variance, 0));
    }
    return out;
  }
}

/**
 * Everything a frame reports, in one object.
 *
 * Holds the reference frame and the running RMSF, so a caller pushes frames in
 * and gets a complete metric set back.
 */
export class LiveAnalysis {
  /**
   * @param {object} system from buildSystem
   * @param {object} selections from buildSelections
   * @param {ArrayLike<number>} reference the frame everything is measured from
   */
  constructor(system, selections, reference) {
    this.system = system;
    this.selections = selections;
    this.reference = Float64Array.from(reference);
    this.masses = system.topology.masses;
    // 🔴 FITTED ON HEAVY SOLUTE ATOMS, NOT ON EVERYTHING. Hydrogens are light,
    // fast and constrained, so including them in a superposition adds noise
    // without adding information - and including the solvent would fit the
    // water box, which is the thing that is not supposed to move.
    const heavy = new Set(selections.groups.heavy);
    this.fit = Int32Array.from([...selections.groups.solute].filter((a) => heavy.has(a)));
    this.receptorFit = Int32Array.from([...selections.receptor].filter((a) => heavy.has(a)));
    this.binderFit = Int32Array.from([...selections.binder].filter((a) => heavy.has(a)));
    this.rmsf = new RunningRmsf(this.reference, selections.residueAtoms, this.fit);
    this.series = [];
  }

  get hasBinder() { return this.binderFit.length > 0; }

  /**
   * @param {ArrayLike<number>} positions 3N, nm
   * @param {object} energy from the simulation
   */
  addFrame(positions, energy) {
    this.rmsf.addFrame(positions);
    const row = {
      step: energy.step,
      time: energy.time,
      potential: energy.potential,
      kinetic: energy.kinetic,
      total: energy.total,
      temperature: energy.temperature,
      volume: energy.volume ?? null,
      rmsd: rmsdFitOn(positions, this.reference, this.fit, this.fit),
      radiusOfGyration: radiusOfGyration(positions, this.masses, this.fit),
      rmsdReceptor: null,
      rmsdBinder: null,
      rmsdBinderOnReceptor: null,
      interfaceDistance: null,
    };
    if (this.hasBinder) {
      row.rmsdReceptor = rmsdFitOn(positions, this.reference, this.receptorFit, this.receptorFit);
      row.rmsdBinder = rmsdFitOn(positions, this.reference, this.binderFit, this.binderFit);
      // The pose-stability number: hold the receptor still, measure the binder.
      row.rmsdBinderOnReceptor = rmsdFitOn(positions, this.reference,
        this.receptorFit, this.binderFit);
      row.interfaceDistance = interfaceDistance(positions, this.receptorFit, this.binderFit);
    }
    this.series.push(row);
    return row;
  }

  /** Every series as CSV, in the layout JIGGLE writes. */
  metricsCsv() {
    const columns = ["step", "time", "potential", "kinetic", "total", "temperature",
      "volume", "rmsd", "radiusOfGyration", "rmsdReceptor", "rmsdBinder",
      "rmsdBinderOnReceptor", "interfaceDistance"];
    const header = ["step", "time_ps", "potential_kJ_mol", "kinetic_kJ_mol",
      "total_kJ_mol", "temperature_K", "volume_nm3", "rmsd_nm", "rg_nm",
      "rmsd_receptor_nm", "rmsd_binder_nm", "rmsd_binder_on_receptor_nm",
      "interface_distance_nm"];
    const lines = [header.join(",")];
    for (const row of this.series) {
      lines.push(columns.map((key) => (row[key] === null || row[key] === undefined
        ? "" : row[key])).join(","));
    }
    return `${lines.join("\n")}\n`;
  }

  /** Per-residue RMSF as CSV, with labels. */
  rmsfCsv() {
    const values = this.rmsf.values();
    const lines = ["residue_index,residue,rmsf_nm"];
    for (let index = 0; index < values.length; index += 1) {
      lines.push(`${index + 1},${this.selections.residueLabels[index]},${values[index]}`);
    }
    return `${lines.join("\n")}\n`;
  }
}
