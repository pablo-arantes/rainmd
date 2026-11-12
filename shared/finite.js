/**
 * Is the state still a number? Asked at a block boundary, and again by every
 * writer before it commits bytes to a file.
 *
 * 🔴 NOTHING ASKED THIS BEFORE, AND A BLOWN-UP RUN COMPLETED NORMALLY. Measured
 * on the page at a 40 fs timestep: every atom non-finite after 50 steps, and
 * then 300 steps encoded at full speed, 13 frames written, both download
 * buttons enabled, the next stage unlocked, and a log line reading
 * "300 steps in 7.7 s". The tiles read "-" and the CSV read `NaN`; nothing said
 * the run was dead. The all-NaN DCD was then accepted by cpptraj
 * ("Read 10 frames and processed 10 frames") and by MDAnalysis without one
 * warning from either.
 *
 * 🔴 THE INFORMATION WAS ALREADY ON THE HOST. `GpuSimulation.#checkSkin` reads
 * the reduced squared displacement every block, and a non-finite position makes
 * that reduction non-finite - so the signal arrived every block and was thrown
 * away. Worse than thrown away: `Math.sqrt(Math.max(NaN, 0))` is `NaN`, which
 * failed `usage <= 1` and so counted every block as a skin violation while
 * leaving `worstDisplacement` at 0, printing "worst displacement 0 pm - 0% of
 * the budget used" directly above "the neighbour list was stale for 300 of 300
 * steps". The one loud line a dead run produced was the wrong diagnosis.
 *
 * 🔴 AND `atomicMax` OVER A BIT PATTERN CATCHES BOTH NaN AND INFINITY, which is
 * why the free detector is sound rather than lucky. The displacement kernel
 * reduces `bitcast<u32>` of a squared displacement; `NaN` is `0x7FC00000`
 * (2143289344) and `+Inf` is `0x7F800000` (2139095040), both above any finite
 * squared displacement a simulation can reach - a displacement of 1 nm is
 * `0x3F800000`, 1.065e9. So a poisoned value always wins the maximum and always
 * reaches the host.
 *
 * Velocities can go non-finite a step before positions do; positions integrate
 * velocities, so the detector lags by at most one step inside the same block.
 * That is why `energy()` checks as well, at every report.
 */

/**
 * The index of the first non-finite value, or -1.
 *
 * Returns the index rather than a boolean because a message naming the atom is
 * what makes the difference between "the run blew up" and a user who can look
 * at their own input and see why.
 *
 * @param {ArrayLike<number>} values
 * @returns {number}
 */
export function firstNonFinite(values) {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) return index;
  }
  return -1;
}

/**
 * Refused by name, with the causes in the order they actually happen.
 *
 * 🔴 THE MESSAGE NAMES CAUSES RATHER THAN DESCRIBING THE SYMPTOM, because a
 * reader who sees "the simulation became non-finite" learns nothing they could
 * not see from the blank panels. Every cause listed here was reachable from the
 * page as shipped: the timestep field had no maximum, the friction field has
 * `min="0"`, and two atoms at the same coordinate is an ordinary artefact of a
 * hand-edited PDB.
 */
export class NonFiniteStateError extends Error {
  /**
   * @param {object} options
   * @param {string} options.what which quantity went non-finite
   * @param {number} [options.step]
   * @param {number} [options.index] component index, if known
   * @param {string} [options.detail] extra context for the caller
   */
  constructor(options) {
    const where = options.index !== undefined && options.index >= 0
      ? ` first at component ${options.index} (atom ${Math.floor(options.index / 3)}, `
        + `${"xyz"[options.index % 3]})`
      : "";
    const when = options.step !== undefined ? ` at step ${options.step}` : "";
    super(
      `the simulation is no longer a number: ${options.what} went non-finite`
      + `${when}${where}. The run has been stopped rather than continued, `
      + "because every frame after this point would be garbage that cpptraj and "
      + "MDAnalysis both read without complaint. The usual causes, commonest "
      + "first: a timestep too large for this system (2 fs is safe with "
      + "HBonds, 4 fs wants hydrogen mass repartitioning, which this port does "
      + "not do); two atoms at the same coordinate in the input, which makes "
      + "the first force evaluation infinite; a structure that was never "
      + "minimised; or a restart file whose velocities do not belong to its "
      + "positions."
      + (options.detail === undefined ? "" : ` ${options.detail}`),
    );
    this.name = "NonFiniteStateError";
    this.what = options.what;
    this.step = options.step;
    this.index = options.index;
  }
}
