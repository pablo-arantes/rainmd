/**
 * The whole potential, on the host, with every term reported separately.
 *
 * The per-term breakdown is the point. A total that disagrees with OpenMM by a
 * few kJ/mol says nothing about where; the same comparison per term says
 * exactly which kernel to look at, and this is the shape the differential
 * tests and the GPU comparison both read.
 */
import { harmonicBondEnergy, harmonicAngleEnergy, periodicTorsionEnergy } from "./forces/bonded.js";
import { directSpaceEnergy, exceptionEnergy } from "./forces/nonbonded.js";
import { reciprocalEnergy, ewaldSelfEnergy } from "./forces/pme.js";
import { pmeParameters } from "../shared/pme/parameters.js";

/**
 * @param {object} system from buildSystem
 * @param {Float64Array} positions 3N, nm
 * @param {object} [options]
 * @param {"PME"|"NoCutoff"} [options.method]
 * @param {number} [options.tolerance] ewald error tolerance
 * @param {number[]} [options.grid] force the reciprocal grid
 * @param {number} [options.alpha] force the splitting parameter
 * @returns {{energy: object, forces: Float64Array}}
 */
export function potentialEnergy(system, positions, options = {}) {
  const method = options.method ?? (system.isPeriodic ? "PME" : "NoCutoff");
  const forces = new Float64Array(positions.length);
  const terms = {};

  terms.harmonicBond = harmonicBondEnergy(system, positions, forces);
  terms.harmonicAngle = harmonicAngleEnergy(system, positions, forces);
  terms.periodicTorsion = periodicTorsionEnergy(system, positions, forces);

  let parameters = null;
  let alpha = null;
  let cutoff = Infinity;
  if (method === "PME") {
    if (!system.isPeriodic) {
      throw new Error("PME needs a periodic box; this system has none");
    }
    parameters = pmeParameters({
      boxVectors: system.boxVectors,
      cutoff: system.options.cutoff,
      tolerance: options.tolerance ?? 0.0005,
      grid: options.grid,
      alpha: options.alpha,
    });
    alpha = parameters.alpha;
    cutoff = system.options.cutoff;
  } else if (method !== "NoCutoff") {
    throw new Error(
      `unsupported nonbonded method ${JSON.stringify(method)}; this reference `
      + "implements PME and NoCutoff",
    );
  }

  const direct = directSpaceEnergy(system, positions, forces, { alpha, cutoff });
  const exception = exceptionEnergy(system, positions, forces, { alpha });
  terms.lennardJones = direct.lennardJones + exception.lennardJones;
  terms.coulombDirect = direct.coulomb + exception.coulomb;

  if (parameters !== null) {
    terms.coulombReciprocal = reciprocalEnergy(system, positions, forces, parameters);
    const selfTerms = ewaldSelfEnergy(system, parameters);
    terms.coulombSelf = selfTerms.self;
    terms.coulombNetCharge = selfTerms.net;
  } else {
    terms.coulombReciprocal = 0;
    terms.coulombSelf = 0;
    terms.coulombNetCharge = 0;
  }

  terms.coulomb = terms.coulombDirect + terms.coulombReciprocal
    + terms.coulombSelf + terms.coulombNetCharge;
  terms.bonded = terms.harmonicBond + terms.harmonicAngle + terms.periodicTorsion;
  terms.total = terms.bonded + terms.lennardJones + terms.coulomb;

  return { energy: terms, forces, pme: parameters };
}
