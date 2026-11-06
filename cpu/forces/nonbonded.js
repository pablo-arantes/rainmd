/**
 * Direct-space nonbonded and the exception corrections, as a reference.
 *
 * O(N^2) over all pairs with a cutoff test. Deliberately: this is the oracle
 * the GPU cell list is checked against, and a reference that shares the
 * neighbour-list machinery with the thing under test cannot find a bug in it.
 * A cell list that drops a pair agrees perfectly with another cell list that
 * drops the same pair.
 */
import { ONE_4PI_EPS0 } from "../../shared/units.js";
import { erf, erfc } from "../../shared/erf.js";
import { minimumImage } from "./geometry.js";
import { buildExclusionSet } from "../../shared/topology/system.js";

/**
 * @param {object} system from buildSystem
 * @param {Float64Array} positions 3N, nm
 * @param {Float64Array} forces accumulated into
 * @param {object} options
 * @param {number|null} options.alpha Ewald parameter, or null for plain Coulomb
 * @param {number} options.cutoff nm; Infinity for NoCutoff
 * @returns {{lennardJones: number, coulomb: number}} kJ/mol
 */
export function directSpaceEnergy(system, positions, forces, options) {
  const { alpha, cutoff } = options;
  const topology = system.topology;
  const exclusions = buildExclusionSet(system);
  const box = system.boxVectors;
  const atomCount = topology.atomCount;
  const typeCount = topology.typeCount;
  const cutoffSquared = cutoff * cutoff;
  const delta = new Float64Array(3);
  let lennardJones = 0;
  let coulomb = 0;

  for (let i = 0; i < atomCount; i += 1) {
    const qi = topology.charges[i];
    const typeRow = typeCount * topology.atomTypes[i];
    for (let j = i + 1; j < atomCount; j += 1) {
      if (exclusions.has(i * 0x4000000 + j)) continue;
      delta[0] = positions[i * 3] - positions[j * 3];
      delta[1] = positions[i * 3 + 1] - positions[j * 3 + 1];
      delta[2] = positions[i * 3 + 2] - positions[j * 3 + 2];
      minimumImage(delta, box);
      const rSquared = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
      if (rSquared > cutoffSquared) continue;
      const r = Math.sqrt(rSquared);
      const inverseR = 1 / r;
      const inverseR2 = inverseR * inverseR;
      const inverseR6 = inverseR2 * inverseR2 * inverseR2;

      const typePair = typeRow + topology.atomTypes[j];
      const a = topology.ljA[typePair];
      const b = topology.ljB[typePair];
      const ljEnergy = a * inverseR6 * inverseR6 - b * inverseR6;
      lennardJones += ljEnergy;
      // dE/dr for A/r^12 - B/r^6 is -12A/r^13 + 6B/r^7; the scalar below is
      // -(1/r)dE/dr so that force = scalar * delta.
      let scalar = (12 * a * inverseR6 * inverseR6 - 6 * b * inverseR6) * inverseR2;

      const qq = ONE_4PI_EPS0 * qi * topology.charges[j];
      if (alpha === null) {
        coulomb += qq * inverseR;
        scalar += qq * inverseR * inverseR2;
      } else {
        // 🔴 THE SCREENED COULOMB DERIVATIVE IS NOT erfc' ALONE. d/dr
        // [erfc(ar)/r] = -erfc(ar)/r^2 - 2a exp(-a^2 r^2)/(sqrt(pi) r), and the
        // second term is the one a port forgets. Without it the direct-space
        // force is wrong by the Gaussian part everywhere inside the cutoff,
        // while the ENERGY stays exactly right - so every energy comparison
        // passes and the simulation heats up.
        const alphaR = alpha * r;
        const screened = erfc(alphaR);
        coulomb += qq * screened * inverseR;
        const gaussian = 2 * alpha * Math.exp(-alphaR * alphaR) / Math.sqrt(Math.PI);
        scalar += qq * (screened * inverseR + gaussian) * inverseR2;
      }

      forces[i * 3] += scalar * delta[0];
      forces[i * 3 + 1] += scalar * delta[1];
      forces[i * 3 + 2] += scalar * delta[2];
      forces[j * 3] -= scalar * delta[0];
      forces[j * 3 + 1] -= scalar * delta[1];
      forces[j * 3 + 2] -= scalar * delta[2];
    }
  }
  return { lennardJones, coulomb };
}

/**
 * The exception pairs: 1-4 interactions and the Ewald exclusion corrections.
 *
 * 🔴 AN EXCEPTION IS COMPUTED WITHOUT A CUTOFF TEST. OpenMM evaluates every
 * exception regardless of distance, and it must: the reciprocal sum included
 * that pair, so the correction has to be subtracted whether the pair happens to
 * be within the cutoff or not. Applying a cutoff here leaves the reciprocal
 * contribution of any 1-4 pair that has stretched past it in the energy, which
 * is rare, intermittent, and therefore read as noise.
 *
 * 🔴 AND THE erf CORRECTION APPLIES TO ALL OF THEM, INCLUDING THE PAIRS WHOSE
 * CHARGE PRODUCT IS ZERO. A fully excluded 1-2 pair contributes nothing to the
 * direct sum and plenty to the reciprocal one; the correction uses the atoms'
 * OWN charges, not the exception's scaled product, because what is being
 * removed is what the reciprocal sum actually put in.
 */
export function exceptionEnergy(system, positions, forces, options) {
  const { alpha } = options;
  const topology = system.topology;
  const exceptions = system.exceptions;
  const box = system.boxVectors;
  const delta = new Float64Array(3);
  let lennardJones = 0;
  let coulomb = 0;

  for (let index = 0; index < exceptions.count; index += 1) {
    const i = exceptions.atomA[index];
    const j = exceptions.atomB[index];
    delta[0] = positions[i * 3] - positions[j * 3];
    delta[1] = positions[i * 3 + 1] - positions[j * 3 + 1];
    delta[2] = positions[i * 3 + 2] - positions[j * 3 + 2];
    minimumImage(delta, box);
    const rSquared = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    const r = Math.sqrt(rSquared);
    const inverseR = 1 / r;
    const inverseR2 = inverseR * inverseR;
    const inverseR6 = inverseR2 * inverseR2 * inverseR2;
    let scalar = 0;

    const a = exceptions.ljA[index];
    const b = exceptions.ljB[index];
    if (a !== 0 || b !== 0) {
      lennardJones += a * inverseR6 * inverseR6 - b * inverseR6;
      scalar += (12 * a * inverseR6 * inverseR6 - 6 * b * inverseR6) * inverseR2;
    }

    const scaledProduct = exceptions.chargeProduct[index];
    if (scaledProduct !== 0) {
      const qq = ONE_4PI_EPS0 * scaledProduct;
      coulomb += qq * inverseR;
      scalar += qq * inverseR * inverseR2;
    }

    if (alpha !== null) {
      const qqFull = ONE_4PI_EPS0 * topology.charges[i] * topology.charges[j];
      if (qqFull !== 0) {
        const alphaR = alpha * r;
        const unscreened = erf(alphaR);
        coulomb -= qqFull * unscreened * inverseR;
        const gaussian = 2 * alpha * Math.exp(-alphaR * alphaR) / Math.sqrt(Math.PI);
        scalar -= qqFull * (unscreened * inverseR - gaussian) * inverseR2;
      }
    }

    forces[i * 3] += scalar * delta[0];
    forces[i * 3 + 1] += scalar * delta[1];
    forces[i * 3 + 2] += scalar * delta[2];
    forces[j * 3] -= scalar * delta[0];
    forces[j * 3 + 1] -= scalar * delta[1];
    forces[j * 3 + 2] -= scalar * delta[2];
  }
  return { lennardJones, coulomb };
}
