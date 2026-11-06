/**
 * The unit system, and every conversion out of AMBER's into it.
 *
 * 🔴 THE INTERNAL UNITS ARE OPENMM'S, NOT AMBER'S, AND THAT IS THE WHOLE POINT.
 * nm, kJ/mol, ps, amu, elementary charge. The oracle this engine is checked
 * against is OpenMM (tools/oracle/dump_openmm_reference.py), and a comparison
 * is only as good as the units on both sides of it. Converting once, at parse
 * time, means every force kernel and every reference reads the same numbers the
 * notebook's OpenMM run read - so a disagreement is a disagreement about
 * physics rather than about 4.184.
 *
 * 🔴 AND AMBER'S HARMONIC TERMS ARE HALF OPENMM'S. AMBER writes
 * `E = k(r - r0)^2` where OpenMM's HarmonicBondForce is `E = (1/2) k(r - r0)^2`,
 * so a bond constant converted by energy and length alone is a factor of two
 * light and every bond in the system is too soft. Same for angles. The two
 * factors are folded into BOND_K_SCALE and ANGLE_K_SCALE and applied nowhere
 * else; a caller that multiplies by 4.184 itself has reintroduced the bug.
 */

/** kJ per kcal. Thermochemical calorie, which is what AMBER means. */
export const KCAL_PER_KJ = 4.184;

/** nm per angstrom. */
export const NM_PER_ANGSTROM = 0.1;

/**
 * 🔴 PRMTOP CHARGES ARE NOT IN ELEMENTARY CHARGE UNITS. They are q*18.2223,
 * chosen so that `sum qi*qj/r` comes out in kcal/mol with r in angstroms -
 * 18.2223^2 is e^2/(4*pi*eps0) in kcal*A/mol. Dividing by it recovers e, which
 * is what this engine stores, and is exactly what OpenMM's AmberPrmtopFile
 * does. A topology that keeps the raw number and a Coulomb kernel that uses
 * ONE_4PI_EPS0 is wrong by 332x on electrostatics alone.
 */
export const AMBER_CHARGE_SCALE = 18.2223;

/**
 * 🔴 AMBER RESTART VELOCITIES ARE IN ANGSTROM PER AMBER TIME UNIT, and the
 * AMBER time unit is 1/20.455 ps. So the file's number is multiplied by 20.455
 * to reach A/ps and then by 0.1 to reach nm/ps. Missing the 20.455 starts a
 * simulation at 1/20th of the intended temperature, which the thermostat then
 * quietly repairs over a few ps - so it does not crash, it just makes the first
 * part of every trajectory wrong.
 */
export const AMBER_VELOCITY_SCALE = 20.455;

/** 1/(4*pi*eps0) in kJ*nm/(mol*e^2). OpenMM's own value. */
export const ONE_4PI_EPS0 = 138.935456;

/** Boltzmann's constant in kJ/(mol*K). OpenMM's MOLAR_GAS_CONSTANT_R. */
export const BOLTZMANN = 0.008314462618;

/** bar -> kJ/(mol*nm^3), for the barostat's PV term. */
export const BAR_TO_KJ_PER_MOL_NM3 = 0.06022140857;

/** kcal/mol -> kJ/mol. */
export const ENERGY_SCALE = KCAL_PER_KJ;

/** kcal/mol/A^2 -> kJ/mol/nm^2, with AMBER's missing factor of two. */
export const BOND_K_SCALE = 2 * KCAL_PER_KJ / (NM_PER_ANGSTROM * NM_PER_ANGSTROM);

/** kcal/mol/rad^2 -> kJ/mol/rad^2, with AMBER's missing factor of two. */
export const ANGLE_K_SCALE = 2 * KCAL_PER_KJ;

/**
 * Lennard-Jones sigma/epsilon out of AMBER's A/B coefficients.
 *
 * AMBER tabulates `A = 4*eps*sigma^12` and `B = 4*eps*sigma^6` per TYPE PAIR,
 * in kcal/mol with angstroms, rather than per atom with a combining rule. So
 * sigma = (A/B)^(1/6) and eps = B^2/(4A).
 *
 * 🔴 B == 0 IS A REAL AND COMMON ENTRY, not a corrupt file: it is how a
 * massless or dummy type with no dispersion is written, and `(A/B)^(1/6)` on it
 * is Infinity. Returning sigma 1 / eps 0 keeps the pair inert, which is what a
 * zero B means, and keeps the number finite so a NaN does not reach the GPU and
 * spread through the whole force array on the first step.
 *
 * @param {number} a ACOEF entry, kcal/mol * A^12
 * @param {number} b BCOEF entry, kcal/mol * A^6
 * @returns {{sigma: number, epsilon: number}} sigma in nm, epsilon in kJ/mol
 */
export function ljFromAmberCoefficients(a, b) {
  if (b === 0 || a === 0) return { sigma: 1, epsilon: 0 };
  const sigmaAngstrom = Math.pow(a / b, 1 / 6);
  const epsilonKcal = (b * b) / (4 * a);
  return {
    sigma: sigmaAngstrom * NM_PER_ANGSTROM,
    epsilon: epsilonKcal * ENERGY_SCALE,
  };
}

/**
 * The A/B coefficients this engine's kernels want, in its own units.
 *
 * The kernels evaluate `A/r^12 - B/r^6` directly rather than reconstructing
 * sigma and epsilon per pair, because that is one fewer pow() per interaction
 * and because it is what AMBER tabulates - a round trip through
 * sigma/epsilon and back is two cancelling sixth roots that lose bits for
 * nothing.
 *
 * @returns {{a: number, b: number}} a in kJ/mol*nm^12, b in kJ/mol*nm^6
 */
export function ljCoefficientsToInternal(a, b) {
  return {
    a: a * ENERGY_SCALE * Math.pow(NM_PER_ANGSTROM, 12),
    b: b * ENERGY_SCALE * Math.pow(NM_PER_ANGSTROM, 6),
  };
}
