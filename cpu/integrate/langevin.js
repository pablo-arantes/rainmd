/**
 * The BAOAB Langevin integrator, with constraints.
 *
 * 🔴 THIS IS LangevinMiddle, NOT THE NOTEBOOKS' `LangevinIntegrator`, AND THAT
 * IS A DELIBERATE DIFFERENCE RATHER THAN AN OVERSIGHT. Amber.ipynb and
 * Protein_ligand.ipynb both construct OpenMM's original `LangevinIntegrator`,
 * which applies the friction and the noise around the position update (an OBABO
 * splitting). BAOAB puts the position update around the thermostat, and its
 * configurational averages are correct to a higher order in the timestep - it
 * is why OpenMM now recommends `LangevinMiddleIntegrator` and why its own
 * documentation calls the older one superseded.
 *
 * What that means for comparison: a BAOAB trajectory and a LangevinIntegrator
 * trajectory from the same seed are DIFFERENT trajectories. They sample the same
 * ensemble, so averages agree and individual frames do not. An energy or force
 * comparison against OpenMM must therefore be a SINGLE POINT comparison - which
 * is what tools/check-against-openmm.mjs does - and never a comparison of
 * frame 1000.
 */
import { BOLTZMANN } from "../../shared/units.js";
import { gaussianTriple } from "../../shared/random.js";
import { applyPositionConstraints, applyVelocityConstraints } from "./constraints.js";

/**
 * @param {object} state {positions, velocities, step}
 * @param {object} system from buildSystem
 * @param {(positions: Float64Array) => {energy: object, forces: Float64Array}} evaluate
 * @param {object} options {timestep (ps), temperature (K), friction (1/ps), seed}
 */
export function langevinStep(state, system, forces, options) {
  const dt = options.timestep;
  const temperature = options.temperature;
  const friction = options.friction;
  const seed = options.seed ?? 0;
  const masses = system.topology.masses;
  const atomCount = system.topology.atomCount;
  const { positions, velocities } = state;
  const halfStep = 0.5 * dt;
  const decay = Math.exp(-friction * dt);
  const noiseScale = Math.sqrt(1 - decay * decay);
  const kT = BOLTZMANN * temperature;
  const reference = state.scratch ?? (state.scratch = new Float64Array(positions.length));
  const normals = state.normals ?? (state.normals = new Float64Array(3));

  // --- B: half kick -------------------------------------------------------
  for (let atom = 0; atom < atomCount; atom += 1) {
    const mass = masses[atom];
    if (mass <= 0) continue;
    const scale = halfStep / mass;
    velocities[atom * 3] += scale * forces[atom * 3];
    velocities[atom * 3 + 1] += scale * forces[atom * 3 + 1];
    velocities[atom * 3 + 2] += scale * forces[atom * 3 + 2];
  }
  applyVelocityConstraints(system, positions, velocities, options);

  // --- A: half drift ------------------------------------------------------
  driftWithConstraints(system, positions, velocities, reference, halfStep, options);

  // --- O: thermostat ------------------------------------------------------
  // 🔴 THE NOISE AMPLITUDE CARRIES 1/sqrt(m), SO A MASSLESS ATOM MUST BE
  // SKIPPED RATHER THAN DIVIDED BY. AMBER topologies really do contain
  // zero-mass extra points, and `sqrt(kT/0)` is an Infinity that the next
  // position update turns into a NaN for the whole system.
  // 🔴 THE NOISE CAN BE SUPPLIED, AND THAT IS WHAT MAKES THE DEVICE
  // INTEGRATOR TESTABLE. WGSL does not specify log, cos and sin to the last
  // bit, so the device's normals differ from these at ~1e-7 even from an
  // identical counter - and a one-step comparison then cannot distinguish a
  // wrong decay constant from a transcendental function's rounding. Handing
  // both sides the same noise array removes the generator from the comparison
  // and leaves only the integrator's arithmetic. A production run supplies
  // nothing and both sides generate their own.
  const supplied = options.noise;
  for (let atom = 0; atom < atomCount; atom += 1) {
    const mass = masses[atom];
    if (mass <= 0) continue;
    if (supplied === undefined) {
      gaussianTriple(normals, seed, atom, state.step);
    } else {
      normals[0] = supplied[atom * 3];
      normals[1] = supplied[atom * 3 + 1];
      normals[2] = supplied[atom * 3 + 2];
    }
    const amplitude = noiseScale * Math.sqrt(kT / mass);
    velocities[atom * 3] = decay * velocities[atom * 3] + amplitude * normals[0];
    velocities[atom * 3 + 1] = decay * velocities[atom * 3 + 1] + amplitude * normals[1];
    velocities[atom * 3 + 2] = decay * velocities[atom * 3 + 2] + amplitude * normals[2];
  }
  applyVelocityConstraints(system, positions, velocities, options);

  // --- A: half drift ------------------------------------------------------
  driftWithConstraints(system, positions, velocities, reference, halfStep, options);

  state.step += 1;
  // The caller recomputes forces at the new positions and calls finishStep,
  // which is the second B. Splitting it that way means one force evaluation per
  // step rather than two.
  return state;
}

/** The second half-kick, after the caller has recomputed forces. */
export function langevinFinishStep(state, system, forces, options) {
  const halfStep = 0.5 * options.timestep;
  const masses = system.topology.masses;
  const { positions, velocities } = state;
  for (let atom = 0; atom < system.topology.atomCount; atom += 1) {
    const mass = masses[atom];
    if (mass <= 0) continue;
    const scale = halfStep / mass;
    velocities[atom * 3] += scale * forces[atom * 3];
    velocities[atom * 3 + 1] += scale * forces[atom * 3 + 1];
    velocities[atom * 3 + 2] += scale * forces[atom * 3 + 2];
  }
  applyVelocityConstraints(system, positions, velocities, options);
}

/**
 * Move, then put the constrained lengths back, then take the velocity from the
 * motion that actually happened.
 *
 * 🔴 THE VELOCITY MUST BE RE-READ FROM THE CONSTRAINED DISPLACEMENT. Projecting
 * the positions and leaving the velocities alone means the velocity no longer
 * describes where the atom went, so the kinetic energy counts motion that the
 * constraint removed - the simulation reports a temperature above the
 * thermostat's and the thermostat fights it for the whole run.
 */
function driftWithConstraints(system, positions, velocities, reference, step, options) {
  reference.set(positions);
  for (let index = 0; index < positions.length; index += 1) {
    positions[index] += step * velocities[index];
  }
  if (system.constraints.count === 0) return;
  const result = applyPositionConstraints(system, positions, reference, options);
  if (!result.converged) {
    const error = new Error(result.message ?? "constraints did not converge");
    error.name = "ConstraintFailure";
    throw error;
  }
  // 🔴 ONLY THE CONSTRAINED ATOMS. For an unconstrained atom
  // `(x - reference)/h` is exactly `v`, so recomputing it buys nothing and
  // costs a division's rounding - and the device solver, which runs one thread
  // per cluster, touches only a cluster's own atoms. Rewriting all of them here
  // would make the two disagree by that rounding on every free atom.
  const inverseStep = 1 / step;
  const { atomA, atomB, count } = system.constraints;
  for (let index = 0; index < count; index += 1) {
    for (const atom of [atomA[index], atomB[index]]) {
      velocities[atom * 3] = (positions[atom * 3] - reference[atom * 3]) * inverseStep;
      velocities[atom * 3 + 1] = (positions[atom * 3 + 1] - reference[atom * 3 + 1]) * inverseStep;
      velocities[atom * 3 + 2] = (positions[atom * 3 + 2] - reference[atom * 3 + 2]) * inverseStep;
    }
  }
}

/**
 * Instantaneous temperature from the kinetic energy.
 *
 * 🔴 THE DENOMINATOR IS THE CONSTRAINED DEGREE-OF-FREEDOM COUNT, NOT 3N. With
 * 1176 constraints on 1190 atoms, 3N over-counts by a third and the reported
 * temperature reads 33% low - so a correctly thermostatted 298 K run appears to
 * be at 200 K and the obvious "fix" is to raise the setpoint, which then really
 * does overheat it. See buildSystem, which computes degreesOfFreedom.
 */
export function instantaneousTemperature(system, velocities) {
  const masses = system.topology.masses;
  let kinetic = 0;
  for (let atom = 0; atom < system.topology.atomCount; atom += 1) {
    const mass = masses[atom];
    if (mass <= 0) continue;
    kinetic += mass * (
      velocities[atom * 3] * velocities[atom * 3]
      + velocities[atom * 3 + 1] * velocities[atom * 3 + 1]
      + velocities[atom * 3 + 2] * velocities[atom * 3 + 2]);
  }
  kinetic *= 0.5;
  return {
    kineticEnergy: kinetic,
    temperature: (2 * kinetic) / (system.degreesOfFreedom * BOLTZMANN),
  };
}

/** Draw velocities from the Maxwell-Boltzmann distribution at `temperature`. */
export function setVelocitiesToTemperature(system, velocities, temperature, seed = 0) {
  const masses = system.topology.masses;
  const normals = new Float64Array(3);
  const kT = BOLTZMANN * temperature;
  for (let atom = 0; atom < system.topology.atomCount; atom += 1) {
    const mass = masses[atom];
    if (mass <= 0) {
      velocities[atom * 3] = 0;
      velocities[atom * 3 + 1] = 0;
      velocities[atom * 3 + 2] = 0;
      continue;
    }
    gaussianTriple(normals, seed, atom, 0x5eed);
    const sigma = Math.sqrt(kT / mass);
    velocities[atom * 3] = sigma * normals[0];
    velocities[atom * 3 + 1] = sigma * normals[1];
    velocities[atom * 3 + 2] = sigma * normals[2];
  }
  return velocities;
}
