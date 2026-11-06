/**
 * The host-side simulation loop: load, minimise, equilibrate, run.
 *
 * This is the reference driver. It is the thing the WebGPU driver must agree
 * with, and it is also a usable simulator in its own right for small systems -
 * which is how the page has something to show for a term whose kernel is not
 * written yet.
 */
import { potentialEnergy } from "./energy.js";
import {
  langevinStep, langevinFinishStep, instantaneousTemperature,
  setVelocitiesToTemperature,
} from "./integrate/langevin.js";
import {
  applyPositionConstraints, applyVelocityConstraints,
} from "./integrate/constraints.js";

export function createState(system, positions, options = {}) {
  const state = {
    positions: Float64Array.from(positions),
    velocities: new Float64Array(positions.length),
    step: 0,
  };
  if (options.velocities !== undefined && options.velocities !== null) {
    state.velocities.set(options.velocities);
  } else if (options.temperature !== undefined) {
    setVelocitiesToTemperature(system, state.velocities, options.temperature,
      options.seed ?? 0);
  }
  // 🔴 THE STARTING STRUCTURE MUST BE PUT ON THE CONSTRAINT MANIFOLD BEFORE
  // STEP ONE, IN VELOCITY AS WELL AS IN POSITION. tleap writes ideal bond
  // lengths, so the position projection is usually a no-op - but
  // setVelocitiesToTemperature draws 3N independent components and the system
  // has only `degreesOfFreedom` of them, so the drawn velocities carry energy
  // along every constrained bond that RATTLE then removes on the first step.
  //
  // Measured on the solvated fixture: 3N components at 298 K carry 4422.7
  // kJ/mol where 2391 degrees of freedom want 2962.1, so the first step threw
  // away **1460 kJ/mol** - and with the friction at zero that appeared as a
  // 1510 kJ/mol drop in the total energy between the first two reports, read as
  // integrator drift. It was an initialisation artifact. Projecting here makes
  // the reported initial energy the energy the trajectory actually starts with.
  if (system.constraints.count > 0) {
    const reference = Float64Array.from(state.positions);
    applyPositionConstraints(system, state.positions, reference, options);
    applyVelocityConstraints(system, state.positions, state.velocities, options);
  }
  return state;
}

/**
 * Steepest descent with a backtracking line search.
 *
 * 🔴 NOT L-BFGS, WHICH IS WHAT OPENMM'S minimizeEnergy USES. This will take
 * more iterations to reach the same gradient norm, and on a structure with a
 * real clash it will take many more. It is here because it is twenty lines that
 * cannot be subtly wrong, and because the number that matters for starting a
 * simulation is the maximum force rather than the energy: the point of
 * minimising is to remove the one contact that would otherwise blow up the
 * first step.
 */
export function minimise(system, state, options = {}) {
  const maxIterations = options.maxIterations ?? 500;
  const forceTolerance = options.forceTolerance ?? 100; // kJ/mol/nm
  let stepSize = options.initialStep ?? 1e-5;
  let current = potentialEnergy(system, state.positions, options);
  const trial = new Float64Array(state.positions.length);
  const history = [];

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    let maxForce = 0;
    for (const value of current.forces) maxForce = Math.max(maxForce, Math.abs(value));
    if (iteration % Math.max(1, Math.floor(maxIterations / 10)) === 0 || maxForce <= forceTolerance) {
      history.push({ iteration, energy: current.energy.total, maxForce });
    }
    if (maxForce <= forceTolerance) {
      return { state, energy: current.energy, iterations: iteration, maxForce, history };
    }
    // Normalise by the largest component so the first trial step is a known
    // displacement in nm rather than a guess scaled by the force magnitude.
    const scale = stepSize / maxForce;
    let accepted = false;
    for (let attempt = 0; attempt < 12; attempt += 1) {
      for (let index = 0; index < trial.length; index += 1) {
        trial[index] = state.positions[index] + scale * current.forces[index];
      }
      if (system.constraints.count > 0) {
        applyPositionConstraints(system, trial, state.positions, options);
      }
      const candidate = potentialEnergy(system, trial, options);
      if (candidate.energy.total < current.energy.total) {
        state.positions.set(trial);
        current = candidate;
        stepSize *= 1.2;
        accepted = true;
        break;
      }
      stepSize *= 0.5;
      if (stepSize < 1e-12) break;
    }
    if (!accepted) break;
  }
  let maxForce = 0;
  for (const value of current.forces) maxForce = Math.max(maxForce, Math.abs(value));
  return { state, energy: current.energy, iterations: maxIterations, maxForce, history };
}

/**
 * Advance `steps` steps, reporting every `reportInterval`.
 *
 * @param {object} options timestep (ps), temperature (K), friction (1/ps),
 *   seed, reportInterval, onReport, stopSignal
 */
export function runDynamics(system, state, options) {
  const steps = options.steps;
  const reportInterval = options.reportInterval ?? Math.max(1, Math.floor(steps / 10));
  const reports = [];
  let evaluation = potentialEnergy(system, state.positions, options);

  for (let step = 0; step < steps; step += 1) {
    options.stopSignal?.throwIfStopped?.();
    langevinStep(state, system, evaluation.forces, options);
    evaluation = potentialEnergy(system, state.positions, options);
    langevinFinishStep(state, system, evaluation.forces, options);

    if ((step + 1) % reportInterval === 0 || step === steps - 1) {
      const kinetic = instantaneousTemperature(system, state.velocities);
      const report = {
        step: state.step,
        time: state.step * options.timestep,
        potential: evaluation.energy.total,
        kinetic: kinetic.kineticEnergy,
        total: evaluation.energy.total + kinetic.kineticEnergy,
        temperature: kinetic.temperature,
      };
      reports.push(report);
      options.onReport?.(report);
    }
  }
  return { state, reports, energy: evaluation.energy };
}
