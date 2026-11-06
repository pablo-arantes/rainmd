/**
 * Does the device minimiser lower the energy, keep the constraints, and leave a
 * structure that runs?
 *
 * 🔴 NOT A COMPARISON AGAINST THE HOST MINIMISER, BECAUSE THEY ARE DIFFERENT
 * ALGORITHMS WITH DIFFERENT ARITHMETIC. Steepest descent is path-dependent, so
 * two implementations reach different points even when both are correct, and
 * the f64 one can keep going past where f32 can tell whether the energy moved.
 * What is testable is monotonicity (the energy never rises), the constraints
 * (an absolute check), and the only thing minimisation is actually for: that
 * dynamics started from the result does not blow up.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { GpuSimulation } from "../../webgpu/simulation.js";

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const simulation = new GpuSimulation(device, system, {
    timestep: 0.002, temperature: 298, friction: 10.0, seed: 13,
    alpha: 2.6282608939112583, grid: [24, 24, 24], skin: 0.2, rebuildInterval: 10,
  });
  let failures = 0;
  try {
    simulation.setState(coordinates.positions, new Float64Array(topology.atomCount * 3));
    await simulation.initialise();
    const before = await simulation.energy();
    const began = Date.now();
    const result = await simulation.minimise({ maxIterations: 200, forceTolerance: 500 });
    const seconds = (Date.now() - began) / 1000;

    // Monotone: every recorded potential must be at or below the one before it.
    let monotone = true;
    for (let index = 1; index < result.history.length; index += 1) {
      if (result.history[index].potential > result.history[index - 1].potential + 1e-6) {
        monotone = false;
      }
    }

    // Constraints, from the minimised geometry.
    const state = await simulation.readState();
    let worstConstraint = 0;
    for (let index = 0; index < system.constraints.count; index += 1) {
      const a = system.constraints.atomA[index];
      const b = system.constraints.atomB[index];
      const length = Math.hypot(
        state.positions[a * 3] - state.positions[b * 3],
        state.positions[a * 3 + 1] - state.positions[b * 3 + 1],
        state.positions[a * 3 + 2] - state.positions[b * 3 + 2]);
      worstConstraint = Math.max(worstConstraint,
        Math.abs(length - system.constraints.distance[index])
          / system.constraints.distance[index]);
    }

    // 🔴 THE REAL TEST: DOES DYNAMICS FROM HERE RUN? Minimisation exists to
    // remove the one contact that would otherwise blow up the first step, so a
    // short run from the minimised structure with a finite temperature is the
    // check that matters. A blown-up run shows as a NaN energy or a temperature
    // in the thousands.
    simulation.setState(state.positions, new Float64Array(topology.atomCount * 3));
    await simulation.initialise();
    await simulation.run(500, { blockSize: 10 });
    const after = await simulation.energy();
    const diagnostics = await simulation.diagnostics();
    const ran = Number.isFinite(after.total) && after.temperature < 1000
      && !diagnostics.shakeHitIterationCap;

    const ok = monotone && worstConstraint < 2e-5 && ran
      && result.potential < before.potential;
    if (!ok) failures += 1;
    return {
      check: "device minimiser: monotone, constrained, and runnable afterwards",
      failures,
      results: {
        potential: { before: before.potential, after: result.potential },
        maxForce: { before: before.maxForce ?? null, after: result.maxForce },
        iterations: result.iterations,
        rejectedTrials: result.rejected,
        stalled: result.stalled,
        converged: result.converged,
        energyResolution: result.energyResolution,
        seconds,
        monotone,
        worstConstraintError: worstConstraint,
        dynamicsAfterwards: {
          ran, steps: 500, temperature: after.temperature, total: after.total,
        },
        ok,
      },
    };
  } finally {
    simulation.release();
  }
}
