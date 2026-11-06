/**
 * Hold the barostat's physics to OpenMM, and then its statistics to itself.
 *
 * 🔴 THE ENERGY AT A SCALED VOLUME IS THE THING TO CHECK, NOT THE ACCEPTANCE
 * RATE. A Monte Carlo barostat is a Metropolis test around one number - the
 * potential energy at a trial volume - and everything that can be wrong is in
 * producing that number: scaling centroids instead of atoms, forgetting to
 * update the reciprocal box, letting the cell list go stale when the cells
 * shrink, or omitting the volume-dependent dispersion term. Each of those gives
 * a plausible acceptance rate and a wrong density. So the first arm forces a
 * known scale factor and compares against an OpenMM energy computed at exactly
 * the same scaled coordinates.
 *
 * 🔴 AND THE ORACLE SCALES THE SAME MOLECULES. Its partition comes from the
 * bond graph by the same union-find, and it weights centroids by AMBER's own
 * masses rather than the periodic table's - otherwise the centroids differ
 * slightly and the comparison measures that.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

const ALPHA = 2.6282608939112583;
const GRID = [24, 24, 24];
const SCALES = ["1.00", "1.02", "0.97"];

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const results = { scaled: [], sampling: null };
  let failures = 0;

  // ---- arm 1: the energy at a forced scale, against OpenMM ---------------
  for (const scale of SCALES) {
    const reference = JSON.parse(await readFixture(`ala-wat-scale-${scale}.json`, import.meta.url));
    const simulation = new GpuSimulation(device, system, {
      timestep: 0.002, temperature: 298, friction: 1.0, seed: 5,
      alpha: ALPHA, grid: GRID, skin: 0.2, rebuildInterval: 10,
    });
    try {
      simulation.setState(coordinates.positions, new Float64Array(topology.atomCount * 3));
      await simulation.initialise();
      const potential = await simulation.scaleVolume(Number(scale));
      const relative = Math.abs(potential - reference.energy.total)
        / Math.abs(reference.energy.total);
      // The same bar as the unscaled force-path comparison: f32 summation over
      // the whole system plus the PME grid's own 8.4e-6 floor.
      const bar = 5e-5;
      const ok = relative <= bar;
      if (!ok) failures += 1;
      results.scaled.push({
        lengthScale: Number(scale),
        volumeRatio: Number(scale) ** 3,
        device: potential,
        openmm: reference.energy.total,
        relative, bar, ok,
        cells: simulation.plan.counts.join("x"),
      });
    } finally {
      simulation.release();
    }
  }

  // ---- arm 2: does the sampler move the volume, and how often ------------
  {
    const simulation = new GpuSimulation(device, system, {
      timestep: 0.002, temperature: 298, friction: 10.0, seed: 31,
      alpha: ALPHA, grid: GRID, skin: 0.2, rebuildInterval: 10,
      pressure: 1.0, barostatInterval: 25,
    });
    try {
      const velocities = new Float64Array(topology.atomCount * 3);
      setVelocitiesToTemperature(system, velocities, 298, 23);
      const start = createState(system, coordinates.positions, { velocities, tolerance: 1e-5 });
      simulation.setState(start.positions, start.velocities);
      await simulation.initialise();
      const first = await simulation.energy();
      const trace = [];
      const began = Date.now();
      for (let block = 0; block < 12; block += 1) {
        await simulation.run(250, { blockSize: 10 });
        const energy = await simulation.energy();
        trace.push({ step: energy.step, volume: energy.volume,
          potential: energy.potential, temperature: energy.temperature });
      }
      const elapsed = (Date.now() - began) / 1000;
      const diagnostics = await simulation.diagnostics();
      const last = trace[trace.length - 1];
      // 🔴 A BAROSTAT MUST ACTUALLY MOVE THE BOX AND MUST NOT ACCEPT
      // EVERYTHING. A rate near zero means the moves are too large or the
      // energy is wrong; a rate near one means the moves are too small to
      // sample anything. The adaptive scale targets the band between.
      const rate = diagnostics.barostat.acceptanceRate;
      const moved = Math.abs(last.volume - first.volume) / first.volume;
      const ok = rate > 0.05 && rate < 0.95 && moved > 1e-4
        && !diagnostics.shakeHitIterationCap;
      if (!ok) failures += 1;
      results.sampling = {
        pressure: 1.0,
        steps: 3000,
        attempts: diagnostics.barostat.attempts,
        accepted: diagnostics.barostat.accepted,
        acceptanceRate: rate,
        volumeScale: diagnostics.barostat.volumeScale,
        startVolume: first.volume,
        endVolume: last.volume,
        fractionalChange: (last.volume - first.volume) / first.volume,
        startDensity: densityOf(system, first.volume),
        endDensity: densityOf(system, last.volume),
        meanTemperature: trace.reduce((a, b) => a + b.temperature, 0) / trace.length,
        millisecondsPerStep: (elapsed / 3000) * 1000,
        finalCells: simulation.plan.counts.join("x"),
        trace,
        ok,
      };
    } finally {
      simulation.release();
    }
  }
  return { check: "Monte Carlo barostat against OpenMM and against itself", failures, results };
}

/** g/cm^3, so the number means something to a chemist. */
function densityOf(system, volume) {
  let mass = 0;
  for (const value of system.topology.masses) mass += value;
  // amu -> g, nm^3 -> cm^3
  return (mass * 1.66053906660e-24) / (volume * 1e-21);
}
