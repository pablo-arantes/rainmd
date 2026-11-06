/**
 * How long can the neighbour list go between rebuilds before the energy drifts?
 *
 * 🔴 A STALE NEIGHBOUR LIST IS A SILENT ENERGY LEAK. Pairs that move inside the
 * cutoff after the list was built are simply absent from the force, so the
 * energy is wrong by however many of them there are - smoothly, with no error
 * and no NaN. The only symptom is drift in a run that should conserve energy,
 * which is why this is measured rather than assumed from the skin.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";
import { createState } from "../../cpu/simulation.js";

const ARMS = [
  { skin: 0.1, rebuildInterval: 20 },
  { skin: 0.1, rebuildInterval: 5 },
  { skin: 0.2, rebuildInterval: 20 },
  { skin: 0.2, rebuildInterval: 10 },
  { skin: 0.3, rebuildInterval: 20 },
];

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const results = [];
  for (const arm of ARMS) {
    const simulation = new GpuSimulation(device, system, {
      timestep: 0.002, temperature: 298, friction: 0, seed: 11,
      alpha: 2.6282608939112583, grid: [24, 24, 24], ...arm,
    });
    try {
      const velocities = new Float64Array(topology.atomCount * 3);
      setVelocitiesToTemperature(system, velocities, 298, 5);
      const start = createState(system, coordinates.positions, { velocities, tolerance: 1e-5 });
      simulation.setState(start.positions, start.velocities);
      await simulation.initialise();
      const trace = [await simulation.energy()];
      const began = Date.now();
      for (let block = 0; block < 8; block += 1) {
        await simulation.run(100, { blockSize: 20 });
        trace.push(await simulation.energy());
      }
      const elapsed = (Date.now() - began) / 1000;
      let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
      for (const report of trace) {
        const x = report.time / 1000;
        sumX += x; sumY += report.total; sumXY += x * report.total; sumXX += x * x;
      }
      const n = trace.length;
      const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
      results.push({
        ...arm,
        neighbourCapacity: simulation.plan.maxNeighbours,
        driftPerNsPerAtom: slope / topology.atomCount,
        millisecondsPerStep: (elapsed / 800) * 1000,
        finalTemperature: trace[trace.length - 1].temperature,
      });
    } finally {
      simulation.release();
    }
  }
  return { check: "energy drift against rebuild interval and skin", failures: 0, results };
}
