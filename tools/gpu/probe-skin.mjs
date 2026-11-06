/**
 * Per-block maximum displacement, printed block by block.
 *
 * Written to settle one question with a measurement: does the rebuild origin
 * re-seed on every rebuild, or is the reported displacement accumulating over
 * the whole run? A resetting origin gives a flat series; an accumulating one
 * grows like sqrt(t).
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";
import { readBuffer } from "../../webgpu/runtime/execution.js";

export async function main(device, args = []) {
  const interval = Number((args.find((a) => a.startsWith("--interval=")) ?? "--interval=10")
    .slice("--interval=".length));
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const simulation = new GpuSimulation(device, system, {
    timestep: 0.002, temperature: 298, friction: 10.0, seed: 4242,
    skin: 10.0, rebuildInterval: interval,   // huge skin: never a violation
  });
  try {
    const velocities = new Float64Array(coordinates.positions.length);
    setVelocitiesToTemperature(system, velocities, 298, 17);
    const start = createState(system, coordinates.positions, { velocities, tolerance: 1e-5 });
    simulation.setState(start.positions, start.velocities);
    await simulation.initialise();
    const series = [];
    for (let block = 0; block < 20; block += 1) {
      await simulation.run(interval, { blockSize: interval });
      const squared = (await readBuffer(device, simulation.allocator,
        simulation.resources.displacement, 4, Float32Array))[0];
      series.push(Number(Math.sqrt(Math.max(squared, 0)).toFixed(5)));
    }
    return {
      check: "per-block max displacement",
      interval,
      steps: interval,
      series,
      // A flat series means the origin re-seeds; a growing one means it does not.
      verdict: series[19] > series[0] * 2 ? "ACCUMULATING - origin is not re-seeding"
        : "flat - origin re-seeds every rebuild",
    };
  } finally { simulation.release(); }
}
