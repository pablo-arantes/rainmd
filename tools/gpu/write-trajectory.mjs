/**
 * Run a short GPU simulation and write a real DCD and restart to disk.
 *
 * 🔴 THE POINT IS THE HANDOFF, NOT THE SIMULATION. Everything upstream of this
 * is checked against OpenMM, but "a run on this page produces something the
 * rest of a user's toolchain can read" is a different claim and needs a
 * different check: the files this writes are opened by MDAnalysis and cpptraj
 * against the same `.prmtop`, with no RainMD code involved on the reading side.
 */
import { writeFileSync } from "node:fs";
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { DcdWriter } from "../../shared/traj/dcd.js";
import { writeRestart } from "../../shared/traj/rst7.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

export async function main(device, args = []) {
  const out = (args.find((a) => a.startsWith("--out=")) ?? "--out=/tmp/rainmd-run")
    .slice("--out=".length);
  const steps = Number((args.find((a) => a.startsWith("--steps=")) ?? "--steps=1000")
    .slice("--steps=".length));
  const interval = Number((args.find((a) => a.startsWith("--interval=")) ?? "--interval=50")
    .slice("--interval=".length));

  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const options = {
    timestep: 0.002, temperature: 298, friction: 10.0, seed: 20261106,
    alpha: 2.6282608939112583, grid: [24, 24, 24], skin: 0.2, rebuildInterval: 10,
  };
  const simulation = new GpuSimulation(device, system, options);
  try {
    const velocities = new Float64Array(topology.atomCount * 3);
    setVelocitiesToTemperature(system, velocities, 298, 17);
    const start = createState(system, coordinates.positions, { velocities, tolerance: 1e-5 });
    simulation.setState(start.positions, start.velocities);
    await simulation.initialise();

    const writer = new DcdWriter({
      atomCount: topology.atomCount, timestep: options.timestep,
      interval, periodic: true, title: "RainMD ala-wat",
    });
    const temperatures = [];
    const began = Date.now();
    for (let done = 0; done < steps; done += interval) {
      await simulation.run(Math.min(interval, steps - done), { blockSize: 10 });
      const state = await simulation.readState();
      writer.appendFrame(state.positions, system.boxVectors);
      temperatures.push((await simulation.energy()).temperature);
    }
    const elapsed = (Date.now() - began) / 1000;
    writeFileSync(`${out}.dcd`, writer.finish());
    const final = await simulation.readState();
    const energy = await simulation.energy();
    writeFileSync(`${out}.rst7`, writeRestart({
      positions: final.positions, velocities: final.velocities,
      box: coordinates.box, time: energy.time, title: "RainMD ala-wat",
    }));
    const mean = temperatures.reduce((a, b) => a + b, 0) / temperatures.length;
    return {
      check: "a GPU run written out as DCD and rst7",
      failures: 0,
      wrote: [`${out}.dcd`, `${out}.rst7`],
      atomCount: topology.atomCount,
      steps,
      frames: writer.frameCount,
      interval,
      picoseconds: energy.time,
      meanTemperature: mean,
      secondsOnDevice: elapsed,
      millisecondsPerStep: (elapsed / steps) * 1000,
      nsPerDay: (energy.time / 1000) / (elapsed / 86400),
      finalEnergy: { potential: energy.potential, kinetic: energy.kinetic, total: energy.total },
    };
  } finally {
    simulation.release();
  }
}
