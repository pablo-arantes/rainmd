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
import { moleculesOf } from "../../shared/traj/imaging.js";
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { DcdWriter } from "../../shared/traj/dcd.js";
import { writeRestart } from "../../shared/traj/rst7.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

/** nm box vectors -> the lengths/angles pair the restart writer wants. */
function boxFromVectors(vectors) {
  const v = Array.from(vectors);
  const a = [v[0], v[1], v[2]];
  const b = [v[3], v[4], v[5]];
  const c = [v[6], v[7], v[8]];
  const norm = (u) => Math.hypot(u[0], u[1], u[2]);
  const dot = (u, w) => u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
  const deg = (u, w) => (Math.acos(dot(u, w) / (norm(u) * norm(w))) * 180) / Math.PI;
  return {
    lengths: [norm(a), norm(b), norm(c)],
    angles: [deg(b, c), deg(a, c), deg(a, b)],
  };
}

export async function main(device, args = []) {
  const out = (args.find((a) => a.startsWith("--out=")) ?? "--out=/tmp/rainmd-run")
    .slice("--out=".length);
  const steps = Number((args.find((a) => a.startsWith("--steps=")) ?? "--steps=1000")
    .slice("--steps=".length));
  const interval = Number((args.find((a) => a.startsWith("--interval=")) ?? "--interval=50")
    .slice("--interval=".length));
  // 🔴 NPT ON DEMAND, BECAUSE A CONSTANT BOX CANNOT TEST A VARYING ONE. Without
  // a barostat the cell never moves, so a writer that ignored the live box
  // produced a correct file by accident - which is exactly why the defect
  // survived: `check:mdanalysis` drives this module's sibling with no pressure.
  const pressureArg = args.find((a) => a.startsWith("--pressure="));
  const pressure = pressureArg === undefined
    ? null : Number(pressureArg.slice("--pressure=".length));

  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const options = {
    timestep: 0.002, temperature: 298, friction: 10.0, seed: 20261106,
    alpha: 2.6282608939112583, grid: [24, 24, 24], rebuildInterval: 20,
    pressure, barostatInterval: 25,
  };
  const simulation = new GpuSimulation(device, system, options);
  try {
    const velocities = new Float64Array(topology.atomCount * 3);
    setVelocitiesToTemperature(system, velocities, 298, 17);
    const start = createState(system, coordinates.positions, { velocities, tolerance: 1e-5 });
    simulation.setState(start.positions, start.velocities);
    await simulation.initialise();

    const writer = new DcdWriter({
        molecules: moleculesOf(topology),
      atomCount: topology.atomCount, timestep: options.timestep,
      interval, periodic: true, title: "RainMD ala-wat",
    });
    const temperatures = [];
    // The volume the simulation actually had when each frame was written, so a
    // reader can be held to it rather than to the construction-time box.
    const volumes = [];
    const began = Date.now();
    for (let done = 0; done < steps; done += interval) {
      await simulation.run(Math.min(interval, steps - done), { blockSize: 10 });
      const state = await simulation.readState();
      // The live box, not the system's construction-time one - see web/app.js.
      // `readState()` returns the simulation's current box, which under a
      // barostat is not the one the file was built with.
      writer.appendFrame(state.positions, state.boxVectors ?? system.boxVectors);
      const reported = await simulation.energy();
      temperatures.push(reported.temperature);
      volumes.push(reported.volume);
    }
    const elapsed = (Date.now() - began) / 1000;
    writeFileSync(`${out}.dcd`, writer.finish());
    const final = await simulation.readState();
    const energy = await simulation.energy();
    // 🔴 THE RESTART'S BOX IS THE LIVE ONE TOO. This wrote `coordinates.box` -
    // the box tleap guessed - so a restart from an NPT run carried the
    // pre-equilibration cell and whatever continued it started at the wrong
    // density. Same defect as the DCD's, one file along.
    writeFileSync(`${out}.rst7`, writeRestart({
      positions: final.positions, velocities: final.velocities,
      box: boxFromVectors(final.boxVectors ?? system.boxVectors),
      time: energy.time, title: "RainMD ala-wat",
    }));
    writeFileSync(`${out}-volumes.json`, JSON.stringify({
      pressure, interval, timestep: options.timestep, volumes,
    }, null, 2));
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
