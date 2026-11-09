/**
 * What a step costs on the shipped defaults, and what each addition costs.
 *
 * 🔴 THE DOCUMENTED 0.8 ms/step WAS MEASURED BEFORE HALF OF THIS EXISTED. It
 * was NVT, with a 0.2 nm skin and a 10-step rebuild, and without the
 * displacement kernel, the restraint pass or the barostat. Every one of those
 * landed afterwards and every one costs something. A reader comparing their own
 * ns/day against that number is comparing against a configuration the page no
 * longer offers, which is why this decomposes rather than reporting one figure.
 *
 * Interleaved, two rounds, minimum per arm: a first round pays for pipeline
 * compilation and a cold cache, which is not what is being compared.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { buildSelections } from "../../shared/analysis/selection.js";
import { soluteHeavyAtoms } from "../../shared/topology/restraints.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

const STEPS = 600;

export async function main(device, args = []) {
  const fixture = (args.find((a) => a.startsWith("--fixture=")) ?? "--fixture=ala-wat")
    .slice("--fixture=".length);
  const topology = parsePrmtop(await readFixture(`${fixture}.prmtop`, import.meta.url));
  const coordinates = parseInpcrd(await readFixture(`${fixture}.inpcrd`, import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const selections = buildSelections(system);
  const restrained = soluteHeavyAtoms(system, selections);

  const ARMS = [
    { key: "shipped default (NPT, derived skin, adaptive)",
      options: { pressure: 1 } },
    { key: "...without the barostat (NVT)",
      options: {} },
    { key: "...NVT, skin 0.2, rebuild every 10 (the old documented setting)",
      options: { skin: 0.2, rebuildInterval: 10, adaptiveRebuild: false } },
    { key: "...NPT with the equilibration restraint on",
      options: { pressure: 1, restraintForce: 500, restraintAtoms: restrained,
        restraintReference: coordinates.positions } },
    { key: "...NVT, rebuild every 40 instead of 20",
      options: { rebuildInterval: 40 } },
    { key: "...NVT at 4 fs instead of 2",
      options: { timestep: 0.004 } },
  ];

  const rows = [];
  for (let round = 0; round < 2; round += 1) {
    for (const arm of ARMS) {
      const timestep = arm.options.timestep ?? 0.002;
      const simulation = new GpuSimulation(device, system, {
        timestep, temperature: 298, friction: 1.0, seed: 7,
        rebuildInterval: 20, barostatInterval: 25, ...arm.options,
      });
      try {
        const velocities = new Float64Array(coordinates.positions.length);
        setVelocitiesToTemperature(system, velocities, 298, 17);
        const start = createState(system, coordinates.positions,
          { velocities, tolerance: 1e-5 });
        simulation.setState(start.positions, start.velocities);
        await simulation.initialise();
        await simulation.run(100, { blockSize: 20 });          // warm
        const began = performance.now();
        await simulation.run(STEPS, { blockSize: 20 });
        const ms = (performance.now() - began) / STEPS;
        const diagnostics = await simulation.diagnostics();
        let row = rows.find((r) => r.arm === arm.key);
        if (row === undefined) { row = { arm: arm.key, msPerStep: Infinity }; rows.push(row); }
        row.msPerStep = Math.min(row.msPerStep, ms);
        row.timestepFs = timestep * 1000;
        row.rebuildEnded = diagnostics.rebuildInterval;
        row.skin = Number(diagnostics.skin.toFixed(3));
        row.neighbours = diagnostics.neighbourRequired;
      } finally { simulation.release(); }
    }
  }
  for (const row of rows) {
    row.msPerStep = Number(row.msPerStep.toFixed(3));
    // ns/day = (timestep in ns per step) * (steps per day)
    const nsPerStep = row.timestepFs * 1e-6;
    row.nsPerDay = Number((nsPerStep * (86400_000 / row.msPerStep)).toFixed(0));
  }
  return {
    check: "throughput on the shipped defaults, decomposed",
    failures: 0,
    fixture,
    atoms: topology.atomCount,
    rows,
  };
}
