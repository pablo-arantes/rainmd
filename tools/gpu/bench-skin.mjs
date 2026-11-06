/**
 * What a safe skin and rebuild interval cost, measured rather than assumed.
 *
 * 🔴 THE TWO KNOBS TRADE AGAINST EACH OTHER AND BOTH BUY THE SAME THING. A
 * wider skin puts more pairs in the list, so every step of the nonbonded kernel
 * does more work; a shorter interval rebuilds more often, so the cell list and
 * the neighbour build run more often. Either one buys margin against a stale
 * list. Choosing between them needs the prices, and nothing here had measured
 * them.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

const ARMS = [
  { skin: 0.2, rebuildInterval: 10, note: "the old default - VIOLATES (needs 0.295)" },
  { skin: 0.35, rebuildInterval: 20, note: "safe but only 2.4% of margin left" },
  { skin: 0.4, rebuildInterval: 20, note: "candidate" },
  { skin: 0.5, rebuildInterval: 20, note: "generous margin" },
  { skin: 0.4, rebuildInterval: 40, note: "fewer rebuilds still" },
  { skin: 0.5, rebuildInterval: 40, note: "fewest rebuilds" },
];

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const rows = [];
  // Two rounds, interleaved, minimum per arm: a first round pays for pipeline
  // compilation and a warm cache, which is not what is being compared.
  for (let round = 0; round < 2; round += 1) {
    for (const arm of ARMS) {
      const simulation = new GpuSimulation(device, system, {
        timestep: 0.002, temperature: 298, friction: 10.0, seed: 4242,
        skin: arm.skin, rebuildInterval: arm.rebuildInterval,
      });
      try {
        const velocities = new Float64Array(coordinates.positions.length);
        setVelocitiesToTemperature(system, velocities, 298, 17);
        const start = createState(system, coordinates.positions, { velocities, tolerance: 1e-5 });
        simulation.setState(start.positions, start.velocities);
        await simulation.initialise();
        await simulation.run(50, { blockSize: arm.rebuildInterval });   // warm
        const began = performance.now();
        await simulation.run(400, { blockSize: arm.rebuildInterval });
        const ms = (performance.now() - began) / 400;
        const diagnostics = await simulation.diagnostics();
        const key = `skin ${arm.skin} / every ${arm.rebuildInterval}`;
        const existing = rows.find((row) => row.arm === key);
        const row = existing ?? { arm: key, note: arm.note, msPerStep: Infinity };
        row.msPerStep = Math.min(row.msPerStep, ms);
        row.neighbours = diagnostics.neighbourRequired;
        row.violations = diagnostics.skinViolations;
        row.intervalEnded = diagnostics.rebuildInterval;
        row.skinUsed = Number(((2 * diagnostics.worstDisplacement) / arm.skin).toFixed(3));
        if (existing === undefined) rows.push(row);
      } finally { simulation.release(); }
    }
  }
  for (const row of rows) row.msPerStep = Number(row.msPerStep.toFixed(3));
  const base = rows[0].msPerStep;
  for (const row of rows) row.relative = Number((row.msPerStep / base).toFixed(3));
  return { check: "the price of a safe neighbour list", failures: 0, rows };
}
