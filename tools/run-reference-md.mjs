/**
 * Run a short simulation on the host reference, end to end.
 *
 * This is the evidence that the pieces compose into a simulation rather than
 * into a set of passing unit checks: a thermostat that holds its setpoint over
 * a few hundred steps is a statement about the forces, the constraints, the
 * degree-of-freedom count and the integrator all being right together.
 *
 *   node tools/run-reference-md.mjs [steps]
 */
import { readFileSync } from "node:fs";
import { parsePrmtop } from "../shared/amber/prmtop.js";
import { parseInpcrd } from "../shared/amber/inpcrd.js";
import { buildSystem } from "../shared/topology/system.js";
import { createState, minimise, runDynamics } from "../cpu/simulation.js";

const steps = Number(process.argv[2] ?? 200);
const base = new URL("../test/fixtures/ala-wat", import.meta.url);
const topology = parsePrmtop(readFileSync(new URL(`${base}.prmtop`), "utf8"));
const coordinates = parseInpcrd(readFileSync(new URL(`${base}.inpcrd`), "utf8"),
  { expectedAtomCount: topology.atomCount });
const system = buildSystem(topology, {
  constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
});

const options = {
  method: "PME",
  alpha: 2.6282608939112583,
  grid: [24, 24, 24],
  timestep: 0.002,
  temperature: 298,
  friction: 1.0,
  seed: 20261106,
  tolerance: 1e-6,
};

console.log(`system: ${topology.atomCount} atoms, ${system.constraints.count} constraints, `
  + `${system.degreesOfFreedom} degrees of freedom, box `
  + `${Array.from(system.boxVectors.filter((_, i) => i % 4 === 0)).map((v) => v.toFixed(3)).join(" x ")} nm`);

const state = createState(system, coordinates.positions, { ...options, temperature: 298 });
const before = Date.now();
const minimised = minimise(system, state, { ...options, maxIterations: 60, forceTolerance: 500 });
console.log(`\nminimisation: ${minimised.iterations} iterations, `
  + `energy ${minimised.history[0].energy.toFixed(2)} -> ${minimised.energy.total.toFixed(2)} kJ/mol, `
  + `max force ${minimised.history[0].maxForce.toFixed(0)} -> ${minimised.maxForce.toFixed(0)} kJ/mol/nm`);

console.log(`\n${steps} steps of Langevin dynamics at ${options.temperature} K, `
  + `${options.timestep * 1000} fs timestep:`);
console.log("  step      time/ps     potential/kJ/mol   kinetic    temperature/K");
const run = runDynamics(system, state, {
  ...options,
  steps,
  reportInterval: Math.max(1, Math.floor(steps / 10)),
  onReport: (report) => {
    console.log(`  ${String(report.step).padStart(5)}  ${report.time.toFixed(4).padStart(9)}  `
      + `${report.potential.toFixed(2).padStart(16)}  ${report.kinetic.toFixed(2).padStart(9)}  `
      + `${report.temperature.toFixed(2).padStart(10)}`);
  },
});
const elapsed = (Date.now() - before) / 1000;
const temperatures = run.reports.map((report) => report.temperature);
const mean = temperatures.reduce((sum, value) => sum + value, 0) / temperatures.length;
const spread = Math.sqrt(temperatures.reduce((sum, value) => sum + (value - mean) ** 2, 0)
  / temperatures.length);
console.log(`\nmean temperature ${mean.toFixed(2)} +/- ${spread.toFixed(2)} K `
  + `against a ${options.temperature} K setpoint`);
console.log(`${elapsed.toFixed(1)} s on the host reference `
  + `(${(elapsed / steps * 1000).toFixed(0)} ms/step for ${topology.atomCount} atoms)`);
