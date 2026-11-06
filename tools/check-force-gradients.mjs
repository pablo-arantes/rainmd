/**
 * Hold every analytic force to the numerical gradient of its OWN energy.
 *
 * 🔴 THIS IS A DIFFERENT QUESTION FROM "DOES THE ENERGY MATCH OPENMM", AND
 * BOTH HAVE TO BE ASKED. An energy can be exactly right while its gradient is
 * wrong, because the energy never reads the gradient - and that combination is
 * the worst case in molecular dynamics: every single-point comparison passes,
 * the simulation runs, and it heats up or drifts for reasons that look like a
 * thermostat problem. Checking each term against a central difference of
 * itself localises the fault to one function, which comparing a total force
 * against OpenMM cannot do.
 *
 * Central differences, so the truncation error is O(h^2); h is chosen per
 * term from the energy scale, and the agreement reported is relative to the
 * term's own typical force rather than to the largest in the system.
 */
import { readFileSync } from "node:fs";
import { parsePrmtop } from "../shared/amber/prmtop.js";
import { parseInpcrd } from "../shared/amber/inpcrd.js";
import { buildSystem } from "../shared/topology/system.js";
import { harmonicBondEnergy, harmonicAngleEnergy, periodicTorsionEnergy } from "../cpu/forces/bonded.js";
import { directSpaceEnergy, exceptionEnergy } from "../cpu/forces/nonbonded.js";
import { reciprocalEnergy } from "../cpu/forces/pme.js";
import { pmeParameters } from "../shared/pme/parameters.js";

const fixture = (name) => new URL(`../test/fixtures/${name}`, import.meta.url);
const topology = parsePrmtop(readFileSync(fixture("ala-vac.prmtop"), "utf8"));
const coordinates = parseInpcrd(readFileSync(fixture("ala-vac.inpcrd"), "utf8"),
  { expectedAtomCount: topology.atomCount });
const system = buildSystem(topology, {
  constraints: "none", rigidWater: false, cutoff: 0, box: coordinates.box,
});

const watTopology = parsePrmtop(readFileSync(fixture("ala-wat.prmtop"), "utf8"));
const watCoordinates = parseInpcrd(readFileSync(fixture("ala-wat.inpcrd"), "utf8"),
  { expectedAtomCount: watTopology.atomCount });
const watSystem = buildSystem(watTopology, {
  constraints: "none", rigidWater: false, cutoff: 1.0, box: watCoordinates.box,
});
const watPme = pmeParameters({
  boxVectors: watSystem.boxVectors, cutoff: 1.0,
  alpha: 2.6282608939112583, grid: [24, 24, 24],
});

const TERMS = [
  {
    name: "harmonic bond", system, positions: coordinates.positions,
    evaluate: (s, p, f) => harmonicBondEnergy(s, p, f),
  },
  {
    name: "harmonic angle", system, positions: coordinates.positions,
    evaluate: (s, p, f) => harmonicAngleEnergy(s, p, f),
  },
  {
    name: "periodic torsion", system, positions: coordinates.positions,
    evaluate: (s, p, f) => periodicTorsionEnergy(s, p, f),
  },
  {
    name: "direct space (NoCutoff)", system, positions: coordinates.positions,
    evaluate: (s, p, f) => {
      const r = directSpaceEnergy(s, p, f, { alpha: null, cutoff: Infinity });
      return r.lennardJones + r.coulomb;
    },
  },
  {
    name: "exceptions (NoCutoff)", system, positions: coordinates.positions,
    evaluate: (s, p, f) => {
      const r = exceptionEnergy(s, p, f, { alpha: null });
      return r.lennardJones + r.coulomb;
    },
  },
  {
    name: "direct space (PME screened)", system: watSystem, positions: watCoordinates.positions,
    atoms: 40,
    evaluate: (s, p, f) => {
      const r = directSpaceEnergy(s, p, f, { alpha: watPme.alpha, cutoff: 1.0 });
      return r.lennardJones + r.coulomb;
    },
  },
  {
    name: "exceptions (PME correction)", system: watSystem, positions: watCoordinates.positions,
    atoms: 40,
    evaluate: (s, p, f) => {
      const r = exceptionEnergy(s, p, f, { alpha: watPme.alpha });
      return r.lennardJones + r.coulomb;
    },
  },
  {
    name: "PME reciprocal", system: watSystem, positions: watCoordinates.positions,
    atoms: 12, step: 2e-6,
    evaluate: (s, p, f) => reciprocalEnergy(s, p, f, watPme),
  },
];

let failures = 0;
console.log("term                             worst |analytic-numeric| / rms(numeric)   atom");
console.log("-".repeat(88));
for (const term of TERMS) {
  const positions = term.positions;
  const analytic = new Float64Array(positions.length);
  term.evaluate(term.system, positions, analytic);
  const step = term.step ?? 1e-7;
  const atomLimit = Math.min(term.atoms ?? term.system.topology.atomCount,
    term.system.topology.atomCount);

  let worst = 0;
  let worstAtom = -1;
  let sumSquares = 0;
  let samples = 0;
  const scratch = new Float64Array(positions.length);
  for (let atom = 0; atom < atomLimit; atom += 1) {
    for (let component = 0; component < 3; component += 1) {
      const index = atom * 3 + component;
      const original = positions[index];
      positions[index] = original + step;
      scratch.fill(0);
      const plus = term.evaluate(term.system, positions, scratch);
      positions[index] = original - step;
      scratch.fill(0);
      const minus = term.evaluate(term.system, positions, scratch);
      positions[index] = original;
      const numeric = -(plus - minus) / (2 * step);
      sumSquares += numeric * numeric;
      samples += 1;
      const difference = Math.abs(analytic[index] - numeric);
      if (difference > worst) { worst = difference; worstAtom = atom; }
    }
  }
  const rms = Math.sqrt(sumSquares / samples);
  const relative = worst / Math.max(rms, 1e-6);
  // Central differences on a double give about sqrt(eps) relative accuracy on a
  // well-scaled term; 1e-5 is a generous bar that still catches a wrong sign,
  // a dropped term or a permuted index, which are what this looks for.
  const ok = relative < 1e-5;
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"} ${term.name.padEnd(28)} `
    + `${relative.toExponential(3).padStart(12)}   (worst ${worst.toExponential(2)}, `
    + `rms ${rms.toExponential(2)})  atom ${worstAtom}`);
}
console.log("-".repeat(88));
console.log(failures === 0 ? "every analytic force is the gradient of its own energy"
  : `${failures} term(s) have a gradient that does not match their energy`);
process.exit(failures > 0 ? 1 : 0);
