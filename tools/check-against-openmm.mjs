/**
 * Hold the host reference to OpenMM's own numbers, term by term.
 *
 * 🔴 PER TERM, AND WITH LJ AND COULOMB SEPARATED BY THE ORACLE'S OWN ZEROING
 * ARMS. A total-energy comparison on this fixture passes while the reciprocal
 * sum is a quarter of its true value and the self energy makes up the
 * difference, because those two are large and opposite. The oracle builds the
 * same system three times - whole, charges zeroed, epsilons zeroed - precisely
 * so that LJ and Coulomb can be compared on their own.
 *
 *   node tools/check-against-openmm.mjs
 */
import { readFileSync } from "node:fs";
import { parsePrmtop } from "../shared/amber/prmtop.js";
import { parseInpcrd } from "../shared/amber/inpcrd.js";
import { buildSystem } from "../shared/topology/system.js";
import { potentialEnergy } from "../cpu/energy.js";

const fixture = (name) => new URL(`../test/fixtures/${name}`, import.meta.url);

const CASES = [
  {
    name: "ala-vac / NoCutoff / unconstrained",
    topology: "ala-vac", reference: "ala-vac-nocutoff.json",
    build: { constraints: "none", rigidWater: false, cutoff: 0 },
    evaluate: { method: "NoCutoff" },
  },
  {
    name: "ala-wat / PME 24^3 / unconstrained",
    topology: "ala-wat", reference: "ala-wat-pme-free.json",
    build: { constraints: "none", rigidWater: false, cutoff: 1.0 },
    evaluate: { method: "PME", alpha: 2.6282608939112583, grid: [24, 24, 24] },
  },
  {
    name: "ala-wat / PME 24^3 / HBonds + rigid water",
    topology: "ala-wat", reference: "ala-wat-pme.json",
    build: { constraints: "HBonds", rigidWater: true, cutoff: 1.0 },
    evaluate: { method: "PME", alpha: 2.6282608939112583, grid: [24, 24, 24] },
  },
];

let failures = 0;
let comparisons = 0;

for (const testCase of CASES) {
  const topology = parsePrmtop(readFileSync(fixture(`${testCase.topology}.prmtop`), "utf8"));
  const coordinates = parseInpcrd(
    readFileSync(fixture(`${testCase.topology}.inpcrd`), "utf8"),
    { expectedAtomCount: topology.atomCount },
  );
  const system = buildSystem(topology, { ...testCase.build, box: coordinates.box });
  const { energy, forces } = potentialEnergy(system, coordinates.positions, testCase.evaluate);
  const reference = JSON.parse(readFileSync(fixture(testCase.reference), "utf8"));

  console.log(`\n=== ${testCase.name} ===`);
  console.log(`    ${topology.atomCount} atoms, ${system.constraints.count} constraints, `
    + `${system.exceptions.count} exceptions`);

  const check = (label, ours, theirs, tolerance) => {
    comparisons += 1;
    const scale = Math.max(Math.abs(theirs), 1);
    const relative = Math.abs(ours - theirs) / scale;
    const ok = relative <= tolerance;
    if (!ok) failures += 1;
    console.log(`  ${ok ? "ok  " : "FAIL"} ${label.padEnd(22)} `
      + `ours ${ours.toFixed(6).padStart(16)}   openmm ${theirs.toFixed(6).padStart(16)}   `
      + `rel ${relative.toExponential(2)}`);
  };

  // 🔴 THE BONDED TERMS ARE READ OUT OF THE ORACLE'S FORCE GROUPS BY CLASS
  // NAME, not by index: the order OpenMM puts forces in a system is its
  // business and has changed between versions.
  const groups = reference.energy.byForceGroup;
  const groupEnergy = (className) => {
    for (const [key, value] of Object.entries(groups)) {
      if (key.endsWith(`:${className}`)) return value.energy;
    }
    return null;
  };
  const bondReference = groupEnergy("HarmonicBondForce");
  const angleReference = groupEnergy("HarmonicAngleForce");
  const torsionReference = groupEnergy("PeriodicTorsionForce");

  if (bondReference !== null) check("harmonic bond", energy.harmonicBond, bondReference, 1e-10);
  if (angleReference !== null) check("harmonic angle", energy.harmonicAngle, angleReference, 1e-10);
  if (torsionReference !== null) {
    check("periodic torsion", energy.periodicTorsion, torsionReference, 1e-10);
  }
  check("lennard-jones", energy.lennardJones, reference.energy.ljOnly, 1e-9);
  check("coulomb (all terms)", energy.coulomb, reference.energy.coulombOnly, 1e-7);
  check("TOTAL", energy.total, reference.energy.total, 1e-7);

  if (reference.forces !== undefined) {
    const theirForces = reference.forces.total;
    let worst = 0;
    let worstAtom = -1;
    let normOurs = 0;
    let normTheirs = 0;
    for (let atom = 0; atom < topology.atomCount; atom += 1) {
      for (let component = 0; component < 3; component += 1) {
        const ours = forces[atom * 3 + component];
        const theirs = theirForces[atom][component];
        normOurs += ours * ours;
        normTheirs += theirs * theirs;
        const difference = Math.abs(ours - theirs);
        if (difference > worst) { worst = difference; worstAtom = atom; }
      }
    }
    const rms = Math.sqrt(normTheirs / (3 * topology.atomCount));
    comparisons += 1;
    const relative = worst / rms;
    const ok = relative <= 1e-6;
    if (!ok) failures += 1;
    console.log(`  ${ok ? "ok  " : "FAIL"} ${"forces (per atom)".padEnd(22)} `
      + `worst component error ${worst.toExponential(3)} kJ/mol/nm at atom ${worstAtom}, `
      + `rms force ${rms.toFixed(3)} -> rel ${relative.toExponential(2)}`);
    console.log(`       |F| ours ${Math.sqrt(normOurs).toFixed(6)}  openmm ${Math.sqrt(normTheirs).toFixed(6)}`);
  }
}

console.log(`\n${comparisons - failures}/${comparisons} comparisons within tolerance`
  + (failures > 0 ? `  --  ${failures} FAILED` : "  --  all pass"));
process.exit(failures > 0 ? 1 : 0);
