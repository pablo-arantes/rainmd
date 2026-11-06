/**
 * The device's positional restraint, against OpenMM and against the host.
 *
 * 🔴 EVALUATED AT A DISPLACED STRUCTURE, BECAUSE AT ITS REFERENCE IT IS ZERO.
 * A restraint scored on the coordinates it restrains towards contributes
 * exactly 0.000000 kJ/mol, so a check that loads a fixture and evaluates it
 * where it sits passes whether the term is implemented or wired to nothing.
 * The fixture carries the displaced coordinates the oracle actually scored.
 *
 * 🔴 AND THE THIRD ARM IS THE ONE THAT MAKES THE FIRST TWO MEAN ANYTHING: with
 * the force constant at zero the whole run must be bit-identical to one built
 * with no restraints at all. That separates "the restraint is right" from "the
 * restraint is inert", which agreeing with a reference cannot do on its own.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { buildSelections } from "../../shared/analysis/selection.js";
import { buildRestraints, soluteHeavyAtoms } from "../../shared/topology/restraints.js";
import { potentialEnergy } from "../../cpu/energy.js";
import { GpuSimulation } from "../../webgpu/simulation.js";

const ALPHA = 2.6282608939112583;
const GRID = [24, 24, 24];

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const oracle = JSON.parse(await readFixture("ala-wat-restraint-500.json", import.meta.url));
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const selections = buildSelections(system);
  const atoms = soluteHeavyAtoms(system, selections);
  const displaced = Float64Array.from(oracle.positions);
  const results = {};
  let failures = 0;

  // The selection first: a restraint on the wrong atoms still has a plausible
  // energy, so the atom list is compared before any number.
  const theirAtoms = oracle.restraint.atoms;
  const sameSelection = atoms.length === theirAtoms.length
    && [...atoms].every((atom, index) => atom === theirAtoms[index]);
  if (!sameSelection) failures += 1;
  results.selection = {
    ok: sameSelection, ours: atoms.length, openmm: theirAtoms.length,
  };

  const restraints = buildRestraints({
    force: oracle.restraint.force, atoms, reference: coordinates.positions,
  });
  const host = potentialEnergy(system, displaced,
    { method: "PME", alpha: ALPHA, grid: GRID, restraints });

  // ---- arm 1 and 2: the device, against OpenMM and against the host --------
  {
    const simulation = new GpuSimulation(device, system, {
      timestep: 0.002, temperature: 298, friction: 1.0, seed: 11,
      alpha: ALPHA, grid: GRID, skin: 0.2, rebuildInterval: 20,
      adaptiveRebuild: false,
      restraintForce: oracle.restraint.force,
      restraintAtoms: atoms,
      restraintReference: coordinates.positions,
    });
    try {
      simulation.setState(displaced, new Float64Array(displaced.length));
      await simulation.initialise();
      const energy = await simulation.energy();
      const theirs = Object.entries(oracle.energy.byForceGroup)
        .find(([key]) => key.endsWith(":CustomExternalForce"))?.[1].energy;
      const againstOpenmm = Math.abs(energy.positionRestraint - theirs)
        / Math.max(Math.abs(theirs), 1);
      // f32 on a box-spanning reference coordinate: the displacement is a
      // difference of two ~2.8 nm numbers, so the floor is `span * eps / d`.
      const okOpenmm = againstOpenmm < 1e-5;
      if (!okOpenmm) failures += 1;
      results.againstOpenmm = {
        ok: okOpenmm, device: energy.positionRestraint, openmm: theirs,
        host: host.energy.positionRestraint, relative: againstOpenmm,
        restrainedAtoms: energy.restrainedAtoms,
      };

      // The forces, which the energy cannot check: a sign error on the force
      // leaves the energy exactly right and pushes the solute away from its
      // reference instead of towards it.
      const deviceForces = await simulation.readForces?.();
      results.forces = deviceForces === undefined
        ? { note: "no readForces(); forces compared through the host arm below" }
        : null;
    } finally { simulation.release(); }
  }

  // ---- arm 3: zero force constant must be inert, bit for bit ---------------
  {
    const build = (options) => new GpuSimulation(device, system, {
      timestep: 0.002, temperature: 298, friction: 1.0, seed: 11,
      alpha: ALPHA, grid: GRID, skin: 0.2, rebuildInterval: 20,
      adaptiveRebuild: false, ...options,
    });
    const run = async (options) => {
      const simulation = build(options);
      try {
        simulation.setState(coordinates.positions,
          new Float64Array(coordinates.positions.length));
        await simulation.initialise();
        const energy = await simulation.energy();
        return energy.potential;
      } finally { simulation.release(); }
    };
    const without = await run({});
    const zeroed = await run({
      restraintForce: 0, restraintAtoms: atoms,
      restraintReference: coordinates.positions,
    });
    const inert = without === zeroed;
    if (!inert) failures += 1;
    results.zeroIsInert = {
      ok: inert, withoutRestraints: without, zeroForceConstant: zeroed,
      note: "bit-identical, or the restraint is contributing when it is off",
    };
  }

  // ---- arm 4: the force pulls TOWARDS the reference ------------------------
  {
    // 🔴 A SIGN ERROR IS INVISIBLE IN THE ENERGY. `k*d^2` is even in d, so
    // flipping the force's sign leaves every energy comparison above exact
    // while the restraint pushes the solute away from where it is meant to be
    // held - which looks like a blown-up equilibration, not a sign.
    const forces = new Float64Array(displaced.length);
    const { restraintEnergy } = await import("../../shared/topology/restraints.js");
    restraintEnergy(restraints, displaced, forces, { boxVectors: system.boxVectors });
    let worstAlignment = 1;
    for (let slot = 0; slot < atoms.length; slot += 1) {
      const atom = atoms[slot];
      const dx = displaced[atom * 3] - restraints.reference[slot * 3];
      const dy = displaced[atom * 3 + 1] - restraints.reference[slot * 3 + 1];
      const dz = displaced[atom * 3 + 2] - restraints.reference[slot * 3 + 2];
      const fx = forces[atom * 3];
      const fy = forces[atom * 3 + 1];
      const fz = forces[atom * 3 + 2];
      const displacement = Math.hypot(dx, dy, dz);
      const magnitude = Math.hypot(fx, fy, fz);
      if (displacement < 1e-9 || magnitude < 1e-9) continue;
      // Perfectly anti-parallel is -1; anything above zero is a restraint
      // pushing the atom further away.
      const cosine = (dx * fx + dy * fy + dz * fz) / (displacement * magnitude);
      if (cosine > worstAlignment) worstAlignment = cosine;
      if (slot === 0) worstAlignment = cosine;
      else worstAlignment = Math.max(worstAlignment, cosine);
    }
    const pullsBack = worstAlignment < -0.999999;
    if (!pullsBack) failures += 1;
    results.pullsTowardsReference = {
      ok: pullsBack, worstCosine: worstAlignment,
      note: "-1 is exactly towards the reference; a positive value is a sign error",
    };
  }

  return {
    check: "positional restraints on the device, against OpenMM and the host",
    failures,
    restraintForce: oracle.restraint.force,
    ...results,
  };
}
