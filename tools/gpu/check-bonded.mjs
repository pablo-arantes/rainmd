/**
 * Hold the bonded KERNEL to the bonded REFERENCE, per atom.
 *
 * 🔴 PER ATOM, NOT ON THE TOTAL. The whole reason the GPU path gathers by atom
 * is that each thread owns its own three force components, so a bug is almost
 * always one role of one term kind being wrong - a torsion's atom 2 taking atom
 * 3's coefficient, say. Summed over the system those cancel to something small,
 * because the bonded forces sum to zero by construction. Comparing the per-atom
 * vectors is what can see it.
 *
 * 🔴 AND THE BAR IS DERIVED FROM f32, NOT CHOSEN. A bond force is k*(r - r0),
 * and r is a difference of two coordinates that span the whole box - so in f32
 * its absolute error is about span*eps and the force error is k times that,
 * whatever the force itself is. For this solvated fixture that is
 *
 *     476976 kJ/mol/nm^2  x  2.82 nm  x  5.96e-8  =  0.16 kJ/mol/nm
 *
 * and the kernel measured 0.157. So the first version of this check, with a
 * relative bar of 2e-5, was asking for an agreement f32 cannot give and would
 * have had someone hunting a kernel bug that is arithmetic. The bar below is
 * that expression with a factor of four of slack for the several terms an atom
 * accumulates - it still refuses a wrong torsion coefficient, which moves a
 * force by tens of percent rather than by a part in a million.
 *
 * 🔴 AND IT IS WHY `HBonds` IS NOT MERELY CONVENTIONAL. Constraining the bonds
 * to hydrogen removes exactly the stiffest springs from the harmonic term -
 * 1185 bonds down to 9 here - so the configuration the notebooks actually run
 * is also the one where this error source is almost entirely gone. The
 * unconstrained arm is kept because it is the only one that exercises every
 * bond term, and it is held to its own, looser, derived bound.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { buildBondedGather } from "../../shared/topology/gather.js";
import { GpuAllocator } from "../../webgpu/runtime/allocator.js";
import { readBuffer, DeferredValidation } from "../../webgpu/runtime/execution.js";
import { encodeBonded } from "../../webgpu/kernels/bonded.js";
import {
  harmonicBondEnergy, harmonicAngleEnergy, periodicTorsionEnergy,
} from "../../cpu/forces/bonded.js";

const CASES = [
  { fixture: "ala-vac", constraints: "none", rigidWater: false, cutoff: 0 },
  { fixture: "ala-wat", constraints: "none", rigidWater: false, cutoff: 1.0 },
  // The notebooks' own configuration, and the one a user will run.
  { fixture: "ala-wat", constraints: "HBonds", rigidWater: true, cutoff: 1.0 },
];

/** f32 mantissa epsilon. */
const F32_EPSILON = 2 ** -24;

/**
 * The largest bonded force error f32 coordinates can produce for this system.
 * See the note at the top of the file.
 */
function f32ForceBound(system, positions) {
  let span = 0;
  for (const value of positions) span = Math.max(span, Math.abs(value));
  let stiffest = 0;
  for (let index = 0; index < system.harmonicBonds.count; index += 1) {
    stiffest = Math.max(stiffest, system.harmonicBonds.k[index]);
  }
  let stiffestAngle = 0;
  for (let index = 0; index < system.topology.angles.count; index += 1) {
    stiffestAngle = Math.max(stiffestAngle, system.topology.angles.k[index]);
  }
  // Bonds dominate: an angle constant is in kJ/mol/rad^2 and its force arm is
  // the bond length, so its contribution is smaller by roughly that factor.
  const bondTerm = stiffest * span * F32_EPSILON * 2;
  const angleTerm = stiffestAngle * F32_EPSILON * 2 / 0.1;
  return { span, stiffest, bound: 4 * (bondTerm + angleTerm) };
}

export async function main(device) {
  const allocator = new GpuAllocator(device);
  const results = [];
  let failures = 0;

  for (const testCase of CASES) {
    const name = testCase.fixture;
    const topology = parsePrmtop(await readFixture(`${name}.prmtop`, import.meta.url));
    const coordinates = parseInpcrd(await readFixture(`${name}.inpcrd`, import.meta.url),
      { expectedAtomCount: topology.atomCount });
    const system = buildSystem(topology, {
      constraints: testCase.constraints, rigidWater: testCase.rigidWater,
      box: coordinates.box, cutoff: testCase.cutoff,
    });
    const precision = f32ForceBound(system, coordinates.positions);
    const gather = buildBondedGather(system);
    const atomCount = topology.atomCount;

    // Host reference, f64.
    const referenceForces = new Float64Array(atomCount * 3);
    const referenceEnergy = harmonicBondEnergy(system, coordinates.positions, referenceForces)
      + harmonicAngleEnergy(system, coordinates.positions, referenceForces)
      + periodicTorsionEnergy(system, coordinates.positions, referenceForces);

    // Device.
    const positions = Float32Array.from(coordinates.positions);
    const held = [];
    const keep = (allocation) => { held.push(allocation); return allocation; };
    try {
      const resources = {
        positions: keep(allocator.upload("positions", positions, GPUBufferUsage.STORAGE)),
        forces: keep(allocator.allocate("forces", atomCount * 3 * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)),
        energies: keep(allocator.allocate("energies", atomCount * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)),
        termIndices: keep(allocator.upload("termIndices", gather.termIndices, GPUBufferUsage.STORAGE)),
        termParams: keep(allocator.upload("termParams",
          Float32Array.from(gather.termParams), GPUBufferUsage.STORAGE)),
        gatherStart: keep(allocator.upload("gatherStart", gather.start, GPUBufferUsage.STORAGE)),
        gatherList: keep(allocator.upload("gatherList", gather.list, GPUBufferUsage.STORAGE)),
      };

      const validation = new DeferredValidation(device, `bonded/${name}`);
      validation.begin();
      const encoder = device.createCommandEncoder({ label: `bonded/${name}` });
      encoder.clearBuffer(resources.forces.buffer);
      encodeBonded(device, encoder, resources, atomCount, gather.layout);
      device.queue.submit([encoder.finish()]);
      validation.end("bonded pass");
      await validation.settle();

      const deviceForces = await readBuffer(device, allocator, resources.forces,
        atomCount * 3 * 4, Float32Array);
      const deviceEnergies = await readBuffer(device, allocator, resources.energies,
        atomCount * 4, Float32Array);

      let deviceEnergy = 0;
      for (let atom = 0; atom < atomCount; atom += 1) deviceEnergy += deviceEnergies[atom];

      let worst = 0;
      let worstAtom = -1;
      let sumSquares = 0;
      for (let index = 0; index < atomCount * 3; index += 1) {
        sumSquares += referenceForces[index] * referenceForces[index];
        const difference = Math.abs(deviceForces[index] - referenceForces[index]);
        if (difference > worst) { worst = difference; worstAtom = Math.floor(index / 3); }
      }
      const rms = Math.sqrt(sumSquares / (atomCount * 3));
      const forceRelative = worst / Math.max(rms, 1e-6);
      const energyRelative = Math.abs(deviceEnergy - referenceEnergy)
        / Math.max(Math.abs(referenceEnergy), 1);

      const forceOk = worst <= precision.bound;
      const energyOk = energyRelative < 2e-6;
      if (!forceOk || !energyOk) failures += 1;
      results.push({
        fixture: `${name} / ${testCase.constraints}`,
        atomCount,
        terms: {
          bonds: gather.layout.bondCount,
          angles: gather.layout.angleCount,
          torsions: gather.layout.torsionCount,
        },
        energy: {
          device: deviceEnergy, reference: referenceEnergy,
          relative: energyRelative, ok: energyOk,
        },
        force: {
          worstComponent: worst,
          f32Bound: precision.bound,
          usedFractionOfBound: worst / precision.bound,
          rms,
          relativeToRms: forceRelative,
          worstAtom,
          ok: forceOk,
        },
      });
    } finally {
      for (let index = held.length - 1; index >= 0; index -= 1) held[index].release();
    }
  }

  return { check: "bonded kernel against host reference", failures, results,
    allocator: allocator.snapshot() };
}
