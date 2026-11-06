/**
 * The whole GPU force path against OpenMM, in the notebooks' configuration.
 *
 * This is the capstone comparison: bonded, cell list, neighbour list,
 * direct-space nonbonded, exceptions and PME, all on the device, against the
 * reference OpenMM itself produced for this system with PME, a 1.0 nm cutoff,
 * HBonds constraints and rigid water.
 *
 * 🔴 PER TERM AS WELL AS IN TOTAL. The total is a sum of a +1795 kJ/mol LJ
 * term, a -13652 Coulomb term and a large negative self energy, so two
 * compensating mistakes cancel in it - which has already happened once in this
 * repository's history. The oracle's zeroed arms give LJ and Coulomb
 * separately and the force groups give the bonded terms, so every line below
 * is independently anchored.
 *
 * 🔴 AND THE SELF ENERGY IS A HOST CONSTANT WITH NO KERNEL. It is
 * -alpha/sqrt(pi) * sum(q^2) * COULOMB, thousands of kJ/mol on this system -
 * larger than the whole Coulomb energy - and it exerts no force at all. So a
 * port can omit it entirely and still run a perfectly stable simulation with a
 * badly wrong energy, which is why it is computed and reported here rather
 * than left implicit.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { buildBondedGather } from "../../shared/topology/gather.js";
import {
  planCellList, grownPlan, buildExclusionGather, buildExceptionGather,
} from "../../shared/topology/cells.js";
import { pmeParameters } from "../../shared/pme/parameters.js";
import { reciprocalBoxVectors } from "../../cpu/forces/geometry.js";
import { ewaldSelfEnergy } from "../../cpu/forces/pme.js";
import { GpuAllocator } from "../../webgpu/runtime/allocator.js";
import { readBuffer, DeferredValidation } from "../../webgpu/runtime/execution.js";
import { packParams } from "../../webgpu/kernels/wgsl-common.js";
import { encodeBonded } from "../../webgpu/kernels/bonded.js";
import { encodeCellList } from "../../webgpu/kernels/cell-list.js";
import { encodeNeighbours, NeighbourOverflowError } from "../../webgpu/kernels/neighbours.js";
import { encodeNonbonded } from "../../webgpu/kernels/nonbonded.js";
import { encodePme } from "../../webgpu/kernels/pme.js";
import { ONE_4PI_EPS0 } from "../../shared/units.js";

const ALPHA = 2.6282608939112583;
const GRID = [24, 24, 24];
const CUTOFF = 1.0;

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const reference = JSON.parse(await readFixture("ala-wat-pme.json", import.meta.url));
  const atomCount = topology.atomCount;
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: CUTOFF, box: coordinates.box,
  });
  const gather = buildBondedGather(system);
  const exclusions = buildExclusionGather(system);
  const exceptions = buildExceptionGather(system);
  const parameters = pmeParameters({
    boxVectors: system.boxVectors, cutoff: CUTOFF, alpha: ALPHA, grid: GRID,
  });
  const selfTerms = ewaldSelfEnergy(system, parameters);
  let plan = planCellList(system, { skin: 0.1 });

  const allocator = new GpuAllocator(device);
  const gridPoints = GRID[0] * GRID[1] * GRID[2];
  const storage = GPUBufferUsage.STORAGE;
  const readable = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const held = [];
  const keep = (allocation) => { held.push(allocation); return allocation; };

  try {
    const ljTable = new Float32Array(topology.typeCount * topology.typeCount * 2);
    for (let index = 0; index < topology.typeCount * topology.typeCount; index += 1) {
      ljTable[index * 2] = topology.ljA[index];
      ljTable[index * 2 + 1] = topology.ljB[index];
    }
    const moduli = new Float32Array(GRID[0] + GRID[1] + GRID[2]);
    let cursor = 0;
    for (let d = 0; d < 3; d += 1) {
      moduli.set(Float32Array.from(parameters.moduli[d]), cursor);
      cursor += GRID[d];
    }
    const partialCount = Math.ceil(gridPoints / 256);

    const resources = {
      params: keep(allocator.upload("params",
        packParams(system.boxVectors, reciprocalBoxVectors(system.boxVectors), {
          cutoff: CUTOFF, listCutoff: plan.listCutoff, alpha: ALPHA, coulomb: ONE_4PI_EPS0,
        }), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)),
      positions: keep(allocator.upload("positions",
        Float32Array.from(coordinates.positions), storage)),
      charges: keep(allocator.upload("charges", Float32Array.from(topology.charges), storage)),
      ljTypes: keep(allocator.upload("ljTypes", Uint32Array.from(topology.atomTypes), storage)),
      ljTable: keep(allocator.upload("ljTable", ljTable, storage)),
      forces: keep(allocator.allocate("forces", atomCount * 12, readable)),
      // bonded
      termIndices: keep(allocator.upload("termIndices", gather.termIndices, storage)),
      termParams: keep(allocator.upload("termParams",
        Float32Array.from(gather.termParams), storage)),
      gatherStart: keep(allocator.upload("gatherStart", gather.start, storage)),
      gatherList: keep(allocator.upload("gatherList", gather.list, storage)),
      energies: keep(allocator.allocate("bondedEnergies", atomCount * 4, readable)),
      // cell and neighbour lists
      cellOfAtom: keep(allocator.allocate("cellOfAtom", atomCount * 4, storage)),
      cellCount: keep(allocator.allocate("cellCount", plan.cellCount * 4, storage)),
      cellStart: keep(allocator.allocate("cellStart", (plan.cellCount + 1) * 4, storage)),
      cellAtoms: keep(allocator.allocate("cellAtoms", atomCount * 4, storage)),
      exclusionStart: keep(allocator.upload("exclusionStart", exclusions.start, storage)),
      exclusionPartners: keep(allocator.upload("exclusionPartners",
        exclusions.partners, storage)),
      neighbours: keep(allocator.allocate("neighbours",
        atomCount * plan.maxNeighbours * 4, storage)),
      neighbourCount: keep(allocator.allocate("neighbourCount", atomCount * 4, storage)),
      required: keep(allocator.allocate("required", 4, readable)),
      // exceptions
      exceptionStart: keep(allocator.upload("exceptionStart", exceptions.start, storage)),
      exceptionPartners: keep(allocator.upload("exceptionPartners",
        exceptions.partners, storage)),
      exceptionParams: keep(allocator.upload("exceptionParams",
        exceptions.parameters, storage)),
      // PME
      moduli: keep(allocator.upload("moduli", moduli, storage)),
      fixedGrid: keep(allocator.allocate("fixedGrid", gridPoints * 2 * 4, storage)),
      gridA: keep(allocator.allocate("gridA", gridPoints * 2 * 4, storage)),
      gridB: keep(allocator.allocate("gridB", gridPoints * 2 * 4, storage)),
      gridEnergy: keep(allocator.allocate("gridEnergy", gridPoints * 4, storage)),
      energyPartials: keep(allocator.allocate("energyPartials", partialCount * 4, readable)),
    };
    // The nonbonded kernel writes its own two-channel energy array.
    const nonbondedEnergies = keep(allocator.allocate("nonbondedEnergies",
      atomCount * 8, readable));

    const encodeAll = () => {
      const validation = new DeferredValidation(device, "forces");
      validation.begin();
      const encoder = device.createCommandEncoder({ label: "forces" });
      // 🔴 EVERY FORCE KERNEL ACCUMULATES, SO THIS CLEAR IS LOAD-BEARING. The
      // allocator pools buffers; without it the first pass adds to whatever
      // the previous run left behind.
      encoder.clearBuffer(resources.forces.buffer);
      encoder.clearBuffer(resources.required.buffer);
      encodeBonded(device, encoder, resources, atomCount, gather.layout);
      encodeCellList(device, encoder, resources, atomCount, plan);
      encodeNeighbours(device, encoder, resources, atomCount, plan);
      encodeNonbonded(device, encoder,
        { ...resources, energies: nonbondedEnergies },
        atomCount, topology.typeCount, plan);
      encodePme(device, encoder, resources, atomCount, GRID);
      device.queue.submit([encoder.finish()]);
      validation.end("full force path");
      return validation.settle();
    };

    await encodeAll();
    let required = (await readBuffer(device, allocator, resources.required, 4, Uint32Array))[0];
    let grew = null;
    if (required > plan.maxNeighbours) {
      grew = { from: plan.maxNeighbours, required };
      plan = grownPlan(plan, required);
      resources.neighbours.release();
      held.splice(held.indexOf(resources.neighbours), 1);
      resources.neighbours = keep(allocator.allocate("neighbours",
        atomCount * plan.maxNeighbours * 4, storage));
      await encodeAll();
      required = (await readBuffer(device, allocator, resources.required, 4, Uint32Array))[0];
      if (required > plan.maxNeighbours) {
        throw new NeighbourOverflowError(required, plan.maxNeighbours);
      }
    }

    const bondedEnergies = await readBuffer(device, allocator, resources.energies,
      atomCount * 4, Float32Array);
    const nonbonded = await readBuffer(device, allocator, nonbondedEnergies,
      atomCount * 8, Float32Array);
    const partials = await readBuffer(device, allocator, resources.energyPartials,
      partialCount * 4, Float32Array);
    const deviceForces = await readBuffer(device, allocator, resources.forces,
      atomCount * 12, Float32Array);

    let bonded = 0;
    let lennardJones = 0;
    let coulombDirect = 0;
    for (let atom = 0; atom < atomCount; atom += 1) {
      bonded += bondedEnergies[atom];
      lennardJones += nonbonded[atom];
      coulombDirect += nonbonded[atomCount + atom];
    }
    let reciprocal = 0;
    for (const value of partials) reciprocal += value;
    const coulomb = coulombDirect + reciprocal + selfTerms.total;
    const total = bonded + lennardJones + coulomb;

    // ---- compare ----------------------------------------------------------
    const groups = reference.energy.byForceGroup;
    const groupEnergy = (className) => {
      for (const [key, value] of Object.entries(groups)) {
        if (key.endsWith(`:${className}`)) return value.energy;
      }
      return 0;
    };
    const referenceBonded = groupEnergy("HarmonicBondForce")
      + groupEnergy("HarmonicAngleForce") + groupEnergy("PeriodicTorsionForce");

    const comparisons = [];
    const compare = (label, ours, theirs, bar) => {
      const relative = Math.abs(ours - theirs) / Math.max(Math.abs(theirs), 1);
      const ok = relative <= bar;
      comparisons.push({ label, device: ours, openmm: theirs, relative, bar, ok });
      return ok;
    };
    let allOk = true;
    allOk = compare("bonded", bonded, referenceBonded, 1e-5) && allOk;
    allOk = compare("lennard-jones", lennardJones, reference.energy.ljOnly, 1e-5) && allOk;
    allOk = compare("coulomb (all terms)", coulomb, reference.energy.coulombOnly, 2e-5) && allOk;
    allOk = compare("TOTAL", total, reference.energy.total, 5e-5) && allOk;

    let worstForce = 0;
    let worstAtom = -1;
    let sumSquares = 0;
    for (let atom = 0; atom < atomCount; atom += 1) {
      for (let component = 0; component < 3; component += 1) {
        const theirs = reference.forces.total[atom][component];
        sumSquares += theirs * theirs;
        const difference = Math.abs(deviceForces[atom * 3 + component] - theirs);
        if (difference > worstForce) { worstForce = difference; worstAtom = atom; }
      }
    }
    const rms = Math.sqrt(sumSquares / (3 * atomCount));
    const forceRelative = worstForce / rms;
    const forceBar = 3e-3;
    const forceOk = forceRelative <= forceBar;
    allOk = allOk && forceOk;

    return {
      check: "the whole GPU force path against OpenMM",
      failures: allOk ? 0 : 1,
      configuration: {
        atoms: atomCount, cutoff: CUTOFF, grid: GRID.join("x"), alpha: ALPHA,
        constraints: system.constraints.count, exceptions: system.exceptions.count,
        cells: plan.counts.join("x"), neighbourCapacity: plan.maxNeighbours,
        observedNeighbourMax: required, grew,
      },
      energy: {
        comparisons,
        breakdown: {
          bonded, lennardJones, coulombDirect, coulombReciprocal: reciprocal,
          coulombSelf: selfTerms.self, coulombNetCharge: selfTerms.net, total,
        },
      },
      force: { worst: worstForce, worstAtom, rms, relative: forceRelative, bar: forceBar, ok: forceOk },
      allocator: allocator.snapshot(),
    };
  } finally {
    for (let index = held.length - 1; index >= 0; index -= 1) held[index].release();
  }
}
