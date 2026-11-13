/**
 * Hold the cell list, the neighbour list and the direct-space kernel to the
 * host reference.
 *
 * 🔴 THE NEIGHBOUR LIST IS CHECKED AS A SET, SEPARATELY FROM THE FORCES. A
 * dropped pair at 0.9 nm contributes almost nothing to any force, so a force
 * comparison can pass with pairs missing - and "almost nothing" times a few
 * thousand missing pairs is a systematically wrong energy. Comparing the list
 * against an O(N^2) host enumeration is exact: every pair is present or it is
 * not, and the report says which atoms differ.
 *
 * 🔴 AND IT RUNS AT THREE CUTOFFS TO GET THREE CELL GEOMETRIES. At the
 * notebooks' 1.0 nm the fixture has only 2x2x2 cells, so the deduplicated
 * stencil covers the entire box and the kernel degenerates to all-pairs - it
 * could not detect a wrong stencil, a wrong wrap, or a miscomputed cell index.
 * 0.7 nm gives 3x3x3 (the stencil wraps but still spans everything) and 0.5 nm
 * gives 4x4x4, where the stencil is 27 cells of 64 and a geometry bug finally
 * has somewhere to hide. The physics of a 0.5 nm cutoff is poor; that is
 * irrelevant, because both sides use it.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import {
  planCellList, grownPlan, buildExclusionGather, buildExceptionGather,
} from "../../shared/topology/cells.js";
import { alphaFromTolerance } from "../../shared/pme/parameters.js";
import { reciprocalBoxVectors, minimumImage } from "../../cpu/forces/geometry.js";
import { directSpaceEnergy, exceptionEnergy } from "../../cpu/forces/nonbonded.js";
import { GpuAllocator } from "../../webgpu/runtime/allocator.js";
import { readBuffer, DeferredValidation } from "../../webgpu/runtime/execution.js";
import { packParams, PARAMS_FLOATS } from "../../webgpu/kernels/wgsl-common.js";
import { encodeCellList } from "../../webgpu/kernels/cell-list.js";
import { encodeNeighbours, NeighbourOverflowError } from "../../webgpu/kernels/neighbours.js";
import { encodeNonbonded } from "../../webgpu/kernels/nonbonded.js";
import { ONE_4PI_EPS0 } from "../../shared/units.js";
import { measureErfcFloor } from "./erfc-floor.js";

/**
 * Each arm is a cutoff and, optionally, a deliberately wrong capacity.
 *
 * 🔴 THE LAST ARM EXISTS TO FALSIFY THE OVERFLOW RECOVERY. The capacity
 * estimate now covers this fixture, so the grow-and-rebuild path would never
 * run - and a recovery path that nothing exercises is not a recovery path. The
 * 64-slot arm forces an overflow of about 8x, and it must come back with
 * EXACTLY the same energies and forces as the arm that never overflowed. If
 * growing silently lost pairs, this is where it shows.
 */
const ARMS = [
  { cutoff: 1.0 },
  { cutoff: 0.7 },
  { cutoff: 0.5 },
  { cutoff: 1.0, forceCapacity: 64, label: "forced overflow" },
];

/** The neighbour list the kernel should have produced, by brute force. */
function hostNeighbourList(system, positions, exclusions, listCutoff) {
  const atomCount = system.topology.atomCount;
  const box = system.boxVectors;
  const limit = listCutoff * listCutoff;
  const delta = new Float64Array(3);
  const lists = [];
  for (let i = 0; i < atomCount; i += 1) {
    const excluded = new Set(
      exclusions.partners.subarray(exclusions.start[i], exclusions.start[i + 1]),
    );
    const list = new Set();
    for (let j = 0; j < atomCount; j += 1) {
      if (j === i || excluded.has(j)) continue;
      delta[0] = positions[i * 3] - positions[j * 3];
      delta[1] = positions[i * 3 + 1] - positions[j * 3 + 1];
      delta[2] = positions[i * 3 + 2] - positions[j * 3 + 2];
      minimumImage(delta, box);
      if (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2 <= limit) list.add(j);
    }
    lists.push(list);
  }
  return lists;
}

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const atomCount = topology.atomCount;
  const allocator = new GpuAllocator(device);
  const results = [];
  let failures = 0;

  // 🔴 THE COULOMB BAR IS MEASURED ON THIS DEVICE, NOT TYPED IN. It used to be
  // `const erfcFloor = 1.19e-6` with a comment claiming the number had been
  // measured, which nothing in the tree could reproduce - and the figure was
  // 21% below what the device actually delivers, because every emulation of it
  // used a correctly-rounded `Math.exp` where WGSL allows `exp` 3 ulp. Dawn on
  // Metal measures 1.44e-6; Chrome on the same machine need not agree, and a
  // bar taken from the running device follows it.
  //
  // 🔴 AND IT IS CAPPED, SO IT CANNOT WIDEN TO ADMIT A BROKEN erfc. The cap is
  // `argumentUlpBound`, derived in erfc-floor.js from the f32 spacing of the
  // exponential's argument; a measurement above it is a failure here, not a
  // licence to raise the tolerance. erfc(z) is evaluated only for r <= cutoff,
  // and alpha*cutoff = sqrt(-ln(2*tolerance)) is the same 2.628 at every one of
  // this check's cutoffs, so one measurement covers all the arms.
  const erfcProbe = await measureErfcFloor(device, allocator,
    { zMax: alphaFromTolerance(1.0, 0.0005) });
  const erfcFloor = Math.min(erfcProbe.worst, erfcProbe.bound);
  if (!(erfcProbe.worst < erfcProbe.bound)) failures += 1;

  for (const arm of ARMS) {
    const cutoff = arm.cutoff;
    const system = buildSystem(topology, {
      constraints: "HBonds", rigidWater: true, cutoff, box: coordinates.box,
    });
    let plan = planCellList(system, {
      skin: 0.1,
      ...(arm.forceCapacity === undefined ? {} : { maxNeighbours: arm.forceCapacity }),
    });
    const exclusions = buildExclusionGather(system);
    const exceptions = buildExceptionGather(system);
    const alpha = alphaFromTolerance(cutoff, 0.0005);

    // ---- host reference -------------------------------------------------
    const referenceForces = new Float64Array(atomCount * 3);
    const direct = directSpaceEnergy(system, coordinates.positions, referenceForces,
      { alpha, cutoff });
    const exception = exceptionEnergy(system, coordinates.positions, referenceForces, { alpha });
    const referenceLj = direct.lennardJones + exception.lennardJones;
    const referenceCoulomb = direct.coulomb + exception.coulomb;
    const hostLists = hostNeighbourList(system, coordinates.positions, exclusions,
      plan.listCutoff);

    // ---- device ---------------------------------------------------------
    const held = [];
    const keep = (allocation) => { held.push(allocation); return allocation; };
    const storage = GPUBufferUsage.STORAGE;
    const readable = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    try {
      const reciprocal = reciprocalBoxVectors(system.boxVectors);
      const params = packParams(system.boxVectors, reciprocal, {
        cutoff, listCutoff: plan.listCutoff, alpha, coulomb: ONE_4PI_EPS0,
      });
      const ljTable = new Float32Array(topology.typeCount * topology.typeCount * 2);
      for (let index = 0; index < topology.typeCount * topology.typeCount; index += 1) {
        ljTable[index * 2] = topology.ljA[index];
        ljTable[index * 2 + 1] = topology.ljB[index];
      }
      const resources = {
        params: keep(allocator.upload("params", params,
          GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)),
        positions: keep(allocator.upload("positions",
          Float32Array.from(coordinates.positions), storage)),
        charges: keep(allocator.upload("charges", Float32Array.from(topology.charges), storage)),
        ljTypes: keep(allocator.upload("ljTypes", Uint32Array.from(topology.atomTypes), storage)),
        ljTable: keep(allocator.upload("ljTable", ljTable, storage)),
        cellOfAtom: keep(allocator.allocate("cellOfAtom", atomCount * 4, storage)),
        cellCount: keep(allocator.allocate("cellCount", plan.cellCount * 4, storage)),
        cellStart: keep(allocator.allocate("cellStart", (plan.cellCount + 1) * 4, readable)),
        cellAtoms: keep(allocator.allocate("cellAtoms", atomCount * 4, readable)),
        exclusionStart: keep(allocator.upload("exclusionStart", exclusions.start, storage)),
        exclusionPartners: keep(allocator.upload("exclusionPartners",
          exclusions.partners.length > 0 ? exclusions.partners : new Uint32Array(1), storage)),
        neighbours: keep(allocator.allocate("neighbours",
          atomCount * plan.maxNeighbours * 4, readable)),
        neighbourCount: keep(allocator.allocate("neighbourCount", atomCount * 4, readable)),
        required: keep(allocator.allocate("required", 4, readable)),
        exceptionStart: keep(allocator.upload("exceptionStart", exceptions.start, storage)),
        exceptionPartners: keep(allocator.upload("exceptionPartners",
          exceptions.partners.length > 0 ? exceptions.partners : new Uint32Array(1), storage)),
        exceptionParams: keep(allocator.upload("exceptionParams",
          exceptions.parameters.length > 0 ? exceptions.parameters : new Float32Array(4), storage)),
        forces: keep(allocator.allocate("forces", atomCount * 12, readable)),
        energies: keep(allocator.allocate("energies", atomCount * 8, readable)),
      };

      const validation = new DeferredValidation(device, `nonbonded/${cutoff}`);
      validation.begin();
      const encoder = device.createCommandEncoder({ label: `nonbonded/${cutoff}` });
      encoder.clearBuffer(resources.forces.buffer);
      encoder.clearBuffer(resources.required.buffer);
      encodeCellList(device, encoder, resources, atomCount, plan);
      encodeNeighbours(device, encoder, resources, atomCount, plan);
      encodeNonbonded(device, encoder, resources, atomCount, topology.typeCount, plan);
      device.queue.submit([encoder.finish()]);
      validation.end("nonbonded");
      await validation.settle();

      // 🔴 GROW AND REBUILD, DO NOT GIVE UP. The capacity is a density
      // estimate and the kernel reports what was actually needed, so an
      // overflow is information rather than a failure - exactly once, because
      // the second attempt is sized from the measurement. A second overflow
      // would mean the requirement is changing between builds, which is a real
      // problem and is where the error belongs.
      let required = (await readBuffer(device, allocator, resources.required, 4, Uint32Array))[0];
      let grew = null;
      if (required > plan.maxNeighbours) {
        grew = { from: plan.maxNeighbours, required };
        plan = grownPlan(plan, required);
        resources.neighbours.release();
        held.splice(held.indexOf(resources.neighbours), 1);
        resources.neighbours = keep(allocator.allocate("neighbours",
          atomCount * plan.maxNeighbours * 4, readable));
        const retryValidation = new DeferredValidation(device, `nonbonded/${cutoff}/retry`);
        retryValidation.begin();
        const retry = device.createCommandEncoder({ label: `nonbonded/${cutoff}/retry` });
        retry.clearBuffer(resources.required.buffer);
        retry.clearBuffer(resources.forces.buffer);
        encodeCellList(device, retry, resources, atomCount, plan);
        encodeNeighbours(device, retry, resources, atomCount, plan);
        encodeNonbonded(device, retry, resources, atomCount, topology.typeCount, plan);
        device.queue.submit([retry.finish()]);
        retryValidation.end("nonbonded retry");
        await retryValidation.settle();
        required = (await readBuffer(device, allocator, resources.required, 4, Uint32Array))[0];
        if (required > plan.maxNeighbours) {
          throw new NeighbourOverflowError(required, plan.maxNeighbours);
        }
      }

      // ---- the cell list, as a partition --------------------------------
      const cellStart = await readBuffer(device, allocator, resources.cellStart,
        (plan.cellCount + 1) * 4, Uint32Array);
      const cellAtoms = await readBuffer(device, allocator, resources.cellAtoms,
        atomCount * 4, Uint32Array);
      const seen = new Uint8Array(atomCount);
      let cellProblems = 0;
      let sortedCells = true;
      for (const atom of cellAtoms) {
        if (atom >= atomCount || seen[atom] === 1) cellProblems += 1;
        else seen[atom] = 1;
      }
      for (let atom = 0; atom < atomCount; atom += 1) if (seen[atom] === 0) cellProblems += 1;
      for (let cell = 0; cell < plan.cellCount; cell += 1) {
        for (let slot = cellStart[cell] + 1; slot < cellStart[cell + 1]; slot += 1) {
          if (cellAtoms[slot - 1] > cellAtoms[slot]) sortedCells = false;
        }
      }
      const cellTotalOk = cellStart[plan.cellCount] === atomCount;

      // ---- the neighbour list, as a set ---------------------------------
      const deviceCounts = await readBuffer(device, allocator, resources.neighbourCount,
        atomCount * 4, Uint32Array);
      const deviceNeighbours = await readBuffer(device, allocator, resources.neighbours,
        atomCount * plan.maxNeighbours * 4, Uint32Array);
      let missing = 0;
      let spurious = 0;
      let duplicated = 0;
      let worstAtom = -1;
      for (let atom = 0; atom < atomCount; atom += 1) {
        const expected = hostLists[atom];
        const got = new Set();
        for (let slot = 0; slot < deviceCounts[atom]; slot += 1) {
          const partner = deviceNeighbours[atom * plan.maxNeighbours + slot];
          if (got.has(partner)) duplicated += 1;
          got.add(partner);
        }
        let atomMissing = 0;
        for (const partner of expected) if (!got.has(partner)) atomMissing += 1;
        for (const partner of got) if (!expected.has(partner)) spurious += 1;
        if (atomMissing > 0 && worstAtom < 0) worstAtom = atom;
        missing += atomMissing;
      }

      // ---- forces and energies ------------------------------------------
      const deviceForces = await readBuffer(device, allocator, resources.forces,
        atomCount * 12, Float32Array);
      const deviceEnergies = await readBuffer(device, allocator, resources.energies,
        atomCount * 8, Float32Array);
      let deviceLj = 0;
      let deviceCoulomb = 0;
      for (let atom = 0; atom < atomCount; atom += 1) {
        deviceLj += deviceEnergies[atom];
        deviceCoulomb += deviceEnergies[atomCount + atom];
      }
      let worstForce = 0;
      let sumSquares = 0;
      for (let index = 0; index < atomCount * 3; index += 1) {
        sumSquares += referenceForces[index] ** 2;
        worstForce = Math.max(worstForce, Math.abs(deviceForces[index] - referenceForces[index]));
      }
      const rms = Math.sqrt(sumSquares / (atomCount * 3));

      // 🔴 THE COULOMB BAR IS SET BY THE f32 erfc, NOT BY f32 EPSILON, and
      // `erfcFloor` is what this device measured a few lines above - a floor no
      // summation order can improve. The LJ bar is f32 accumulation over a few
      // hundred terms with cancellation between the r^-12 and r^-6 parts.
      const ljBar = 2e-5;
      const coulombBar = 4 * erfcFloor;
      const forceBar = 1e-4;

      const ljRelative = Math.abs(deviceLj - referenceLj) / Math.abs(referenceLj);
      const coulombRelative = Math.abs(deviceCoulomb - referenceCoulomb)
        / Math.abs(referenceCoulomb);
      const forceRelative = worstForce / rms;
      const listOk = missing === 0 && spurious === 0 && duplicated === 0;
      const cellsOk = cellProblems === 0 && sortedCells && cellTotalOk;
      const ok = listOk && cellsOk && ljRelative < ljBar
        && coulombRelative < coulombBar && forceRelative < forceBar;
      if (!ok) failures += 1;

      results.push({
        cutoff,
        arm: arm.label ?? "normal",
        geometry: `${plan.counts.join("x")} cells, stencil ${plan.stencil.length} of ${plan.cellCount}`,
        occupancy: Number(plan.meanOccupancy.toFixed(1)),
        neighbours: { capacity: plan.maxNeighbours, observedMax: required, grew },
        cellList: { ok: cellsOk, problems: cellProblems, sorted: sortedCells, totalOk: cellTotalOk },
        list: { ok: listOk, missing, spurious, duplicated, firstBadAtom: worstAtom },
        lennardJones: { device: deviceLj, reference: referenceLj, relative: ljRelative, bar: ljBar },
        coulomb: {
          device: deviceCoulomb, reference: referenceCoulomb,
          relative: coulombRelative, bar: coulombBar,
        },
        force: { worst: worstForce, rms, relative: forceRelative, bar: forceBar },
        ok,
      });
    } finally {
      for (let index = held.length - 1; index >= 0; index -= 1) held[index].release();
    }
  }

  return {
    check: "cell list, neighbour list and direct-space nonbonded against the host reference",
    failures,
    // Where the Coulomb bar came from, so the report carries its own origin.
    erfcFloor: {
      measured: erfcProbe.worst,
      cap: erfcProbe.bound,
      used: erfcFloor,
      coulombBar: 4 * erfcFloor,
      atAlphaR: Number(erfcProbe.worstAt.toFixed(4)),
      hostEmulation: erfcProbe.worstEmulated,
      note: "measured on this device by tools/gpu/erfc-floor.js; the host "
        + "emulation is a lower bound because WGSL allows exp 3 ulp",
    },
    results,
    allocator: allocator.snapshot(),
  };
}
