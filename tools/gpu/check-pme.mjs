/**
 * Hold the PME kernels to the host reference, stage by stage.
 *
 * 🔴 THE CHARGE GRID IS COMPARED BEFORE THE ENERGY IS. Spread, transform,
 * convolve and gather can each produce a wrong reciprocal energy, and the
 * energy is one number - so comparing only that leaves four candidates. The
 * grid after spreading is comparable directly against the host's, which
 * separates the spline arithmetic and the wrap convention from everything
 * downstream. The FFT is already isolated by tools/gpu/check-fft.mjs.
 *
 * 🔴 AND THE FIXED-POINT HEADROOM IS ASSERTED, NOT ASSUMED. An i32 atomicAdd
 * wraps silently on overflow, so a grid value past 128 charge units would
 * produce a large wrong number with nothing to say why. The maximum is read
 * back and reported as a fraction of the limit.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { pmeParameters } from "../../shared/pme/parameters.js";
import { reciprocalBoxVectors } from "../../cpu/forces/geometry.js";
import { reciprocalEnergy } from "../../cpu/forces/pme.js";
import { GpuAllocator } from "../../webgpu/runtime/allocator.js";
import { readBuffer, DeferredValidation } from "../../webgpu/runtime/execution.js";
import { packParams } from "../../webgpu/kernels/wgsl-common.js";
import { encodePme, FIXED_SCALE, FIXED_LIMIT } from "../../webgpu/kernels/pme.js";
import { ONE_4PI_EPS0 } from "../../shared/units.js";

/** Grids that exercise a cubic case, a non-cubic one and radix 3 and 5. */
const GRIDS = [[24, 24, 24], [24, 32, 20], [27, 25, 16]];

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const atomCount = topology.atomCount;
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const allocator = new GpuAllocator(device);
  const results = [];
  let failures = 0;

  for (const grid of GRIDS) {
    const parameters = pmeParameters({
      boxVectors: system.boxVectors, cutoff: 1.0, alpha: 2.6282608939112583, grid,
    });
    const gridPoints = grid[0] * grid[1] * grid[2];

    // ---- host reference ---------------------------------------------------
    const referenceForces = new Float64Array(atomCount * 3);
    const referenceEnergy = reciprocalEnergy(system, coordinates.positions,
      referenceForces, parameters);

    // The host's spread, re-run so the grid can be compared directly. Charge
    // only, before any transform.
    const hostGrid = new Float64Array(gridPoints);
    {
      const recip = reciprocalBoxVectors(system.boxVectors);
      const { bsplineWeights } = await import("../../shared/pme/parameters.js");
      const order = parameters.order;
      for (let atom = 0; atom < atomCount; atom += 1) {
        const q = topology.charges[atom];
        const fraction = [0, 1, 2].map((d) => {
          const value = coordinates.positions[atom * 3] * recip[d]
            + coordinates.positions[atom * 3 + 1] * recip[3 + d]
            + coordinates.positions[atom * 3 + 2] * recip[6 + d];
          return value - Math.floor(value);
        });
        const base = [];
        const weights = [];
        for (let d = 0; d < 3; d += 1) {
          const scaled = fraction[d] * grid[d];
          const integral = Math.floor(scaled);
          base.push(integral);
          weights.push(bsplineWeights(scaled - integral, order).weights);
        }
        for (let kx = 0; kx < order; kx += 1) {
          let ix = (base[0] + kx - order + 1) % grid[0];
          if (ix < 0) ix += grid[0];
          for (let ky = 0; ky < order; ky += 1) {
            let iy = (base[1] + ky - order + 1) % grid[1];
            if (iy < 0) iy += grid[1];
            for (let kz = 0; kz < order; kz += 1) {
              let iz = (base[2] + kz - order + 1) % grid[2];
              if (iz < 0) iz += grid[2];
              hostGrid[(ix * grid[1] + iy) * grid[2] + iz]
                += q * weights[0][kx] * weights[1][ky] * weights[2][kz];
            }
          }
        }
      }
    }

    // ---- device -----------------------------------------------------------
    const held = [];
    const keep = (allocation) => { held.push(allocation); return allocation; };
    const storage = GPUBufferUsage.STORAGE;
    const readable = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    try {
      const reciprocal = reciprocalBoxVectors(system.boxVectors);
      const moduli = new Float32Array(grid[0] + grid[1] + grid[2]);
      let cursor = 0;
      for (let d = 0; d < 3; d += 1) {
        moduli.set(Float32Array.from(parameters.moduli[d]), cursor);
        cursor += grid[d];
      }
      const partialCount = Math.ceil(gridPoints / 256);
      const resources = {
        params: keep(allocator.upload("params",
          packParams(system.boxVectors, reciprocal, {
            cutoff: 1.0, listCutoff: 1.1, alpha: parameters.alpha, coulomb: ONE_4PI_EPS0,
          }), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)),
        positions: keep(allocator.upload("positions",
          Float32Array.from(coordinates.positions), storage)),
        charges: keep(allocator.upload("charges", Float32Array.from(topology.charges), storage)),
        moduli: keep(allocator.upload("moduli", moduli, storage)),
        fixedGrid: keep(allocator.allocate("fixedGrid", gridPoints * 2 * 4, readable)),
        gridA: keep(allocator.allocate("gridA", gridPoints * 2 * 4, readable)),
        gridB: keep(allocator.allocate("gridB", gridPoints * 2 * 4, readable)),
        gridEnergy: keep(allocator.allocate("gridEnergy", gridPoints * 4, readable)),
        energyPartials: keep(allocator.allocate("energyPartials", partialCount * 4, readable)),
        forces: keep(allocator.allocate("forces", atomCount * 12, readable)),
      };

      const validation = new DeferredValidation(device, `pme/${grid.join("x")}`);
      validation.begin();
      const encoder = device.createCommandEncoder({ label: `pme/${grid.join("x")}` });
      // The gather accumulates, so the forces start at zero for an isolated test.
      encoder.clearBuffer(resources.forces.buffer);
      const { partialCount: partials } = encodePme(device, encoder, resources, atomCount, grid);
      device.queue.submit([encoder.finish()]);
      validation.end("pme");
      await validation.settle();

      // ---- the spread grid, in fixed point --------------------------------
      const fixedGrid = await readBuffer(device, allocator, resources.fixedGrid,
        gridPoints * 2 * 4, Int32Array);
      let worstGrid = 0;
      let gridScale = 0;
      let maxCharge = 0;
      for (let point = 0; point < gridPoints; point += 1) {
        const value = fixedGrid[point * 2] / FIXED_SCALE;
        maxCharge = Math.max(maxCharge, Math.abs(value));
        gridScale = Math.max(gridScale, Math.abs(hostGrid[point]));
        worstGrid = Math.max(worstGrid, Math.abs(value - hostGrid[point]));
      }

      // ---- energy and forces ----------------------------------------------
      const energyPartials = await readBuffer(device, allocator, resources.energyPartials,
        partials * 4, Float32Array);
      let deviceEnergy = 0;
      for (const value of energyPartials) deviceEnergy += value;
      const deviceForces = await readBuffer(device, allocator, resources.forces,
        atomCount * 12, Float32Array);

      let worstForce = 0;
      let sumSquares = 0;
      for (let index = 0; index < atomCount * 3; index += 1) {
        sumSquares += referenceForces[index] ** 2;
        worstForce = Math.max(worstForce, Math.abs(deviceForces[index] - referenceForces[index]));
      }
      const rms = Math.sqrt(sumSquares / (atomCount * 3));

      const gridRelative = worstGrid / gridScale;
      const energyRelative = Math.abs(deviceEnergy - referenceEnergy)
        / Math.abs(referenceEnergy);
      const forceRelative = worstForce / rms;
      // The reciprocal sum is a transform, a convolution and an inverse
      // transform in f32 over 9-12 stages, so the error is an accumulation
      // rather than a single rounding: ~sqrt(stages) * eps, times a modest
      // constant for the convolution's dynamic range.
      const gridBar = 1e-5;
      const energyBar = 2e-4;
      const forceBar = 3e-3;
      const headroomOk = maxCharge < 0.25 * FIXED_LIMIT;
      const ok = gridRelative < gridBar && energyRelative < energyBar
        && forceRelative < forceBar && headroomOk;
      if (!ok) failures += 1;

      results.push({
        grid: grid.join("x"),
        spreadGrid: { worst: worstGrid, scale: gridScale, relative: gridRelative, bar: gridBar },
        fixedPoint: {
          maxChargeOnAPoint: maxCharge, limit: FIXED_LIMIT,
          usedFractionOfHeadroom: maxCharge / FIXED_LIMIT, ok: headroomOk,
        },
        energy: {
          device: deviceEnergy, reference: referenceEnergy,
          relative: energyRelative, bar: energyBar,
        },
        force: { worst: worstForce, rms, relative: forceRelative, bar: forceBar },
        ok,
      });
    } finally {
      for (let index = held.length - 1; index >= 0; index -= 1) held[index].release();
    }
  }
  return { check: "PME kernels against the host reference", failures, results };
}
