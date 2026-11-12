import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { erf, erfc } from "../shared/erf.js";
import {
  alphaFromTolerance, bsplineWeights, bsplineModuli, isTransformableSize,
  nextTransformableSize, PME_ORDER,
} from "../shared/pme/parameters.js";
import { gaussianTriple } from "../shared/random.js";
import { parsePrmtop } from "../shared/amber/prmtop.js";
import { parseInpcrd } from "../shared/amber/inpcrd.js";
import { buildSystem } from "../shared/topology/system.js";
import { potentialEnergy } from "../cpu/energy.js";
import { applyPositionConstraints } from "../cpu/integrate/constraints.js";
import { instantaneousTemperature } from "../cpu/integrate/langevin.js";

const read = (name) => readFileSync(new URL(`fixtures/${name}`, import.meta.url), "utf8");

test("erfc matches known values to near machine precision", () => {
  // Reference values from SciPy.
  // Printed by SciPy; see tools/ for the command that produced them.
  const cases = [
    [0.0, 1],
    [0.5, 0.47950012218695348],
    [1.0, 0.15729920705028516],
    [2.0, 0.0046777349810472662],
    [2.628261, 0.00020166438881469927],
  ];
  for (const [x, expected] of cases) {
    const relative = Math.abs(erfc(x) - expected) / Math.max(expected, 1e-300);
    assert.ok(relative < 1e-13, `erfc(${x}) off by ${relative}`);
  }
  assert.ok(Math.abs(erf(0.5) + erfc(0.5) - 1) < 1e-15);
  assert.ok(Math.abs(erfc(-1) - (2 - erfc(1))) < 1e-15);
});

test("alpha reproduces OpenMM's rule", () => {
  assert.ok(Math.abs(alphaFromTolerance(1.0, 0.0005) - 2.628261) < 1e-6);
  assert.throws(() => alphaFromTolerance(0, 0.0005));
  assert.throws(() => alphaFromTolerance(1, 0.6));
});

test("B-splines are a partition of unity with vanishing derivative sum", () => {
  for (let index = 0; index <= 32; index += 1) {
    const { weights, derivatives } = bsplineWeights(index / 32);
    const weightSum = Array.from(weights).reduce((a, b) => a + b, 0);
    const derivativeSum = Array.from(derivatives).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(weightSum - 1) < 1e-14, `weights sum to ${weightSum}`);
    assert.ok(Math.abs(derivativeSum) < 1e-14, `derivatives sum to ${derivativeSum}`);
  }
});

test("the Nyquist B-spline modulus is patched, not zero", () => {
  // Order 5 cancels exactly at m = K/2; an unpatched modulus there divided the
  // convolution and produced 2.9e57 kJ/mol.
  const moduli = bsplineModuli(24, PME_ORDER);
  for (let m = 0; m < 24; m += 1) {
    assert.ok(moduli[m] > 1e-6, `modulus ${m} is ${moduli[m]}`);
  }
  assert.ok(Math.abs(moduli[12] - 0.5 * (moduli[11] + moduli[13])) < 1e-9);
});

test("only 2/3/5-smooth grid sizes are accepted", () => {
  assert.equal(isTransformableSize(24), true);
  assert.equal(isTransformableSize(22), false, "22 = 2 x 11");
  assert.equal(isTransformableSize(21), false, "21 = 3 x 7");
  assert.equal(nextTransformableSize(22), 24);
  assert.equal(nextTransformableSize(21), 24);
});

test("counter-based noise is standard normal and reproducible", () => {
  const out = new Float64Array(3);
  let sum = 0;
  let sumSquares = 0;
  let count = 0;
  for (let atom = 0; atom < 4000; atom += 1) {
    gaussianTriple(out, 99, atom, 3);
    for (const value of out) { sum += value; sumSquares += value * value; count += 1; }
  }
  assert.ok(Math.abs(sum / count) < 0.05, `mean ${sum / count}`);
  assert.ok(Math.abs(sumSquares / count - 1) < 0.05, `variance ${sumSquares / count}`);
  gaussianTriple(out, 99, 7, 3);
  const first = Array.from(out);
  gaussianTriple(out, 99, 7, 3);
  assert.deepEqual(Array.from(out), first);
});

test("vacuum energy matches the OpenMM reference per term", () => {
  const topology = parsePrmtop(read("ala-vac.prmtop"));
  const coordinates = parseInpcrd(read("ala-vac.inpcrd"),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, { constraints: "none", rigidWater: false, cutoff: 0 });
  const { energy } = potentialEnergy(system, coordinates.positions, { method: "NoCutoff" });
  const reference = JSON.parse(read("ala-vac-nocutoff.json"));
  const close = (ours, theirs, tolerance, label) => {
    const relative = Math.abs(ours - theirs) / Math.max(Math.abs(theirs), 1);
    assert.ok(relative <= tolerance, `${label}: ours ${ours} theirs ${theirs} rel ${relative}`);
  };
  close(energy.lennardJones, reference.energy.ljOnly, 1e-9, "lennard-jones");
  close(energy.coulomb, reference.energy.coulombOnly, 1e-7, "coulomb");
  close(energy.total, reference.energy.total, 1e-7, "total");
});

test("PME energy and forces match the OpenMM reference", () => {
  const topology = parsePrmtop(read("ala-wat.prmtop"));
  const coordinates = parseInpcrd(read("ala-wat.inpcrd"),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const { energy, forces } = potentialEnergy(system, coordinates.positions, {
    method: "PME", alpha: 2.6282608939112583, grid: [24, 24, 24],
  });
  const reference = JSON.parse(read("ala-wat-pme.json"));
  const relative = Math.abs(energy.total - reference.energy.total)
    / Math.abs(reference.energy.total);
  assert.ok(relative < 1e-7, `total energy relative error ${relative}`);

  let worst = 0;
  let sumSquares = 0;
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    for (let component = 0; component < 3; component += 1) {
      const theirs = reference.forces.total[atom][component];
      sumSquares += theirs * theirs;
      worst = Math.max(worst, Math.abs(forces[atom * 3 + component] - theirs));
    }
  }
  const rms = Math.sqrt(sumSquares / (3 * topology.atomCount));
  assert.ok(worst / rms < 1e-6, `worst force error ${worst} against rms ${rms}`);
});

test("SHAKE restores every constrained length after a perturbation", () => {
  const topology = parsePrmtop(read("ala-wat.prmtop"));
  const coordinates = parseInpcrd(read("ala-wat.inpcrd"),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const positions = Float64Array.from(coordinates.positions);
  const reference = Float64Array.from(coordinates.positions);
  for (let index = 0; index < positions.length; index += 1) {
    positions[index] += Math.sin(index * 12.9898) * 0.004;
  }
  const result = applyPositionConstraints(system, positions, reference, { tolerance: 1e-8 });
  assert.ok(result.converged, "SHAKE must converge");
  for (let index = 0; index < system.constraints.count; index += 1) {
    const a = system.constraints.atomA[index];
    const b = system.constraints.atomB[index];
    const length = Math.hypot(
      positions[a * 3] - positions[b * 3],
      positions[a * 3 + 1] - positions[b * 3 + 1],
      positions[a * 3 + 2] - positions[b * 3 + 2],
    );
    const error = Math.abs(length - system.constraints.distance[index])
      / system.constraints.distance[index];
    assert.ok(error < 1e-7, `constraint ${index} off by ${error}`);
  }
});

test("temperature uses the constrained degree-of-freedom count", () => {
  const topology = parsePrmtop(read("ala-wat.prmtop"));
  const coordinates = parseInpcrd(read("ala-wat.inpcrd"),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  // Unit velocities give a kinetic energy of (1/2) sum(m), so the temperature
  // is a pure statement about the denominator.
  const velocities = new Float64Array(topology.atomCount * 3).fill(1);
  const { kineticEnergy, temperature } = instantaneousTemperature(system, velocities);
  let massSum = 0;
  for (const mass of topology.masses) massSum += mass;
  assert.ok(Math.abs(kineticEnergy - 1.5 * massSum) < 1e-6);
  const expected = (2 * kineticEnergy) / (system.degreesOfFreedom * 0.008314462618);
  assert.ok(Math.abs(temperature - expected) < 1e-9);
});

test("the cell stencil is deduplicated so no cell is visited twice", async () => {
  const { planCellList } = await import("../shared/topology/cells.js");
  const topology = parsePrmtop(read("ala-wat.prmtop"));
  const coordinates = parseInpcrd(read("ala-wat.inpcrd"),
    { expectedAtomCount: topology.atomCount });
  for (const cutoff of [1.0, 0.7, 0.5]) {
    const system = buildSystem(topology, {
      constraints: "HBonds", rigidWater: true, cutoff, box: coordinates.box,
    });
    const plan = planCellList(system, { skin: 0.1 });
    // Every stencil entry must land on a distinct cell after wrapping. A naive
    // 3x3x3 on a 2-cell axis visits the same cell twice and double-counts
    // every pair in it.
    const visited = new Set();
    for (const [dx, dy, dz] of plan.stencil) {
      const wrap = (value, count) => ((value % count) + count) % count;
      visited.add(`${wrap(dx, plan.counts[0])},${wrap(dy, plan.counts[1])},${wrap(dz, plan.counts[2])}`);
    }
    assert.equal(visited.size, plan.stencil.length,
      `cutoff ${cutoff}: stencil visits ${plan.stencil.length} entries but only ${visited.size} distinct cells`);
    // The cell must be at least the list cutoff, or a one-deep stencil misses pairs.
    for (let axis = 0; axis < 3; axis += 1) {
      const size = Math.abs(plan.widths[axis]) / plan.counts[axis];
      assert.ok(size >= plan.listCutoff - 1e-12,
        `cutoff ${cutoff}: cell size ${size} on axis ${axis} is below the list cutoff ${plan.listCutoff}`);
    }
  }
});

test("exclusions are symmetric and sorted for the kernel's binary search", async () => {
  const { buildExclusionGather } = await import("../shared/topology/cells.js");
  const topology = parsePrmtop(read("ala-vac.prmtop"));
  const system = buildSystem(topology, { constraints: "none", rigidWater: false, cutoff: 0 });
  const gather = buildExclusionGather(system);
  assert.equal(gather.partners.length, 2 * topology.exclusions.pairCount,
    "every one-sided AMBER exclusion must appear in both directions");
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    const slice = gather.partners.subarray(gather.start[atom], gather.start[atom + 1]);
    for (let index = 1; index < slice.length; index += 1) {
      assert.ok(slice[index - 1] <= slice[index], `atom ${atom} slice is not sorted`);
    }
  }
  // Spot-check both directions of every stored pair.
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    const begin = topology.exclusions.start[atom];
    const end = topology.exclusions.start[atom + 1];
    for (let cursor = begin; cursor < end; cursor += 1) {
      const partner = topology.exclusions.partners[cursor];
      const forward = gather.partners.subarray(gather.start[atom], gather.start[atom + 1]);
      const backward = gather.partners.subarray(gather.start[partner], gather.start[partner + 1]);
      assert.ok(forward.includes(partner), `${atom} -> ${partner} missing`);
      assert.ok(backward.includes(atom), `${partner} -> ${atom} missing`);
    }
  }
});

test("exception gather duplicates each pair onto both atoms with the right products", async () => {
  const { buildExceptionGather } = await import("../shared/topology/cells.js");
  const topology = parsePrmtop(read("ala-vac.prmtop"));
  const system = buildSystem(topology, { constraints: "none", rigidWater: false, cutoff: 0 });
  const gather = buildExceptionGather(system);
  assert.equal(gather.count, 2 * system.exceptions.count);
  // The Ewald correction uses the atoms' FULL charges, which is the fourth
  // parameter and is NOT the scaled product in the first.
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    for (let slot = gather.start[atom]; slot < gather.start[atom + 1]; slot += 1) {
      const partner = gather.partners[slot];
      const expected = topology.charges[atom] * topology.charges[partner];
      const stored = gather.parameters[slot * 4 + 3];
      assert.ok(Math.abs(stored - expected) < 1e-6 * Math.max(Math.abs(expected), 1e-6),
        `full charge product for ${atom}-${partner}: stored ${stored}, expected ${expected}`);
    }
  }
});

test("the DCD header and unit cell match the layout OpenMM writes", async () => {
  const { DcdWriter, cellFromVectors } = await import("../shared/traj/dcd.js");
  const writer = new DcdWriter({
    atomCount: 3, timestep: 0.002, interval: 5, periodic: true,
  });
  const positions = Float64Array.from([0, 0, 0, 0.1, 0.2, 0.3, -0.4, 0.5, -0.6]);
  const box = Float64Array.from([2.5, 0, 0, 0.3, 2.6, 0, 0.4, 0.5, 2.7]);
  writer.appendFrame(positions, box);
  writer.appendFrame(positions, box);
  const bytes = writer.finish();
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  assert.equal(view.getInt32(0, true), 84, "leading record marker");
  assert.equal(String.fromCharCode(...bytes.slice(4, 8)), "CORD");
  assert.equal(view.getInt32(8, true), 2, "frame count");
  assert.equal(view.getInt32(16, true), 5, "interval");
  // (NSET - 1) * NSAVC, which is how OpenMM fills this field.
  assert.equal(view.getInt32(20, true), 5, "last frame's step");
  // delta is in AKMA units, not picoseconds - 2 fs stores as 0.0409.
  assert.ok(Math.abs(view.getFloat32(44, true) - 0.002 / 0.04888821) < 1e-6);
  assert.equal(view.getInt32(48, true), 1, "unit-cell flag");
  assert.equal(view.getInt32(84, true), 24, "format version");
  assert.equal(view.getInt32(268, true), 3, "atom count");

  // 🔴 THE CELL ORDER IS [a, cos(gamma), b, cos(beta), cos(alpha), c], which is
  // invisible on a rectangular box because all three cosines are zero there.
  const cell = cellFromVectors(box);
  assert.ok(Math.abs(cell[0] - 25) < 1e-9, "a in angstroms");
  assert.ok(Math.abs(cell[2] - Math.hypot(0.3, 2.6) * 10) < 1e-9, "b in angstroms");
  assert.ok(Math.abs(cell[5] - Math.hypot(0.4, 0.5, 2.7) * 10) < 1e-9, "c in angstroms");
  const cosGamma = 0.3 * 2.5 / (2.5 * Math.hypot(0.3, 2.6));
  assert.ok(Math.abs(cell[1] - cosGamma) < 1e-12, "slot 1 is cos(gamma)");
});

test("a periodic DCD refuses a frame with no box", async () => {
  const { DcdWriter } = await import("../shared/traj/dcd.js");
  const writer = new DcdWriter({ atomCount: 1, timestep: 0.002, interval: 1, periodic: true });
  assert.throws(() => writer.appendFrame(Float64Array.from([0, 0, 0])), /needs its box/);
});

test("an AMBER restart round-trips through this port's own reader", async () => {
  const { writeRestart } = await import("../shared/traj/rst7.js");
  const coordinates = parseInpcrd(read("ala-wat.inpcrd"));
  const velocities = new Float64Array(coordinates.positions.length);
  for (let index = 0; index < velocities.length; index += 1) {
    velocities[index] = Math.sin(index * 0.7) * 0.5;
  }
  const text = writeRestart({
    positions: coordinates.positions, velocities, box: coordinates.box, time: 1.5,
  });
  const again = parseInpcrd(text, { expectedAtomCount: coordinates.atomCount });
  assert.equal(again.atomCount, coordinates.atomCount);
  assert.equal(again.time, 1.5);
  for (let index = 0; index < coordinates.positions.length; index += 1) {
    // The format stores seven decimals of an angstrom, i.e. 1e-8 nm.
    assert.ok(Math.abs(again.positions[index] - coordinates.positions[index]) < 1e-7);
  }
  for (let index = 0; index < velocities.length; index += 1) {
    assert.ok(Math.abs(again.velocities[index] - velocities[index]) < 1e-6,
      `velocity ${index}: wrote ${velocities[index]}, read ${again.velocities[index]}`);
  }
  for (let axis = 0; axis < 3; axis += 1) {
    assert.ok(Math.abs(again.box.lengths[axis] - coordinates.box.lengths[axis]) < 1e-7);
  }
});

test("a coordinate that overflows the restart's 12 columns is refused", async () => {
  const { writeRestart } = await import("../shared/traj/rst7.js");
  // 🔴 A TRUNCATED FIELD SHIFTS EVERY VALUE AFTER IT BY ONE COLUMN, producing a
  // file that parses and whose coordinates are nonsense from that point on.
  assert.throws(
    () => writeRestart({ positions: Float64Array.from([1e6, 0, 0]) }),
    /does not fit the 12-column restart format/,
  );
});

test("a NaN coordinate is refused by the restart writer, which the width guard cannot catch", async () => {
  const { writeRestart } = await import("../shared/traj/rst7.js");
  // 🔴 THE WIDTH GUARD PASSES NaN, WHICH IS WHY THIS IS A SEPARATE ARM.
  // `Number.NaN.toFixed(7)` is "NaN" - three characters, comfortably inside
  // twelve - so the file was written, and this port's own reader then refused
  // it: `non-numeric coordinate field "NaN" on line 3`. Falsified by removing
  // the finiteness check, which makes this test fail and the round-trip below
  // throw from the READER instead of the writer.
  assert.throws(
    () => writeRestart({ positions: Float64Array.from([1, Number.NaN, 0]) }),
    /no longer a number.*restart writer/s,
  );
  assert.throws(
    () => writeRestart({
      positions: Float64Array.from([1, 2, 3]),
      velocities: Float64Array.from([0, Number.POSITIVE_INFINITY, 0]),
    }),
    /no longer a number/,
  );
  // The width guard still does its own job, on a finite value.
  assert.throws(
    () => writeRestart({ positions: Float64Array.from([1e6, 0, 0]) }),
    /does not fit the 12-column restart format/,
  );
  // And a finite frame is still written.
  assert.match(writeRestart({ positions: Float64Array.from([1, 2, 3]) }),
    /RainMD restart/);
});

test("a NaN frame is refused by the DCD writer, and the frames before it survive", async () => {
  const { DcdWriter } = await import("../shared/traj/dcd.js");
  // 🔴 BOTH READERS ACCEPT AN ALL-NaN DCD IN SILENCE. Measured: cpptraj reports
  // "Read 10 frames and processed 10 frames" with no warning, and MDAnalysis
  // loads it and reports its own n_frames. Neither will ever tell the user, so
  // the refusal has to be here.
  const writer = new DcdWriter({ atomCount: 2, timestep: 0.002, interval: 1 });
  writer.appendFrame(Float32Array.from([0, 0, 0, 1, 1, 1]));
  writer.appendFrame(Float32Array.from([0, 0, 0, 1, 1, 1.1]));
  assert.equal(writer.frameCount, 2);
  assert.throws(
    () => writer.appendFrame(Float32Array.from([0, 0, 0, 1, Number.NaN, 1])),
    /no longer a number.*DCD writer/s,
  );
  // 🔴 THE REFUSAL MUST NOT COST THE GOOD FRAMES. A run goes non-finite partway
  // through, so the frames before that point are exactly the ones worth
  // keeping - and the header is built at finish() from the real count, so the
  // file stays self-consistent.
  assert.equal(writer.frameCount, 2);
  const bytes = writer.finish();
  assert.ok(bytes.byteLength > 0);
});

test("firstNonFinite names the component, because a message naming the atom is the point", async () => {
  const { firstNonFinite } = await import("../shared/finite.js");
  assert.equal(firstNonFinite(Float64Array.from([1, 2, 3])), -1);
  assert.equal(firstNonFinite(Float64Array.from([1, Number.NaN, 3])), 1);
  assert.equal(firstNonFinite(Float64Array.from([Number.POSITIVE_INFINITY, 2])), 0);
  assert.equal(firstNonFinite(Float64Array.from([1, 2, Number.NEGATIVE_INFINITY])), 2);
  // An f32 array cannot hold a NaN it was not given, so the scan is exact.
  assert.equal(firstNonFinite(Float32Array.from([0.1, 0.2])), -1);
});

test("the PME reciprocal sum matches an exact lattice sum on a SHEARED cell", async () => {
  const { pmeParameters } = await import("../shared/pme/parameters.js");
  const { reciprocalEnergy } = await import("../cpu/forces/pme.js");
  const { reciprocalBoxVectors } = await import("../cpu/forces/geometry.js");
  const { ONE_4PI_EPS0 } = await import("../shared/units.js");

  // 🔴 AN EXACT k-SPACE SUM, WRITTEN FROM THE TEXTBOOK FORMULA, SHARING NO CODE
  // WITH THE THING IT VALIDATES. The reciprocal sum was transposed - built from
  // the COLUMNS of the inverse box where `fractional` indexes its ROWS - and
  // nothing in this repository could see it:
  //   - every fixture was rectangular, where the inverse box is diagonal and a
  //     matrix equals its transpose, so the error was exactly zero;
  //   - the WGSL kernel carried the same construction as its host reference, so
  //     the differential test between them agreed about a shared error;
  //   - and no gradient check can ever catch it, because the wave vector enters
  //     only the kernel WEIGHTS, which do not depend on position - so the
  //     analytic force stays the exact gradient of the WRONG energy, and a
  //     central difference on this very box passes at 0.00e+0.
  // The only instrument that works is an independent energy oracle on a sheared
  // cell. The one shared dependency is `reciprocalBoxVectors`, which is checked
  // against OpenMM's own box vectors digit for digit and is not where the bug
  // was.
  const exactEwald = (system, positions, alpha, nmax, rowForm) => {
    const b = system.boxVectors;
    const recip = reciprocalBoxVectors(b);
    const volume = Math.abs(b[0] * b[4] * b[8]);
    const charges = system.topology.charges;
    const atoms = system.topology.atomCount;
    const factor = (Math.PI * Math.PI) / (alpha * alpha);
    let sum = 0;
    for (let na = -nmax; na <= nmax; na += 1) {
      for (let nb = -nmax; nb <= nmax; nb += 1) {
        for (let nc = -nmax; nc <= nmax; nc += 1) {
          if (na === 0 && nb === 0 && nc === 0) continue;
          // rowForm = the correct convention; the other arm is the bug.
          const mx = rowForm ? na * recip[0] + nb * recip[1] + nc * recip[2]
            : na * recip[0] + nb * recip[3] + nc * recip[6];
          const my = rowForm ? na * recip[3] + nb * recip[4] + nc * recip[5]
            : na * recip[1] + nb * recip[4] + nc * recip[7];
          const mz = rowForm ? na * recip[6] + nb * recip[7] + nc * recip[8]
            : na * recip[2] + nb * recip[5] + nc * recip[8];
          const m2 = mx * mx + my * my + mz * mz;
          if (m2 === 0) continue;
          const weight = Math.exp(-factor * m2) / m2;
          if (weight < 1e-18) continue;
          let re = 0;
          let im = 0;
          for (let atom = 0; atom < atoms; atom += 1) {
            const phase = 2 * Math.PI * (mx * positions[atom * 3]
              + my * positions[atom * 3 + 1] + mz * positions[atom * 3 + 2]);
            re += charges[atom] * Math.cos(phase);
            im += charges[atom] * Math.sin(phase);
          }
          sum += weight * (re * re + im * im);
        }
      }
    }
    // E = (1/(2 pi V)) * ONE_4PI_EPS0 * sum, because k = 2 pi m so k^2 carries
    // 4 pi^2. Writing (2 pi / V) instead is out by exactly 4 pi^2 = 39.478, and
    // the first version of this oracle did - caught because it disagreed with
    // the RECTANGULAR fixture, which is why that arm is here and runs first.
    return (1 / (2 * Math.PI * volume)) * ONE_4PI_EPS0 * sum;
  };

  const alpha = 2.6282608939112584;
  const grid = [48, 48, 48];
  // 🔴 THE TOLERANCE IS THE TWO ORACLES' OWN FLOORS, NOT A CHOSEN NUMBER. The
  // exact sum converges to 1.6e-6 between nmax 8 and 11, and a 48^3 grid sits
  // about 2e-6 from a 64^3 one, so 1e-5 is a few times the better of the two.
  // Measured at 24^3 the grid alone costs 1.5e-4, which is why the gate does
  // not run there.
  const bar = 1e-5;
  const measured = {};
  for (const name of ["ala-wat", "ala-oct"]) {
    const topology = parsePrmtop(read(`${name}.prmtop`));
    const coordinates = parseInpcrd(read(`${name}.inpcrd`),
      { expectedAtomCount: topology.atomCount });
    const system = buildSystem(topology, {
      nonbonded: "PME", cutoff: 1.0, constraints: "HBonds", rigidWater: true,
      box: coordinates.box,
    });
    const parameters = pmeParameters({
      boxVectors: system.boxVectors, cutoff: 1.0, alpha, grid,
    });
    const pme = reciprocalEnergy(system, coordinates.positions,
      new Float64Array(topology.atomCount * 3), parameters);
    const exact = exactEwald(system, coordinates.positions, alpha, 8, true);
    const relative = Math.abs(pme - exact) / Math.abs(exact);
    measured[name] = { pme, exact, relative };
    assert.ok(relative < bar,
      `${name}: PME reciprocal ${pme} against the exact lattice sum ${exact} `
      + `is ${relative.toExponential(2)}, over the ${bar} floor`);
  }

  // The off-diagonal box components are what makes the second fixture a test.
  const oct = buildSystem(parsePrmtop(read("ala-oct.prmtop")), {
    nonbonded: "PME", cutoff: 1.0, constraints: "HBonds", rigidWater: true,
    box: parseInpcrd(read("ala-oct.inpcrd"),
      { expectedAtomCount: 1315 }).box,
  });
  const offDiagonal = Math.max(...[1, 2, 3, 5, 6, 7].map((i) => Math.abs(oct.boxVectors[i])));
  assert.ok(offDiagonal > 0.5,
    `ala-oct must actually be sheared, off-diagonal max ${offDiagonal}`);

  // 🔴 AND THE TRANSPOSED FORM MUST FAIL HERE AND PASS ON THE RECTANGULAR BOX.
  // Without this arm the test cannot distinguish the fix from the bug, because
  // the bug is invisible on every other fixture in the repository.
  const topology = parsePrmtop(read("ala-oct.prmtop"));
  const coordinates = parseInpcrd(read("ala-oct.inpcrd"),
    { expectedAtomCount: topology.atomCount });
  const transposed = exactEwald(oct, coordinates.positions, alpha, 8, false);
  const transposedError = Math.abs(transposed - measured["ala-oct"].exact)
    / Math.abs(measured["ala-oct"].exact);
  assert.ok(transposedError > 1e-2,
    "the transposed wave vector must be grossly wrong on a sheared cell, or "
    + `this test cannot see the defect it exists for - got ${transposedError}`);

  const flat = parsePrmtop(read("ala-wat.prmtop"));
  const flatCoordinates = parseInpcrd(read("ala-wat.inpcrd"),
    { expectedAtomCount: flat.atomCount });
  const flatSystem = buildSystem(flat, {
    nonbonded: "PME", cutoff: 1.0, constraints: "HBonds", rigidWater: true,
    box: flatCoordinates.box,
  });
  const flatTransposed = exactEwald(flatSystem, flatCoordinates.positions, alpha, 8, false);
  assert.equal(flatTransposed, measured["ala-wat"].exact,
    "on a rectangular box the two conventions must be BIT-IDENTICAL - that is "
    + "why the defect survived every validated number in docs/STATUS.md");
});

test("the dispersion correction reproduces OpenMM's own value", async () => {
  const { dispersionCoefficients, dispersionEnergy } = await import("../shared/dispersion.js");
  const topology = parsePrmtop(read("ala-wat.prmtop"));
  const coordinates = parseInpcrd(read("ala-wat.inpcrd"),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const volume = Math.abs(system.boxVectors[0] * system.boxVectors[4] * system.boxVectors[8]);
  const energy = dispersionEnergy(dispersionCoefficients(system), volume, 1.0);
  // Isolated by running the oracle with and without
  // NonbondedForce.useDispersionCorrection.
  assert.ok(Math.abs(energy - -41.537158) / 41.537158 < 1e-6,
    `dispersion correction ${energy}, OpenMM reports -41.537158`);
  // 🔴 IT SCALES AS 1/V, WHICH IS WHY THE BAROSTAT NEEDS IT: a term that
  // depends on the volume contributes to the pressure.
  const scaled = dispersionEnergy(dispersionCoefficients(system), volume * 1.05, 1.0);
  assert.ok(Math.abs(scaled / energy - 1 / 1.05) < 1e-12);
});

test("the Metropolis weight includes the molecular volume entropy", async () => {
  const { metropolisWeight } = await import("../webgpu/kernels/barostat.js");
  // 🔴 THE `N ln(V'/V)` TERM COUNTS MOLECULES, NOT ATOMS, and omitting it
  // biases every NPT run towards smaller volumes. With no energy change and no
  // pressure, an expansion must be favourable purely from that term.
  const expand = metropolisWeight(
    { deltaEnergy: 0, deltaVolume: 0.1, volumeRatio: 1.01 }, 0, 2.478, 393);
  assert.ok(expand < 0, `an entropically favourable expansion gave ${expand}`);
  const withoutEntropy = metropolisWeight(
    { deltaEnergy: 0, deltaVolume: 0.1, volumeRatio: 1.01 }, 0, 2.478, 0);
  assert.equal(withoutEntropy, 0, "with no molecules the term must vanish");
  // Pressure opposes expansion.
  const compressed = metropolisWeight(
    { deltaEnergy: 0, deltaVolume: 0.1, volumeRatio: 1.01 }, 100, 2.478, 393);
  assert.ok(compressed > expand);
});
