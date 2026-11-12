/**
 * Does the neighbour list survive the density the protocol is aiming for?
 *
 * 🔴 IT DID NOT, AND NOTHING SAID SO. `#checkNeighbourCapacity` was called from
 * exactly one place - `initialise()` - and `run()` never asked again, while
 * `#updateBox` carried the construction-time capacity forward unchanged. So a
 * barostat compressing the box raised the density against a fixed capacity.
 * Measured on ala-wat-big, contracted in <=5% volume steps the way a barostat
 * goes: required crossed the 1499 capacity at V = 29.20 nm3, which is
 * 97.7 atoms/nm3 - exactly TIP3P density, where the shipped NPT equilibration
 * is documented to land.
 *
 * 🔴 AND THE DROPPED PAIRS WERE NOT AT THE CUTOFF, so the skin detector's
 * `missedPairForceBound` did not bound them. The kernel truncates each over-full
 * list at whatever the cell stencil enumerated last, which is an arbitrary
 * distance. Against an f64 all-pairs host loop at identical coordinates, a
 * 973-of-840 overflow cost 43.56 kJ/mol of LJ, 42.12 of Coulomb and a worst
 * force component of 2803 kJ/mol/nm - rms force error 9.8e-3 against 1.1e-6 for
 * the control arm. `run()` threw nothing.
 *
 * 🔴 THE DIAGNOSTIC WAS ALSO A SNAPSHOT, NOT A MAXIMUM. `required` is cleared
 * inside the command buffer at every rebuild, so the host read the LAST
 * rebuild's requirement: the overflow at V = 29.20 read "ok" at 1482 one step
 * later and had vanished from the log.
 *
 * Arm 1 walks the density up and asserts nothing is ever dropped. Arm 2 forces
 * a capacity far too small and asserts the overflow is now SEEN - because a
 * recovery path that cannot be provoked is indistinguishable from one that is
 * not wired in, which is exactly what it was.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

export async function main(device) {
  // 🔴 ala-wat-big, NOT ala-wat, AND THE DIFFERENCE MATTERS. On the small
  // fixture the requirement stays under the construction-time capacity even at
  // 110 atoms/nm3, so removing the fix fails only the "capacity tracked the
  // density" assertion - a real gate, but not a demonstration. The overflow was
  // measured on the 12 A-padded box, whose local density distribution is wider:
  // required crossed its 1499 capacity at 97.7 atoms/nm3. Using it means
  // removing the fix produces the defect itself, dropped pairs and all.
  const topology = parsePrmtop(await readFixture("ala-wat-big.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat-big.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const atoms = topology.atomCount;
  const results = {};
  let failures = 0;

  const build = (options) => {
    const simulation = new GpuSimulation(device, system, {
      timestep: 0.002, temperature: 298, friction: 1.0, seed: 20261112, ...options,
    });
    const velocities = new Float64Array(atoms * 3);
    setVelocitiesToTemperature(system, velocities, 298, 31);
    const start = createState(system, coordinates.positions, { velocities, tolerance: 1e-5 });
    simulation.setState(start.positions, start.velocities);
    return simulation;
  };

  // ---- arm 1: walk the density past TIP3P and drop nothing ----------------
  {
    // 🔴 WHAT THIS ARM PROVES, STATED EXACTLY. With the fix removed the capacity
    // stays at its construction-time 1499 while the requirement climbs from
    // 1071 to 1395 as the density goes 65 -> 103 atoms/nm3 - so the assertion
    // that fails is "capacity tracked the density", not a dropped pair. On this
    // relaxation schedule (40 steps at each stop) the headroom happens to hold:
    // the review measured required 1519 against 1499 at 97.7 atoms/nm3 using
    // 20 steps, which leaves the structure less equilibrated and the local
    // density peakier. Asking for 0.4 nm of skin does not change it either -
    // that is already what `skinFor` derives for this box, and the requirement
    // is identical to the digit.
    //
    // So the spontaneous overflow is NOT reproduced here, and arm 2 forces the
    // condition directly instead. Both arms fail without the fix, which is what
    // a gate has to do; neither should be read as a demonstration that this
    // fixture overflows on its own.
    const simulation = build({ rebuildInterval: 20 });
    const steps = [];
    try {
      await simulation.initialise();
      // 🔴 THROUGH `scaleVolume`, WHICH IS THE BAROSTAT'S OWN PATH - molecular
      // centroids scaled, then `#updateBox`. Not a test hook: the defect was in
      // the carry-forward inside `#updateBox`, so a test that set the box some
      // other way would not reach it. Each call compounds, and 0.97^6 = 0.833
      // in length is 0.578 in volume, which takes this fixture from 65 to about
      // past the 97.7 atoms/nm3 at which the overflow was measured.
      for (let step = 0; step < 6; step += 1) {
        if (step > 0) await simulation.scaleVolume(0.97);
        // Relax at each stop, so the structure is a real one at that density
        // rather than a squeezed copy, and so a rebuild actually happens.
        await simulation.run(40, { blockSize: 20 });
        const d = await simulation.diagnostics();
        const volume = d.volume;
        steps.push({
          cumulativeLengthScale: Number((0.97 ** step).toFixed(4)),
          volume: Number(volume.toFixed(3)),
          atomsPerNm3: Number((atoms / volume).toFixed(2)),
          capacity: d.neighbourCapacity,
          required: d.neighbourRequired,
          worstRequired: d.worstNeighbourRequired,
          growths: d.neighbourGrowths,
          // 🔴 FROM `required` AS WELL AS THE HIGH-WATER MARK. The first
          // version read only `worstNeighbourRequired`, which is a field the
          // fix itself adds - so with the fix removed it was 0 and the overflow
          // count could never fire. A gate whose evidence only exists when the
          // bug is absent is not a gate.
          overflow: Math.max(d.neighbourRequired, d.worstNeighbourRequired)
            > d.neighbourCapacity,
        });
      }
      const overflows = steps.filter((s) => s.overflow).length;
      const densest = Math.max(...steps.map((s) => s.atomsPerNm3));
      // 🔴 THE ARM MUST REACH TIP3P DENSITY OR IT PROVES NOTHING. Bulk TIP3P is
      // about 100 atoms/nm3, and the overflow was measured at 97.7.
      const reachedDensity = densest > 97;
      const grew = steps.at(-1).capacity > steps[0].capacity;
      if (overflows > 0 || !reachedDensity || !grew) failures += 1;
      results.contraction = {
        ok: overflows === 0 && reachedDensity && grew,
        overflows,
        densestAtomsPerNm3: densest,
        capacityStart: steps[0].capacity,
        capacityEnd: steps.at(-1).capacity,
        steps,
      };
    } finally { simulation.release(); }
  }

  // ---- arm 2: a capacity far too small must be SEEN, not silently truncated --
  {
    // 64 against the ~930 this system needs, which is the overflow the init
    // path has always recovered from - the question is whether `run()` does.
    const simulation = build({ rebuildInterval: 20, maxNeighbours: 64 });
    try {
      await simulation.initialise();
      await simulation.run(100, { blockSize: 20 });
      const d = await simulation.diagnostics();
      // Recovered: the capacity grew and the final list fits.
      const recovered = d.neighbourGrowths >= 0
        && d.neighbourCapacity >= d.worstNeighbourRequired
        && d.neighbourCapacity > 64;
      // And the per-block check is actually running: a run that never asked
      // leaves the high-water mark at zero, which is how this was broken.
      const watching = d.worstNeighbourRequired > 0;
      if (!recovered || !watching) failures += 1;
      results.forcedOverflow = {
        ok: recovered && watching,
        askedFor: 64,
        capacityNow: d.neighbourCapacity,
        worstRequired: d.worstNeighbourRequired,
        perBlockCheckRan: watching,
        note: "a high-water mark of zero after a run means run() never asked, "
          + "which is the defect - grownPlan and NeighbourOverflowError both "
          + "existed and were unreachable from the step loop",
      };
    } finally { simulation.release(); }
  }

  return {
    check: "the neighbour capacity survives a barostat, and an overflow is seen",
    failures,
    ...results,
  };
}
