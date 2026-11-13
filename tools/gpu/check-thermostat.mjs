/**
 * Does the reported temperature hold the setpoint across the timestep range?
 *
 * 🔴 IT DID NOT, AND THE CAUSE WAS WHERE THE ESTIMATOR WAS SAMPLED. This
 * integrator runs B A O A B and used to read the kinetic energy after the
 * closing half kick; OpenMM's LangevinMiddleIntegrator ends its step at the
 * position update and the velocity projection, so it reads one half kick
 * earlier. Same splitting, same trajectory - but a half kick is an O(dt) change
 * in v and O(dt^2) in energy, with the sign set by which side of it you stand
 * on. Measured offsets from a 298 K setpoint, friction 10/ps:
 *
 *   dt      before      now      OpenMM
 *   1 fs    -0.02      +0.46     +0.42
 *   2 fs    -2.08      -0.33     +1.63
 *   4 fs    -6.36      +0.88     +4.45
 *
 * 🔴 AND THE TOTAL ENERGY STILL USES THE POST-STEP SAMPLE, which is not a
 * detail. Pairing the early velocity with the end-of-step position is a
 * half-step mismatch that inflated the measured NVE energy fluctuation from
 * 9.31 to 15.51 kJ/mol - a spurious oscillation that reads as drift. The two
 * samples answer different questions and `energy()` keeps both.
 *
 * The review measured, device NVT, friction 10/ps, against OpenMM
 * LangevinMiddle on the same system: RainMD 296.76 +- 0.24 at 2 fs and
 * 291.37 +- 0.23 at 4 fs, where OpenMM gave 300.10 +- 0.23 and 301.57 +- 0.23.
 * 14a was an f32 heating that grew as the step shrank; this is a cooling that
 * grows as the step grows, so it should be untouched - but that has to be
 * measured, not assumed, because the two could have shared a cause.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  // One minimised, equilibrated start shared by every timestep, so the arms
  // differ only in dt.
  const warm = new GpuSimulation(device, system, {
    timestep: 0.002, temperature: 298, friction: 10.0, seed: 3, rebuildInterval: 20,
  });
  const v0 = new Float64Array(topology.atomCount * 3);
  setVelocitiesToTemperature(system, v0, 298, 3);
  const st = createState(system, coordinates.positions, { velocities: v0, tolerance: 1e-5 });
  warm.setState(st.positions, st.velocities);
  await warm.initialise();
  await warm.minimise({ maxIterations: 400, forceTolerance: 200 });
  await warm.run(5000, { blockSize: 20 });
  const warmState = await warm.readState();
  const start = {
    positions: Float64Array.from(warmState.positions),
    velocities: Float64Array.from(warmState.velocities),
  };
  warm.release();

  const arms = [];
  for (const dtFs of [1, 2, 4]) {
    const dt = dtFs / 1000;
    // 60 ps at each dt, sampled every 0.2 ps, first 20% discarded as burn-in.
    const steps = Math.round(60 / dt);
    const every = Math.max(1, Math.round(0.2 / dt));
    const simulation = new GpuSimulation(device, system, {
      timestep: dt, temperature: 298, friction: 10.0, seed: 11, rebuildInterval: 20,
    });
    simulation.setState(start.positions, start.velocities);
    await simulation.initialise();
    try {
      const temps = [];
      const pots = [];
      for (let done = 0; done < steps; done += every) {
        await simulation.run(Math.min(every, steps - done), { blockSize: 20 });
        const e = await simulation.energy();
        temps.push(e.temperature);
        // 🔴 THE POTENTIAL IS A CONFIGURATIONAL OBSERVABLE, WHICH IS THE POINT.
        // The reported temperature is a kinetic estimator and can carry an
        // O(dt^2) bias while the trajectory still samples the right
        // configurations. <U> cannot: if the two engines agree on it while
        // their reported temperatures diverge, the ensemble is sound and the
        // defect is in the reporting. If <U> diverges too, it is not.
        pots.push(e.potential);
      }
      const keep = temps.slice(Math.floor(temps.length * 0.2));
      const mean = keep.reduce((a, b) => a + b, 0) / keep.length;
      const sd = Math.sqrt(keep.reduce((a, b) => a + (b - mean) ** 2, 0) / keep.length);
      const keptP = pots.slice(Math.floor(pots.length * 0.2));
      const meanP = keptP.reduce((a, b) => a + b, 0) / keptP.length;
      const sdP = Math.sqrt(keptP.reduce((a, b) => a + (b - meanP) ** 2, 0) / keptP.length);
      arms.push({
        dtFs,
        meanPotential: Number(meanP.toFixed(2)),
        potentialSem: Number((sdP / Math.sqrt(keptP.length)).toFixed(3)),
        picoseconds: 60,
        samples: keep.length,
        meanTemperature: Number(mean.toFixed(2)),
        sem: Number((sd / Math.sqrt(keep.length)).toFixed(3)),
        offsetFrom298: Number((mean - 298).toFixed(2)),
      });
    } finally { simulation.release(); }
  }
  // 🔴 THE BAR IS 2 K, DERIVED FROM THE SAMPLING ERROR AND NOT FROM TASTE. Each
  // arm averages 240 samples over 60 ps and reports a standard error near
  // 0.55 K, so 2 K is between three and four standard errors - tight enough to
  // catch the defect, which was -2.08 K at 2 fs and -6.36 at 4 fs, and loose
  // enough not to flake on the sampling. OpenMM itself would fail a 2 K bar at
  // 4 fs (+4.45), which is worth knowing before anyone tightens it further.
  const bar = 2.0;
  let failures = 0;
  const bad = arms.filter((a) => Math.abs(a.offsetFrom298) > bar);
  if (bad.length > 0) failures += 1;
  // And the offset must not GROW with the timestep, which is the signature.
  const growth = Math.abs(arms.at(-1).offsetFrom298) - Math.abs(arms[0].offsetFrom298);
  if (growth > bar) failures += 1;
  return {
    check: "the reported temperature holds the setpoint across the timestep range",
    failures,
    bar,
    offsets: arms.map((a) => ({ dtFs: a.dtFs, offset: a.offsetFrom298, sem: a.sem })),
    outsideBar: bad.map((a) => `${a.dtFs} fs off by ${a.offsetFrom298} K`),
    growthFrom1fsTo4fs: Number(growth.toFixed(2)),
    beforeTheFix: { "1fs": -0.02, "2fs": -2.08, "4fs": -6.36 },
    openmmLangevinMiddle: { "1fs": 0.42, "2fs": 1.63, "4fs": 4.45 },
    arms,
  };
}
