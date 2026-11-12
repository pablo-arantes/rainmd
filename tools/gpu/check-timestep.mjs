/**
 * Blocker 14: localise the heating to the constrained-velocity re-read.
 *
 * The host in f64 shows drift FALLING as dt falls (4 fs -9.4, 2 fs -0.58,
 * 1 fs +0.063, 0.5 fs +0.089 kJ/mol/ns/atom) - a correct integrator. The device
 * in f32 shows it RISING steeply. If the cause is
 * `v = (x - reference) / halfStep` after SHAKE, then with constraints OFF that
 * code never runs and the scaling must flatten. Neighbour list rebuilt every
 * step so staleness cannot contribute.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

const slope = (t, y) => {
  const n = t.length, mt = t.reduce((a, b) => a + b) / n, my = y.reduce((a, b) => a + b) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i += 1) { num += (t[i] - mt) * (y[i] - my); den += (t[i] - mt) ** 2; }
  return num / den;
};

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const results = {};

  for (const constraints of ["HBonds", "none"]) {
    const system = buildSystem(topology, {
      constraints, rigidWater: constraints === "HBonds",
      cutoff: 1.0, box: coordinates.box,
    });
    // One warm, minimised, equilibrated starting point per constraint mode.
    // 🔴 THE WARM-UP TIMESTEP MUST SUIT THE MODE. 2 fs is invalid without
    // constraints - an X-H bond's period is ~10 fs - and the first version of
    // this probe warmed both modes at 2 fs and blew up in its own setup.
    const warmDt = constraints === "HBonds" ? 0.002 : 0.00025;
    const warm = new GpuSimulation(device, system, {
      timestep: warmDt, temperature: 298, friction: 1.0, seed: 5,
      rebuildInterval: 10,
    });
    const v0 = new Float64Array(topology.atomCount * 3);
    setVelocitiesToTemperature(system, v0, 298, 5);
    const st = createState(system, coordinates.positions, { velocities: v0, tolerance: 1e-5 });
    warm.setState(st.positions, st.velocities);
    await warm.initialise();
    await warm.minimise({ maxIterations: 400, forceTolerance: 200 });
    await warm.run(constraints === "HBonds" ? 3000 : 8000, { blockSize: 10 });
    const warmState = await warm.readState();
    const start = {
      positions: Float64Array.from(warmState.positions),
      velocities: Float64Array.from(warmState.velocities),
    };
    warm.release();

    const arms = [];
    for (const dtFs of [1, 0.5, 0.25]) {
      // 2 ps of NVE at every dt, so each arm covers the same physical time.
      const dt = dtFs / 1000;
      const steps = Math.round(1 / dt);
      const sim = new GpuSimulation(device, system, {
        timestep: dt, temperature: 298, friction: 0, seed: 9,
        rebuildInterval: 1, adaptiveRebuild: false, skin: 0.5,
      });
      sim.setState(start.positions, start.velocities);
      await sim.initialise();
      try {
        process.stderr.write(`  ${constraints} ${dtFs} fs ...\n`);
        const t = [], y = [];
        const every = Math.max(10, Math.floor(steps / 20));
        for (let done = 0; done < steps; done += every) {
          const e = await sim.energy();
          t.push(e.time); y.push(e.total);
          await sim.run(Math.min(every, steps - done), { blockSize: 10 });
        }
        const last = await sim.energy();
        t.push(last.time); y.push(last.total);
        const d = await sim.diagnostics();
        arms.push({
          dtFs,
          drift: Number((slope(t, y) * 1000 / topology.atomCount).toPrecision(4)),
          temperatureEnd: Number(last.temperature.toFixed(1)),
          maxShake: d.maxShakeIterations,
          stalledResidual: d.worstStalledResidual ?? null,
          skinViolations: d.skinViolations,
        });
      } catch (error) {
        // Reported, not thrown: an arm that is unstable at this timestep is
        // itself a result, and losing the other arms to it would be a waste.
        arms.push({ dtFs, drift: null, blewUp: `${error.name}` });
      } finally { sim.release(); }
    }
    for (let i = 1; i < arms.length; i += 1) {
      if (arms[i].drift === null || arms[i - 1].drift === null) continue;
      arms[i].ratioPerHalving =
        Number(Math.abs(arms[i].drift / arms[i - 1].drift).toFixed(2));
    }
    results[constraints] = arms;
  }
  // 🔴 THE ASSERTION IS ON THE SCALING, NOT ON A DRIFT VALUE. A correct
  // integrator's error falls as the timestep falls; this one's Rose, because
  // the constrained velocity was recovered as (x - x_ref)/halfStep and that
  // error is span*eps whatever the step size, so dividing by a smaller half
  // step amplified it. Measured before the fix, NVE with the list rebuilt every
  // step: 2.2 -> 258.7 -> 1811 kJ/mol/ns/atom over 1 -> 0.5 -> 0.25 fs, about
  // 29x per halving, while the f64 host FELL over the same range and the
  // constraints-off arm was flat. After accumulating the correction instead:
  // 2.747 -> 3.627 -> 5.583, and the end temperature holds near the 298 K
  // setpoint at every step where 0.25 fs previously reached 398 K.
  //
  // The bar is 10x over a 4x reduction in dt. Measured 2.03x; the defect gave
  // 824x. A tighter bar would be measuring this machine's noise - the
  // constraints-off arm, which never touches the re-read, varies by 45x over
  // the same range on a 1 ps run.
  let failures = 0;
  const constrained = results.HBonds;
  const first = constrained.find((a) => a.drift !== null);
  const last = [...constrained].reverse().find((a) => a.drift !== null);
  const growth = first === undefined || last === undefined ? null
    : Math.abs(last.drift / first.drift);
  const hot = constrained.filter((a) => a.temperatureEnd !== undefined
    && Math.abs(a.temperatureEnd - 298) > 40);
  if (growth === null || growth > 10 || hot.length > 0) failures += 1;
  results.scalingWithTimestep = {
    ok: growth !== null && growth <= 10 && hot.length === 0,
    driftRatioOverFourfoldSmallerStep: growth === null ? null : Number(growth.toFixed(2)),
    bar: 10,
    endTemperatures: constrained.map((a) => a.temperatureEnd),
    tooHot: hot.map((a) => `${a.dtFs} fs reached ${a.temperatureEnd} K`),
    note: "before the fix this ratio was 824 and 0.25 fs ended at 398 K",
  };

  return {
    check: "the NVE drift does not grow as the timestep shrinks",
    failures,
    ...results,
  };
}
