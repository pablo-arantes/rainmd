/**
 * Does a blown-up run STOP, or does it keep writing?
 *
 * 🔴 IT KEPT WRITING, AND THAT IS WHY THIS FILE EXISTS. Before the finiteness
 * check, a 40 fs run on the page went entirely non-finite after 50 steps and
 * then completed: 300 steps encoded at full speed, 13 frames written, both
 * download buttons enabled, the next stage unlocked, and a log line reading
 * "300 steps in 7.7 s - 25.79 ms/step". The tiles read "-" and the CSV read
 * `NaN`. The DCD it produced was then read by cpptraj without a single warning
 * ("Read 10 frames and processed 10 frames") and by MDAnalysis without one
 * either, so no tool downstream would ever have told the user.
 *
 * 🔴 THE GATE MUST FIRE AND MUST ALSO STAY QUIET, which is why arm 2 exists. A
 * detector that refuses everything passes arm 1 and is useless; the cliff on
 * this system is between 5 and 6 fs, so a 2 fs run has to come through clean.
 *
 * 🔴 AND THE TWO DETECTORS ARE TESTED SEPARATELY, because they catch different
 * things. Arm 1 reaches the one in `#checkSkin`, which reads the reduced
 * displacement every block. Arm 3 reaches the one in `energy()` by placing two
 * atoms at the same coordinate - that is infinite on the FIRST force
 * evaluation, before any step has run, so there is no displacement to reduce
 * and arm 1's detector cannot see it.
 */
import { readFixture } from "./fixture.js";
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

async function build(device, system, positions, options) {
  const simulation = new GpuSimulation(device, system, {
    timestep: 0.002, temperature: 298, friction: 1.0, seed: 20261112, ...options,
  });
  const velocities = new Float64Array(positions.length);
  setVelocitiesToTemperature(system, velocities, options.temperature ?? 298, 23);
  const start = createState(system, positions, { velocities, tolerance: 1e-5 });
  simulation.setState(start.positions, start.velocities);
  await simulation.initialise();
  return simulation;
}

/** Run and report what came back, rather than letting a throw escape. */
async function attempt(body) {
  try {
    const value = await body();
    return { threw: false, value };
  } catch (error) {
    return { threw: true, name: error.name, message: error.message };
  }
}

export async function main(device) {
  const topology = parsePrmtop(await readFixture("ala-wat.prmtop", import.meta.url));
  const coordinates = parseInpcrd(await readFixture("ala-wat.inpcrd", import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const results = {};
  let failures = 0;

  // ---- arm 1: a run that blows up must be refused, by name ---------------
  {
    // 40 fs, which is the timestep the page accepted without comment and which
    // reached all-NaN in 50 steps. Twenty times past the measured 5-6 fs cliff,
    // so this arm is about the detector rather than about where the cliff is.
    const simulation = await build(device, system, coordinates.positions,
      { timestep: 0.040 });
    try {
      const outcome = await attempt(() => simulation.run(200, { blockSize: 20 }));
      // The NAME is asserted, not just that something threw: a WebGPU
      // validation failure or an allocator error would also throw here and
      // would mean something completely different.
      const named = outcome.threw && outcome.name === "NonFiniteStateError";
      // 🔴 AND THE MESSAGE IS GATED TOO. "Refuse by name rather than
      // approximating" is only worth anything if the refusal says what to do,
      // and the commonest cause of this one is the timestep.
      const actionable = named && /timestep too large/.test(outcome.message);
      if (!named || !actionable) failures += 1;
      results.blowUpRefused = {
        ok: named && actionable,
        threw: outcome.threw,
        name: outcome.name ?? null,
        namesACause: actionable,
        stoppedAtStep: simulation.step,
        message: outcome.message ?? null,
      };
    } finally { simulation.release(); }
  }

  // ---- arm 2: a sane run must NOT be refused -----------------------------
  {
    const simulation = await build(device, system, coordinates.positions,
      { timestep: 0.002 });
    try {
      const outcome = await attempt(() => simulation.run(200, { blockSize: 20 }));
      const energy = outcome.threw ? null : await simulation.energy();
      const diagnostics = await simulation.diagnostics();
      // Not merely "did not throw": the diagnostics the detector sits in front
      // of must still carry real numbers, because the failure mode being
      // guarded against is a NaN that poisons them.
      const quiet = !outcome.threw
        && Number.isFinite(diagnostics.worstDisplacement)
        && diagnostics.worstDisplacement > 0
        && Number.isFinite(energy.potential)
        && Number.isFinite(energy.temperature);
      if (!quiet) failures += 1;
      results.saneRunPasses = {
        ok: quiet,
        threw: outcome.threw,
        name: outcome.name ?? null,
        steps: simulation.step,
        worstDisplacement: diagnostics.worstDisplacement,
        potential: energy === null ? null : energy.potential,
        temperature: energy === null ? null : energy.temperature,
      };
    } finally { simulation.release(); }
  }

  // ---- arm 3: coincident atoms, caught before a step runs ----------------
  {
    // 🔴 TWO ATOMS AT THE SAME COORDINATE IS AN ORDINARY INPUT ARTEFACT, not a
    // contrived one - a hand-edited PDB, a badly merged complex, or a ligand
    // placed on top of a water. At r = 0 the direct-space term is 1/r, so the
    // first force evaluation is already infinite and nothing has moved yet.
    // 🔴 A WHOLE MOLECULE, NOT ONE ATOM, AND THAT DISTINCTION COST AN HOUR.
    // Moving a single water oxygen onto another atom does NOT reach the kernel:
    // `createState` projects onto the constraint manifold before the run, and
    // an oxygen 2 nm from its own two hydrogens is a grossly violated O-H
    // constraint, so SHAKE drags it straight back to its own molecule. Copying
    // all three atoms of one water onto another leaves both molecules
    // internally valid - SHAKE has nothing to do - so the two oxygens stay
    // exactly coincident and `1.0 / rSquared` in the kernel is `+Inf`.
    const broken = Float64Array.from(coordinates.positions);
    const last = topology.atomCount - 3;
    const previous = topology.atomCount - 6;
    for (let component = 0; component < 9; component += 1) {
      broken[last * 3 + component] = broken[previous * 3 + component];
    }
    const simulation = await build(device, system, broken, { timestep: 0.002 });
    try {
      // Measured AFTER createState, so this arm proves its own premise rather
      // than assuming the perturbation survived the projection.
      const settled = (await simulation.readState()).positions;
      const separation = Math.hypot(
        settled[last * 3] - settled[previous * 3],
        settled[last * 3 + 1] - settled[previous * 3 + 1],
        settled[last * 3 + 2] - settled[previous * 3 + 2]);
      const outcome = await attempt(() => simulation.energy());
      const named = outcome.threw && outcome.name === "NonFiniteStateError";
      // The premise and the conclusion are both asserted: a separation that
      // survived projection, AND a refusal.
      const ok = named && separation < 1e-6;
      if (!ok) failures += 1;
      results.coincidentAtomsRefused = {
        ok,
        threw: outcome.threw,
        name: outcome.name ?? null,
        atStep: simulation.step,
        separationAfterProjection: separation,
        // Reported so a reader can see this arm reaches the OTHER detector:
        // step 0 means no block has run, so `#checkSkin` was never consulted.
        detector: "energy(), with no step taken",
        message: outcome.message ?? null,
      };
    } finally { simulation.release(); }
  }

  return {
    check: "a blown-up run is refused by name rather than completing",
    failures,
    ...results,
  };
}
