/**
 * Hold the live analysis to MDAnalysis, on a trajectory this engine produced.
 *
 * 🔴 AN INDEPENDENT READER ON AN INDEPENDENT FILE. The metrics are computed
 * live, frame by frame, from coordinates that never touch the disk; MDAnalysis
 * then recomputes them afterwards from the DCD, with its own superposition, its
 * own mass table and its own RMSF accumulator. Nothing is shared but the
 * trajectory - which is itself byte-compatible with OpenMM's writer.
 *
 * 🔴 AND THE RMSF COMPARISON IS THE SHARP ONE. RMSD and Rg are single numbers
 * per frame and a systematic error shows up immediately; RMSF is accumulated
 * incrementally here and in one pass over stored coordinates there, so agreeing
 * is a statement about the running-sums identity rather than about arithmetic.
 */
import { readFixture } from "./fixture.js";

/**
 * 🔴 WRITTEN ONLY WHERE THERE IS A FILESYSTEM, AND THE IMPORT IS DYNAMIC FOR
 * THAT REASON. A static `import { writeFileSync } from "node:fs"` made this
 * whole module unloadable in the page - the browser lane failed with
 * "Failed to fetch dynamically imported module", which names the file and not
 * the cause - so the GPU work here, which is lane-agnostic, could only ever run
 * through Dawn. The comparison against MDAnalysis is a Node-lane step; the
 * physics belongs to both lanes.
 */
async function writeOut(files) {
  if (typeof process === "undefined" || process.release?.name !== "node") return [];
  const { writeFileSync } = await import("node:fs");
  for (const [path, data] of files) writeFileSync(path, data);
  return files.map(([path]) => path);
}
import { parsePrmtop } from "../../shared/amber/prmtop.js";
import { parseInpcrd } from "../../shared/amber/inpcrd.js";
import { buildSystem } from "../../shared/topology/system.js";
import { buildSelections } from "../../shared/analysis/selection.js";
import { LiveAnalysis } from "../../shared/analysis/metrics.js";
import { DcdWriter } from "../../shared/traj/dcd.js";
import { GpuSimulation } from "../../webgpu/simulation.js";
import { createState } from "../../cpu/simulation.js";
import { setVelocitiesToTemperature } from "../../cpu/integrate/langevin.js";

export async function main(device, args = []) {
  const out = (args.find((a) => a.startsWith("--out=")) ?? "--out=/tmp/rainmd-analysis")
    .slice("--out=".length);
  const fixture = (args.find((a) => a.startsWith("--fixture=")) ?? "--fixture=two-body")
    .slice("--fixture=".length);
  const steps = 1500;
  const interval = 50;

  const topology = parsePrmtop(await readFixture(`${fixture}.prmtop`, import.meta.url));
  const coordinates = parseInpcrd(await readFixture(`${fixture}.inpcrd`, import.meta.url),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  const selections = buildSelections(system);
  const simulation = new GpuSimulation(device, system, {
    timestep: 0.002, temperature: 298, friction: 10.0, seed: 909,
    skin: 0.2, rebuildInterval: 10,
  });
  try {
    const velocities = new Float64Array(topology.atomCount * 3);
    setVelocitiesToTemperature(system, velocities, 298, 71);
    const start = createState(system, coordinates.positions, { velocities, tolerance: 1e-5 });
    simulation.setState(start.positions, start.velocities);
    await simulation.initialise();

    // The reference is the first frame of the run, which is what MDAnalysis
    // will also be pointed at.
    const first = await simulation.readState();
    const reference = Float64Array.from(first.positions);
    const analysis = new LiveAnalysis(system, selections, reference);
    // 🔴 EVERY ATOM GOES INTO THE DCD, not just the solute, so the file pairs
    // with the `.prmtop` the user already has and MDAnalysis can make its own
    // selection. The METRICS use the solute alone; the trajectory does not have
    // to make that choice for whoever reads it later.
    const writer = new DcdWriter({
      atomCount: topology.atomCount, timestep: 0.002, interval,
      periodic: true, title: `RainMD ${fixture}`,
    });
    writer.appendFrame(first.positions, system.boxVectors);
    analysis.addFrame(reference, await simulation.energy());

    for (let done = 0; done < steps; done += interval) {
      await simulation.run(interval, { blockSize: 10 });
      const state = await simulation.readState();
      const energy = await simulation.energy();
      writer.appendFrame(state.positions, system.boxVectors);
      analysis.addFrame(Float64Array.from(state.positions), energy);
    }
    const wrote = await writeOut([
      [`${out}.dcd`, writer.finish()],
      [`${out}.metrics.csv`, analysis.metricsCsv()],
      [`${out}.rmsf.csv`, analysis.rmsfCsv()],
      // The fit selection, so MDAnalysis superposes on exactly the same atoms.
      [`${out}.fit.json`, JSON.stringify({
        fit: [...analysis.fit],
        receptor: [...analysis.receptorFit],
        binder: [...analysis.binderFit],
        residueAtoms: [...selections.residueAtoms],
        residueLabels: selections.residueLabels,
        frames: writer.frameCount,
        interval,
      })],
    ]);
    const last = analysis.series[analysis.series.length - 1];
    return {
      check: "live analysis written out for comparison against MDAnalysis",
      failures: 0,
      fixture,
      atomCount: topology.atomCount,
      counts: selections.counts,
      soluteMolecules: selections.soluteMoleculeCount,
      hasBinder: analysis.hasBinder,
      frames: writer.frameCount,
      residues: selections.residueLabels.length,
      // Empty in the browser lane, which has nowhere to write.
      wrote,
      lastFrame: last,
    };
  } finally {
    simulation.release();
  }
}
