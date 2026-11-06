/**
 * The page.
 *
 * 🔴 IT RUNS THE GPU ENGINE, AND IT SAYS WHICH ENGINE RAN. For several
 * iterations this page ran the host reference while a validated GPU engine sat
 * beside it unreachable, which is the one genuinely misleading state this
 * repository could be in. The engine selector exists so the host path is still
 * available for comparison, and every log line names the path it took.
 */
import { parsePrmtop } from "../shared/amber/prmtop.js";
import { parseInpcrd } from "../shared/amber/inpcrd.js";
import { buildSystem } from "../shared/topology/system.js";
import { DcdWriter } from "../shared/traj/dcd.js";
import { writeRestart } from "../shared/traj/rst7.js";
import { requestMdDevice } from "../webgpu/runtime/device.js";
import { yieldToBrowser, StopSignal } from "../webgpu/runtime/yield.js";
import { GpuSimulation } from "../webgpu/simulation.js";
import { potentialEnergy } from "../cpu/energy.js";
import { createState, minimise } from "../cpu/simulation.js";
import {
  langevinStep, langevinFinishStep, instantaneousTemperature,
} from "../cpu/integrate/langevin.js";
import { Panel, PanelStack } from "./plot.js";
import { StructureViewer } from "./viewer.js";
import { buildSelections, VIEW_PRESETS, BODY_COLOURS } from "../shared/analysis/selection.js";
import { secondaryOfAtoms } from "../shared/analysis/secondary.js";
import { buildRibbon } from "../shared/analysis/ribbon.js";
import { LiveAnalysis } from "../shared/analysis/metrics.js";
import { writePdb } from "../shared/traj/pdb.js";

const element = (id) => document.getElementById(id);
const logBox = element("log");
let gpu = null;
let loaded = null;
let hostState = null;
let stopSignal = null;
let trajectory = null;
let finalState = null;
let plot = null;
let viewer = null;
let selections = null;
let analysis = null;
let paused = false;
let resumeGate = null;
let lastPositions = null;
let selectionSystem = null;
let binderOverride = null;

function log(line = "") {
  logBox.textContent += `${line}\n`;
  logBox.scrollTop = logBox.scrollHeight;
}

function describe(list, rows) {
  list.innerHTML = "";
  for (const [term, definition] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = definition;
    list.append(dt, dd);
  }
  list.hidden = false;
}

function showLive(rows) {
  const container = element("live");
  container.innerHTML = "";
  for (const [label, value] of rows) {
    const cell = document.createElement("div");
    const name = document.createElement("span");
    name.className = "label";
    name.textContent = label;
    const text = document.createElement("span");
    text.className = "value";
    text.textContent = value;
    cell.append(name, text);
    container.append(cell);
  }
}

function download(name, data, type) {
  const blob = new Blob([data], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  // 🔴 REVOKED ON A TIMER, NOT IMMEDIATELY. Revoking in the same task cancels
  // the download in Safari before it has read the blob.
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ---------------------------------------------------------------- device ----
(async () => {
  const report = element("device-report");
  try {
    gpu = await requestMdDevice(navigator.gpu);
    const rows = [
      ["WebGPU", "available"],
      ["adapter", gpu.report.description
        || `${gpu.report.vendor ?? "?"} / ${gpu.report.architecture ?? "?"}`],
      ["max storage binding",
        `${(gpu.report.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`],
      ["invocations / workgroup", String(gpu.report.limits.maxComputeInvocationsPerWorkgroup)],
    ];
    if (gpu.report.adapterOfferedF16 !== gpu.report.deviceHasF16) {
      rows.push(["shader-f16", "the adapter offers it and a default device does not "
        + "— requested explicitly here"]);
    }
    describe(report, rows);
  } catch (error) {
    describe(report, [["WebGPU", "unavailable"], ["reason", error.message]]);
    element("engine").value = "host";
    log(`WebGPU unavailable, falling back to the host reference: ${error.message}`);
  }
})();

// ------------------------------------------------------------------ load ----
const dropZone = element("drop-zone");
for (const name of ["dragenter", "dragover"]) {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("over");
  });
}
for (const name of ["dragleave", "drop"]) {
  dropZone.addEventListener(name, () => dropZone.classList.remove("over"));
}
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  void loadFiles([...event.dataTransfer.files]);
});
element("file-input").addEventListener("change", (event) => {
  void loadFiles([...event.target.files]);
});

/**
 * Load one of the bundled fixtures, so the page is usable by someone who has
 * never run tleap.
 *
 * 🔴 RELATIVE URLs, NOT ABSOLUTE. GitHub Pages serves a project site from
 * `/<repo>/`, so a leading slash resolves to the account root and every fetch
 * 404s - the page would work on the dev server and be broken everywhere it is
 * actually published. The test harness uses absolute paths because it drives a
 * server rooted at the repository; the page cannot assume that.
 *
 * 🔴 AND IT GOES THROUGH `loadFiles`, THE SAME PATH A DROPPED FILE TAKES. An
 * example that bypassed the loader would not exercise the format detection, the
 * warnings or the error reporting - which is most of what could be wrong with
 * a file a visitor supplies.
 */
element("load-example").addEventListener("click", async () => {
  const button = element("load-example");
  const name = element("example").value;
  const errorBox = element("load-error");
  button.disabled = true;
  try {
    const [topology, coordinates] = await Promise.all([
      fetch(`test/fixtures/${name}.prmtop`).then((response) => {
        if (!response.ok) throw new Error(`${name}.prmtop: HTTP ${response.status}`);
        return response.text();
      }),
      fetch(`test/fixtures/${name}.inpcrd`).then((response) => {
        if (!response.ok) throw new Error(`${name}.inpcrd: HTTP ${response.status}`);
        return response.text();
      }),
    ]);
    await loadFiles([
      new File([topology], `${name}.prmtop`),
      new File([coordinates], `${name}.inpcrd`),
    ]);
  } catch (error) {
    errorBox.textContent = `could not load the example: ${error.message}`;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
  }
});

async function loadFiles(files) {
  const errorBox = element("load-error");
  errorBox.hidden = true;
  try {
    const topologyFile = files.find((file) => /\.(prmtop|parm7|top)$/i.test(file.name));
    const coordinateFile = files.find((file) => /\.(crd|inpcrd|rst7|rst)$/i.test(file.name));
    if (topologyFile === undefined || coordinateFile === undefined) {
      throw new Error(
        "two files are needed: a topology (.prmtop or .parm7) and coordinates "
        + "(.crd, .inpcrd or .rst7) from the same tleap run.",
      );
    }
    const topology = parsePrmtop(await topologyFile.text());
    const coordinates = parseInpcrd(await coordinateFile.text(),
      { expectedAtomCount: topology.atomCount });
    loaded = { topology, coordinates, name: topologyFile.name.replace(/\.[^.]+$/, "") };
    // 🔴 SELECTIONS ARE BUILT ONCE, FROM THE TOPOLOGY, and everything else
    // reads them: the viewer's visibility, the analysis's fit set, the RMSF's
    // residue list and the receptor/binder split. Recomputing them per frame
    // would be the obvious place for the viewer and the metrics to drift apart.
    selectionSystem = buildSystem(topology, {
      constraints: "HBonds", rigidWater: true,
      cutoff: coordinates.box === null ? 0 : 1.0, box: coordinates.box,
    });
    binderOverride = null;
    selections = buildSelections(selectionSystem);
    analysis = null;
    hostState = null;
    trajectory = null;
    finalState = null;
    element("download-dcd").disabled = true;
    element("download-state").disabled = true;

    let chargeSum = 0;
    for (const charge of topology.charges) chargeSum += charge;
    const box = coordinates.box;
    describe(element("system-report"), [
      ["files", `${topologyFile.name} + ${coordinateFile.name}`],
      ["atoms", topology.atomCount.toLocaleString()],
      ["residues", `${topology.residueNames.length.toLocaleString()} in `
        + `${topology.moleculeCount.toLocaleString()} molecules`],
      ["bonds / angles / torsions",
        `${topology.bonds.count} / ${topology.angles.count} / ${topology.dihedrals.count}`],
      ["net charge", `${chargeSum.toFixed(4)} e`],
      ["box", box === null ? "none (non-periodic)"
        : `${box.lengths.map((v) => v.toFixed(3)).join(" × ")} nm, angles `
          + `${box.angles.map((v) => v.toFixed(1)).join("/")}°`],
      ["velocities in file", coordinates.velocities === null ? "no" : "yes"],
      ["solute / water / ions",
        `${selections.counts.solute} / ${selections.counts.water} / ${selections.counts.ion}`],
      ["protein / ligand", `${selections.counts.protein} / ${selections.counts.ligand}`],
      ["receptor / binder", selections.binder.length === 0
        ? `${selections.receptor.length} (single body — no binder split)`
        : `${selections.receptor.length} / ${selections.binder.length} `
          + `(${selections.soluteMoleculeCount} solute molecules)`],
    ]);
    populateBinders();
    populatePresets();
    // 🔴 DRAW ON LOAD, NOT ON RUN. The viewer used to be created by
    // `attachViewer` inside Minimise and Run, so dropping a topology showed a
    // black rectangle until the user started something - and a reader with a
    // file they are unsure about has no way to check they loaded the right
    // thing. JIGGLE shows the structure the moment it has one. This draws the
    // file's own coordinates from its own buffer; the run then rebinds the
    // viewer to the simulation's.
    showLoadedStructure();
    element("run-panel").hidden = false;
    for (const id of ["download-dcd", "download-state", "download-pdb",
      "download-metrics", "download-rmsf"]) element(id).disabled = true;
    element("method").value = box === null ? "NoCutoff" : "PME";
    logBox.textContent = "";
    log(`loaded ${topology.atomCount} atoms from ${topologyFile.name}`);
    for (const warning of topology.warnings) log(`note: ${warning}`);
    for (const note of coordinates.ambiguous) log(`note: ${note}`);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
    loaded = null;
    element("run-panel").hidden = true;
  }
}

// ---------------------------------------------------------------- system ----
function numberOf(id) { return Number(element(id).value); }

function assembleSystem() {
  const method = element("method").value;
  const system = buildSystem(loaded.topology, {
    constraints: element("constraints").value,
    rigidWater: true,
    cutoff: method === "NoCutoff" ? 0 : numberOf("cutoff"),
    box: loaded.coordinates.box,
  });
  return { system, method };
}

function runOptions(method) {
  const pressureText = element("pressure").value.trim();
  return {
    method,
    timestep: numberOf("timestep") / 1000,
    temperature: numberOf("temperature"),
    friction: numberOf("friction"),
    // Blank means NVT. An explicit 0 bar is a legitimate (if unusual) request,
    // so emptiness rather than falsiness is what switches the barostat off.
    pressure: pressureText === "" ? null : Number(pressureText),
    seed: 20261106,
    tolerance: 1e-5,
  };
}

/** Draw the coordinates as loaded, before anything has been run. */
function showLoadedStructure() {
  const view = ensureViewer();
  if (view === null || loaded === null) return;
  applyView(view, { topology: loaded.topology }, loaded.coordinates.positions);
  // 🔴 ITS OWN BUFFER, UPLOADED FROM THE HOST. There is no simulation yet, so
  // there is no device buffer to copy from - `setStructure` has already sized
  // the viewer's own positions buffer, and this fills it.
  // No frame() call here: `setStructure` already framed on the visible atoms,
  // and `frame` takes the styles array it built - calling it with one argument
  // throws on `styles[...]`, which the load handler's catch then swallows into
  // a blank log and a null `loaded`.
  view.upload(loaded.coordinates.positions);
  view.draw();
}

function ensureViewer() {
  if (viewer === null && gpu !== null) {
    viewer = new StructureViewer(gpu.device, element("viewer"));
  }
  return viewer;
}

/**
 * Six panels, following JIGGLE's set: RMSD (split four ways), radius of
 * gyration, potential energy, per-residue RMSF, interface distance, and
 * temperature.
 */
function ensurePlot() {
  if (plot !== null) return plot;
  plot = new PanelStack([
    new Panel(element("plot-rmsd"), {
      title: "RMSD", unit: "nm",
      series: [
        // 🔴 THE SAME PALETTE THE VIEWER USES, so a reader can map a trace onto
        // a molecule. "binder on receptor" is the exception and is deliberately
        // a warning red: it is not a body, it is the pose-stability number.
        { key: "rmsd", label: "complex", colour: BODY_COLOURS.complex.line },
        { key: "rmsdReceptor", label: "receptor", colour: BODY_COLOURS.receptor.line },
        { key: "rmsdBinder", label: "binder", colour: BODY_COLOURS.binder.line },
        { key: "rmsdBinderOnReceptor", label: "binder on receptor", colour: "#b3443a" },
      ],
    }),
    new Panel(element("plot-rg"), {
      title: "Radius of gyration", unit: "nm",
      series: [{ key: "radiusOfGyration", label: "Rg", colour: "#3f7f4f" }],
    }),
    new Panel(element("plot-energy"), {
      title: "Potential energy", unit: "kJ/mol",
      series: [{ key: "potential", label: "potential", colour: "#7a4fa3" }],
    }),
    new Panel(element("plot-rmsf"), {
      title: "Per-residue RMSF", unit: "nm", kind: "bar",
      series: [{ key: "rmsf", label: "RMSF", colour: BODY_COLOURS.receptor.line }],
      labels: selections?.residueLabels ?? null,
    }),
    new Panel(element("plot-iface"), {
      title: "Interface distance", unit: "nm",
      series: [{ key: "interfaceDistance", label: "centroid separation", colour: "#c07a3a" }],
    }),
    new Panel(element("plot-temperature"), {
      title: "Temperature", unit: "K",
      series: [{ key: "temperature", label: "T", colour: "#b3443a" }],
    }),
  ]);
  return plot;
}

/**
 * Offer the molecule list in place of the chain letters a prmtop does not have.
 *
 * 🔴 THE DERIVED SPLIT IS RIGHT FOR A PROTEIN AND A LIGAND AND WRONG FOR A
 * DIMERIC RECEPTOR, and until now there was no way to say so. "Largest solute
 * molecule is the receptor" reports one chain of a two-chain receptor as the
 * binder, so `binder on receptor` measures the two halves of the receptor
 * moving relative to each other and reads like a pose coming apart. The row is
 * hidden for a single-body system, because then there is nothing to choose.
 */
function populateBinders() {
  const row = element("binder-row");
  const select = element("binder");
  const molecules = selections.soluteMolecules ?? [];
  row.hidden = molecules.length < 2;
  if (row.hidden) return;
  select.innerHTML = "";
  const auto = document.createElement("option");
  auto.value = "auto";
  auto.textContent = `auto — largest of ${molecules.length} is the receptor`;
  select.append(auto);
  for (const molecule of molecules) {
    const option = document.createElement("option");
    option.value = String(molecule.molecule);
    option.textContent = `${molecule.label} (${molecule.atomCount} atoms)`;
    select.append(option);
  }
  select.value = binderOverride === null ? "auto" : String(binderOverride);
}

/** Re-split the solute, then rebuild everything that reads the split. */
function applyBinderChoice() {
  if (loaded === null) return;
  const value = element("binder").value;
  binderOverride = value === "auto" ? null : Number(value);
  selections = buildSelections(selectionSystem,
    binderOverride === null ? {} : { binderMolecules: [binderOverride] });
  populatePresets();
  log(`split: receptor ${selections.receptor.length} atoms / binder `
    + `${selections.binder.length}`
    + (binderOverride === null ? " (derived from the molecule partition)"
      : " (chosen)"));
  if (viewer !== null) {
    applyView(viewer, { topology: loaded.topology },
      lastPositions ?? hostState?.positions ?? loaded.coordinates.positions);
    viewer.draw();
  }
}

function populatePresets() {
  const select = element("view-preset");
  select.innerHTML = "";
  for (const preset of VIEW_PRESETS) {
    const visible = preset.groups.reduce(
      (total, name) => total + (selections.counts[name] ?? 0), 0);
    if (visible === 0) continue;
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = `${preset.label} (${visible})`;
    select.append(option);
  }
  select.value = selections.counts.ligand > 0 ? "protein-ligand" : "solute";
}

function currentPreset() {
  return VIEW_PRESETS.find((preset) => preset.id === element("view-preset").value)
    ?? VIEW_PRESETS[0];
}

const number = (value, digits, unit) => (value === null || value === undefined
  || !Number.isFinite(value) ? "–" : `${value.toFixed(digits)}${unit}`);

function report(row, extra = []) {
  const tiles = [
    ["step", row.step.toLocaleString()],
    ["time", `${row.time.toFixed(3)} ps`],
    ["potential", number(row.potential, 1, " kJ/mol")],
    ["total energy", number(row.total, 1, " kJ/mol")],
    ["temperature", number(row.temperature, 1, " K")],
    ["RMSD", number(row.rmsd, 4, " nm")],
    ["Rg", number(row.radiusOfGyration, 4, " nm")],
  ];
  if (row.rmsdBinderOnReceptor !== null && row.rmsdBinderOnReceptor !== undefined) {
    tiles.push(["receptor RMSD", number(row.rmsdReceptor, 4, " nm")]);
    tiles.push(["binder RMSD", number(row.rmsdBinder, 4, " nm")]);
    // The pose-stability number, labelled so it is not mistaken for the others.
    tiles.push(["binder on receptor", number(row.rmsdBinderOnReceptor, 4, " nm")]);
    tiles.push(["interface", number(row.interfaceDistance, 4, " nm")]);
  }
  showLive([...tiles, ...extra]);
  const stack = ensurePlot();
  stack.push(row);
  if (analysis !== null) stack.setBars("rmsf", analysis.rmsf.values());
  stack.draw();
}

// ------------------------------------------------------- GPU kernel check ----
element("check-gpu").addEventListener("click", async () => {
  if (gpu === null) { log("no WebGPU device on this machine"); return; }
  const button = element("check-gpu");
  button.disabled = true;
  try {
    log("");
    log("running the kernel checks on your GPU (bundled test system, not your files)…");
    // 🔴 THE SAME CHECK MODULES THE COMMAND LINE RUNS. They carry their own
    // fixtures and their own tolerances, so what a reader sees here is the same
    // statement `npm run test:gpu` makes - about their hardware rather than
    // about this code.
    for (const name of ["check-forces", "check-integrator"]) {
      const module = await import(`../tools/gpu/${name}.mjs`);
      const result = await module.main(gpu.device);
      log(`  ${name}: ${result.failures === 0 ? "PASS" : `${result.failures} FAILURE(S)`}`);
      if (name === "check-forces") {
        for (const comparison of result.energy.comparisons) {
          log(`    ${comparison.ok ? "ok  " : "FAIL"} ${comparison.label.padEnd(20)} `
            + `relative ${comparison.relative.toExponential(2)}`);
        }
        log(`    ${result.force.ok ? "ok  " : "FAIL"} ${"per-atom forces".padEnd(20)} `
          + `relative ${result.force.relative.toExponential(2)}`);
      } else {
        const conservative = result.results.conservative;
        log(`    one step vs host: ${result.results.oneStep.worstPosition.toExponential(2)} nm`);
        log(`    energy drift: ${conservative.energyDriftPerNsPerAtom.toFixed(3)} `
          + `kJ/mol/ns/atom over ${conservative.steps} steps`);
        log(`    ${conservative.millisecondsPerStep.toFixed(2)} ms/step on this GPU`);
      }
      await yieldToBrowser();
    }
  } catch (error) {
    log(`error: ${error.stack ?? error.message}`);
  } finally {
    button.disabled = false;
  }
});

// -------------------------------------------------------------- minimise ----
element("minimise").addEventListener("click", async () => {
  if (loaded === null) return;
  const button = element("minimise");
  button.disabled = true;
  try {
    const { system, method } = assembleSystem();
    const options = runOptions(method);
    const engine = element("engine").value;
    log("");
    if (engine === "gpu" && gpu !== null && method === "PME") {
      const simulation = new GpuSimulation(gpu.device, system, {
        ...options, rebuildInterval: Math.max(1, numberOf("rebuild")),
      });
      try {
        simulation.setState(hostState?.positions ?? loaded.coordinates.positions,
          new Float64Array(system.topology.atomCount * 3));
        await simulation.initialise();
        attachViewer(simulation, system);
        log("minimising on the GPU (steepest descent, backtracking)…");
        await yieldToBrowser();
        const result = await simulation.minimise({ maxIterations: 200, forceTolerance: 500 });
        log(`  potential ${result.history[0].potential.toFixed(2)} → `
          + `${result.potential.toFixed(2)} kJ/mol in ${result.iterations} iterations`);
        log(`  max force ${result.history[0].maxForce.toFixed(0)} → `
          + `${result.maxForce.toFixed(0)} kJ/mol/nm`);
        if (result.stalled && !result.converged) {
          log(`  stalled: f32 resolves the energy to about `
            + `${result.energyResolution.toFixed(3)} kJ/mol, below which the line `
            + `search cannot tell whether a step helped. The host engine (f64) `
            + `minimises further.`);
        }
        const state = await simulation.readState();
        // 🔴 NULL, NOT A ZERO ARRAY - SEE `startState`. A minimised state has
        // no velocities; recording that as zeros made it a valid draw at 0 K.
        hostState = { positions: Float64Array.from(state.positions),
          velocities: null, step: 0 };
        if (viewer !== null) {
          viewer.updateFrom(simulation.resources.positions);
          viewer.draw();
        }
      } finally {
        simulation.release();
      }
      return;
    }
    if (hostState === null) {
      hostState = createState(system, loaded.coordinates.positions, {
        ...options, velocities: loaded.coordinates.velocities,
        temperature: loaded.coordinates.velocities === null ? options.temperature : undefined,
      });
    }
    log("minimising on the host (steepest descent, f64)…");
    await yieldToBrowser();
    const result = minimise(system, hostState, {
      ...options, maxIterations: 80, forceTolerance: 500,
    });
    log(`  energy ${result.history[0].energy.toFixed(2)} → ${result.energy.total.toFixed(2)} kJ/mol`);
    log(`  max force ${result.history[0].maxForce.toFixed(0)} → ${result.maxForce.toFixed(0)} `
      + `kJ/mol/nm in ${result.iterations} iterations`);
    // The velocities in this state were drawn for the structure that went IN;
    // what came out is a different structure, and the run temperature may have
    // been changed since. Draw fresh at run time.
    hostState.velocities = null;
  } catch (error) {
    log(`error: ${error.stack ?? error.message}`);
  } finally {
    button.disabled = false;
  }
});

/**
 * Point the viewer at a simulation's position buffer.
 *
 * 🔴 THE BUFFER, NOT A COPY. The viewer binds the same GPUBuffer the
 * integrator writes, so a redraw after a block of steps shows the current state
 * with no readback at all - which is why `draw()` can be called every report
 * without costing anything measurable.
 */
function attachViewer(simulation, system) {
  const view = ensureViewer();
  if (view === null) return;
  applyView(view, system, hostState?.positions ?? loaded.coordinates.positions);
  view.updateFrom(simulation.resources.positions);
  view.draw();
}

function applyView(view, system, positions) {
  const preset = currentPreset();
  const colourBy = element("colour-by").value;
  const representation = element("representation").value;
  // 🔴 ASSIGNED FROM THE FRAME BEING DRAWN, not once at load. Secondary
  // structure is the one colouring that CHANGES during a run - watching a helix
  // fray is the point of having it - so it is recomputed whenever the view is
  // rebuilt. It costs a pass over the alpha carbons, one per residue, which is
  // three orders of magnitude smaller than the atom count it colours.
  // 🔴 THE RIBBON NEEDS THE ASSIGNMENT WHATEVER THE COLOUR MODE IS. A cartoon
  // whose widths did not follow the structure would be a tube, so the
  // assignment is computed whenever either the colouring or the representation
  // asks for it - not only when the colour dropdown says "structure".
  const wantsRibbon = representation !== "spheres";
  const secondary = (colourBy === "structure" || wantsRibbon) && selections !== null
    ? secondaryOfAtoms({ topology: system.topology }, positions, selections)
    : null;
  view.setStructure(system.topology, {
    selections,
    groups: preset.groups,
    heavyOnly: preset.heavyOnly === true,
    colourBy,
    secondary: secondary?.perAtom,
    // 🔴 THIN ATOMS BESIDE A RIBBON, OR THE RIBBON IS INVISIBLE. A van der
    // Waals sphere is 0.11 to 0.17 nm and the ribbon is 0.12 nm half-width, so
    // at full radius the atoms swallow the cartoon completely - "ribbon +
    // atoms" drew exactly the same picture as "atoms". This is why PyMOL pairs
    // a cartoon with STICKS rather than spheres.
    radiusScale: representation === "both" ? 0.22 : 0.65,
    box: loaded.coordinates.box,
    positions,
  });
  view.representation = representation;
  if (wantsRibbon && secondary !== null) {
    // The ribbon follows the view: hiding a chain or a ligand must break the
    // ribbon there rather than draw it across the gap. `visibleAtoms` is the
    // same flag array the styles buffer was built from.
    const ribbon = buildRibbon(system.topology, positions, selections, secondary,
      { visible: view.visibleAtoms });
    view.setRibbon(ribbon);
    log(`ribbon: ${ribbon.guideCount} guides in ${ribbon.runCount} chain run(s), `
      + `${ribbon.segmentCount} segments`);
  } else {
    view.setRibbon({ guides: new Float32Array(0), segments: new Uint32Array(0),
      segmentCount: 0, guideCount: 0, runCount: 0 });
  }
  if (secondary !== null) {
    log(`structure: ${secondary.counts.helix} helix / ${secondary.counts.sheet} sheet / `
      + `${secondary.counts.coil} coil residues (alpha-carbon geometry, not DSSP)`);
  }
  const shown = preset.groups.reduce(
    (total, name) => total + (selections.counts[name] ?? 0), 0);
  element("view-counts").textContent =
    `${shown.toLocaleString()} of ${system.topology.atomCount.toLocaleString()} atoms drawn`
    + (preset.heavyOnly === true ? ", hydrogens hidden" : "");
}

/**
 * The state a run starts from.
 *
 * 🔴 ABSENT VELOCITIES ARE `null`, AND A ZERO ARRAY IS NOT THE SAME THING.
 * Minimisation destroys velocities - it is a walk downhill in position with no
 * dynamics - and the GPU minimiser recorded that as `new Float64Array(3N)`.
 * That is a VALID Maxwell-Boltzmann draw at 0 K, so the run's presence test
 * (`=== null`) accepted it and every trajectory after a minimisation started at
 * ABSOLUTE ZERO: measured on the two-body fixture, frame 0 came back at 0.00 K
 * with 0.00 kJ/mol of kinetic energy while the box was asked for 298 K.
 * Langevin friction heats it back within about a picosecond, which is why
 * nothing looked broken - but frame 0 is the reference every RMSD is measured
 * from, so the early rise in every curve was partly the system warming up, and
 * the temperature panel's axis ran from -24 K. The same shape as any `?? 0` on
 * a missing value: an ABSENT thing collapsed onto a valid one.
 *
 * 🔴 AND ONE RESOLVER, BECAUSE THE TWO HALVES DISAGREED. The run read
 * `hostState?.velocities ?? loaded.coordinates.velocities` for the array but
 * `(hostState ?? loaded.coordinates).velocities === null` for the temperature,
 * so a minimised state beside an inpcrd that carries velocities would have
 * taken the FILE's pre-minimisation velocities while reporting that it drew
 * fresh ones. Whichever state is current decides both, or neither is right.
 */
function startState(system, options) {
  const current = hostState ?? loaded.coordinates;
  const supplied = current.velocities ?? null;
  const state = createState(system, hostState?.positions ?? loaded.coordinates.positions, {
    ...options,
    velocities: supplied,
    temperature: supplied === null ? options.temperature : undefined,
  });
  log(supplied === null
    ? `  velocities drawn at ${options.temperature} K and projected onto the `
      + `constraint manifold`
    : "  velocities carried over from the current state");
  return state;
}

// --------------------------------------------------------------- dynamics ----
element("stop").addEventListener("click", () => {
  // Release a held pause so the loop can observe the stop.
  paused = false;
  resumeGate?.();
  resumeGate = null;
  element("pause").textContent = "Pause";
  stopSignal?.stop();
});

element("pause").addEventListener("click", () => {
  paused = !paused;
  element("pause").textContent = paused ? "Resume" : "Pause";
  if (!paused) { resumeGate?.(); resumeGate = null; }
  log(paused ? "  paused — the integrator is not being advanced" : "  resumed");
});

element("run").addEventListener("click", async () => {
  if (loaded === null) return;
  const runButton = element("run");
  const stopButton = element("stop");
  runButton.disabled = true;
  stopButton.disabled = false;
  element("pause").disabled = false;
  // 🔴 THE SPLIT IS FROZEN WHILE A RUN IS IN FLIGHT. `LiveAnalysis` takes the
  // receptor and binder sets once, at the first frame, and every row after
  // that is measured against them - so re-splitting mid-run would leave the
  // series with two different meanings for one column and no way to tell
  // which rows were which. The view presets stay live; only the split locks.
  element("binder").disabled = true;
  paused = false;
  element("pause").textContent = "Pause";
  stopSignal = new StopSignal();
  ensurePlot().clear();
  try {
    const engine = element("engine").value;
    if (engine === "gpu" && gpu === null) throw new Error("no WebGPU device");
    if (engine === "gpu") await runOnDevice();
    else await runOnHost();
  } catch (error) {
    log(error.name === "SimulationStopped" ? `  stopped` : `error: ${error.stack ?? error.message}`);
  } finally {
    runButton.disabled = false;
    stopButton.disabled = true;
    element("pause").disabled = true;
    element("binder").disabled = false;
    paused = false;
    resumeGate = null;
    stopSignal = null;
  }
});

async function runOnDevice() {
  const { system, method } = assembleSystem();
  if (method !== "PME") {
    throw new Error(
      "the device engine implements PME, so it needs a periodic system. A "
      + "non-periodic one can still be run on the host reference.",
    );
  }
  const options = runOptions(method);
  const steps = numberOf("steps");
  const trajectoryInterval = numberOf("trajectory");
  const rebuildInterval = Math.max(1, numberOf("rebuild"));

  const simulation = new GpuSimulation(gpu.device, system, {
    ...options, rebuildInterval, barostatInterval: 25,
  });
  try {
    const start = startState(system, options);
    simulation.setState(start.positions, start.velocities);
    log("");
    log(`GPU: ${steps} steps, ${options.timestep * 1000} fs, ${options.temperature} K, `
      + `friction ${options.friction}/ps`
      + (options.pressure === null ? " (NVT)" : `, ${options.pressure} bar (NPT)`));
    log(`  ${system.constraints.count} constraints in ${simulation.clusters.layout.clusterCount} `
      + `clusters, ${simulation.plan.counts.join("×")} cells, PME grid `
      + `${simulation.pme.grid.join("×")}, α ${simulation.pme.alpha.toFixed(4)}`);
    await simulation.initialise();
    attachViewer(simulation, system);
    const diagnostics = await simulation.diagnostics();
    if (diagnostics.toleranceWasClamped) {
      log(`  note: constraint tolerance raised to ${diagnostics.constraintTolerance.toExponential(2)} `
        + `— ${diagnostics.requestedTolerance.toExponential(0)} is below what f32 coordinates `
        + `can reach on this box (floor ${diagnostics.toleranceFloor.toExponential(2)})`);
    }

    const writer = trajectoryInterval > 0
      ? new DcdWriter({
        atomCount: system.topology.atomCount,
        timestep: options.timestep,
        interval: trajectoryInterval,
        periodic: true,
        title: `RainMD ${loaded.name}`,
      })
      : null;

    const chunk = trajectoryInterval > 0 ? trajectoryInterval
      : Math.max(1, Math.floor(steps / 20));

    // 🔴 THE REFERENCE FRAME IS THE FIRST FRAME OF THIS RUN, not the file on
    // disk. Every RMSD is measured from it, so taking the input coordinates
    // instead would fold the minimisation and the constraint projection into
    // frame one - a step change of a tenth of a nanometre before any dynamics
    // has happened.
    lastPositions = await simulation.readPositions();
    analysis = new LiveAnalysis(system, selections, lastPositions);
    log(`  analysis: fitting on ${analysis.fit.length} heavy solute atoms, `
      + `${selections.residueLabels.length} residues tracked`
      + (analysis.hasBinder
        ? `; receptor ${analysis.receptorFit.length} / binder ${analysis.binderFit.length} `
          + "— RMSD is split four ways"
        : "; single body — no binder split"));

    const began = performance.now();
    let done = 0;
    report(analysis.addFrame(lastPositions, await simulation.energy()));
    if (writer !== null) writer.appendFrame(lastPositions, system.boxVectors);
    while (done < steps) {
      stopSignal.throwIfStopped();
      // 🔴 A TRUE PAUSE: the gate is awaited BETWEEN blocks, so the integrator
      // is not advanced while it is held and resuming continues from the
      // identical state. Pausing inside a block would leave positions updated
      // and velocities not.
      if (paused) {
        await new Promise((resolve) => { resumeGate = resolve; });
      }
      const take = Math.min(chunk, steps - done);
      await simulation.run(take, { blockSize: rebuildInterval, stopSignal });
      done += take;
      const energy = await simulation.energy();
      lastPositions = await simulation.readPositions();
      const row = analysis.addFrame(lastPositions, energy);
      const elapsed = (performance.now() - began) / 1000;
      const extra = [
        ["ms / step", ((elapsed / done) * 1000).toFixed(2)],
        ["ns / day", ((energy.time / 1000) / (elapsed / 86400)).toFixed(1)],
        ["frames", String(writer?.frameCount ?? 0)],
      ];
      if (options.pressure !== null) {
        extra.push(["volume", `${energy.volume.toFixed(3)} nm³`]);
        extra.push(["density", `${densityOf(system, energy.volume).toFixed(4)} g/cm³`]);
      }
      report(row, extra);
      if (viewer !== null) {
        viewer.updateFrom(simulation.resources.positions);
        // Re-assign only in the one mode whose colours depend on the frame;
        // every other mode's styles buffer is still valid.
        if (element("colour-by").value === "structure"
          || element("representation").value !== "spheres") {
          applyView(viewer, { topology: system.topology }, lastPositions);
          viewer.updateFrom(simulation.resources.positions);
        }
        viewer.draw();
      }
      if (writer !== null) writer.appendFrame(lastPositions, system.boxVectors);
      element("download-pdb").disabled = false;
      element("download-metrics").disabled = false;
      element("download-rmsf").disabled = analysis.rmsf.frames < 2;
      await yieldToBrowser();
    }
    const elapsed = (performance.now() - began) / 1000;
    log(`  ${done} steps in ${elapsed.toFixed(1)} s — ${((elapsed / done) * 1000).toFixed(2)} ms/step`);

    const after = await simulation.diagnostics();
    if (after.barostat !== null) {
      log(`  barostat: ${after.barostat.accepted}/${after.barostat.attempts} moves accepted `
        + `(${(after.barostat.acceptanceRate * 100).toFixed(0)}%), `
        + `${after.barostat.refusedForCutoff} refused because the cutoff would not fit; `
        + `volume ${after.volume.toFixed(3)} nm³, density `
        + `${densityOf(system, after.volume).toFixed(4)} g/cm³`);
    }
    log(`  constraints: ${after.maxShakeIterations} SHAKE iterations at most, `
      + `iteration cap reached: ${after.shakeHitIterationCap || after.rattleHitIterationCap}`);
    log(`  neighbours: ${after.neighbourRequired} needed of ${after.neighbourCapacity} capacity`);
    // 🔴 THE SKIN LINE IS NOT OPTIONAL OUTPUT. If the neighbour list was ever
    // stale, this run's forces were missing pairs inside the cutoff - a loss of
    // exactness the energies and the trajectory will not show. Reporting the
    // margin on every run, and the violation loudly when there is one, is the
    // whole user-facing point of measuring it.
    log(`  skin: ${(after.skin * 1000).toFixed(0)} pm, worst displacement `
      + `${(after.worstDisplacement * 1000).toFixed(0)} pm — `
      + `${(after.worstSkinUsage * 100).toFixed(0)}% of the budget used; `
      + `rebuild every ${after.rebuildInterval} steps `
      + `(asked ${after.requestedRebuildInterval})`);
    if (after.skinViolations > 0) {
      log(`  ⚠ the neighbour list was stale for ${after.stepsWithStaleList} of `
        + `${done} steps, over ${after.skinViolations} block(s). Some pair may `
        + `have been inside the cutoff and not counted. A pair that close is `
        + `worth at most ${after.missedPairForceBound.toFixed(2)} kJ/mol/nm `
        + `against an rms of a few hundred, so this costs exactness rather than `
        + `stability — but the run is not exact. The rebuild interval was `
        + `shortened to ${after.rebuildInterval}; a larger box or a shorter `
        + `cutoff would give the skin room to grow instead.`);
    }
    log(`  device memory: peak ${(after.allocator.peakBytes / 2 ** 20).toFixed(1)} MiB`);

    const state = await simulation.readState();
    finalState = { positions: state.positions, velocities: state.velocities,
      time: simulation.step * options.timestep, box: loaded.coordinates.box };
    element("download-state").disabled = false;
    if (writer !== null && writer.frameCount > 0) {
      trajectory = writer.finish();
      element("download-dcd").disabled = false;
      log(`  trajectory: ${writer.frameCount} frames, `
        + `${(trajectory.length / 2 ** 20).toFixed(2)} MiB — ready to download`);
    }
  } finally {
    simulation.release();
  }
}

async function runOnHost() {
  const { system, method } = assembleSystem();
  const options = runOptions(method);
  const steps = numberOf("steps");
  // Unconditional: on a second Run the current state's velocities ARE supplied,
  // so this continues the trajectory rather than restarting it.
  hostState = startState(system, options);
  log("");
  log(`host reference: ${steps} steps — this is the oracle, not a simulator, `
    + `and it is roughly sixty times slower than the GPU path`);
  const interval = Math.max(1, Math.floor(steps / 20));
  let evaluation = potentialEnergy(system, hostState.positions, options);
  const began = performance.now();
  for (let step = 0; step < steps; step += 1) {
    stopSignal.throwIfStopped();
    langevinStep(hostState, system, evaluation.forces, options);
    evaluation = potentialEnergy(system, hostState.positions, options);
    langevinFinishStep(hostState, system, evaluation.forces, options);
    if ((step + 1) % interval === 0 || step === steps - 1) {
      const kinetic = instantaneousTemperature(system, hostState.velocities);
      const elapsed = (performance.now() - began) / 1000;
      report({
        step: hostState.step,
        time: hostState.step * options.timestep,
        potential: evaluation.energy.total,
        kinetic: kinetic.kineticEnergy,
        total: evaluation.energy.total + kinetic.kineticEnergy,
        temperature: kinetic.temperature,
      }, [["ms / step", ((elapsed / (step + 1)) * 1000).toFixed(1)]]);
      await yieldToBrowser();
    }
  }
  const elapsed = (performance.now() - began) / 1000;
  log(`  ${steps} steps in ${elapsed.toFixed(1)} s — ${((elapsed / steps) * 1000).toFixed(0)} ms/step`);
  finalState = { positions: hostState.positions, velocities: hostState.velocities,
    time: hostState.step * options.timestep, box: loaded.coordinates.box };
  element("download-state").disabled = false;
}

// --------------------------------------------------------------- downloads ----
element("download-dcd").addEventListener("click", () => {
  if (trajectory === null) return;
  download(`${loaded.name}.dcd`, trajectory, "application/octet-stream");
  log(`  wrote ${loaded.name}.dcd — open it with the .prmtop you loaded, in `
    + `cpptraj, MDAnalysis, VMD or PyMOL`);
});

element("download-state").addEventListener("click", () => {
  if (finalState === null) return;
  const text = writeRestart({
    positions: finalState.positions,
    velocities: finalState.velocities,
    box: finalState.box,
    time: finalState.time,
    title: `RainMD ${loaded.name}`,
  });
  download(`${loaded.name}.rst7`, text, "text/plain");
  log(`  wrote ${loaded.name}.rst7 — an AMBER restart, so this run can be `
    + `continued here, in the notebooks, or in AMBER`);
});

/** g/cm^3, so the number means something to a chemist. */
function densityOf(system, volume) {
  let mass = 0;
  for (const value of system.topology.masses) mass += value;
  return (mass * 1.66053906660e-24) / (volume * 1e-21);
}

/**
 * The atoms a preset shows, in index order.
 *
 * 🔴 ONE RESOLVER FOR THE VIEWER AND THE PDB. The download button sits beside
 * the view bar and writes what is on screen, so deriving the two atom sets
 * separately would eventually hand the reader a file that is not the thing they
 * were looking at. `viewer.setStructure` marks the same groups the same way;
 * this is that loop with the result kept as indices.
 */
function visibleIndices(preset) {
  const topology = loaded.topology;
  if (selections === null) return null;
  const flags = new Uint8Array(topology.atomCount);
  for (const name of preset.groups) {
    const group = selections.groups[name];
    if (group === undefined) continue;
    for (const atom of group) flags[atom] = 1;
  }
  const heavyOnly = preset.heavyOnly === true;
  const shown = [];
  for (let atom = 0; atom < flags.length; atom += 1) {
    if (flags[atom] === 0) continue;
    if (heavyOnly && topology.atomicNumbers[atom] === 1) continue;
    shown.push(atom);
  }
  return shown;
}

// ------------------------------------------------------------------ view ----
element("binder").addEventListener("change", applyBinderChoice);

for (const control of ["view-preset", "colour-by", "representation"]) {
  element(control).addEventListener("change", () => {
    if (loaded === null || viewer === null) return;
    // 🔴 THE STYLES BUFFER IS REBUILT AND THE POSITIONS ARE NOT. Switching the
    // view while a run is in flight must not touch the buffer the integrator is
    // writing, so this re-derives colour and visibility only - which is what
    // makes it safe to call mid-run, and why it needs no readback. It is handed
    // the topology rather than a rebuilt system for the same reason: a dropdown
    // must not re-run `buildSystem` over every exclusion in the box.
    applyView(viewer, { topology: loaded.topology },
      lastPositions ?? hostState?.positions ?? loaded.coordinates.positions);
    viewer.draw();
  });
}

element("download-pdb").addEventListener("click", () => {
  if (loaded === null) return;
  const preset = currentPreset();
  const indices = visibleIndices(preset);
  // The last frame the run read back, so the file is the structure on screen
  // rather than the one that was loaded.
  const positions = lastPositions ?? hostState?.positions
    ?? loaded.coordinates.positions;
  const text = writePdb({ topology: loaded.topology }, positions, {
    indices,
    box: loaded.coordinates.box,
    remarks: [
      `RainMD ${loaded.name}`,
      `view ${preset.label}: ${indices === null ? loaded.topology.atomCount : indices.length}`
      + ` of ${loaded.topology.atomCount} atoms`,
    ],
  });
  download(`${loaded.name}.pdb`, text, "chemical/x-pdb");
  log(`  wrote ${loaded.name}.pdb - ${indices === null ? loaded.topology.atomCount : indices.length}`
    + ` atoms, the view on screen, as the topology for the .dcd`);
});

element("download-metrics").addEventListener("click", () => {
  if (analysis === null) return;
  download(`${loaded.name}-metrics.csv`, analysis.metricsCsv(), "text/csv");
  log(`  wrote ${loaded.name}-metrics.csv - ${analysis.series.length} frames, `
    + `one column per panel`);
});

element("download-rmsf").addEventListener("click", () => {
  if (analysis === null) return;
  download(`${loaded.name}-rmsf.csv`, analysis.rmsfCsv(), "text/csv");
  log(`  wrote ${loaded.name}-rmsf.csv - per-residue fluctuation over `
    + `${analysis.rmsf.frames} frames`);
});

window.addEventListener("resize", () => { plot?.draw(); viewer?.draw(); });

/**
 * A debugging handle.
 *
 * 🔴 IT IS HERE SO THE BROWSER LANE CAN CHECK WHAT THE PAGE ACTUALLY DID, and
 * because a reader with the console open should be able to reach the same
 * objects. `tools/gpu-page.mjs --simulate` drives the controls and then reads
 * this to confirm the viewer validated and the right number of atoms reached
 * it - neither of which is visible from the DOM.
 */
window.rainmd = {
  get device() { return gpu?.device ?? null; },
  get viewer() { return viewer; },
  get loaded() { return loaded; },
  get trajectoryBytes() { return trajectory === null ? 0 : trajectory.length; },
  get selections() { return selections; },
  get analysis() { return analysis; },
};
