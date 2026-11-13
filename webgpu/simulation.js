/**
 * A GPU-resident simulation: forces and integration both on the device.
 *
 * 🔴 A BLOCK OF STEPS IS ENCODED INTO ONE COMMAND BUFFER, WHICH IS THE WHOLE
 * POINT. A fold runs its graph a few times; a simulation runs it a million.
 * Submitting one step at a time puts a host round trip between every step, and
 * at a million steps a 1 ms round trip is seventeen minutes of nothing. Nothing
 * in a step reads back to the host, so a hundred steps go in one submit - which
 * is also why the step counter the thermostat indexes lives in a storage buffer
 * rather than a uniform.
 *
 * 🔴 AND THE YIELD IS BETWEEN BLOCKS, NEVER INSIDE ONE. A browser tab that
 * never yields does not repaint and its Stop button does not respond; a tab
 * that yields per step is throttled to a crawl in the background. One real task
 * yield per block is both.
 */
import { GpuAllocator } from "./runtime/allocator.js";
import { readBuffer, DeferredValidation } from "./runtime/execution.js";
import { yieldToBrowser } from "./runtime/yield.js";
import { packParams } from "./kernels/wgsl-common.js";
import { encodeBonded } from "./kernels/bonded.js";
import { encodeCellList } from "./kernels/cell-list.js";
import { encodeNeighbours, NeighbourOverflowError } from "./kernels/neighbours.js";
import { erfc } from "../shared/erf.js";
import { encodeNonbonded } from "./kernels/nonbonded.js";
import { encodeDisplacement } from "./kernels/displacement.js";
import { encodeRestraints } from "./kernels/restraints.js";
import { buildRestraints } from "../shared/topology/restraints.js";
import { encodePme } from "./kernels/pme.js";
import {
  encodeLangevinStep, encodeKineticEnergy, packIntegrator, INTEGRATOR_FLOATS,
  encodeConstrainPositions, forceMagnitudeShader, maxReduceShader, displaceShader,
} from "./kernels/integrate.js";
import { reduceShader, REDUCE_WORKGROUP } from "./kernels/pme.js";
import { pipelineCacheFor } from "./runtime/pipeline-cache.js";
import { dispatchGrid } from "./runtime/execution.js";
import { buildBondedGather } from "../shared/topology/gather.js";
import {
  planCellList, grownPlan, skinFor, buildExclusionGather, buildExceptionGather,
} from "../shared/topology/cells.js";
import {
  buildClusterTables, constraintToleranceFloor,
} from "../shared/topology/clusters.js";
import { pmeParameters } from "../shared/pme/parameters.js";
import { reciprocalBoxVectors } from "../cpu/forces/geometry.js";
import { ewaldSelfEnergy } from "../cpu/forces/pme.js";
import {
  ONE_4PI_EPS0, BOLTZMANN, BAR_TO_KJ_PER_MOL_NM3,
} from "../shared/units.js";
import { dispersionCoefficients, dispersionEnergy } from "../shared/dispersion.js";
import { uniformAt } from "../shared/random.js";
import { NonFiniteStateError } from "../shared/finite.js";
import { encodeVolumeScale, metropolisWeight } from "./kernels/barostat.js";

/**
 * 🔴 RESOLVED LAZILY, NOT AT MODULE SCOPE. `GPUBufferUsage` is a global that
 * exists only once WebGPU is present - in a browser always, in Node only after
 * the Dawn binding's globals have been installed. Reading it at module scope
 * makes merely IMPORTING this file throw in any context that has not set
 * WebGPU up yet, which breaks a plain `import` smoke test and would break a
 * server-side render of the page.
 */
/**
 * How many steps the first block runs before anything is known.
 *
 * 🔴 SHORT, BECAUSE THE FIRST BLOCK IS THE ONE THAT CANNOT BE CHECKED IN
 * ADVANCE. Four steps of extra rebuilds is a cost paid once; a first block of
 * forty steps on a list that turns out to have been stale is a wrong answer
 * nobody sees.
 */
const CALIBRATION_INTERVAL = 4;

function usages() {
  const storage = GPUBufferUsage.STORAGE;
  return {
    STORAGE: storage,
    READABLE: storage | GPUBufferUsage.COPY_SRC,
    WRITABLE: storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  };
}

export class GpuSimulation {
  constructor(device, system, options = {}) {
    this.device = device;
    this.system = system;
    this.allocator = new GpuAllocator(device);
    this.held = [];
    this.step = 0;
    this.stepsSinceRebuild = Infinity;

    const topology = system.topology;
    this.atomCount = topology.atomCount;
    this.timestep = options.timestep ?? 0.002;
    this.temperature = options.temperature ?? 298;
    this.friction = options.friction ?? 1.0;
    this.seed = options.seed ?? 0;
    // 🔴 CLAMPED TO WHAT f32 CAN REACH, AND THE CLAMP IS REPORTED. A caller
    // asking for 1e-6 on a 2.8 nm box is asking for something below the
    // arithmetic's floor; honouring it literally makes every cluster run to the
    // iteration cap and sets the failure flag on a correct result.
    this.requestedTolerance = options.tolerance ?? 1e-5;
    this.toleranceFloor = 0;
    this.tolerance = this.requestedTolerance;
    this.maxConstraintIterations = options.maxConstraintIterations ?? 150;
    // 🔴 THE REQUESTED INTERVAL IS A CEILING, AND THE RUN STARTS BELOW IT. No
    // model predicts the maximum displacement well enough to pick the interval
    // up front: measured on the solvated fixture the worst displacement scales
    // as about n^0.25 over the useful range - slower than diffusive, because it
    // is a maximum over a thousand atoms and the fastest one's ballistic phase
    // is short - so a linear model overshoots by 3x and a diffusive one by 2x.
    // Starting at a short interval and growing only while the measurement says
    // there is room never puts a stale list under a step, which starting high
    // and shrinking necessarily does for one block.
    this.requestedRebuildInterval = options.rebuildInterval ?? 20;
    this.adaptiveRebuild = options.adaptiveRebuild !== false;
    this.rebuildInterval = this.adaptiveRebuild
      ? Math.min(this.requestedRebuildInterval, CALIBRATION_INTERVAL)
      : this.requestedRebuildInterval;
    this.rebuildGrowths = 0;
    this.calmBlocks = 0;
    // Skin accounting. `worstDisplacement` is over the whole run, not a block.
    this.worstDisplacement = 0;
    this.worstSkinRatio = 0;
    this.worstSkinUsage = 0;
    this.skinViolations = 0;
    this.stepsWithStaleList = 0;
    this.rebuildShrinks = 0;
    // Monte Carlo barostat: off unless a pressure is given.
    this.pressure = options.pressure ?? null;
    this.barostatInterval = options.barostatInterval ?? 25;
    this.volumeScale = null;
    this.barostatAttempts = 0;
    // 🔴 A HIGH-WATER MARK, BECAUSE `required` IS CLEARED AT EVERY REBUILD.
    // `#encodeForces` clears it inside the command buffer, so the value the
    // host reads is the LAST rebuild's requirement, not the block's worst - an
    // overflow at one rebuild had vanished from the diagnostics by the time
    // anyone looked. Measured: required 1519 of 1499 at V = 29.20 nm3, then
    // "ok" at 1482 one step later.
    // The step whose in-step kinetic sample `resources.kinetic` holds, or null.
    this.kineticSampledAtStep = null;
    this.worstNeighbourRequired = 0;
    this.neighbourGrowths = 0;
    this.barostatAccepted = 0;
    this.barostatWindow = { attempts: 0, accepted: 0 };
    this.boxVectors = Float64Array.from(system.boxVectors ?? []);
    this.dispersion = dispersionCoefficients(system);
    this.useDispersionCorrection = options.dispersionCorrection !== false;

    this.gather = buildBondedGather(system);
    this.moleculeOfAtom = Uint32Array.from(system.topology.moleculeOfAtom);
    this.exclusions = buildExclusionGather(system);
    this.exceptions = buildExceptionGather(system);
    this.clusters = buildClusterTables(system);
    // 🔴 THE SKIN IS DERIVED FROM THE RUN, NOT A CONSTANT - see `skinFor`. The
    // old `?? 0.1` was half what this system needs at 2 fs and a fifth of what
    // it needs at 4 fs, and nothing in the force path could tell.
    // 🔴 RESTRAINTS BELONG TO THE RUN, NOT TO THE SYSTEM. The equilibration
    // stage restrains and the production stage does not, over the same
    // topology, so `buildSystem` is the wrong place for them - the restraint
    // reference is a configuration, which a topology has no notion of.
    this.restraints = buildRestraints({
      force: options.restraintForce ?? 0,
      atoms: options.restraintAtoms ?? [],
      reference: options.restraintReference ?? [],
    });
    this.skin = options.skin ?? skinFor(system, {
      timestep: options.timestep, temperature: options.temperature,
    });
    // 🔴 THE CAPACITY CONTROLS ARE FORWARDED, WHICH THEY WERE NOT. A caller
    // could not raise `neighbourSlack` or pin `maxNeighbours` at all: this
    // called `planCellList(system, { skin })` and dropped both. That also made
    // a mid-run overflow impossible to provoke on purpose, so the detection
    // path below could not be tested - which is a large part of why it was
    // missing.
    this.plan = planCellList(system, {
      skin: this.skin,
      neighbourSlack: options.neighbourSlack,
      maxNeighbours: options.maxNeighbours,
    });
    this.pme = pmeParameters({
      boxVectors: system.boxVectors,
      cutoff: system.options.cutoff,
      tolerance: options.ewaldTolerance ?? 0.0005,
      alpha: options.alpha,
      grid: options.grid,
    });
    this.selfEnergy = ewaldSelfEnergy(system, this.pme);
    this.gridPoints = this.pme.grid.reduce((a, b) => a * b, 1);
    this.#allocate();
  }

  #keep(allocation) { this.held.push(allocation); return allocation; }

  #allocate() {
    const { allocator } = this;
    const { STORAGE, READABLE, WRITABLE } = usages();
    const atoms = this.atomCount;
    const topology = this.system.topology;
    const keep = (allocation) => this.#keep(allocation);

    const ljTable = new Float32Array(topology.typeCount * topology.typeCount * 2);
    for (let index = 0; index < topology.typeCount * topology.typeCount; index += 1) {
      ljTable[index * 2] = topology.ljA[index];
      ljTable[index * 2 + 1] = topology.ljB[index];
    }
    const moduli = new Float32Array(this.pme.grid.reduce((total, size) => total + size, 0));
    let cursor = 0;
    for (let d = 0; d < 3; d += 1) {
      moduli.set(Float32Array.from(this.pme.moduli[d]), cursor);
      cursor += this.pme.grid[d];
    }
    this.partialCount = Math.ceil(this.gridPoints / REDUCE_WORKGROUP);
    this.kineticPartialCount = Math.ceil(atoms / REDUCE_WORKGROUP);

    this.resources = {
      params: keep(allocator.upload("params",
        packParams(this.system.boxVectors, reciprocalBoxVectors(this.system.boxVectors), {
          cutoff: this.system.options.cutoff,
          listCutoff: this.plan.listCutoff,
          alpha: this.pme.alpha,
          coulomb: ONE_4PI_EPS0,
        }), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)),
      integrator: keep(allocator.allocate("integrator", INTEGRATOR_FLOATS * 4,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)),
      positions: keep(allocator.allocate("positions", atoms * 12, WRITABLE)),
      velocities: keep(allocator.allocate("velocities", atoms * 12, WRITABLE)),
      reference: keep(allocator.allocate("reference", atoms * 12, STORAGE)),
      // 🔴 NOT `reference`, WHICH IS THE CONSTRAINT SOLVER'S AND IS REWRITTEN
      // EVERY STEP. This one holds the positions as they were when the
      // neighbour list was last built, which is a different lifetime entirely.
      rebuildOrigin: keep(allocator.allocate("rebuildOrigin", atoms * 12,
        STORAGE | GPUBufferUsage.COPY_DST)),
      displacement: keep(allocator.allocate("displacement", 8, READABLE)),
      // 🔴 THE CONSTRAINT CORRECTION, SO THE VELOCITY UPDATE NEVER DIFFERENCES
      // TWO BOX-SPANNING COORDINATES. See the note in the shake shader: the old
      // form was algebraically right and badly conditioned, and the error it
      // injected grew as the box did.
      corrections: keep(allocator.allocate("corrections",
        atoms * 12, STORAGE)),
      // 🔴 ALLOCATED EVEN WHEN THERE ARE NO RESTRAINTS, at one slot. A zero
      // length buffer is invalid in WebGPU, and making the whole resource table
      // conditional would put a branch in every bind group that reads it.
      restraintAtoms: keep(allocator.upload("restraintAtoms",
        this.restraints.atoms.length > 0
          ? Uint32Array.from(this.restraints.atoms) : new Uint32Array(1),
        STORAGE)),
      restraintReference: keep(allocator.upload("restraintReference",
        this.restraints.reference.length > 0
          ? Float32Array.from(this.restraints.reference) : new Float32Array(3),
        STORAGE)),
      restraintEnergies: keep(allocator.allocate("restraintEnergies",
        Math.max(this.restraints.atoms.length, 1) * 4, READABLE)),
      restraintSettings: keep(allocator.upload("restraintSettings",
        Float32Array.from([this.restraints.force, 0, 0, 0]),
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)),
      forces: keep(allocator.allocate("forces", atoms * 12, READABLE)),
      masses: keep(allocator.upload("masses", Float32Array.from(topology.masses), STORAGE)),
      charges: keep(allocator.upload("charges", Float32Array.from(topology.charges), STORAGE)),
      ljTypes: keep(allocator.upload("ljTypes", Uint32Array.from(topology.atomTypes), STORAGE)),
      ljTable: keep(allocator.upload("ljTable", ljTable, STORAGE)),
      // bonded
      termIndices: keep(allocator.upload("termIndices", this.gather.termIndices, STORAGE)),
      termParams: keep(allocator.upload("termParams",
        Float32Array.from(this.gather.termParams), STORAGE)),
      gatherStart: keep(allocator.upload("gatherStart", this.gather.start, STORAGE)),
      gatherList: keep(allocator.upload("gatherList", this.gather.list, STORAGE)),
      energies: keep(allocator.allocate("bondedEnergies", atoms * 4, READABLE)),
      nonbondedEnergies: keep(allocator.allocate("nonbondedEnergies", atoms * 8, READABLE)),
      // cells and neighbours
      cellOfAtom: keep(allocator.allocate("cellOfAtom", atoms * 4, STORAGE)),
      cellCount: keep(allocator.allocate("cellCount", this.plan.cellCount * 4, STORAGE)),
      cellStart: keep(allocator.allocate("cellStart", (this.plan.cellCount + 1) * 4, STORAGE)),
      cellAtoms: keep(allocator.allocate("cellAtoms", atoms * 4, STORAGE)),
      exclusionStart: keep(allocator.upload("exclusionStart", this.exclusions.start, STORAGE)),
      exclusionPartners: keep(allocator.upload("exclusionPartners",
        this.exclusions.partners, STORAGE)),
      neighbours: keep(allocator.allocate("neighbours",
        atoms * this.plan.maxNeighbours * 4, STORAGE)),
      neighbourCount: keep(allocator.allocate("neighbourCount", atoms * 4, STORAGE)),
      required: keep(allocator.allocate("required", 4, READABLE)),
      exceptionStart: keep(allocator.upload("exceptionStart", this.exceptions.start, STORAGE)),
      exceptionPartners: keep(allocator.upload("exceptionPartners",
        this.exceptions.partners, STORAGE)),
      exceptionParams: keep(allocator.upload("exceptionParams",
        this.exceptions.parameters, STORAGE)),
      // PME
      moduli: keep(allocator.upload("moduli", moduli, STORAGE)),
      fixedGrid: keep(allocator.allocate("fixedGrid", this.gridPoints * 8, STORAGE)),
      gridA: keep(allocator.allocate("gridA", this.gridPoints * 8, STORAGE)),
      gridB: keep(allocator.allocate("gridB", this.gridPoints * 8, STORAGE)),
      gridEnergy: keep(allocator.allocate("gridEnergy", this.gridPoints * 4, STORAGE)),
      energyPartials: keep(allocator.allocate("energyPartials", this.partialCount * 4, READABLE)),
      // integrator
      clusterData: keep(allocator.upload("clusterData",
        this.clusters.clusterData.length > 0 ? this.clusters.clusterData : new Uint32Array(1),
        STORAGE)),
      constraintAtoms: keep(allocator.upload("constraintAtoms",
        this.clusters.constraintAtoms.length > 0
          ? this.clusters.constraintAtoms : new Uint32Array(1), STORAGE)),
      constraintParams: keep(allocator.upload("constraintParams",
        this.clusters.constraintParams.length > 0
          ? this.clusters.constraintParams : new Float32Array(4), STORAGE)),
      diagnostics: keep(allocator.allocate("diagnostics", 16, WRITABLE)),
      stepCounter: keep(allocator.allocate("stepCounter", 8, WRITABLE)),
      // 🔴 ALLOCATED EVEN IN PRODUCTION, where the thermostat generates its own
      // noise and never reads it. A binding that exists but is unread costs one
      // descriptor; a conditional bind group costs a second pipeline.
      noise: keep(allocator.allocate("noise", atoms * 12, WRITABLE)),
      kinetic: keep(allocator.allocate("kinetic", atoms * 4, STORAGE)),
      // 🔴 A SECOND KINETIC ARRAY, BECAUSE THE TWO SAMPLES ANSWER DIFFERENT
      // QUESTIONS. `kinetic` holds the post-step velocity, which is what the
      // conserved total energy needs; `kineticEarly` holds the sample taken
      // inside the step before the closing half kick, which is where OpenMM
      // reports its temperature. Pairing the early velocity with the
      // end-of-step position inflated the measured NVE fluctuation from 6.36 to
      // 15.51 kJ/mol, so the total must not use it.
      kineticEarly: keep(allocator.allocate("kineticEarly", atoms * 4, STORAGE)),
      kineticPartials: keep(allocator.allocate("kineticPartials",
        this.kineticPartialCount * 4, READABLE)),
      kineticEarlyPartials: keep(allocator.allocate("kineticEarlyPartials",
        this.kineticPartialCount * 4, READABLE)),
      // barostat
      moleculeStart: keep(allocator.upload("moleculeStart",
        Uint32Array.from(this.system.molecules.start), STORAGE)),
      moleculeAtoms: keep(allocator.upload("moleculeAtoms",
        Uint32Array.from(this.system.molecules.atoms), STORAGE)),
      moleculeOfAtom: keep(allocator.upload("moleculeOfAtom", this.moleculeOfAtom, STORAGE)),
      centroids: keep(allocator.allocate("centroids",
        Math.max(1, this.system.molecules.count) * 12, STORAGE)),
      scaling: keep(allocator.allocate("scaling", 16,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)),
      savedPositions: keep(allocator.allocate("savedPositions", atoms * 12, WRITABLE)),
      // minimiser
      stepping: keep(allocator.allocate("stepping", 16,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)),
      magnitudes: keep(allocator.allocate("magnitudes", atoms * 4, STORAGE)),
      magnitudePartials: keep(allocator.allocate("magnitudePartials",
        this.kineticPartialCount * 4, READABLE)),
    };
    this.#writeIntegrator();
  }

  #writeIntegrator() {
    const decay = Math.exp(-this.friction * this.timestep);
    this.allocator.write(this.resources.integrator, packIntegrator({
      halfStep: 0.5 * this.timestep,
      decay,
      noiseAmplitude: Math.sqrt(Math.max(0, 1 - decay * decay)),
      kT: BOLTZMANN * this.temperature,
      tolerance: this.tolerance,
      seed: this.seed,
      maxIterations: this.maxConstraintIterations,
    }));
  }

  /** Set the state, and the noise mode (0 = generate, 1 = read the buffer). */
  setState(positions, velocities, options = {}) {
    this.toleranceFloor = constraintToleranceFloor(this.system, positions);
    this.tolerance = Math.max(this.requestedTolerance, this.toleranceFloor);
    this.#writeIntegrator();
    this.allocator.write(this.resources.positions, Float32Array.from(positions));
    this.allocator.write(this.resources.velocities,
      Float32Array.from(velocities ?? new Float64Array(positions.length)));
    this.allocator.write(this.resources.stepCounter,
      Uint32Array.from([options.step ?? 0, options.suppliedNoise === true ? 1 : 0]));
    this.allocator.write(this.resources.diagnostics, new Uint32Array(4));
    this.step = options.step ?? 0;
    this.stepsSinceRebuild = Infinity;
  }

  setNoise(noise) { this.allocator.write(this.resources.noise, Float32Array.from(noise)); }

  #encodeForces(encoder, rebuild) {
    const { device, resources, atomCount } = this;
    encoder.clearBuffer(resources.forces.buffer);
    encodeBonded(device, encoder, resources, atomCount, this.gather.layout);
    if (rebuild) {
      encoder.clearBuffer(resources.required.buffer);
      encodeCellList(device, encoder, resources, atomCount, this.plan);
      encodeNeighbours(device, encoder, resources, atomCount, this.plan);
      // 🔴 THE ORIGIN IS THE POSITIONS THE LIST WAS BUILT FROM, COPIED HERE AND
      // NOWHERE ELSE. Seeding it anywhere but beside the rebuild lets the two
      // drift apart, and a displacement measured from the wrong origin is
      // worse than no displacement at all - it reads as safe.
      encoder.copyBufferToBuffer(resources.positions.buffer, 0,
        resources.rebuildOrigin.buffer, 0, atomCount * 12);
      // 🔴 THE CLEAR IS FOR THE CONTROLLER, NOT FOR THE DIAGNOSTIC, and that
      // distinction was measured by removing it: `worstDisplacement` barely
      // moved, because the host already keeps a running maximum and each
      // interval's displacement is similar to the last. What breaks is growth -
      // an uncleared buffer never shows a calm block again, so the interval can
      // only ever shrink. A mutation that leaves a number almost right while
      // disabling the mechanism it feeds is the kind a differential test on the
      // number alone cannot see.
      // 🔴 ONLY WORD 0, WHICH IS THE SINCE-THE-REBUILD SLOT. Clearing the
      // whole buffer here is what made the host's measurement cover only the
      // steps since the last rebuild in the block: word 1 is the block's
      // high-water mark and must survive every rebuild inside it.
      encoder.clearBuffer(resources.displacement.buffer, 0, 4);
    }
    // Every step, because the maximum over the whole rebuild period is what the
    // skin has to cover, and an atom can move out and back inside one block.
    encodeDisplacement(device, encoder, resources, atomCount);
    encodeRestraints(device, encoder, resources, this.restraints.atoms.length);
    encodeNonbonded(device, encoder,
      { ...resources, energies: resources.nonbondedEnergies },
      atomCount, this.system.topology.typeCount, this.plan);
    encodePme(device, encoder, resources, atomCount, this.pme.grid);
  }

  /** One force evaluation, with a fresh neighbour list. Call before stepping. */
  async initialise() {
    const validation = new DeferredValidation(this.device, "initialise");
    validation.begin();
    const encoder = this.device.createCommandEncoder({ label: "rainmd.initialise" });
    this.#encodeForces(encoder, true);
    this.device.queue.submit([encoder.finish()]);
    validation.end("initial forces");
    await validation.settle();
    this.stepsSinceRebuild = 0;
    await this.#checkNeighbourCapacity();
  }

  async #checkNeighbourCapacity() {
    const required = (await readBuffer(this.device, this.allocator,
      this.resources.required, 4, Uint32Array))[0];
    if (required <= this.plan.maxNeighbours) return required;
    // Grow and rebuild: the capacity is a density estimate and the kernel
    // reported what was actually needed.
    const grown = grownPlan(this.plan, required);
    this.resources.neighbours.release();
    this.held.splice(this.held.indexOf(this.resources.neighbours), 1);
    this.plan = grown;
    this.resources.neighbours = this.#keep(this.allocator.allocate("neighbours",
      this.atomCount * grown.maxNeighbours * 4, usages().STORAGE));
    await this.initialise();
    const again = (await readBuffer(this.device, this.allocator,
      this.resources.required, 4, Uint32Array))[0];
    if (again > this.plan.maxNeighbours) {
      throw new NeighbourOverflowError(again, this.plan.maxNeighbours);
    }
    return again;
  }

  /**
   * Advance `count` steps, in blocks of `blockSize` submits.
   *
   * 🔴 THE NEIGHBOUR LIST IS REBUILT ON A SCHEDULE AND VERIFIED AFTERWARDS,
   * NOT BEFORE. Checking the maximum displacement against the skin before each
   * step would need a readback per step, which is the round trip this design
   * exists to avoid. Rebuilding every `rebuildInterval` steps and reading the
   * reported requirement at the end of a block catches a capacity problem
   * within one block instead of within one step - and the skin is what makes
   * that safe.
   */
  async run(count, options = {}) {
    const blockSize = options.blockSize ?? this.rebuildInterval;
    let remaining = count;
    this.stepsSinceBarostat = this.stepsSinceBarostat ?? 0;
    while (remaining > 0) {
      options.stopSignal?.throwIfStopped?.();
      // 🔴 THE VOLUME MOVE HAPPENS AT A BLOCK BOUNDARY, NOT MID-BLOCK. It needs
      // the trial energy back on the host, so it cannot sit inside a command
      // buffer of encoded steps - and the block size is the rebuild interval,
      // which is already a natural synchronisation point.
      if (this.pressure !== null && this.stepsSinceBarostat >= this.barostatInterval) {
        this.lastVolumeMove = await this.attemptVolumeChange();
        this.stepsSinceBarostat = 0;
      }
      const steps = Math.min(blockSize, remaining);
      const validation = new DeferredValidation(this.device, "run");
      validation.begin();
      const encoder = this.device.createCommandEncoder({ label: `rainmd.run.${steps}` });
      // 🔴 THE BLOCK'S HIGH-WATER SLOT IS CLEARED ONCE, HERE. Word 0 is reset at
      // every rebuild inside the block below; word 1 survives them and so
      // records the worst per-interval displacement anywhere in this block,
      // which is the quantity the host is entitled to report. Clearing it here
      // rather than after the read keeps it inside the command buffer.
      encoder.clearBuffer(this.resources.displacement.buffer, 4, 4);
      for (let index = 0; index < steps; index += 1) {
        const rebuild = this.stepsSinceRebuild >= this.rebuildInterval;
        if (rebuild) this.stepsSinceRebuild = 0;
        // 🔴 ONLY ON THE LAST STEP OF THE BLOCK, BECAUSE THAT IS THE ONE THE
        // HOST REPORTS. The kinetic sample has to be taken inside the step -
        // see the note in integrate.js - and taking it on every step would pay
        // for a reduction the host never reads. `energy()` below checks that
        // the sample it finds belongs to the current step and falls back to a
        // post-step one if not, so the number is never of mixed provenance.
        const lastOfBlock = index === steps - 1;
        encodeLangevinStep(this.device, encoder, this.resources, this.atomCount,
          this.clusters.layout, () => this.#encodeForces(encoder, rebuild),
          { sampleKinetic: lastOfBlock });
        if (lastOfBlock) this.kineticSampledAtStep = this.step + 1;
        this.stepsSinceRebuild += 1;
        this.step += 1;
      }
      this.device.queue.submit([encoder.finish()]);
      validation.end(`block of ${steps}`);
      await validation.settle();
      // The block boundary already synchronises for validation, so reading two
      // u32s here costs nothing a step pays for.
      await this.#checkSkin(steps);
      // 🔴 AND CHECK THE NEIGHBOUR CAPACITY, WHICH ONLY `initialise()` DID.
      // `grownPlan` and `NeighbourOverflowError` both existed and were
      // unreachable from `run()`, so an overflow mid-run dropped pairs at
      // ARBITRARY distances - whatever the cell stencil enumerated last, not
      // pairs at the cutoff - and `missedPairForceBound` does not bound those.
      // Measured against an f64 all-pairs host loop at identical coordinates,
      // a 973-of-840 overflow cost 43.56 kJ/mol of LJ, 42.12 of Coulomb and a
      // worst force component of 2803 kJ/mol/nm, with rms force error 9.8e-3
      // against 1.1e-6 for the control. `run()` threw nothing.
      //
      // The readback is one u32 at a boundary that already synchronises for
      // validation, so it costs what `#checkSkin` costs: nothing a step pays
      // for. On overflow this grows and rebuilds, which is what the init path
      // has always done - and counts it, because the block that overflowed
      // computed its forces from a truncated list and was not exact.
      const neededNow = await this.#checkNeighbourCapacity();
      if (neededNow > this.worstNeighbourRequired) {
        this.worstNeighbourRequired = neededNow;
      }
      remaining -= steps;
      this.stepsSinceBarostat += steps;
      await yieldToBrowser();
    }
    return this.step;
  }

  /**
   * Was the neighbour list still valid over the block that just ran?
   *
   * 🔴 THE ANSWER CAN ONLY BE AFTER THE FACT, AND SAYING SO IS THE POINT. The
   * decision to rebuild is baked into the command buffer as it is encoded, so
   * nothing can be known about a step's displacement until the block holding it
   * has run - that is the price of not reading back per step, and it is the
   * right price. What is NOT acceptable is the previous behaviour, where the
   * question was never asked: the skin was assumed to cover the motion, and if
   * it did not, the forces were quietly missing pairs.
   *
   * So: measure, shrink the interval so it cannot continue, force a rebuild
   * before the next step, and COUNT the violations so a run that was ever
   * invalid says so in its own diagnostics rather than looking clean.
   */
  async #checkSkin(steps) {
    // 🔴 SLOT 1, THE BLOCK MAXIMUM - NOT SLOT 0, WHICH IS ONLY THE STEPS SINCE
    // THE LAST REBUILD IN THIS BLOCK. Reading slot 0 understated every
    // user-facing staleness number by 2.3x to 5.2x purely by the choice of
    // submit granularity. Controlled measurement, the same 100 steps with a
    // rebuild every 4 and only the submit boundary moved - final position
    // checksum identical to 13 digits in all three, so nothing about the
    // physics changed:
    //
    //   blockSize   4 : 25 violations, worst usage 47.23, bound 5.649
    //   blockSize  20 :  5 violations, worst usage 20.34, bound 1.346
    //   blockSize 100 :  1 violation , worst usage 16.37, bound 1.077
    //
    // And it got worse exactly when it mattered: after a violation the interval
    // halves, so the fraction of the block the host could see fell further.
    const words = await readBuffer(this.device, this.allocator,
      this.resources.displacement, 8, Float32Array);
    const squared = Math.max(words[1], words[0]);
    // 🔴 BEFORE THE ARITHMETIC, NOT AFTER IT. A non-finite position makes this
    // reduction non-finite, so this one `u32` is a free whole-state finiteness
    // check at every block - see shared/finite.js for why the bit pattern
    // always wins the `atomicMax`. It has to come first because `NaN` poisons
    // every line below it: `Math.sqrt(Math.max(NaN, 0))` is `NaN`, which leaves
    // `worstDisplacement` at 0 while failing `usage <= 1`, so a dead run
    // reported "worst displacement 0 pm - 0% of the budget used" and "stale for
    // 300 of 300 steps" as consecutive lines, and the rebuild interval was
    // driven to 1 so it also got 20x slower and read as a performance problem.
    if (!Number.isFinite(squared)) {
      throw new NonFiniteStateError({
        what: "the maximum atomic displacement over the last block",
        step: this.step,
        detail: `The block of ${steps} steps that ended at step ${this.step} is `
          + "the last one that may have been valid; everything written during "
          + "it should be discarded too.",
      });
    }
    const worst = Math.sqrt(Math.max(squared, 0));
    if (worst > this.worstDisplacement) this.worstDisplacement = worst;
    // A pair outside the list closes by at most the sum of two displacements.
    const usage = (2 * worst) / this.plan.skin;
    if (usage > this.worstSkinUsage) this.worstSkinUsage = usage;
    if (usage <= 1) {
      // 🔴 GROWTH IS GATED AT HALF THE SKIN, NOT AT THE SKIN. The next block
      // runs at the LONGER interval, so growing while the margin is thin would
      // be deciding on evidence that no longer applies. Half leaves room for
      // the growth factor even if displacement were linear in the interval,
      // which it is not - it is far slower - so this is doubly conservative.
      this.calmBlocks += 1;
      // 🔴 AND GROWTH WAITS FOR THREE CALM BLOCKS, BECAUSE THE SYSTEM ITSELF
      // MOVES. A strained start converts potential energy into kinetic, so the
      // temperature overshoots a few hundred steps in and every atom speeds up
      // at once - measured, the interval grew to 20 during the calm phase and
      // was then caught by that spike. Growing on one quiet block is reading a
      // transient as a trend.
      if (this.adaptiveRebuild && usage < 0.4 && this.calmBlocks >= 3
        && this.rebuildInterval < this.requestedRebuildInterval) {
        this.rebuildInterval = Math.min(this.requestedRebuildInterval,
          Math.max(this.rebuildInterval + 1, Math.floor(this.rebuildInterval * 1.25)));
        this.rebuildGrowths += 1;
        this.calmBlocks = 0;
      }
      return;
    }
    this.skinViolations += 1;
    this.calmBlocks = 0;
    this.stepsWithStaleList += steps;
    const ratio = (2 * worst) / this.plan.skin;
    if (ratio > this.worstSkinRatio) this.worstSkinRatio = ratio;
    // Halve the interval rather than divide by the ratio: displacement over n
    // steps is between sqrt(n) and n depending on how ballistic the motion is,
    // so a single derived jump would overshoot for one regime and undershoot
    // for the other. Halving converges in a few blocks either way.
    const shrunk = Math.max(1, Math.floor(this.rebuildInterval / 2));
    if (this.adaptiveRebuild && shrunk !== this.rebuildInterval) {
      this.rebuildInterval = shrunk;
      this.rebuildShrinks += 1;
    }
    // The list in hand was built too long ago; do not step on it again.
    this.stepsSinceRebuild = Infinity;
  }

  /**
   * The screened-Coulomb force at the deepest a missed pair could have reached,
   * for a unit charge product, in kJ/mol/nm.
   *
   * 🔴 AT THE DEPTH THE OVERSHOOT ALLOWS, NOT AT THE CUTOFF. If the worst block
   * consumed `1 + e` of the skin, then a pair that started just outside the
   * list could have closed by `skin * (1 + e)` and so could be `e * skin`
   * INSIDE the cutoff - not at it. That depth is what sets the magnitude of the
   * error, and evaluating at the cutoff instead would understate it by exactly
   * the amount that matters. With no overshoot this is the force at the cutoff,
   * which is the right answer there too: the list was exact.
   */
  #missedPairForce() {
    const rc = this.system.options.cutoff;
    if (!(rc > 0)) return 0;
    const overshoot = Math.max(0, this.worstSkinUsage - 1);
    // Never report a depth past the cutoff itself; a gross violation says
    // nothing useful about a bound and the violation count is what matters.
    const depth = Math.min(overshoot * this.plan.skin, rc * 0.5);
    const r = rc - depth;
    const alpha = this.pme.alpha;
    const x = alpha * r;
    // -d/dr [erfc(ar)/r] = erfc(ar)/r^2 + 2a exp(-a^2 r^2)/(sqrt(pi) r)
    const screened = erfc(x) / (r * r)
      + (2 * alpha * Math.exp(-x * x)) / (Math.sqrt(Math.PI) * r);
    return ONE_4PI_EPS0 * screened;
  }

  #volume() {
    if (this.boxVectors.length < 9) return 0;
    return Math.abs(this.boxVectors[0] * this.boxVectors[4] * this.boxVectors[8]);
  }

  #dispersion() {
    if (!this.useDispersionCorrection || this.boxVectors.length < 9) return 0;
    return dispersionEnergy(this.dispersion, this.#volume(), this.system.options.cutoff);
  }

  /**
   * Adopt a new box: rewrite the uniform, re-derive what depends on volume, and
   * re-plan the cell list if its geometry changed.
   *
   * 🔴 THE CELL COUNT CAN CHANGE AND THE SHADER BAKES IT IN. Cell size is
   * `boxWidth / cellCount`, so an accepted volume move shrinks the cells - and
   * once one falls below the list cutoff, a one-deep stencil stops covering the
   * interaction sphere and pairs go missing silently. Re-planning recompiles a
   * kernel (cached, so only the first time at each geometry) and reallocates
   * the two per-cell buffers.
   *
   * 🔴 THE PME GRID IS DEPARTED FROM: IT STAYS FIXED. OpenMM keeps alpha and
   * the grid constant across a barostat's moves, so the reciprocal vectors in
   * the uniform are the only thing that changes - and the B-spline moduli,
   * which depend on the grid size alone, stay valid. Re-deriving the grid from
   * the new box would change the energy discontinuously at a volume move and
   * make the Metropolis test compare two different approximations.
   */
  async #updateBox(vectors) {
    this.boxVectors = Float64Array.from(vectors);
    // 🔴 THE CAPACITY IS RE-ESTIMATED FOR THE NEW BOX, NOT CARRIED FORWARD.
    // This passed `maxNeighbours: this.plan.maxNeighbours` - the estimate made
    // from the box at CONSTRUCTION - so a barostat compressing the system
    // raised the density while the neighbour capacity stayed where it started.
    // Measured on ala-wat-big, contracted in <=5% volume steps the way a
    // barostat goes: required crossed the 1499 capacity at V = 29.20 nm3, which
    // is 97.7 atoms/nm3 - exactly TIP3P density, where the shipped NPT
    // equilibration is documented to land. Taking the larger of the two keeps
    // the buffer from churning when the box expands again.
    const estimate = planCellList(this.system, {
      skin: this.plan.skin, boxVectors: this.boxVectors,
    });
    const capacity = Math.max(this.plan.maxNeighbours, estimate.maxNeighbours);
    const plan = planCellList(this.system, {
      skin: this.plan.skin, maxNeighbours: capacity,
      boxVectors: this.boxVectors,
    });
    if (capacity > this.plan.maxNeighbours) {
      this.resources.neighbours.release();
      this.held.splice(this.held.indexOf(this.resources.neighbours), 1);
      this.resources.neighbours = this.#keep(this.allocator.allocate("neighbours",
        this.atomCount * capacity * 4, usages().STORAGE));
      this.neighbourGrowths = (this.neighbourGrowths ?? 0) + 1;
    }
    const geometryChanged = plan.counts.join(",") !== this.plan.counts.join(",");
    if (geometryChanged) {
      for (const name of ["cellCount", "cellStart"]) {
        this.resources[name].release();
        this.held.splice(this.held.indexOf(this.resources[name]), 1);
      }
      const { STORAGE } = usages();
      this.resources.cellCount = this.#keep(this.allocator.allocate(
        "cellCount", plan.cellCount * 4, STORAGE));
      this.resources.cellStart = this.#keep(this.allocator.allocate(
        "cellStart", (plan.cellCount + 1) * 4, STORAGE));
    }
    this.plan = plan;
    this.selfEnergy = ewaldSelfEnergy(
      { ...this.system, boxVectors: this.boxVectors }, this.pme);
    this.allocator.write(this.resources.params,
      packParams(this.boxVectors, reciprocalBoxVectors(this.boxVectors), {
        cutoff: this.system.options.cutoff,
        listCutoff: this.plan.listCutoff,
        alpha: this.pme.alpha,
        coulomb: ONE_4PI_EPS0,
      }));
    this.stepsSinceRebuild = Infinity;
    return geometryChanged;
  }

  /** A full force evaluation with a fresh neighbour list, and its potential. */
  async #evaluatePotential() {
    const validation = new DeferredValidation(this.device, "barostat");
    validation.begin();
    const encoder = this.device.createCommandEncoder({ label: "rainmd.barostat.forces" });
    encoder.clearBuffer(this.resources.required.buffer);
    this.#encodeForces(encoder, true);
    this.device.queue.submit([encoder.finish()]);
    validation.end("trial forces");
    await validation.settle();
    this.stepsSinceRebuild = 0;

    const atomCount = this.atomCount;
    const bonded = await readBuffer(this.device, this.allocator, this.resources.energies,
      atomCount * 4, Float32Array);
    const nonbonded = await readBuffer(this.device, this.allocator,
      this.resources.nonbondedEnergies, atomCount * 8, Float32Array);
    const partials = await readBuffer(this.device, this.allocator,
      this.resources.energyPartials, this.partialCount * 4, Float32Array);
    let total = this.selfEnergy.total + this.#dispersion();
    for (let atom = 0; atom < atomCount; atom += 1) {
      total += bonded[atom] + nonbonded[atom] + nonbonded[atomCount + atom];
    }
    for (const value of partials) total += value;

    // 🔴 THE POSITIONAL RESTRAINT BELONGS IN HERE, AND IT WAS MISSING FROM THE
    // ONE PLACE IT MATTERS. This function is what `attemptVolumeChange` feeds
    // to `metropolisWeight`, and it summed bonded + LJ + direct + reciprocal +
    // self + dispersion and never read `resources.restraintEnergies` -
    // while `energy()`, which only REPORTS, did. So through the whole shipped
    // NPT equilibration the reported potential was right and the accept/reject
    // decision was wrong. OpenMM's MonteCarloBarostat takes
    // `getState(State::Energy, ..., integrator.getIntegrationForceGroups())`,
    // which includes the notebooks' own `CustomExternalForce`.
    //
    // 🔴 AND IT IS A BIAS ON THE VOLUME, NOT A SYMMETRIC PENALTY. Centroid
    // scaling sends each displacement d0 -> d0 + (s-1)R, so the restraint
    // energy picks up 2k(s-1)*sum(d0.R) - linear in (s-1), so directional - as
    // well as k(s-1)^2*sum(|R|^2), which resists any move. Dropping both means
    // the barostat neither feels the restraint's resistance to changing the
    // volume nor its preferred direction. Measured at the page's own
    // equilibration settings the omitted term was 6.906 kJ/mol, about 2.8 kT,
    // against a whole Metropolis weight of -2.510.
    //
    // 🔴 `#encodeForces` ALREADY ENCODED THE RESTRAINT PASS, so the trial
    // FORCES carried it and only the energy sum did not - which is why the
    // trajectory looked right. Nothing in the output showed it.
    const restraintSlots = this.restraints.atoms.length;
    if (restraintSlots > 0) {
      const restraintEnergies = await readBuffer(this.device, this.allocator,
        this.resources.restraintEnergies, restraintSlots * 4, Float32Array);
      for (let slot = 0; slot < restraintSlots; slot += 1) {
        total += restraintEnergies[slot];
      }
    }
    return total;
  }

  /**
   * Scale molecular centroids and the box by a length factor, and return the
   * potential energy there.
   *
   * Separated from the Metropolis test so a differential check can force a
   * known factor: a barostat that proposes random moves cannot be compared
   * against an oracle, and the physics worth checking - the centroid scaling,
   * the box update, PME at a new cell and the volume-dependent dispersion term -
   * is all in here rather than in the accept/reject arithmetic.
   */
  async scaleVolume(lengthScale) {
    this.allocator.write(this.resources.scaling,
      Float32Array.from([lengthScale, 0, 0, 0]));
    const encoder = this.device.createCommandEncoder({ label: "rainmd.barostat.scale" });
    encodeVolumeScale(this.device, encoder, this.resources,
      this.atomCount, this.system.molecules.count);
    this.device.queue.submit([encoder.finish()]);
    await this.#updateBox(
      Float64Array.from(this.boxVectors, (value) => value * lengthScale));
    return this.#evaluatePotential();
  }

  /** The largest force on any atom, from the last force evaluation. */
  async #maxForce() {
    const { device, allocator, resources, atomCount } = this;
    const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
    const run = (key, code, bindings, items, workgroup) => {
      const workgroups = Math.ceil(items / workgroup);
      const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
      const gridWidth = x === workgroups ? workgroups : maxPerDimension;
      const pipeline = pipelineCacheFor(device).get(`${key}:${gridWidth}`, code(gridWidth));
      const pass = encoderRef.beginComputePass({ label: key });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((allocation, binding) => ({
          binding, resource: { buffer: allocation.buffer },
        })),
      }));
      pass.dispatchWorkgroups(x, y, z);
      pass.end();
    };
    const encoderRef = device.createCommandEncoder({ label: "rainmd.maxforce" });
    run(`minimise:magnitude:${atomCount}`,
      (gridWidth) => forceMagnitudeShader(atomCount, gridWidth),
      [resources.forces, resources.magnitudes], atomCount, 64);
    run(`minimise:maxreduce:${atomCount}`,
      (gridWidth) => maxReduceShader(atomCount, gridWidth),
      [resources.magnitudes, resources.magnitudePartials], atomCount, REDUCE_WORKGROUP);
    device.queue.submit([encoderRef.finish()]);
    const partials = await readBuffer(device, allocator, resources.magnitudePartials,
      this.kineticPartialCount * 4, Float32Array);
    let worst = 0;
    for (const value of partials) worst = Math.max(worst, value);
    return worst;
  }

  /**
   * Steepest descent with a backtracking line search, on the device.
   *
   * 🔴 ONE ROUND TRIP PER TRIAL, WHICH IS AFFORDABLE HERE AND NOWHERE ELSE. A
   * line search is a decision about whether the energy went down, so the number
   * has to come back to the host - but minimisation is hundreds of iterations
   * rather than millions of steps, so a millisecond each is a fraction of a
   * second in total. The step loop keeps its no-readback rule precisely because
   * it cannot afford what this can.
   *
   * 🔴 AND IT IS STILL NOT L-BFGS. OpenMM's minimizeEnergy converges in fewer
   * iterations, especially on a bad structure. This is twenty lines that cannot
   * be subtly wrong, and the number that matters for starting dynamics is the
   * MAXIMUM force rather than the energy - one clashing pair is what blows up
   * the first step.
   *
   * 🔴 IT STALLS BEFORE A TIGHT FORCE TOLERANCE, AND THAT IS f32 RATHER THAN
   * THE ALGORITHM. The line search decides by comparing two potential
   * energies, and the potential is a sum of ~1190 f32 per-atom contributions
   * whose total is of order 13000 kJ/mol - so its resolution is about
   * `13000 * 1.2e-7 = 1.6e-3` kJ/mol at best, and in practice a few hundredths
   * once the summation is accounted for. Once a step would improve the energy
   * by less than that, "did it go down" is a coin flip and backtracking
   * collapses. Measured on the solvated fixture: it reaches a potential of
   * -13450 kJ/mol where the f64 host minimiser reaches -13408 in forty
   * iterations, and then stalls at a max force near 2100 kJ/mol/nm.
   *
   * That is good enough for its purpose - the structure it leaves runs stably,
   * and the unminimised one at 4029 kJ/mol/nm also did - but a tighter
   * minimisation needs the f64 host path, and `stalled` in the result says
   * which happened rather than leaving a caller to infer it from the force.
   */
  async minimise(options = {}) {
    const maxIterations = options.maxIterations ?? 200;
    const forceTolerance = options.forceTolerance ?? 500;
    // 🔴 THE STEP IS A DISPLACEMENT, NOT A FORCE SCALE, AND THE DIFFERENCE
    // BROKE THE LINE SEARCH. A fixed force scale with a per-atom displacement
    // clamp moves every atom above the clamp by exactly the same distance
    // regardless of its force - so the step is no longer along the gradient and
    // the energy is not guaranteed to decrease for a small enough step, which
    // is the one property backtracking relies on. Measured: the search failed
    // after 11 iterations with 21 rejected trials and a max force of 2725.
    // Sizing the step so the LARGEST force moves exactly `maxStep` keeps the
    // direction exact; the clamp stays only as a cap against a single
    // pathological pair.
    let maxStep = options.maxStep ?? 0.002;
    const displacementCap = options.maxDisplacement ?? 0.05;
    const bytes = this.atomCount * 12;
    let potential = await this.#evaluatePotential();
    const history = [];
    let iterations = 0;
    let rejected = 0;
    let stalled = false;

    for (let iteration = 0; iteration < maxIterations; iteration += 1) {
      iterations = iteration + 1;
      const worst = await this.#maxForce();
      if (iteration === 0 || iteration % 20 === 0) {
        history.push({ iteration, potential, maxForce: worst });
      }
      if (worst <= forceTolerance) break;

      // Keep both the pre-move positions (for the rollback) and a copy for
      // SHAKE to project against.
      const save = this.device.createCommandEncoder({ label: "rainmd.minimise.save" });
      save.copyBufferToBuffer(this.resources.positions.buffer, 0,
        this.resources.savedPositions.buffer, 0, bytes);
      save.copyBufferToBuffer(this.resources.positions.buffer, 0,
        this.resources.reference.buffer, 0, bytes);
      this.device.queue.submit([save.finish()]);

      let accepted = false;
      for (let attempt = 0; attempt < 14; attempt += 1) {
        this.allocator.write(this.resources.stepping,
          Float32Array.from([maxStep / Math.max(worst, 1e-6), displacementCap, 0, 0]));
        const encoder = this.device.createCommandEncoder({ label: "rainmd.minimise.step" });
        const maxPerDimension = this.device.limits.maxComputeWorkgroupsPerDimension;
        const workgroups = Math.ceil(this.atomCount / 64);
        const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
        const gridWidth = x === workgroups ? workgroups : maxPerDimension;
        const key = `minimise:displace:${this.atomCount}:${gridWidth}`;
        const pipeline = pipelineCacheFor(this.device).get(key,
          displaceShader(this.atomCount, gridWidth));
        const pass = encoder.beginComputePass({ label: key });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [this.resources.stepping, this.resources.positions, this.resources.forces]
            .map((allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
        }));
        pass.dispatchWorkgroups(x, y, z);
        pass.end();
        encodeConstrainPositions(this.device, encoder, this.resources,
          this.atomCount, this.clusters.layout);
        this.device.queue.submit([encoder.finish()]);

        const trial = await this.#evaluatePotential();
        if (trial < potential) {
          potential = trial;
          maxStep *= 1.3;
          accepted = true;
          break;
        }
        rejected += 1;
        // Put the pre-move positions back before trying a smaller step: the
        // displacement is applied IN PLACE, so a rejected trial must be undone
        // or the next attempt steps from the rejected geometry.
        const restore = this.device.createCommandEncoder({ label: "rainmd.minimise.restore" });
        restore.copyBufferToBuffer(this.resources.savedPositions.buffer, 0,
          this.resources.positions.buffer, 0, bytes);
        this.device.queue.submit([restore.finish()]);
        maxStep *= 0.5;
        if (maxStep < 1e-9) break;
      }
      if (!accepted) { stalled = true; break; }
    }
    const worst = await this.#maxForce();
    history.push({ iteration: iterations, potential, maxForce: worst });
    // The minimiser leaves stale velocities behind; zero them so a run from
    // here starts from the temperature it is asked for rather than from
    // whatever the last trajectory had.
    this.allocator.write(this.resources.velocities,
      new Float32Array(this.atomCount * 3));
    return {
      iterations, rejected, potential, maxForce: worst, maxStep, history,
      stalled,
      converged: worst <= forceTolerance,
      // The energy resolution that bounds the line search, so a caller can see
      // why it stopped rather than guessing.
      energyResolution: Math.abs(potential) * 2 ** -24 * Math.sqrt(this.atomCount),
    };
  }

  /**
   * One Monte Carlo volume move.
   *
   * 🔴 THIS IS THE ONE PLACE A STEP SYNCHRONISES WITH THE HOST, AND IT HAS TO.
   * A Metropolis test is a decision about a number, so the trial energy must
   * come back before the next step can be encoded. At the default interval of
   * 25 steps that is one round trip per 25, which is why the barostat is cheap
   * despite breaking the no-readback rule the rest of the loop keeps.
   *
   * 🔴 AND THE SCALE ADAPTS, WITH A CEILING. An acceptance rate far from a half
   * wastes the moves; OpenMM widens the attempted volume change when more than
   * three quarters are accepted and narrows it below a quarter, capped at 30% of
   * the volume so a single move can never invert the box.
   */
  async attemptVolumeChange() {
    if (this.pressure === null || this.boxVectors.length < 9) return null;
    const volume = this.#volume();
    if (this.volumeScale === null) this.volumeScale = 0.01 * volume;
    const kT = BOLTZMANN * this.temperature;
    const pressure = this.pressure * BAR_TO_KJ_PER_MOL_NM3;

    const oldPotential = await this.#evaluatePotential();
    const encoder = this.device.createCommandEncoder({ label: "rainmd.barostat.save" });
    encoder.copyBufferToBuffer(this.resources.positions.buffer, 0,
      this.resources.savedPositions.buffer, 0, this.atomCount * 12);
    this.device.queue.submit([encoder.finish()]);

    this.barostatAttempts += 1;
    this.barostatWindow.attempts += 1;
    const draw = uniformAt(this.seed ^ 0x5a17, this.barostatAttempts, 0, 0);
    const deltaVolume = this.volumeScale * 2 * (draw - 0.5);
    const newVolume = volume + deltaVolume;
    if (!(newVolume > 0)) return null;
    const lengthScale = Math.cbrt(newVolume / volume);

    // 🔴 A BAROSTAT CAN SHRINK THE BOX PAST THE CUTOFF, AND NOTHING DOWNSTREAM
    // WOULD SAY SO. buildSystem refuses a cutoff over half the smallest box
    // width at construction, but the barostat changes the box afterwards - and
    // once it is too small each pair interacts with more than one image of its
    // partner, which breaks the minimum-image convention silently. Observed
    // reaching a 2x2x1 cell grid on this deliberately underdense fixture, which
    // was well inside the limit but on the way towards it. A move that would
    // cross it is rejected like any other, so the run continues at the smallest
    // box the cutoff supports instead of failing or going quietly wrong.
    const smallestWidth = lengthScale * Math.min(
      Math.abs(this.boxVectors[0]), Math.abs(this.boxVectors[4]), Math.abs(this.boxVectors[8]));
    if (this.system.options.cutoff > 0.5 * smallestWidth) {
      this.barostatRefusals = (this.barostatRefusals ?? 0) + 1;
      return {
        accepted: false, refused: "the cutoff would not fit the proposed box",
        deltaVolume, lengthScale, volume,
        volumeScale: this.volumeScale,
        acceptanceRate: this.barostatAccepted / this.barostatAttempts,
      };
    }

    const newPotential = await this.scaleVolume(lengthScale);
    const weight = metropolisWeight({
      deltaEnergy: newPotential - oldPotential,
      deltaVolume,
      volumeRatio: newVolume / volume,
    }, pressure, kT, this.system.molecules.count);
    const accept = weight <= 0
      || uniformAt(this.seed ^ 0x5a17, this.barostatAttempts, 1, 0) < Math.exp(-weight / kT);

    if (accept) {
      this.barostatAccepted += 1;
      this.barostatWindow.accepted += 1;
    } else {
      const restore = this.device.createCommandEncoder({ label: "rainmd.barostat.restore" });
      restore.copyBufferToBuffer(this.resources.savedPositions.buffer, 0,
        this.resources.positions.buffer, 0, this.atomCount * 12);
      this.device.queue.submit([restore.finish()]);
      await this.#updateBox(Float64Array.from(this.boxVectors, (value) => value / lengthScale));
      await this.#evaluatePotential();
    }

    if (this.barostatWindow.attempts >= 10) {
      const rate = this.barostatWindow.accepted / this.barostatWindow.attempts;
      if (rate < 0.25) this.volumeScale /= 1.1;
      else if (rate > 0.75) this.volumeScale = Math.min(this.volumeScale * 1.1, 0.3 * volume);
      this.barostatWindow = { attempts: 0, accepted: 0 };
    }
    return {
      accepted: accept, weight, deltaVolume, lengthScale,
      oldPotential, newPotential, volume: this.#volume(),
      volumeScale: this.volumeScale,
      acceptanceRate: this.barostatAccepted / this.barostatAttempts,
    };
  }

  /** Potential breakdown, kinetic energy and temperature. */
  async energy() {
    const { device, allocator, resources, atomCount } = this;
    const encoder = device.createCommandEncoder({ label: "rainmd.energy" });
    // 🔴 THE IN-STEP SAMPLE IS USED WHEN IT BELONGS TO THIS STEP, AND A
    // POST-STEP ONE IS ENCODED WHEN IT DOES NOT - which happens at step 0, in
    // the minimiser, and either side of a barostat move. The two differ by a
    // half kick and so by an O(dt^2) bias of opposite sign, so which was used
    // is reported rather than left to be guessed.
    const inStep = this.kineticSampledAtStep === this.step;
    // The post-step sample is always taken: it is what the total energy needs.
    encodeKineticEnergy(device, encoder, resources, atomCount);
    const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
    const workgroups = Math.ceil(atomCount / REDUCE_WORKGROUP);
    const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
    const gridWidth = x === workgroups ? workgroups : maxPerDimension;
    const key = `reduce:kinetic:${atomCount}:${gridWidth}`;
    const pipeline = pipelineCacheFor(device).get(key, reduceShader(atomCount, gridWidth));
    const reduce = (source, into) => {
      const pass = encoder.beginComputePass({ label: key });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [source, into]
          .map((allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
      }));
      pass.dispatchWorkgroups(x, y, z);
      pass.end();
    };
    reduce(resources.kinetic, resources.kineticPartials);
    if (inStep) reduce(resources.kineticEarly, resources.kineticEarlyPartials);
    device.queue.submit([encoder.finish()]);

    // 🔴 THE PER-ATOM ENERGY ARRAYS ARE READ BACK WHOLE, WHICH IS A REPORTING
    // COST AND NOT A STEP COST. At a thousand atoms that is 14 KiB per report;
    // at a hundred thousand it is 1.2 MiB, which is why reports are infrequent
    // and why reducing these two on the device is on the list.
    const bonded = await readBuffer(device, allocator, resources.energies,
      atomCount * 4, Float32Array);
    const nonbonded = await readBuffer(device, allocator, resources.nonbondedEnergies,
      atomCount * 8, Float32Array);
    const reciprocalPartials = await readBuffer(device, allocator, resources.energyPartials,
      this.partialCount * 4, Float32Array);
    const kineticPartials = await readBuffer(device, allocator, resources.kineticPartials,
      this.kineticPartialCount * 4, Float32Array);
    const kineticEarlyPartials = inStep
      ? await readBuffer(device, allocator, resources.kineticEarlyPartials,
        this.kineticPartialCount * 4, Float32Array) : null;
    const restraintSlots = this.restraints.atoms.length;
    const restraintEnergies = restraintSlots === 0 ? null
      : await readBuffer(device, allocator, resources.restraintEnergies,
        restraintSlots * 4, Float32Array);

    let bondedTotal = 0;
    let lennardJones = 0;
    let coulombDirect = 0;
    for (let atom = 0; atom < atomCount; atom += 1) {
      bondedTotal += bonded[atom];
      lennardJones += nonbonded[atom];
      coulombDirect += nonbonded[atomCount + atom];
    }
    let reciprocal = 0;
    for (const value of reciprocalPartials) reciprocal += value;
    let kinetic = 0;
    for (const value of kineticPartials) kinetic += value;
    // 🔴 THE TEMPERATURE AND THE TOTAL ENERGY USE DIFFERENT SAMPLES, ON PURPOSE.
    // `kinetic` is the post-step velocity, which pairs with the end-of-step
    // positions and is what a conserved total needs - using the early sample
    // there inflated the measured NVE fluctuation from 6.36 to 15.51 kJ/mol,
    // a spurious half-step oscillation that reads as drift. The temperature
    // uses the sample taken inside the step, before the closing half kick,
    // which is where OpenMM reports: that moved the offset from a 298 K
    // setpoint from -0.02 / -2.08 / -6.36 K at 1 / 2 / 4 fs to
    // +0.46 / -0.33 / +0.88, against OpenMM's own +0.42 / +1.63 / +4.45.
    let kineticForTemperature = kinetic;
    let sampledAt = "after the step";
    if (kineticEarlyPartials !== null) {
      let early = 0;
      for (const value of kineticEarlyPartials) early += value;
      // 🔴 A GUARD, BECAUSE THE FLAG AND THE PASS ARE SET IN TWO PLACES. The
      // host marks the step it asked for an in-step sample and the encoder
      // emits the pass; if those ever disagree the buffer is read unwritten and
      // the temperature comes out as 0 K, which is how the first falsification
      // of this failed - it reported -298.00 K rather than falling back. A
      // kinetic energy of zero beside a non-zero post-step one cannot be
      // physical, so take the post-step value and say so instead.
      if (early > 0 || kinetic <= 0) {
        kineticForTemperature = early;
        sampledAt = "before the closing half kick";
      }
    }

    const coulomb = coulombDirect + reciprocal + this.selfEnergy.total;
    const dispersion = this.#dispersion();
    // 🔴 SUMMED ON THE HOST IN f64, over as many slots as there are restrained
    // atoms. The restrained set is the solute's heavy atoms - 2% of a solvated
    // box - so this is a few hundred adds at a reporting interval, not a
    // reduction worth a kernel.
    let restraintEnergy = 0;
    if (restraintEnergies !== null) {
      for (let slot = 0; slot < restraintSlots; slot += 1) {
        restraintEnergy += restraintEnergies[slot];
      }
    }
    const potential = bondedTotal + lennardJones + dispersion + coulomb
      + restraintEnergy;
    // 🔴 THE SECOND DETECTOR, AND IT IS NOT REDUNDANT. `#checkSkin` sees
    // positions; this sees the energies and the kinetic sum, so it catches a
    // velocity that went non-finite in the step before the positions followed
    // it, and it catches a first evaluation that was already infinite - two
    // atoms at the same coordinate give `Inf` before any step has run, where
    // there is no displacement to reduce yet.
    //
    // 🔴 DELIBERATELY NOT IN `#evaluatePotential`, which the minimiser's line
    // search calls on trial points. A line search must be allowed to evaluate a
    // bad step and reject it; throwing there would turn a working back-off into
    // a failed minimisation. That path is already safe by accident worth
    // stating: the accept test is `trial < potential`, and every comparison
    // against `NaN` is false, so a non-finite trial is always rejected.
    for (const [what, value] of [["the potential energy", potential],
      ["the kinetic energy", kinetic]]) {
      if (Number.isFinite(value)) continue;
      throw new NonFiniteStateError({
        what,
        step: this.step,
        detail: this.step === 0
          ? "This was the first force evaluation, before any step ran, so the "
            + "input itself is what cannot be evaluated - not the dynamics."
          : "Measured at a reporting interval, so the blow-up happened "
            + "somewhere in the block that ended here.",
      });
    }
    return {
      step: this.step,
      time: this.step * this.timestep,
      bonded: bondedTotal,
      lennardJones,
      dispersion,
      coulombDirect,
      coulombReciprocal: reciprocal,
      coulombSelf: this.selfEnergy.total,
      coulomb,
      potential,
      kinetic,
      total: potential + kinetic,
      positionRestraint: restraintEnergy,
      restrainedAtoms: restraintSlots,
      temperature: (2 * kineticForTemperature)
        / (this.system.degreesOfFreedom * BOLTZMANN),
      kineticForTemperature,
      // Which point in B A O A B the kinetic energy came from. OpenMM's
      // LangevinMiddleIntegrator reports at "before the closing half kick",
      // which is what a run reports; the other is what a one-off evaluation
      // outside a step can offer.
      kineticSampledAt: sampledAt,
      volume: this.#volume(),
      boxVectors: Float64Array.from(this.boxVectors),
    };
  }

  /**
   * Positions alone.
   *
   * 🔴 THE ANALYSIS NEEDS COORDINATES AND NOTHING ELSE, so reading the full
   * state would move three times the bytes at every report. On a 30000-atom
   * system that is 1.1 MiB a frame against 360 KiB.
   */
  async readPositions() {
    return readBuffer(this.device, this.allocator, this.resources.positions,
      this.atomCount * 12, Float32Array);
  }

  async readState() {
    const bytes = this.atomCount * 12;
    return {
      positions: await readBuffer(this.device, this.allocator,
        this.resources.positions, bytes, Float32Array),
      velocities: await readBuffer(this.device, this.allocator,
        this.resources.velocities, bytes, Float32Array),
      forces: await readBuffer(this.device, this.allocator,
        this.resources.forces, bytes, Float32Array),
      // 🔴 THE BOX IS PART OF THE STATE. A barostat changes it, so a caller
      // that reads positions and velocities and not the box has two thirds of
      // what it needs to continue - and will silently continue in the volume
      // the run STARTED from. OpenMM serialises all three into its .rst for
      // exactly this reason.
      boxVectors: Float64Array.from(this.boxVectors),
      volume: this.#volume(),
    };
  }

  /** Constraint convergence and the neighbour high-water mark. */
  async diagnostics() {
    const words = await readBuffer(this.device, this.allocator,
      this.resources.diagnostics, 16, Uint32Array);
    const required = (await readBuffer(this.device, this.allocator,
      this.resources.required, 4, Uint32Array))[0];
    return {
      maxShakeIterations: words[0],
      shakeHitIterationCap: words[1] === 1,
      rattleHitIterationCap: words[2] === 1,
      // 🔴 THE WORST PRE-CORRECTION RESIDUAL AT WHICH SOME CLUSTER STALLED,
      // which is not the same as the error in the final geometry and is larger.
      // It is measured on |d|^2/L^2 - 1 before that sweep's corrections are
      // applied, so it roughly doubles the length error and takes a maximum
      // over every cluster and every half-step. The authoritative number is the
      // geometry itself: measured 5.0e-6 of a bond length where this reports
      // 3.5e-5.
      worstStalledResidual: words[3] / 1e9,
      neighbourRequired: required,
      neighbourCapacity: this.plan.maxNeighbours,
      // The worst requirement seen at any block boundary of this run, not just
      // the last rebuild's - see the note beside worstNeighbourRequired.
      worstNeighbourRequired: this.worstNeighbourRequired,
      neighbourGrowths: this.neighbourGrowths ?? 0,
      // 🔴 `skinViolations > 0` MEANS THE RUN WAS WRONG FOR A WHILE, not that a
      // threshold was approached. It counts blocks over which some atom moved
      // far enough that the neighbour list could have been missing a pair
      // inside the cutoff. A caller that reports a trajectory without reading
      // this is reporting forces it has not checked.
      skin: this.plan.skin,
      worstDisplacement: this.worstDisplacement,
      worstSkinRatio: this.worstSkinRatio,
      skinViolations: this.skinViolations,
      stepsWithStaleList: this.stepsWithStaleList,
      rebuildInterval: this.rebuildInterval,
      requestedRebuildInterval: this.requestedRebuildInterval,
      rebuildShrinks: this.rebuildShrinks,
      rebuildGrowths: this.rebuildGrowths,
      // The fraction of the skin the worst block actually consumed. Under 1 is
      // exact; over 1 means some pair may have been missed, and the force that
      // pair would have contributed is bounded by the interaction strength AT
      // the cutoff - erfc(alpha*rc)/rc and the LJ tail, both small - so a
      // violation is a loss of exactness rather than an explosion. It is
      // reported, and corrected, rather than tolerated silently.
      worstSkinUsage: this.worstSkinUsage,
      adaptiveRebuild: this.adaptiveRebuild,
      // 🔴 WHAT A MISSED PAIR WOULD HAVE BEEN WORTH, so a violation has a
      // magnitude instead of only a flag. A pair that sneaks inside the cutoff
      // arrives AT the cutoff, where both interactions are nearly zero by
      // construction: the direct-space Coulomb force is screened by
      // erfc(alpha*rc) and the LJ force has fallen off as 1/r^7. This is that
      // force for a unit charge product, against an rms force of ~500
      // kJ/mol/nm on the test system - which is why a 1% overshoot is a loss
      // of exactness and a 95x one is a wrong answer.
      missedPairForceBound: this.#missedPairForce(),
      volume: this.#volume(),
      barostat: this.pressure === null ? null : {
        pressure: this.pressure,
        attempts: this.barostatAttempts,
        accepted: this.barostatAccepted,
        acceptanceRate: this.barostatAttempts === 0
          ? null : this.barostatAccepted / this.barostatAttempts,
        volumeScale: this.volumeScale,
        refusedForCutoff: this.barostatRefusals ?? 0,
      },
      constraintTolerance: this.tolerance,
      requestedTolerance: this.requestedTolerance,
      toleranceFloor: this.toleranceFloor,
      toleranceWasClamped: this.tolerance > this.requestedTolerance,
      allocator: this.allocator.snapshot(),
    };
  }

  release() {
    for (let index = this.held.length - 1; index >= 0; index -= 1) this.held[index].release();
    this.held = [];
    this.allocator.destroyPooled();
  }
}
