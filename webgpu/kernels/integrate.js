/**
 * BAOAB Langevin dynamics and the constraint solver, on the device.
 *
 * A transcription of cpu/integrate/langevin.js and cpu/integrate/constraints.js.
 * The step is B A O A [forces] B, which is one force evaluation per step and
 * two half-kicks around it.
 *
 * 🔴 `target` IS A RESERVED KEYWORD IN WGSL, AND IT IS AN EASY NAME TO REACH
 * FOR. It caught the FFT's output binding and then both constraint solvers'
 * natural name for a bond's rest length. The parser names the identifier, so it
 * is one line to fix - but only because DeferredValidation surfaces the message
 * instead of letting the pipeline silently fail to be created.
 *
 * 🔴 ONE THREAD PER ATOM, OR ONE THREAD PER CLUSTER. Nothing here uses an
 * atomic except the two diagnostic counters, because every thread writes only
 * the atoms it owns: an atom belongs to exactly one constraint cluster, and
 * clusters that share no atom can be solved in any order. So a step gives the
 * same bits every time, which is what makes a trajectory reproducible from its
 * seed and what lets a single step be compared against the host.
 *
 * 🔴 THE STEP NUMBER LIVES IN A STORAGE BUFFER, NOT A UNIFORM. The thermostat
 * needs it to index the noise stream, and a uniform cannot be rewritten between
 * passes inside one command buffer - which is exactly what encoding a block of
 * steps per submit requires. A one-thread kernel increments it instead, so a
 * hundred steps can be encoded and submitted together with no host round trip.
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";

export const INTEGRATE_WORKGROUP = 64;

/**
 * The integrator's parameters.
 *
 * 🔴 PADDED TO 32 BYTES. A uniform struct's size must be a multiple of its
 * alignment, and getting the host layout out of step with the shader's does not
 * fail validation - it reads the next field's bytes as a friction coefficient.
 */
export const INTEGRATOR_STRUCT = `
struct Integrator {
  halfStep: f32,
  decay: f32,
  noiseAmplitude: f32,
  kT: f32,
  tolerance: f32,
  seed: u32,
  maxIterations: u32,
  reserved: u32,
};
`;
export const INTEGRATOR_FLOATS = 8;

export function packIntegrator(options) {
  const buffer = new ArrayBuffer(INTEGRATOR_FLOATS * 4);
  const floats = new Float32Array(buffer);
  const words = new Uint32Array(buffer);
  floats[0] = options.halfStep;
  floats[1] = options.decay;
  floats[2] = options.noiseAmplitude;
  floats[3] = options.kT;
  floats[4] = options.tolerance;
  words[5] = options.seed >>> 0;
  words[6] = options.maxIterations;
  words[7] = 0;
  return new Uint8Array(buffer);
}

/**
 * The counter-based normal generator, identical to shared/random.js.
 *
 * 🔴 THE UNIFORM SHIFTS BY NINE AND THE HOST DOES TOO. `(state >> 9) + 0.5` is
 * below 2^23 + 1 and therefore exact in f32 as well as f64, so the host and the
 * device draw the SAME uniform from the same counter - bit for bit, verified on
 * 40000 draws. Shifting by eight instead agrees on only half of them, because
 * `integer + 0.5` needs 25 significand bits above 2^23 and f32 has 24. Without
 * that agreement a thermostat comparison cannot distinguish a wrong decay
 * constant from a rounding mode.
 *
 * What still differs between the lanes is `log`, `cos` and `sin`, which WGSL
 * does not specify to the last bit - so the NORMALS agree to about 1e-7 rather
 * than exactly. That is why the integrator check supplies its own noise.
 */
export const RANDOM_FUNCTIONS = `
const TWO_PI: f32 = 6.283185307179586;
const UNIFORM_SCALE: f32 = 1.1920928955078125e-7;

fn rainmdMix(value: u32) -> u32 {
  var state = value;
  state = state ^ (state >> 16u);
  state = state * 0x7feb352du;
  state = state ^ (state >> 15u);
  state = state * 0x846ca68bu;
  state = state ^ (state >> 16u);
  return state;
}

fn rainmdUniform(seed: u32, a: u32, b: u32, c: u32) -> f32 {
  var state = rainmdMix(seed ^ (a * 0x9e3779b1u));
  state = rainmdMix(state ^ (b * 0x85ebca6bu));
  state = rainmdMix(state ^ (c * 0xc2b2ae35u));
  return (f32(state >> 9u) + 0.5) * UNIFORM_SCALE;
}

fn rainmdGaussian(seed: u32, atom: u32, step: u32) -> vec3<f32> {
  let u1 = rainmdUniform(seed, atom, step, 0u);
  let u2 = rainmdUniform(seed, atom, step, 1u);
  let u3 = rainmdUniform(seed, atom, step, 2u);
  let u4 = rainmdUniform(seed, atom, step, 3u);
  let radius1 = sqrt(-2.0 * log(u1));
  let radius2 = sqrt(-2.0 * log(u3));
  return vec3<f32>(
    radius1 * cos(TWO_PI * u2),
    radius1 * sin(TWO_PI * u2),
    radius2 * cos(TWO_PI * u4));
}
`;

function preamble(atomCount, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
${INTEGRATOR_STRUCT}`;
}

/** B: a half kick. */
export function kickShader(atomCount, gridWidth) {
  return `${preamble(atomCount, gridWidth)}
@group(0) @binding(0) var<uniform> integrator: Integrator;
@group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> forces: array<f32>;
@group(0) @binding(3) var<storage, read> masses: array<f32>;
@compute @workgroup_size(${INTEGRATE_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${INTEGRATE_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let mass = masses[atom];
  // 🔴 A ZERO-MASS ATOM IS SKIPPED, NOT DIVIDED BY. AMBER topologies really do
  // carry massless extra points, and f/0 is an Infinity that the next drift
  // turns into a NaN for every atom within two steps.
  if (mass <= 0.0) { return; }
  let scale = integrator.halfStep / mass;
  velocities[atom * 3u] = velocities[atom * 3u] + scale * forces[atom * 3u];
  velocities[atom * 3u + 1u] = velocities[atom * 3u + 1u] + scale * forces[atom * 3u + 1u];
  velocities[atom * 3u + 2u] = velocities[atom * 3u + 2u] + scale * forces[atom * 3u + 2u];
}
`;
}

/**
 * A: a half drift, keeping where every atom came from.
 *
 * 🔴 THE REFERENCE IS WHAT MAKES SHAKE A RIGID MOTION. The projection is taken
 * along the OLD bond direction, so the pre-drift positions have to survive into
 * the solver - projecting along the new direction instead makes the correction
 * slide the cluster rather than rotate it, and the constraint converges to a
 * slightly wrong geometry.
 */
export function driftShader(atomCount, gridWidth) {
  return `${preamble(atomCount, gridWidth)}
@group(0) @binding(0) var<uniform> integrator: Integrator;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> reference: array<f32>;
@group(0) @binding(3) var<storage, read> velocities: array<f32>;
@compute @workgroup_size(${INTEGRATE_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let component = id.x + id.y * GRID_WIDTH * ${INTEGRATE_WORKGROUP}u;
  if (component >= ATOM_COUNT * 3u) { return; }
  let current = positions[component];
  reference[component] = current;
  positions[component] = current + integrator.halfStep * velocities[component];
}
`;
}

/**
 * SHAKE: put the constrained lengths back, then read the velocity off the
 * motion that actually happened.
 *
 * 🔴 THE VELOCITY IS REWRITTEN ONLY FOR ATOMS IN A CLUSTER. For an
 * unconstrained atom `(x - reference)/h` is exactly `v` in exact arithmetic, so
 * recomputing it buys nothing and costs a division's worth of rounding. The
 * host reference does the same, which is what lets the two be compared.
 *
 * 🔴 AND THE CONSTRAINT VECTOR IS NOT MINIMUM-IMAGED. A constraint is always
 * within one molecule, and this engine never wraps positions during a run -
 * imaging here would be harmless today and wrong the moment someone wraps.
 */
export function shakeShader(atomCount, layout, gridWidth) {
  return `${preamble(atomCount, gridWidth)}
const CLUSTER_COUNT: u32 = ${layout.clusterCount}u;
const CONSTRAINT_START_BASE: u32 = ${layout.constraintStartBase}u;
const CONSTRAINT_LIST_BASE: u32 = ${layout.constraintListBase}u;
const ATOM_START_BASE: u32 = ${layout.atomStartBase}u;
const ATOM_LIST_BASE: u32 = ${layout.atomListBase}u;
@group(0) @binding(0) var<uniform> integrator: Integrator;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read> reference: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(4) var<storage, read> clusterData: array<u32>;
@group(0) @binding(5) var<storage, read> constraintAtoms: array<u32>;
@group(0) @binding(6) var<storage, read> constraintParams: array<f32>;
@group(0) @binding(7) var<storage, read_write> diagnostics: array<atomic<u32>>;
// 🔴 THE CONSTRAINT CORRECTION, KEPT SEPARATELY, AND IT IS A CONDITIONING FIX
// RATHER THAN A CHANGE OF ALGORITHM. The velocity used to be recovered as
// (positions - reference) / halfStep, which is algebraically exactly
// v + correction / halfStep - but computes a ~1e-3 nm displacement by
// subtracting two coordinates that span the box, so its absolute f32 error is
// span * eps however small the answer is. Divided by the half step that is a
// 2.4e-4 nm/ps kick, every step, on every constrained atom - against a
// hydrogen's thermal 1.6 nm/ps. Measured: NVE drift rose 29x per halving of the
// timestep with constraints on and was flat with them off, and the f64 host
// shows drift FALLING as the timestep falls. Accumulating the correction keeps
// a ~1e-5 nm number in its own float, where it has relative precision: the
// per-step velocity error falls from 5.89e-5 to 7.23e-9 nm/ps on this box, and
// the gain grows with the box because the old error scales with the span.
@group(0) @binding(8) var<storage, read_write> corrections: array<f32>;

fn loadVector(source: ptr<storage, array<f32>, read_write>, atom: u32) -> vec3<f32> {
  return vec3<f32>((*source)[atom * 3u], (*source)[atom * 3u + 1u], (*source)[atom * 3u + 2u]);
}

@compute @workgroup_size(${INTEGRATE_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let cluster = id.x + id.y * GRID_WIDTH * ${INTEGRATE_WORKGROUP}u;
  if (cluster >= CLUSTER_COUNT) { return; }
  let constraintBegin = clusterData[CONSTRAINT_START_BASE + cluster];
  let constraintEnd = clusterData[CONSTRAINT_START_BASE + cluster + 1u];

  // The accumulator is per drift, so this thread clears its own atoms first.
  // There are two drifts per BAOAB step and each needs its own correction.
  {
    let zeroBegin = clusterData[ATOM_START_BASE + cluster];
    let zeroEnd = clusterData[ATOM_START_BASE + cluster + 1u];
    for (var slot = zeroBegin; slot < zeroEnd; slot = slot + 1u) {
      let atom = clusterData[ATOM_LIST_BASE + slot];
      corrections[atom * 3u] = 0.0;
      corrections[atom * 3u + 1u] = 0.0;
      corrections[atom * 3u + 2u] = 0.0;
    }
  }

  var iterations = 0u;
  var converged = false;
  var stalled = false;
  var achieved = 0.0;
  // 🔴 A STALL BREAK, BECAUSE A TOLERANCE BELOW THE f32 FLOOR IS REACHABLE
  // ONLY BY ASKING FOR IT. SHAKE converges roughly linearly, so once the
  // residual stops shrinking it will never shrink again - and without this the
  // solver burns every one of its 150 iterations per cluster per half-step on a
  // result that was already as good as f32 allows.
  var previousWorst = 1e30;
  for (var iteration = 0u; iteration < integrator.maxIterations; iteration = iteration + 1u) {
    var worst = 0.0;
    for (var slot = constraintBegin; slot < constraintEnd; slot = slot + 1u) {
      let k = clusterData[CONSTRAINT_LIST_BASE + slot];
      let a = constraintAtoms[k * 2u];
      let b = constraintAtoms[k * 2u + 1u];
      let restLength = constraintParams[k * 4u];
      let inverseA = constraintParams[k * 4u + 1u];
      let inverseB = constraintParams[k * 4u + 2u];
      let delta = vec3<f32>(
        positions[a * 3u] - positions[b * 3u],
        positions[a * 3u + 1u] - positions[b * 3u + 1u],
        positions[a * 3u + 2u] - positions[b * 3u + 2u]);
      let restSquared = restLength * restLength;
      let error = dot(delta, delta) - restSquared;
      let relative = abs(error) / restSquared;
      worst = max(worst, relative);
      if (relative <= integrator.tolerance) { continue; }
      let old = vec3<f32>(
        reference[a * 3u] - reference[b * 3u],
        reference[a * 3u + 1u] - reference[b * 3u + 1u],
        reference[a * 3u + 2u] - reference[b * 3u + 2u]);
      let projection = dot(delta, old);
      let denominator = 2.0 * (inverseA + inverseB) * projection;
      if (abs(denominator) < 1e-20) { continue; }
      let lambda = -error / denominator;
      let shiftA = lambda * inverseA * old;
      let shiftB = lambda * inverseB * old;
      positions[a * 3u] = positions[a * 3u] + shiftA.x;
      positions[a * 3u + 1u] = positions[a * 3u + 1u] + shiftA.y;
      positions[a * 3u + 2u] = positions[a * 3u + 2u] + shiftA.z;
      positions[b * 3u] = positions[b * 3u] - shiftB.x;
      positions[b * 3u + 1u] = positions[b * 3u + 1u] - shiftB.y;
      positions[b * 3u + 2u] = positions[b * 3u + 2u] - shiftB.z;
      corrections[a * 3u] = corrections[a * 3u] + shiftA.x;
      corrections[a * 3u + 1u] = corrections[a * 3u + 1u] + shiftA.y;
      corrections[a * 3u + 2u] = corrections[a * 3u + 2u] + shiftA.z;
      corrections[b * 3u] = corrections[b * 3u] - shiftB.x;
      corrections[b * 3u + 1u] = corrections[b * 3u + 1u] - shiftB.y;
      corrections[b * 3u + 2u] = corrections[b * 3u + 2u] - shiftB.z;
    }
    iterations = iteration + 1u;
    achieved = worst;
    if (worst <= integrator.tolerance) { converged = true; break; }
    // 🔴 A STALL IS NOT A FAILURE AND MUST NOT BE REPORTED AS ONE. Once the
    // residual stops shrinking it never shrinks again, and if the tolerance is
    // near the f32 floor the solver will stall just above it having already
    // produced the best geometry f32 can represent. Treating that as
    // non-convergence flagged a correct result on every one of 394 clusters.
    // Only running out of iterations is a genuine failure.
    if (iteration >= 2u && worst >= previousWorst * 0.99) { stalled = true; break; }
    previousWorst = worst;
  }

  let inverseStep = 1.0 / integrator.halfStep;
  let atomBegin = clusterData[ATOM_START_BASE + cluster];
  let atomEnd = clusterData[ATOM_START_BASE + cluster + 1u];
  // 🔴 ONE THREAD OWNS A CLUSTER, SO NO ATOMICS ARE NEEDED HERE. A cluster is a
  // connected set of constraints and no atom belongs to two clusters, so this
  // thread is the only writer of these components - which is also why the loop
  // below is over UNIQUE atoms and the host's equivalent had to be taught to
  // deduplicate: iterating constraints instead touches a water oxygen twice,
  // harmless when the velocity was overwritten and a factor-of-two error the
  // moment the update becomes an addition.
  for (var slot = atomBegin; slot < atomEnd; slot = slot + 1u) {
    let atom = clusterData[ATOM_LIST_BASE + slot];
    velocities[atom * 3u] = velocities[atom * 3u] + corrections[atom * 3u] * inverseStep;
    velocities[atom * 3u + 1u] = velocities[atom * 3u + 1u] + corrections[atom * 3u + 1u] * inverseStep;
    velocities[atom * 3u + 2u] = velocities[atom * 3u + 2u] + corrections[atom * 3u + 2u] * inverseStep;
  }

  // 🔴 NON-CONVERGENCE MUST REACH THE HOST. A cluster whose atoms have been
  // pushed through each other has no solution, and silently stopping after the
  // iteration cap produces a trajectory that is not physics. These two counters
  // are the only atomics in the integrator and they touch no simulation state.
  atomicMax(&diagnostics[0], iterations);
  // Only the iteration cap counts as failure; a stall is reported through the
  // achieved residual instead, in fixed point so atomicMax can carry it.
  if (!converged && !stalled) { atomicMax(&diagnostics[1], 1u); }
  atomicMax(&diagnostics[3], u32(min(achieved * 1e9, 4.0e9)));
}
`;
}

/** RATTLE: remove the velocity component along every constrained bond. */
export function rattleShader(atomCount, layout, gridWidth) {
  return `${preamble(atomCount, gridWidth)}
const CLUSTER_COUNT: u32 = ${layout.clusterCount}u;
const CONSTRAINT_START_BASE: u32 = ${layout.constraintStartBase}u;
const CONSTRAINT_LIST_BASE: u32 = ${layout.constraintListBase}u;
@group(0) @binding(0) var<uniform> integrator: Integrator;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(3) var<storage, read> clusterData: array<u32>;
@group(0) @binding(4) var<storage, read> constraintAtoms: array<u32>;
@group(0) @binding(5) var<storage, read> constraintParams: array<f32>;
@group(0) @binding(6) var<storage, read_write> diagnostics: array<atomic<u32>>;

@compute @workgroup_size(${INTEGRATE_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let cluster = id.x + id.y * GRID_WIDTH * ${INTEGRATE_WORKGROUP}u;
  if (cluster >= CLUSTER_COUNT) { return; }
  let begin = clusterData[CONSTRAINT_START_BASE + cluster];
  let end = clusterData[CONSTRAINT_START_BASE + cluster + 1u];

  var converged = false;
  var stalled = false;
  var previousWorst = 1e30;
  for (var iteration = 0u; iteration < integrator.maxIterations; iteration = iteration + 1u) {
    var worst = 0.0;
    for (var slot = begin; slot < end; slot = slot + 1u) {
      let k = clusterData[CONSTRAINT_LIST_BASE + slot];
      let a = constraintAtoms[k * 2u];
      let b = constraintAtoms[k * 2u + 1u];
      let restLength = constraintParams[k * 4u];
      let inverseA = constraintParams[k * 4u + 1u];
      let inverseB = constraintParams[k * 4u + 2u];
      let bond = vec3<f32>(
        positions[a * 3u] - positions[b * 3u],
        positions[a * 3u + 1u] - positions[b * 3u + 1u],
        positions[a * 3u + 2u] - positions[b * 3u + 2u]);
      let relative = vec3<f32>(
        velocities[a * 3u] - velocities[b * 3u],
        velocities[a * 3u + 1u] - velocities[b * 3u + 1u],
        velocities[a * 3u + 2u] - velocities[b * 3u + 2u]);
      let projection = dot(bond, relative);
      let scaled = abs(projection) / (restLength * restLength);
      worst = max(worst, scaled);
      if (scaled <= integrator.tolerance) { continue; }
      let denominator = (inverseA + inverseB) * dot(bond, bond);
      if (denominator < 1e-20) { continue; }
      let lambda = -projection / denominator;
      velocities[a * 3u] = velocities[a * 3u] + lambda * inverseA * bond.x;
      velocities[a * 3u + 1u] = velocities[a * 3u + 1u] + lambda * inverseA * bond.y;
      velocities[a * 3u + 2u] = velocities[a * 3u + 2u] + lambda * inverseA * bond.z;
      velocities[b * 3u] = velocities[b * 3u] - lambda * inverseB * bond.x;
      velocities[b * 3u + 1u] = velocities[b * 3u + 1u] - lambda * inverseB * bond.y;
      velocities[b * 3u + 2u] = velocities[b * 3u + 2u] - lambda * inverseB * bond.z;
    }
    if (worst <= integrator.tolerance) { converged = true; break; }
    if (iteration >= 2u && worst >= previousWorst * 0.99) { stalled = true; break; }
    previousWorst = worst;
  }
  if (!converged && !stalled) { atomicMax(&diagnostics[2], 1u); }
}
`;
}

/**
 * O: the Ornstein-Uhlenbeck step.
 *
 * 🔴 THE NOISE AMPLITUDE CARRIES 1/sqrt(m), SO A MASSLESS ATOM IS SKIPPED.
 * `sqrt(kT/0)` is an Infinity that the next drift spreads to the whole system.
 */
export function thermostatShader(atomCount, gridWidth) {
  return `${preamble(atomCount, gridWidth)}
@group(0) @binding(0) var<uniform> integrator: Integrator;
@group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> masses: array<f32>;
@group(0) @binding(3) var<storage, read> stepCounter: array<u32>;
@group(0) @binding(4) var<storage, read> noise: array<f32>;
${RANDOM_FUNCTIONS}
@compute @workgroup_size(${INTEGRATE_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${INTEGRATE_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let mass = masses[atom];
  if (mass <= 0.0) { return; }
  // 🔴 A SUPPLIED-NOISE PATH, FOR THE DIFFERENTIAL TEST ONLY. log, cos and sin
  // are not specified to the last bit in WGSL, so device normals differ from
  // host normals at ~1e-7 even with an identical counter. Reading the noise
  // from a buffer lets the integrator's ARITHMETIC be compared exactly; a
  // production run sets stepCounter[1] to zero and generates it here.
  var gaussian: vec3<f32>;
  if (stepCounter[1u] == 1u) {
    gaussian = vec3<f32>(noise[atom * 3u], noise[atom * 3u + 1u], noise[atom * 3u + 2u]);
  } else {
    gaussian = rainmdGaussian(integrator.seed, atom, stepCounter[0u]);
  }
  let amplitude = integrator.noiseAmplitude * sqrt(integrator.kT / mass);
  velocities[atom * 3u] = integrator.decay * velocities[atom * 3u] + amplitude * gaussian.x;
  velocities[atom * 3u + 1u] = integrator.decay * velocities[atom * 3u + 1u] + amplitude * gaussian.y;
  velocities[atom * 3u + 2u] = integrator.decay * velocities[atom * 3u + 2u] + amplitude * gaussian.z;
}
`;
}

/** Per-atom kinetic energy, for the existing reduction kernel to sum. */
export function kineticShader(atomCount, gridWidth) {
  return `${preamble(atomCount, gridWidth)}
@group(0) @binding(0) var<storage, read> velocities: array<f32>;
@group(0) @binding(1) var<storage, read> masses: array<f32>;
@group(0) @binding(2) var<storage, read_write> kinetic: array<f32>;
@compute @workgroup_size(${INTEGRATE_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${INTEGRATE_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let v = vec3<f32>(velocities[atom * 3u], velocities[atom * 3u + 1u], velocities[atom * 3u + 2u]);
  kinetic[atom] = 0.5 * masses[atom] * dot(v, v);
}
`;
}

/** One thread, advancing the noise stream's counter. */
export function stepCounterShader() {
  return `
@group(0) @binding(0) var<storage, read_write> stepCounter: array<u32>;
@compute @workgroup_size(1)
fn main() {
  stepCounter[0u] = stepCounter[0u] + 1u;
}
`;
}

/** Encode one BAOAB step, given a callback that encodes the force evaluation. */
export function encodeLangevinStep(device, encoder, resources, atomCount, layout, encodeForces, options = {}) {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const cache = pipelineCacheFor(device);
  const dispatchFor = (items) => {
    const workgroups = Math.ceil(items / INTEGRATE_WORKGROUP);
    const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
    return { x, y, z, gridWidth: x === workgroups ? workgroups : maxPerDimension };
  };
  const atoms = dispatchFor(atomCount);
  const components = dispatchFor(atomCount * 3);
  const clusters = dispatchFor(Math.max(1, layout.clusterCount));

  const run = (key, code, bindings, dispatch) => {
    const pipeline = cache.get(key, code);
    const pass = encoder.beginComputePass({ label: key });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((allocation, binding) => ({
        binding, resource: { buffer: allocation.buffer },
      })),
    }));
    pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    pass.end();
  };

  const kick = () => run(`integrate:kick:${atomCount}:${atoms.gridWidth}`,
    kickShader(atomCount, atoms.gridWidth),
    [resources.integrator, resources.velocities, resources.forces, resources.masses], atoms);
  const drift = () => run(`integrate:drift:${atomCount}:${components.gridWidth}`,
    driftShader(atomCount, components.gridWidth),
    [resources.integrator, resources.positions, resources.reference, resources.velocities],
    components);
  const shake = () => {
    if (layout.clusterCount === 0) return;
    run(`integrate:shake:${atomCount}:${layout.clusterCount}:${clusters.gridWidth}`,
      shakeShader(atomCount, layout, clusters.gridWidth),
      [resources.integrator, resources.positions, resources.reference, resources.velocities,
        resources.clusterData, resources.constraintAtoms, resources.constraintParams,
        resources.diagnostics, resources.corrections], clusters);
  };
  const rattle = () => {
    if (layout.clusterCount === 0) return;
    run(`integrate:rattle:${atomCount}:${layout.clusterCount}:${clusters.gridWidth}`,
      rattleShader(atomCount, layout, clusters.gridWidth),
      [resources.integrator, resources.positions, resources.velocities,
        resources.clusterData, resources.constraintAtoms, resources.constraintParams,
        resources.diagnostics], clusters);
  };
  const thermostat = () => run(`integrate:thermostat:${atomCount}:${atoms.gridWidth}`,
    thermostatShader(atomCount, atoms.gridWidth),
    [resources.integrator, resources.velocities, resources.masses,
      resources.stepCounter, resources.noise], atoms);

  // B A O A [forces] B, each half-kick and each drift followed by its
  // constraint projection - the same order as cpu/integrate/langevin.js.
  kick(); rattle();
  drift(); shake();
  thermostat(); rattle();
  drift(); shake();
  // 🔴 THE KINETIC SAMPLE IS TAKEN HERE, WHICH IS WHERE OPENMM TAKES IT. Its
  // LangevinMiddleIntegrator ends a step at the position update and the
  // velocity projection; this integrator then applies a closing half kick, so a
  // sample read after the step is one half kick later and its O(dt^2) bias has
  // the opposite sign. Measured by moving nothing but the sampling point, in
  // f64 on the identical trajectory: the dt dependence of the reported
  // temperature went from -37.89 K to +3.68 K over 1 -> 4 fs, against OpenMM's
  // own +3.01 K.
  //
  // 🔴 AND IT MUST BE SAMPLED IN THE STEP RATHER THAN RECONSTRUCTED AFTER IT.
  // The two velocities differ by (halfStep/m)*F, so subtracting that from the
  // post-step value looks equivalent - and on a 12-constraint vacuum fixture it
  // agrees to 0.2-0.8 K. It is wrong on a real system: the closing kick is
  // followed by RATTLE, so v_post = RATTLE(v_pre + (halfStep/m)*F) and
  // subtracting the kick does not invert the projection, it adds back the
  // bond-directed velocity RATTLE removed. On the solvated fixture, where
  // constraints are a third of 3N, that reconstruction reported +86.85 K at
  // 4 fs against a true +4.45. Hence this pass.
  if (options.sampleKinetic === true) {
    // 🔴 ITS OWN BUFFER, BECAUSE BOTH SAMPLES ARE WANTED. The TOTAL energy needs
    // the post-step velocity: pairing a pre-kick velocity with an end-of-step
    // position is a half-step mismatch, and doing so inflated the measured NVE
    // energy fluctuation from 6.36 to 15.51 kJ/mol - a 2.4x spurious
    // oscillation that reads as drift. So the temperature comes from here and
    // the conserved quantity keeps coming from the post-step sample.
    encodeKineticEnergy(device, encoder, resources, atomCount, "kineticEarly");
  }
  encodeForces();
  kick(); rattle();
  run("integrate:step-counter", stepCounterShader(), [resources.stepCounter],
    { x: 1, y: 1, z: 1 });
}

/** Encode the kinetic-energy pass; the caller reduces it. */
export function encodeKineticEnergy(device, encoder, resources, atomCount,
  target = "kinetic") {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const workgroups = Math.ceil(atomCount / INTEGRATE_WORKGROUP);
  const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
  const gridWidth = x === workgroups ? workgroups : maxPerDimension;
  const key = `integrate:kinetic:${atomCount}:${gridWidth}`;
  const pipeline = pipelineCacheFor(device).get(key, kineticShader(atomCount, gridWidth));
  const pass = encoder.beginComputePass({ label: key });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [resources.velocities, resources.masses, resources[target]]
      .map((allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
  }));
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
}

/**
 * Per-atom force magnitude, for the minimiser's convergence test.
 *
 * 🔴 THE MAXIMUM, NOT THE NORM. What decides whether a structure is safe to
 * start dynamics from is the single worst force in it - one clashing pair at
 * 1e6 kJ/mol/nm will blow up the first step however small the rms is. A
 * root-mean-square over a solvated system divides that spike by the thousands
 * of well-behaved atoms around it and reports a comfortable number.
 */
export function forceMagnitudeShader(atomCount, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
@group(0) @binding(0) var<storage, read> forces: array<f32>;
@group(0) @binding(1) var<storage, read_write> magnitudes: array<f32>;
@compute @workgroup_size(${INTEGRATE_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${INTEGRATE_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let f = vec3<f32>(forces[atom * 3u], forces[atom * 3u + 1u], forces[atom * 3u + 2u]);
  magnitudes[atom] = length(f);
}
`;
}

/** A tree maximum, mirroring the sum reduction in pme.js. */
export function maxReduceShader(count, gridWidth, workgroup = 256) {
  return `
const COUNT: u32 = ${count}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
var<workgroup> scratch: array<f32, ${workgroup}>;
@compute @workgroup_size(${workgroup})
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * ${workgroup}u;
  var value = 0.0;
  if (index < COUNT) { value = source[index]; }
  scratch[local.x] = value;
  workgroupBarrier();
  for (var stride = ${workgroup / 2}u; stride > 0u; stride = stride >> 1u) {
    if (local.x < stride) {
      scratch[local.x] = max(scratch[local.x], scratch[local.x + stride]);
    }
    workgroupBarrier();
  }
  if (local.x == 0u) { partials[group.x + group.y * GRID_WIDTH] = scratch[0]; }
}
`;
}

/**
 * Move every atom along the force by a scaled step.
 *
 * 🔴 THE STEP IS CLAMPED PER ATOM, NOT ONLY SCALED GLOBALLY. A single clashing
 * pair can carry a force six orders of magnitude above the rest, and a step
 * sized to move it safely moves everything else by nothing at all - so
 * minimisation stalls. Capping the displacement lets the clash relax over
 * several iterations while the rest of the system keeps making progress.
 */
export function displaceShader(atomCount, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
struct Step { scale: f32, maxDisplacement: f32, reserved0: f32, reserved1: f32 };
@group(0) @binding(0) var<uniform> stepping: Step;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read> forces: array<f32>;
@compute @workgroup_size(${INTEGRATE_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${INTEGRATE_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let f = vec3<f32>(forces[atom * 3u], forces[atom * 3u + 1u], forces[atom * 3u + 2u]);
  var shift = stepping.scale * f;
  let distance = length(shift);
  if (distance > stepping.maxDisplacement) {
    shift = shift * (stepping.maxDisplacement / distance);
  }
  positions[atom * 3u] = positions[atom * 3u] + shift.x;
  positions[atom * 3u + 1u] = positions[atom * 3u + 1u] + shift.y;
  positions[atom * 3u + 2u] = positions[atom * 3u + 2u] + shift.z;
}
`;
}

/** SHAKE alone, for the minimiser, which moves positions without a drift. */
export function encodeConstrainPositions(device, encoder, resources, atomCount, layout) {
  if (layout.clusterCount === 0) return;
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const workgroups = Math.ceil(layout.clusterCount / INTEGRATE_WORKGROUP);
  const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
  const gridWidth = x === workgroups ? workgroups : maxPerDimension;
  const key = `integrate:shake:${atomCount}:${layout.clusterCount}:${gridWidth}`;
  const pipeline = pipelineCacheFor(device).get(key,
    shakeShader(atomCount, layout, gridWidth));
  const pass = encoder.beginComputePass({ label: `${key}:minimise` });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [resources.integrator, resources.positions, resources.reference,
      resources.velocities, resources.clusterData, resources.constraintAtoms,
      resources.constraintParams, resources.diagnostics, resources.corrections]
      .map((allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
  }));
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
}
