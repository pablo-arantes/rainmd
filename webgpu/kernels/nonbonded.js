/**
 * Direct-space Lennard-Jones and screened Coulomb, plus the exception pass.
 *
 * A transcription of cpu/forces/nonbonded.js, which is validated against
 * OpenMM to 2.7e-11 on the LJ energy and 1.2e-8 on Coulomb, and against its own
 * numerical gradient to 3.4e-9. The expressions must stay a transcription.
 *
 * 🔴 ONE THREAD PER ATOM, GATHERING, SO EACH PAIR IS COMPUTED TWICE AND NO
 * ATOMIC IS NEEDED. The alternative - one thread per pair, atomically adding
 * into both atoms' forces - halves the arithmetic and costs determinism: WGSL
 * has no f32 atomicAdd, so it would need fixed point, and fixed-point
 * accumulation in a nondeterministic order gives a different answer in the last
 * bits every run. Here each thread writes only its own atom's three
 * components, so the same step gives the same bits on every device, and the
 * differential test has a fixed value to compare against.
 *
 * 🔴 THE ENERGY IS HALVED AND THE FORCE IS NOT. Each pair appears in both
 * atoms' neighbour lists, so its ENERGY would be counted twice - hence the 0.5.
 * Its FORCE is not double-counted: atom i's loop computes the force on i, and
 * atom j's loop computes the force on j, which are different quantities.
 * Halving the force as well is the obvious mistake here and gives a force field
 * at half strength that still looks entirely plausible.
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";
import { PARAMS_STRUCT, GEOMETRY_FUNCTIONS, ERFC_FUNCTION } from "./wgsl-common.js";

export const NONBONDED_WORKGROUP = 64;

export function directSpaceShader(atomCount, typeCount, maxNeighbours, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const TYPE_COUNT: u32 = ${typeCount}u;
const MAX_NEIGHBOURS: u32 = ${maxNeighbours}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> neighbours: array<u32>;
@group(0) @binding(3) var<storage, read> neighbourCount: array<u32>;
@group(0) @binding(4) var<storage, read> charges: array<f32>;
@group(0) @binding(5) var<storage, read> ljTypes: array<u32>;
@group(0) @binding(6) var<storage, read> ljTable: array<f32>;
@group(0) @binding(7) var<storage, read_write> forces: array<f32>;
@group(0) @binding(8) var<storage, read_write> energies: array<f32>;
${GEOMETRY_FUNCTIONS}
${ERFC_FUNCTION}

@compute @workgroup_size(${NONBONDED_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${NONBONDED_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }

  let centre = atomPosition(atom);
  let charge = charges[atom];
  let typeRow = TYPE_COUNT * ljTypes[atom];
  let cutoff = params.scalars[PARAMS_CUTOFF];
  let cutoffSquared = cutoff * cutoff;
  let alpha = params.scalars[PARAMS_ALPHA];
  let coulombConstant = params.scalars[PARAMS_COULOMB];
  let count = neighbourCount[atom];
  let base = atom * MAX_NEIGHBOURS;

  var force = vec3<f32>(0.0, 0.0, 0.0);
  var lennardJones = 0.0;
  var coulomb = 0.0;

  for (var index = 0u; index < count; index = index + 1u) {
    let other = neighbours[base + index];
    let delta = minimumImage(centre - atomPosition(other));
    let rSquared = dot(delta, delta);
    // The list holds pairs out to cutoff + skin; the cutoff is applied here.
    if (rSquared > cutoffSquared) { continue; }
    let inverseR2 = 1.0 / rSquared;
    let inverseR = sqrt(inverseR2);
    let r = rSquared * inverseR;
    let inverseR6 = inverseR2 * inverseR2 * inverseR2;

    let pair = typeRow + ljTypes[other];
    let a = ljTable[pair * 2u];
    let b = ljTable[pair * 2u + 1u];
    lennardJones = lennardJones + 0.5 * (a * inverseR6 * inverseR6 - b * inverseR6);
    var scalar = (12.0 * a * inverseR6 * inverseR6 - 6.0 * b * inverseR6) * inverseR2;

    let qq = coulombConstant * charge * charges[other];
    let alphaR = alpha * r;
    let screened = erfcApprox(alphaR);
    coulomb = coulomb + 0.5 * qq * screened * inverseR;
    // 🔴 d/dr [erfc(ar)/r] HAS TWO TERMS. The second, 2a*exp(-a^2r^2)/sqrt(pi),
    // is the one a port forgets - and leaving it out keeps the ENERGY exactly
    // right while making the force wrong by the Gaussian part everywhere
    // inside the cutoff, so every energy comparison passes and the simulation
    // heats up.
    let gaussian = 2.0 * alpha * exp(-alphaR * alphaR) * INVERSE_SQRT_PI;
    scalar = scalar + qq * (screened * inverseR + gaussian) * inverseR2;

    force = force + scalar * delta;
  }

  // Accumulated so the bonded, direct-space, exception and PME passes can all
  // write into one force array; the caller clears it. See bonded.js.
  forces[atom * 3u] = forces[atom * 3u] + force.x;
  forces[atom * 3u + 1u] = forces[atom * 3u + 1u] + force.y;
  forces[atom * 3u + 2u] = forces[atom * 3u + 2u] + force.z;
  energies[atom] = lennardJones;
  energies[ATOM_COUNT + atom] = coulomb;
}
`;
}

/**
 * The exception pass: 1-4 interactions and the Ewald exclusion corrections.
 *
 * 🔴 NO CUTOFF TEST, DELIBERATELY. The reciprocal sum includes every pair, so
 * the correction has to be subtracted whether the pair is inside the cutoff or
 * not. Applying a cutoff here leaves the reciprocal contribution of any 1-4
 * pair that has stretched past it in the energy - rare, intermittent, and
 * therefore read as noise rather than as a bug.
 *
 * 🔴 AND THE CORRECTION USES THE ATOMS' FULL CHARGES, NOT THE EXCEPTION'S
 * SCALED PRODUCT. What is being removed is what the reciprocal sum actually put
 * in, which knows nothing about SCEE. The two products are carried separately
 * in the parameter block for exactly this reason.
 *
 * It ADDS to the forces the direct-space pass wrote. Safe without atomics
 * because each thread still owns one atom's three components.
 */
export function exceptionShader(atomCount, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> exceptionStart: array<u32>;
@group(0) @binding(3) var<storage, read> exceptionPartners: array<u32>;
@group(0) @binding(4) var<storage, read> exceptionParams: array<f32>;
@group(0) @binding(5) var<storage, read_write> forces: array<f32>;
@group(0) @binding(6) var<storage, read_write> energies: array<f32>;
${GEOMETRY_FUNCTIONS}
${ERFC_FUNCTION}

@compute @workgroup_size(${NONBONDED_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${NONBONDED_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let begin = exceptionStart[atom];
  let end = exceptionStart[atom + 1u];
  if (begin == end) { return; }

  let centre = atomPosition(atom);
  let alpha = params.scalars[PARAMS_ALPHA];
  let coulombConstant = params.scalars[PARAMS_COULOMB];
  var force = vec3<f32>(0.0, 0.0, 0.0);
  var lennardJones = 0.0;
  var coulomb = 0.0;

  for (var cursor = begin; cursor < end; cursor = cursor + 1u) {
    let other = exceptionPartners[cursor];
    let chargeProduct = exceptionParams[cursor * 4u];
    let a = exceptionParams[cursor * 4u + 1u];
    let b = exceptionParams[cursor * 4u + 2u];
    let fullProduct = exceptionParams[cursor * 4u + 3u];

    let delta = minimumImage(centre - atomPosition(other));
    let rSquared = dot(delta, delta);
    let inverseR2 = 1.0 / rSquared;
    let inverseR = sqrt(inverseR2);
    let r = rSquared * inverseR;
    let inverseR6 = inverseR2 * inverseR2 * inverseR2;
    var scalar = 0.0;

    if (a != 0.0 || b != 0.0) {
      lennardJones = lennardJones + 0.5 * (a * inverseR6 * inverseR6 - b * inverseR6);
      scalar = scalar + (12.0 * a * inverseR6 * inverseR6 - 6.0 * b * inverseR6) * inverseR2;
    }
    if (chargeProduct != 0.0) {
      let qq = coulombConstant * chargeProduct;
      coulomb = coulomb + 0.5 * qq * inverseR;
      scalar = scalar + qq * inverseR * inverseR2;
    }
    if (alpha > 0.0 && fullProduct != 0.0) {
      let qq = coulombConstant * fullProduct;
      let alphaR = alpha * r;
      let unscreened = 1.0 - erfcApprox(alphaR);
      coulomb = coulomb - 0.5 * qq * unscreened * inverseR;
      let gaussian = 2.0 * alpha * exp(-alphaR * alphaR) * INVERSE_SQRT_PI;
      scalar = scalar - qq * (unscreened * inverseR - gaussian) * inverseR2;
    }
    force = force + scalar * delta;
  }

  forces[atom * 3u] = forces[atom * 3u] + force.x;
  forces[atom * 3u + 1u] = forces[atom * 3u + 1u] + force.y;
  forces[atom * 3u + 2u] = forces[atom * 3u + 2u] + force.z;
  energies[atom] = energies[atom] + lennardJones;
  energies[ATOM_COUNT + atom] = energies[ATOM_COUNT + atom] + coulomb;
}
`;
}

export function encodeNonbonded(device, encoder, resources, atomCount, typeCount, plan) {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const workgroups = Math.ceil(atomCount / NONBONDED_WORKGROUP);
  const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
  const gridWidth = x === workgroups ? workgroups : maxPerDimension;
  const cache = pipelineCacheFor(device);

  const directCode = directSpaceShader(atomCount, typeCount, plan.maxNeighbours, gridWidth);
  const directKey = `nonbonded:direct:${atomCount}:${typeCount}:${plan.maxNeighbours}:${gridWidth}`;
  let pipeline = cache.get(directKey, directCode);
  let pass = encoder.beginComputePass({ label: directKey });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      resources.params, resources.positions, resources.neighbours, resources.neighbourCount,
      resources.charges, resources.ljTypes, resources.ljTable,
      resources.forces, resources.energies,
    ].map((allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
  }));
  pass.dispatchWorkgroups(x, y, z);
  pass.end();

  const exceptionCode = exceptionShader(atomCount, gridWidth);
  const exceptionKey = `nonbonded:exceptions:${atomCount}:${gridWidth}`;
  pipeline = cache.get(exceptionKey, exceptionCode);
  pass = encoder.beginComputePass({ label: exceptionKey });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      resources.params, resources.positions,
      resources.exceptionStart, resources.exceptionPartners, resources.exceptionParams,
      resources.forces, resources.energies,
    ].map((allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
  }));
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
}
